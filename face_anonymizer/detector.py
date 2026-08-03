"""YOLO-FaceV2 기반 얼굴 검출기.

YOLO-FaceV2 (Krasjet-Yu, clibdev fork) 는 YOLOv5 계열이라 자체 model/utils 코드가
필요하다. 체크포인트를 unpickle 하려면 해당 리포가 sys.path 에 있어야 하므로,
setup_weights.py 로 third_party/YOLO-FaceV2 에 클론해 둔 뒤 사용한다.

블러/모자이크 용도에는 박스만 있으면 되므로, 리포별로 이름이 다른 face-NMS 함수에
의존하지 않고 letterbox + torchvision NMS 로 직접 디코딩한다(랜드마크 열은 무시).

이 모듈은 torch 를 임포트한다. torch 없이도 파이프라인 로직을 테스트할 수 있도록
좌표 변환은 geometry.py 로, 픽셀 처리는 anonymize.py 로 분리해 두었다.
"""

import logging
import os
import sys

import numpy as np
import torch
import torchvision

from .geometry import letterbox, unletterbox

log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.abspath(os.path.join(_HERE, "..", "third_party", "YOLO-FaceV2"))
DEFAULT_WEIGHTS = os.path.abspath(os.path.join(_HERE, "..", "weights", "yolo-facev2.pt"))


class FaceDetector:
    """YOLO-FaceV2 얼굴 검출기.

    Parameters
    ----------
    weights : str
        .pt 가중치 경로.
    repo_path : str
        YOLO-FaceV2 리포 경로 (model/utils 코드 로드용).
    device : str | None
        'cuda:0' / 'cpu'. None 이면 자동 감지.
    half : bool | None
        FP16 추론. None 이면 CUDA 에서만 자동 활성화. CPU 에서는 항상 무시된다.
    imgsz, conf, iou : 추론 기본값. detect() 호출 시 개별 오버라이드 가능.
    """

    def __init__(self, weights=DEFAULT_WEIGHTS, repo_path=DEFAULT_REPO,
                 device=None, half=None, imgsz=960, conf=0.25, iou=0.45):
        repo_path = os.path.abspath(repo_path)
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(
                f"YOLO-FaceV2 repo not found at {repo_path}. "
                "Run `python setup_weights.py` first."
            )
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"weights not found at {weights}. Run `python setup_weights.py` first."
            )
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from models.experimental import attempt_load  # noqa: E402 (repo 로드 후)

        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # torch>=2.6 은 torch.load 의 weights_only 기본값이 True 라 YOLOv5 형식
        # (모델 객체를 담은) 체크포인트 로드가 실패한다. 체크포인트 로드 구간에서만
        # False 로 강제하고, 끝나면 finally 로 원래 torch.load 를 반드시 되돌린다.
        # (전역 상태를 영구히 오염시키지 않기 위함)
        _orig_load = torch.load
        torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
        try:
            model = None
            for kwargs in ({"device": self.device}, {"map_location": self.device}, {}):
                try:
                    model = attempt_load(weights, **kwargs)
                    break
                except TypeError:
                    continue
        finally:
            torch.load = _orig_load
        if model is None:
            raise RuntimeError("attempt_load() signature mismatch for this repo version.")

        self.model = model.to(self.device).eval()

        # FP16 은 CUDA 에서만 의미가 있다. CPU 에서 half 를 켜면 오히려 느리거나
        # 커널이 없어 죽으므로 강제로 끈다.
        self.half = bool(half if half is not None else self.device.type == "cuda")
        if self.half and self.device.type != "cuda":
            self.half = False
        if self.half:
            self.model.half()

        self.stride = int(self.model.stride.max()) if hasattr(self.model, "stride") else 32
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        log.info("FaceDetector ready: device=%s half=%s imgsz=%s",
                 self.device, self.half, self.imgsz)

    # ------------------------------------------------------------------ #

    def warmup(self, imgsz=None, batch_size=1):
        """더미 입력으로 1회 추론해 CUDA 커널/메모리를 미리 잡아 둔다.

        서버로 띄울 때 첫 요청만 유독 느려지는 것을 막는다.
        """
        imgsz = imgsz or self.imgsz
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.detect_batch([dummy] * batch_size)
        return self

    @torch.no_grad()
    def detect(self, frame, imgsz=None, conf=None, iou=None):
        """단일 BGR 프레임 → [(x1, y1, x2, y2, score), ...] (원본 좌표계)."""
        return self.detect_batch([frame], imgsz=imgsz, conf=conf, iou=iou)[0]

    @torch.no_grad()
    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        """BGR 프레임 리스트 → 프레임별 검출 리스트.

        같은 영상의 프레임은 크기가 같으므로 letterbox 파라미터를 공유해
        한 번의 forward 로 처리한다. GPU 에서 프레임 단위 호출 대비 처리량이
        크게 올라간다.
        """
        if not frames:
            return []
        imgsz = imgsz or self.imgsz
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou

        shapes = {f.shape[:2] for f in frames}
        if len(shapes) != 1:
            raise ValueError(f"detect_batch() expects uniform frame sizes, got {shapes}")
        h, w = frames[0].shape[:2]

        ims, r, padx, pady = [], None, None, None
        for f in frames:
            im, r, padx, pady = letterbox(f, imgsz)
            ims.append(im)

        # BGR(HWC) x N -> RGB(NCHW)
        batch = np.ascontiguousarray(
            np.stack(ims)[:, :, :, ::-1].transpose(0, 3, 1, 2)
        )
        t = torch.from_numpy(batch).to(self.device)
        t = t.half() if self.half else t.float()
        t /= 255.0

        pred = self.model(t)[0]
        if pred.ndim == 2:          # (N, C) -> (1, N, C)
            pred = pred.unsqueeze(0)

        return [self._decode(pred[i], conf, iou, r, padx, pady, w, h)
                for i in range(pred.shape[0])]

    # ------------------------------------------------------------------ #

    @staticmethod
    def _score(pred):
        """objectness x class conf.

        단일 클래스(얼굴)이므로 클래스 conf 는 마지막 열이지만, 포크에 따라
        랜드마크 열이 뒤에 오는 변형이 있다. 값이 확률 범위를 벗어나면
        랜드마크로 보고 objectness 만 사용한다.
        """
        scores = pred[:, 4]
        if pred.shape[0] > 0 and pred.shape[1] > 5:
            cls_conf = pred[:, -1]
            if float(cls_conf.min()) >= 0.0 and float(cls_conf.max()) <= 1.0:
                scores = scores * cls_conf
        return scores

    def _decode(self, pred, conf, iou, r, padx, pady, width, height):
        """모델 출력 한 장 → [(x1, y1, x2, y2, score), ...] (원본 좌표계)."""
        pred = pred.float()
        scores = self._score(pred)

        keep = scores > conf
        pred, scores = pred[keep], scores[keep]
        if pred.shape[0] == 0:
            return []

        xy, wh = pred[:, :2], pred[:, 2:4]        # box: 0~3열 (cx,cy,w,h)
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)
        idx = torchvision.ops.nms(boxes, scores, iou)
        boxes, scores = boxes[idx], scores[idx]

        mapped = unletterbox(boxes.cpu().numpy(), r, padx, pady, width, height)
        return [(float(x1), float(y1), float(x2), float(y2), float(sc))
                for (x1, y1, x2, y2), sc in zip(mapped, scores.tolist())]
