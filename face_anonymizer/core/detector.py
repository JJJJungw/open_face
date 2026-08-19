"""YOLO-FaceV2 기반 얼굴 검출기.

YOLO-FaceV2 (Krasjet-Yu, clibdev fork) 는 YOLOv5 계열이라 자체 model/utils 코드가
필요하다. 체크포인트를 unpickle 하려면 해당 리포가 sys.path 에 있어야 하므로,
scripts/setup_weights.py 로 third_party/YOLO-FaceV2 에 클론해 둔 뒤 사용한다.

블러/모자이크 용도에는 박스만 있으면 되므로, 리포별로 이름이 다른 face-NMS 함수에
의존하지 않고 letterbox + torchvision NMS 로 직접 디코딩한다(랜드마크 열은 무시).

이 모듈은 torch 를 임포트한다. torch 없이도 파이프라인 로직을 테스트할 수 있도록
좌표 변환은 geometry.py 로, 픽셀 처리는 anonymize.py 로 분리해 두었다.
"""

import contextlib
import logging
import os
import sys
import threading

import numpy as np
import torch
import torchvision

from .geometry import letterbox, snap_to_stride, unletterbox

log = logging.getLogger(__name__)

from .paths import DEFAULT_REPO, DEFAULT_WEIGHTS, ROOT  # noqa: F401

# RLock: 중첩 호출이 생겨도 데드락으로 프로세스 전체를 막지 않는다.
_LOAD_LOCK = threading.RLock()


@contextlib.contextmanager
def _patched_torch_load():
    """체크포인트 로드 동안만 ``weights_only=False`` 를 강제한다.

    torch>=2.6 은 ``torch.load`` 의 ``weights_only`` 기본값이 True 라 YOLOv5
    형식(모델 객체를 담은) 체크포인트를 못 읽는다.

    단순히 감쌌다 되돌리면 **동시에 두 검출기를 만들 때 영구히 오염된다**:
    T1 이 원본을 저장하고 L1 을 설치, T2 가 L1 을 "원본"으로 저장하고 L2 설치,
    T1 이 원본 복원, T2 가 L1 복원 — 이후 프로세스의 모든 torch.load 가
    조용히 임의 객체를 unpickle 한다. 서버에서 워커별로 지연 로드하면
    실제로 일어나는 순서다. 그래서 락으로 직렬화하고, 복원할 때는 내가 설치한
    함수가 그대로 있을 때만 되돌린다.
    """
    with _LOAD_LOCK:
        original = torch.load

        def _loader(*a, **k):
            return original(*a, **{**k, "weights_only": False})

        torch.load = _loader
        try:
            yield
        finally:
            if torch.load is _loader:
                torch.load = original
            else:                       # 누군가 그 사이에 또 바꿨다 — 덮지 않는다
                log.warning("torch.load was replaced during model load; "
                            "leaving the current implementation in place")


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
                "Run `python scripts/setup_weights.py` first."
            )
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"weights not found at {weights}. Run `python scripts/setup_weights.py` first."
            )
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from models.experimental import attempt_load  # noqa: E402 (repo 로드 후)

        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        with _patched_torch_load():
            model = None
            for kwargs in ({"device": self.device}, {"map_location": self.device}, {}):
                try:
                    model = attempt_load(weights, **kwargs)
                    break
                except TypeError:
                    continue
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
        # YOLOv5 계열은 입력이 stride 배수여야 한다. 아니면 forward 에서
        # 어긋나거나 터진다. 항상 위로 올림해 안전한 쪽으로 맞춘다.
        self.imgsz = self.snap_imgsz(imgsz)
        if self.imgsz != imgsz:
            log.info("imgsz %d -> %d (stride %d 배수로 스냅)",
                     imgsz, self.imgsz, self.stride)
        self.conf, self.iou = conf, iou
        log.info("FaceDetector ready: device=%s half=%s imgsz=%s stride=%s",
                 self.device, self.half, self.imgsz, self.stride)

    def snap_imgsz(self, imgsz):
        return snap_to_stride(imgsz, self.stride)

    # ------------------------------------------------------------------ #


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
        # __init__ 에서 스냅한 값(self.imgsz)은 호출자가 imgsz 를 넘기면 무시됐다.
        # 파이프라인이 매번 원본 값을 넘기므로 스냅이 사실상 적용되지 않았고,
        # 로그만 "스냅했다" 고 찍혔다. 실제로 쓰는 지점에서 맞춘다.
        imgsz = snap_to_stride(imgsz or self.imgsz, self.stride)
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
        """스코어 = objectness.

        원래는 objectness x class-conf 를 쓰려고 마지막 열을 클래스 conf 로
        해석했다. 그런데 포크에 따라 그 자리에 랜드마크 좌표가 온다. 값이
        확률 범위 안이면 클래스 conf 로 보는 휴리스틱을 넣었더니, 정규화된
        랜드마크가 [0,1] 에 들어오는 경우 obj 0.95 x 0.01 = 0.0095 로 스코어가
        무너져 **영상 내내 얼굴이 검출되지 않는** 조용한 사고가 났다.

        단일 클래스(얼굴) 모델에서 class-conf 는 거의 항상 1 에 가깝고 곱해도
        실익이 없다. 틀렸을 때 과검출(안전)로 기우는 objectness 단독을 쓴다.
        """
        return pred[:, 4]

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
