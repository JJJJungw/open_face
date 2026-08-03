"""YOLO-FaceV2 기반 얼굴 검출기.

YOLO-FaceV2 (Krasjet-Yu, clibdev fork) 는 YOLOv5 계열이라 자체 model/utils 코드가
필요하다. 체크포인트를 unpickle 하려면 해당 리포가 sys.path 에 있어야 하므로,
setup_weights.py 로 third_party/YOLO-FaceV2 에 클론해 둔 뒤 사용한다.

블러/모자이크 용도에는 박스만 있으면 되므로, 리포별로 이름이 다른 face-NMS 함수에
의존하지 않고 letterbox + torchvision NMS 로 직접 디코딩한다(랜드마크 열은 무시).
"""

import os
import sys

import cv2
import numpy as np
import torch
import torchvision

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
    imgsz, conf, iou : 추론 기본값. detect() 호출 시 개별 오버라이드 가능.
    """

    def __init__(self, weights=DEFAULT_WEIGHTS, repo_path=DEFAULT_REPO,
                 device=None, imgsz=960, conf=0.25, iou=0.45):
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

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.stride = int(self.model.stride.max()) if hasattr(self.model, "stride") else 32
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    @staticmethod
    def _letterbox(im, new, color=(114, 114, 114)):
        """비율 유지 리사이즈 + 패딩 → (new x new). 반환: (img, ratio, pad_x, pad_y)."""
        h, w = im.shape[:2]
        r = min(new / h, new / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (new - nw) / 2, (new - nh) / 2
        if (w, h) != (nw, nh):
            im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
        right, bottom = int(round(dw + 0.1)), int(round(dh + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
        return im, r, left, top

    @torch.no_grad()
    def detect(self, frame, imgsz=None, conf=None, iou=None):
        """단일 BGR 프레임 → [(x1, y1, x2, y2, score), ...] (원본 좌표계)."""
        imgsz = imgsz or self.imgsz
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou

        im, r, padx, pady = self._letterbox(frame, imgsz)
        t = torch.from_numpy(
            np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))
        ).to(self.device).float() / 255.0
        t = t.unsqueeze(0)

        pred = self.model(t)[0]
        if pred.ndim == 3:
            pred = pred[0]

        # 스코어 = objectness x class conf. 단일 클래스(얼굴)이므로 클래스 conf 는
        # 마지막 열이지만, 포크에 따라 랜드마크 열이 뒤에 오는 변형이 있다.
        # 값이 확률 범위를 벗어나면 랜드마크로 보고 objectness 만 사용한다.
        scores = pred[:, 4]
        if pred.shape[0] > 0 and pred.shape[1] > 5:
            cls_conf = pred[:, -1]
            if float(cls_conf.min()) >= 0.0 and float(cls_conf.max()) <= 1.0:
                scores = scores * cls_conf

        keep = scores > conf
        pred, scores = pred[keep], scores[keep]
        if pred.shape[0] == 0:
            return []

        xy, wh = pred[:, :2], pred[:, 2:4]        # box: 0~3열 (cx,cy,w,h), 랜드마크 무시
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)
        idx = torchvision.ops.nms(boxes, scores, iou)
        boxes, scores = boxes[idx], scores[idx]

        boxes[:, [0, 2]] -= padx
        boxes[:, [1, 3]] -= pady
        boxes /= r

        H, W = frame.shape[:2]
        out = []
        for (x1, y1, x2, y2), sc in zip(boxes.tolist(), scores.tolist()):
            out.append((max(0, min(W, x1)), max(0, min(H, y1)),
                        max(0, min(W, x2)), max(0, min(H, y2)), float(sc)))
        return out
