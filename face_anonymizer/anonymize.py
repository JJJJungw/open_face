"""얼굴 영역 익명화 방법들.

- mosaic  : 픽셀화 (기본). 약한 블러는 복원 위험이 있으므로 강한 익명화엔 이걸 권장.
- blur    : 가우시안 블러.
- box     : 단색 박스로 완전히 가림 (가장 강함, 유틸리티 0).
"""

import cv2
import numpy as np


def pad_box(box, pad, w, h):
    """박스를 pad 비율만큼 확장하고 프레임 안으로 클램프."""
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad; x2 += bw * pad
    y1 -= bh * pad; y2 += bh * pad
    return (int(max(0, x1)), int(max(0, y1)),
            int(min(w, x2)), int(min(h, y2)))


def mosaic(frame, box, scale=0.06):
    """box 영역을 다운스케일 후 최근접 업스케일 → 픽셀화. scale↓ = 블록↑ = 더 강함."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    small = cv2.resize(roi, (max(1, int(rw * scale)), max(1, int(rh * scale))),
                       interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


def blur(frame, box, ksize=None):
    """가우시안 블러. ksize 미지정 시 박스 크기에 비례해 강하게."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if ksize is None:
        k = max(3, (min(x2 - x1, y2 - y1) // 2) | 1)  # 홀수 보장
    else:
        k = ksize | 1
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def solid_box(frame, box, color=(0, 0, 0)):
    """단색 박스로 완전히 덮음."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)


METHODS = {"mosaic": mosaic, "blur": blur, "box": solid_box}


def apply(frame, box, method="mosaic", **kwargs):
    """method 이름으로 익명화 적용."""
    fn = METHODS.get(method)
    if fn is None:
        raise ValueError(f"unknown method: {method}. choose one of {list(METHODS)}")
    fn(frame, box, **kwargs)
