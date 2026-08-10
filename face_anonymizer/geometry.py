"""letterbox 리사이즈와 좌표 역변환.

detector.py 에서 분리해 둔 이유는 두 가지다.

1. torch 없이 임포트할 수 있어야 좌표 변환 로직을 단위 테스트할 수 있다.
   (검출기 자체는 가중치 + torch 가 있어야 돌지만, 좌표 계산이 맞는지는
   합성 데이터로 얼마든지 검증할 수 있고 실제로 여기서 틀리면 박스가
   얼굴에서 어긋나 프라이버시 누출로 이어진다.)
2. 배치 추론에서 프레임마다 같은 변환을 재사용하기 위해 파라미터를
   명시적으로 주고받는 편이 낫다.
"""

import math

import cv2
import numpy as np

PAD_COLOR = (114, 114, 114)


def snap_to_stride(imgsz, stride=32):
    """추론 해상도를 stride 배수로 올림.

    YOLOv5 계열은 입력이 stride 배수가 아니면 skip-connection concat 에서
    크기가 어긋난다. 항상 **위로** 올린다 — 내리면 해상도가 줄어 작은 얼굴을
    놓치는 쪽으로 틀리기 때문이다.
    """
    stride = max(1, int(stride))
    return max(stride, int(math.ceil(int(imgsz) / stride) * stride))


def letterbox(im, new, color=PAD_COLOR):
    """비율을 유지한 채 (new x new) 로 리사이즈 + 패딩.

    Returns
    -------
    (img, r, pad_x, pad_y)
        img   : (new, new, 3) uint8
        r     : 적용된 스케일 (원본 -> 리사이즈)
        pad_x : 좌측 패딩 픽셀 수
        pad_y : 상단 패딩 픽셀 수
    """
    h, w = im.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid frame shape: {im.shape}")
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


def unletterbox(boxes, r, pad_x, pad_y, width, height):
    """letterbox 좌표계의 xyxy 박스를 원본 프레임 좌표계로 되돌리고 클램프.

    boxes : (N, 4) array-like — letterbox 이미지 기준 (x1, y1, x2, y2)
    width, height : 원본 프레임 크기

    Returns (N, 4) float ndarray.
    """
    b = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    b[:, [0, 2]] -= pad_x
    b[:, [1, 3]] -= pad_y
    b /= r
    # 팬시 인덱싱은 복사본을 만들기 때문에 np.clip(..., out=b[:, [0, 2]]) 는
    # 임시 배열에 쓰고 버려진다. 반드시 대입으로 되돌려 놓아야 한다.
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, width)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, height)
    return b
