"""letterbox 좌표 변환 테스트.

여기가 틀리면 박스가 얼굴에서 어긋난 위치에 찍힌다. 모델 없이도 검증 가능한
순수 기하 계산이고, 실패 시 증상이 '조금 어긋난 모자이크' 라 눈으로 놓치기 쉬워서
자동 테스트의 가치가 특히 크다.
"""

import numpy as np
import pytest

from face_anonymizer.geometry import letterbox, unletterbox


@pytest.mark.parametrize("shape", [(240, 320), (1080, 1920), (720, 720), (100, 400)])
@pytest.mark.parametrize("size", [320, 640, 960])
def test_letterbox_output_is_square(shape, size):
    im = np.zeros((*shape, 3), np.uint8)
    out, r, px, py = letterbox(im, size)
    assert out.shape[:2] == (size, size)
    assert r == pytest.approx(min(size / shape[0], size / shape[1]))


@pytest.mark.parametrize("shape", [(240, 320), (1080, 1920), (480, 480), (100, 400)])
@pytest.mark.parametrize("size", [320, 640])
def test_roundtrip_maps_box_back_to_original(shape, size):
    """원본 좌표의 박스를 letterbox 좌표로 옮겼다가 되돌리면 제자리여야 한다."""
    h, w = shape
    im = np.zeros((h, w, 3), np.uint8)
    _, r, px, py = letterbox(im, size)

    original = np.array([
        [10.0, 20.0, 60.0, 90.0],
        [w * 0.5, h * 0.5, w * 0.6, h * 0.7],
    ])
    # 원본 -> letterbox 좌표 (unletterbox 의 역연산)
    lb = original * r
    lb[:, [0, 2]] += px
    lb[:, [1, 3]] += py

    back = unletterbox(lb, r, px, py, w, h)
    assert np.allclose(back, original, atol=1e-6)


def test_unletterbox_clamps_to_frame():
    boxes = [[-50.0, -50.0, 10_000.0, 10_000.0]]
    out = unletterbox(boxes, r=1.0, pad_x=0, pad_y=0, width=320, height=240)
    assert out.tolist() == [[0.0, 0.0, 320.0, 240.0]]


def test_unletterbox_accepts_single_box():
    out = unletterbox([1.0, 2.0, 3.0, 4.0], r=1.0, pad_x=0, pad_y=0,
                      width=100, height=100)
    assert out.shape == (1, 4)


def test_unletterbox_does_not_mutate_input():
    boxes = np.array([[10.0, 10.0, 20.0, 20.0]])
    keep = boxes.copy()
    unletterbox(boxes, r=0.5, pad_x=4, pad_y=8, width=100, height=100)
    assert np.array_equal(boxes, keep)


def test_letterbox_centers_the_image():
    """세로로 긴 영상은 좌우에, 가로로 긴 영상은 상하에 패딩이 붙어야 한다."""
    wide, _, px_w, py_w = letterbox(np.zeros((100, 400, 3), np.uint8), 400)
    assert px_w == 0 and py_w > 0

    tall, _, px_t, py_t = letterbox(np.zeros((400, 100, 3), np.uint8), 400)
    assert py_t == 0 and px_t > 0


def test_letterbox_rejects_empty_frame():
    with pytest.raises(ValueError):
        letterbox(np.zeros((0, 10, 3), np.uint8), 320)
