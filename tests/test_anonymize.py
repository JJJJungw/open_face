"""익명화 연산 단위 테스트 — 순수 픽셀 처리라 torch/가중치가 필요 없다."""

import cv2
import numpy as np
import pytest

from face_anonymizer.anonymize import METHODS, apply, blur, mosaic, pad_box, solid_box


def noisy(h=80, w=80):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------- pad_box

def test_pad_box_expands_by_ratio():
    # 100x100 박스를 10% 확장하면 각 변이 10px 씩 늘어난다
    assert pad_box((100, 100, 200, 200), 0.1, 1000, 1000) == (90, 90, 210, 210)


def test_pad_box_clamps_to_frame():
    x1, y1, x2, y2 = pad_box((5, 5, 50, 50), 0.5, 40, 40)
    assert (x1, y1) == (0, 0)
    assert (x2, y2) == (40, 40)          # 프레임 밖으로 나가지 않는다


def test_pad_box_zero_pad_is_identity():
    assert pad_box((10.0, 20.0, 30.0, 40.0), 0.0, 100, 100) == (10, 20, 30, 40)


def test_pad_box_ignores_extra_columns():
    """검출 결과는 (x1,y1,x2,y2,score) 5-튜플로 들어온다."""
    assert pad_box((10, 10, 20, 20, 0.99), 0.0, 100, 100) == (10, 10, 20, 20)


# ---------------------------------------------------------------------- mosaic

def test_mosaic_destroys_detail():
    img = noisy()
    before = img[20:60, 20:60].std()
    mosaic(img, (20, 20, 60, 60), scale=0.1)
    after = img[20:60, 20:60].std()
    assert after < before / 2


def test_mosaic_smaller_scale_is_stronger():
    """scale 을 낮출수록 블록이 커져 고유 색 수가 줄어야 한다."""
    def unique_colors(scale):
        img = noisy()
        mosaic(img, (0, 0, 80, 80), scale=scale)
        return len(np.unique(img.reshape(-1, 3), axis=0))

    assert unique_colors(0.03) < unique_colors(0.3)


def test_mosaic_leaves_outside_untouched():
    img = noisy()
    keep = img[0:10, 0:10].copy()
    mosaic(img, (20, 20, 60, 60), scale=0.1)
    assert np.array_equal(img[0:10, 0:10], keep)


# ------------------------------------------------------------------------ blur

def test_blur_reduces_variance():
    img = noisy()
    before = img[10:70, 10:70].std()
    blur(img, (10, 10, 70, 70))
    assert img[10:70, 10:70].std() < before


def test_blur_accepts_even_ksize():
    """가우시안 커널은 홀수여야 한다 — 짝수를 줘도 죽지 않아야."""
    img = noisy()
    blur(img, (10, 10, 70, 70), ksize=8)     # 예외가 나면 실패


# ------------------------------------------------------------------- solid_box

def test_solid_box_fills_completely():
    img = noisy()
    solid_box(img, (10, 10, 50, 50))
    assert np.all(img[10:50, 10:50] == 0)


def test_solid_box_custom_color():
    img = noisy()
    solid_box(img, (10, 10, 50, 50), color=(255, 0, 0))
    assert np.all(img[10:50, 10:50] == np.array([255, 0, 0]))


# ------------------------------------------------------- degenerate / dispatch

@pytest.mark.parametrize("fn", [mosaic, blur, solid_box])
@pytest.mark.parametrize("box", [(50, 50, 50, 50), (60, 10, 20, 70), (0, 0, 0, 0)])
def test_degenerate_boxes_are_noops(fn, box):
    """폭/높이가 0 이거나 뒤집힌 박스에서 죽지 않고 조용히 넘어가야 한다.

    검출/보간 결과가 프레임 경계에서 이렇게 될 수 있는데, 여기서 예외가 나면
    영상 한 편이 통째로 실패한다.
    """
    img = noisy()
    keep = img.copy()
    fn(img, box)
    assert np.array_equal(img, keep)


def test_apply_dispatches_all_methods():
    for name in METHODS:
        img = noisy()
        kw = {"scale": 0.1} if name == "mosaic" else {}
        apply(img, (10, 10, 60, 60), method=name, **kw)
        assert not np.array_equal(img[10:60, 10:60], noisy()[10:60, 10:60])


def test_apply_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        apply(noisy(), (0, 0, 10, 10), method="nope")
