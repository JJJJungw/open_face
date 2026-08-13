"""스모크 테스트 — 가중치도 torch 도 없이 파이프라인 전 구간을 돌린다.

검출기를 주입할 수 있으므로, 얼굴 위치를 아는 합성 영상 + 가짜 검출기면
"검출 → 추적 → 보간 → 렌더 → 오디오" 배선을 확인하는 데 충분하다.
모델 정확도는 가중치가 필요한 별도 관심사라 여기서 보지 않는다.
"""

import numpy as np
import pytest

from conftest import FakeDetector, face_rect, read_frames, region_is_obscured
from face_anonymizer import VideoAnonymizer, VideoOpenError
from face_anonymizer.core.geometry import snap_to_stride
from face_anonymizer.core.anonymize import pad_box
from face_anonymizer.core.geometry import letterbox, unletterbox
from face_anonymizer.core.pipeline import sane_fps


def test_every_face_is_obscured(make_video, tmp_path):
    path, frames, size = make_video(frames=30)
    out = str(tmp_path / "out.mp4")

    res = VideoAnonymizer(detector=FakeDetector(size)).process(
        path, out, method="box", pad=0.0, linger=0)

    assert res.frames == frames and res.raw_boxes == frames
    rendered = read_frames(out)
    assert len(rendered) == frames
    for i, frame in enumerate(rendered):
        assert region_is_obscured(frame, face_rect(i, *size)), f"frame {i} 노출"


@pytest.mark.parametrize("method", ["mosaic", "blur", "box"])
def test_all_methods_obscure(make_video, tmp_path, method):
    path, _, size = make_video(frames=12)
    out = str(tmp_path / f"{method}.mp4")
    VideoAnonymizer(detector=FakeDetector(size)).process(
        path, out, method=method, mosaic_scale=0.05, linger=0)
    assert all(region_is_obscured(f, face_rect(i, *size))
               for i, f in enumerate(read_frames(out)))


def test_output_keeps_size_and_frame_count(make_video, tmp_path):
    path, frames, (w, h) = make_video(frames=25)
    out = str(tmp_path / "o.mp4")
    VideoAnonymizer(detector=FakeDetector((w, h))).process(path, out)
    rendered = read_frames(out)
    assert len(rendered) == frames and rendered[0].shape[:2] == (h, w)


def test_interpolation_covers_missed_detections(make_video, tmp_path):
    """이 프로젝트가 존재하는 이유. 검출기가 프레임 12~16 을 통째로 놓쳐도
    추적 보간이 그 구간을 덮어야 한다."""
    missed = {12, 13, 14, 15, 16}
    path, _, size = make_video(frames=30)
    out = str(tmp_path / "out.mp4")

    res = VideoAnonymizer(detector=FakeDetector(size, missed)).process(
        path, out, method="box", pad=0.0, interp=True, linger=3)

    assert res.filled_boxes >= len(missed)
    rendered = read_frames(out)
    for i in sorted(missed):
        assert region_is_obscured(rendered[i], face_rect(i, *size)), \
            f"놓친 frame {i} 이 그대로 노출됐다"


def test_without_interpolation_it_actually_leaks(make_video, tmp_path):
    """음성 대조 — 위 테스트가 '보간 덕분에' 통과한 것인지 확인한다.

    이게 없으면 애초에 검출 누락이 재현되지 않았을 가능성을 배제할 수 없다.
    """
    missed = {12, 13, 14, 15, 16}
    path, _, size = make_video(frames=30)
    out = str(tmp_path / "out.mp4")

    VideoAnonymizer(detector=FakeDetector(size, missed)).process(
        path, out, method="box", pad=0.0, interp=False, linger=0)

    rendered = read_frames(out)
    assert any(not region_is_obscured(rendered[i], face_rect(i, *size))
               for i in sorted(missed)), "누락 프레임이 재현되지 않았다"


def test_missing_input_raises(tmp_path):
    with pytest.raises(VideoOpenError):
        VideoAnonymizer(detector=FakeDetector((320, 240))).process(
            str(tmp_path / "nope.mp4"), str(tmp_path / "o.mp4"))
    assert not (tmp_path / "o.mp4").exists()


def test_params_that_would_disable_anonymization_are_rejected(make_video, tmp_path):
    """음수 pad 는 박스를 뒤집고 mosaic_scale>=1 은 축소를 없앤다. 둘 다
    익명화 함수가 조용히 return 해서 픽셀을 하나도 안 건드린다."""
    path, _, size = make_video(frames=5)
    anon = VideoAnonymizer(detector=FakeDetector(size))
    with pytest.raises(ValueError, match="pad"):
        anon.process(path, str(tmp_path / "a.mp4"), pad=-0.6)
    with pytest.raises(ValueError, match="mosaic_scale"):
        anon.process(path, str(tmp_path / "b.mp4"), mosaic_scale=1.0)


@pytest.mark.parametrize("bad", [0, -1, None, float("nan"), float("inf"), "x"])
def test_sane_fps_rejects_garbage(bad):
    """`fps or 30.0` 은 NaN 을 통과시킨다 — NaN fps 는 깨진 파일을 만든다."""
    assert sane_fps(bad) == 30.0


def test_box_geometry():
    """좌표가 틀리면 박스가 얼굴에서 어긋나는데 눈으로 놓치기 쉽다.
    정수화는 항상 바깥쪽으로 — int() 는 오른쪽/아래를 최대 1px 깎는다."""
    h, w = 240, 320
    _, r, px, py = letterbox(np.zeros((h, w, 3), np.uint8), 640)
    original = np.array([[10.0, 20.0, 60.0, 90.0]])
    lb = original * r
    lb[:, [0, 2]] += px
    lb[:, [1, 3]] += py
    assert np.allclose(unletterbox(lb, r, px, py, w, h), original)

    assert pad_box((0.4, 0.4, 10.9, 10.9), 0.0, 100, 100) == (0, 0, 11, 11)
    assert pad_box((5, 5, 50, 50), 0.5, 40, 40) == (0, 0, 40, 40)


def test_batch_size_does_not_change_result(make_video, tmp_path):
    """배치 크기는 성능 손잡이일 뿐 결과를 바꾸면 안 된다.

    비교는 허용 오차로 한다. 결과물은 손실 압축을 거치므로 픽셀이 완전히
    같기를 요구하면 인코더 설정이 바뀔 때마다 깨진다 — 실제로 목표
    비트레이트를 올리자(압축이 약해지자) 이 테스트가 먼저 넘어졌다.

    반대로 배치 크기가 정말로 결과를 바꾸면 박스 위치가 어긋나므로 차이가
    국소적으로 수십~수백 단위로 뜬다. 인코딩 잡음(한 자릿수)과 섞이지 않는다.
    """
    path, _, size = make_video(frames=18)
    outs = []
    for bs in (1, 6):
        out = str(tmp_path / f"bs{bs}.mp4")
        VideoAnonymizer(detector=FakeDetector(size)).process(
            path, out, method="box", pad=0.0, batch_size=bs, linger=0)
        outs.append(read_frames(out))

    assert len(outs[0]) == len(outs[1])
    worst = max(int(np.abs(a.astype(int) - b.astype(int)).max())
                for a, b in zip(*outs))
    assert worst <= 12, f"배치 크기에 따라 결과가 달라진다 (최대 차이 {worst})"


def test_pipeline_does_not_need_torch():
    """CI 에서 2GB 짜리 torch 없이 회귀를 잡을 수 있는 근거.

    ``sys.modules`` 를 그냥 들여다보면 **테스트 순서에 따라 답이 바뀐다.**
    torch 가 깔린 기계(=GPU 서버)에서는 앞선 테스트가 진짜 검출기를 올리면서
    이미 넣어 두기 때문에, 배선이 멀쩡해도 실패한다. 정작 묻고 싶은 것은
    "이 모듈만 임포트했을 때 딸려 오는가" 이므로 새 프로세스에서 확인한다.
    """
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys, face_anonymizer.core.pipeline;"
         " print('torch' in sys.modules)"],
        cwd=root, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": root})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False", "pipeline 임포트만으로 torch 가 딸려 온다"


def test_snap_to_stride_rounds_up():
    """항상 위로 올린다 — 내리면 해상도가 줄어 작은 얼굴을 놓치는 쪽으로 틀린다."""
    assert snap_to_stride(1000, 32) == 1024
    assert snap_to_stride(1280, 32) == 1280
    assert snap_to_stride(960, 32) == 960
    assert snap_to_stride(1, 32) == 32
    assert snap_to_stride(700, 16) == 704
    assert snap_to_stride(1000.4, 32) == 1024


def test_snap_to_stride_never_below_stride():
    assert snap_to_stride(0, 32) == 32
    assert snap_to_stride(-10, 32) == 32
