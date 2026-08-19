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

    **최대 픽셀 차이로 재면 안 된다.** 예전에는 그렇게 재면서 "인코딩 잡음은
    한 자릿수라 진짜 차이와 안 섞인다" 고 적어 뒀는데, 재 보니 사실이 아니었다 —
    **같은 배치끼리 비교해도** 최대값이 0~46 사이에서 튄다. 결과물이 손실
    압축을 거치는 데다 인코딩이 실행마다 비트 단위로 같지 않아서, 경계 픽셀
    하나가 그만큼 흔들린다. 그래서 이 테스트는 전체 실행의 25% 쯤 실패했다.

        같은 배치 1v1   최대  10 · 8 · 0      평균절대차 0.005 · 0.003 · 0.000
        다른 배치 1v6   최대  37 · 37 · 8     평균절대차 0.008 · 0.010 · 0.003

    **평균절대차는 안정적이다.** 0.01 아래에 머문다. 반대로 배치 크기가 정말로
    결과를 바꾸면 박스가 통째로 어긋나므로 — 320x240 에서 박스 하나가 밀리면
    화면의 4% 가 150 단위로 바뀌어 평균절대차가 6 쯤 된다 — 잡음보다 세 자릿수
    크다. 자를 바꾸면 두 신호가 겹치지 않는다.

    자가 실제로 듣는지도 확인했다 — 박스를 15px 밀어 보면 평균절대차가
    **2.276**(최대 203)이 된다. 잡음 0.01 과 임계값 0.5 사이가 50배, 임계값과
    진짜 차이 사이가 4.5배다.
    """
    path, _, size = make_video(frames=18)
    outs = []
    for bs in (1, 6):
        out = str(tmp_path / f"bs{bs}.mp4")
        VideoAnonymizer(detector=FakeDetector(size)).process(
            path, out, method="box", pad=0.0, batch_size=bs, linger=0)
        outs.append(read_frames(out))

    assert len(outs[0]) == len(outs[1])
    mae = max(float(np.abs(a.astype(int) - b.astype(int)).mean())
              for a, b in zip(*outs))
    # 잡음 0.01 이하 · 박스 15px 이동 2.276. 0.5 는 그 사이를 넉넉히 가른다.
    assert mae <= 0.5, f"배치 크기에 따라 결과가 달라진다 (평균절대차 {mae:.3f})"


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
