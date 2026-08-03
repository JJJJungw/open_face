"""파이프라인 E2E 테스트 — 가중치도 torch 도 없이 전 구간을 돌린다.

가짜 검출기를 주입하므로 여기서 검증하는 건 모델 정확도가 아니라 **배선**이다.
검출 결과가 올바른 프레임에 반영되는가, 끊긴 구간이 실제로 가려지는가,
실패가 조용히 넘어가지 않는가 — 서빙에서 사고가 나는 지점들이다.
"""

import os

import cv2
import numpy as np
import pytest

from conftest import FakeDetector, IndexedFakeDetector, face_rect, region_is_obscured
from face_anonymizer import VideoAnonymizer, VideoOpenError, probe
from face_anonymizer.pipeline import (
    Cancelled,
    VideoWriteError,
    sane_fps,
)


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


# ------------------------------------------------------------------ sane_fps

@pytest.mark.parametrize("bad", [0, -1, None, float("nan"), float("inf"), 1e9, "x"])
def test_sane_fps_rejects_garbage(bad):
    """`fps or 30.0` 은 NaN 을 통과시킨다 — NaN fps 는 깨진 출력 파일을 만든다."""
    assert sane_fps(bad) == 30.0


def test_sane_fps_keeps_valid():
    assert sane_fps(23.976) == pytest.approx(23.976)


# --------------------------------------------------------------------- probe

def test_probe_reads_metadata(make_video):
    path, frames, (w, h) = make_video(fps=15.0)
    info = probe(path)
    assert (info.width, info.height) == (w, h)
    assert info.fps == pytest.approx(15.0, abs=0.1)


def test_probe_missing_file_raises():
    with pytest.raises(VideoOpenError, match="does not exist"):
        probe("/nonexistent/nope.mp4")


def test_probe_garbage_file_raises(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is definitely not a video" * 100)
    with pytest.raises(VideoOpenError):
        probe(str(junk))


# ------------------------------------------------------------------ 정상 경로

def test_process_anonymizes_every_face(make_video, tmp_path):
    path, frames, size = make_video(frames=30)
    out = str(tmp_path / "out.mp4")

    res = VideoAnonymizer(detector=FakeDetector(size)).process(
        path, out, method="box", pad=0.0, linger=0,
    )

    assert res.frames == frames
    assert res.raw_boxes == frames
    assert os.path.exists(out)

    rendered = read_frames(out)
    assert len(rendered) == frames
    for i, frame in enumerate(rendered):
        assert region_is_obscured(frame, face_rect(i, *size)), \
            f"frame {i} 의 얼굴이 가려지지 않았다"


def test_process_preserves_frame_count_and_size(make_video, tmp_path):
    path, frames, (w, h) = make_video(frames=25)
    out = str(tmp_path / "out.mp4")
    VideoAnonymizer(detector=FakeDetector((w, h))).process(path, out)
    info = probe(out)
    assert (info.width, info.height) == (w, h)
    assert len(read_frames(out)) == frames


@pytest.mark.parametrize("method", ["mosaic", "blur", "box"])
def test_all_methods_obscure(make_video, tmp_path, method):
    path, frames, size = make_video(frames=12)
    out = str(tmp_path / f"{method}.mp4")
    VideoAnonymizer(detector=FakeDetector(size)).process(
        path, out, method=method, mosaic_scale=0.05, linger=0,
    )
    rendered = read_frames(out)
    assert all(region_is_obscured(f, face_rect(i, *size))
               for i, f in enumerate(rendered))


def test_unknown_method_rejected_before_work(make_video, tmp_path):
    path, _, size = make_video(frames=3)
    with pytest.raises(ValueError, match="unknown method"):
        VideoAnonymizer(detector=FakeDetector(size)).process(
            path, str(tmp_path / "o.mp4"), method="pixelate",
        )


# --------------------------------------------------- 핵심: 검출 누락 시 누출 방지

def test_interpolation_covers_missed_detections(make_video, tmp_path):
    """이 프로젝트가 존재하는 이유 그 자체.

    검출기가 프레임 12~16 을 통째로 놓쳐도, 추적 보간이 그 구간을 덮어야 한다.
    이게 깨지면 README 의 프라이버시 주장이 거짓이 된다.
    """
    missed = {12, 13, 14, 15, 16}
    path, frames, size = make_video(frames=30)
    out = str(tmp_path / "out.mp4")

    res = VideoAnonymizer(
        detector=FakeDetector(size, miss_frames=missed)
    ).process(path, out, method="box", pad=0.0, interp=True, linger=3)

    assert res.filled_boxes >= len(missed)
    rendered = read_frames(out)
    for i in sorted(missed):
        assert region_is_obscured(rendered[i], face_rect(i, *size)), \
            f"검출을 놓친 frame {i} 이 그대로 노출됐다"


def test_without_interpolation_missed_frames_leak(make_video, tmp_path):
    """대조군 — 보간을 끄면 실제로 노출된다.

    위 테스트가 '보간 덕분에' 통과한 것인지, 아니면 애초에 검출 누락이
    재현되지 않은 것인지 구분하기 위한 음성 대조다.
    """
    missed = {12, 13, 14, 15, 16}
    path, frames, size = make_video(frames=30)
    out = str(tmp_path / "out.mp4")

    VideoAnonymizer(
        detector=FakeDetector(size, miss_frames=missed)
    ).process(path, out, method="box", pad=0.0, interp=False, linger=0)

    rendered = read_frames(out)
    leaked = [i for i in sorted(missed)
              if not region_is_obscured(rendered[i], face_rect(i, *size))]
    assert leaked, "음성 대조가 성립하지 않는다 — 누락 프레임이 재현되지 않았다"


# --------------------------------------------------------------- 프레임 스킵

def test_detect_every_skips_inference_but_still_covers(make_video, tmp_path):
    path, frames, size = make_video(frames=30)
    out = str(tmp_path / "out.mp4")
    det = IndexedFakeDetector(size, frame_indices=range(0, frames, 3))

    res = VideoAnonymizer(detector=det).process(
        path, out, method="box", detect_every=3, linger=2,
    )

    assert res.detected_frames == 10           # 30 프레임 / 3
    assert res.frames == frames

    rendered = read_frames(out)
    for i, frame in enumerate(rendered):
        assert region_is_obscured(frame, face_rect(i, *size)), \
            f"스킵된 frame {i} 이 보간으로 덮이지 않았다"


def test_detect_every_without_interp_is_refused(make_video, tmp_path):
    """스킵 + 보간 off 는 프레임 대부분이 무방비로 나가는 조합이다.
    조용히 허용하면 안 된다."""
    path, _, size = make_video(frames=6)
    with pytest.raises(ValueError, match="보간"):
        VideoAnonymizer(detector=FakeDetector(size)).process(
            path, str(tmp_path / "o.mp4"), detect_every=5, interp=False,
        )


def test_batch_size_is_respected(make_video, tmp_path):
    path, frames, size = make_video(frames=20)
    det = FakeDetector(size)
    VideoAnonymizer(detector=det).process(
        path, str(tmp_path / "o.mp4"), batch_size=8,
    )
    assert max(det.batch_sizes) == 8
    assert sum(det.batch_sizes) == frames


def test_batch_size_does_not_change_result(make_video, tmp_path):
    """배치 크기는 성능 손잡이일 뿐 결과를 바꾸면 안 된다."""
    path, frames, size = make_video(frames=18)
    outs = []
    for bs in (1, 6):
        out = str(tmp_path / f"bs{bs}.mp4")
        VideoAnonymizer(detector=FakeDetector(size)).process(
            path, out, method="box", pad=0.0, batch_size=bs, linger=0,
        )
        outs.append(read_frames(out))
    assert all(np.array_equal(a, b) for a, b in zip(*outs))


# ------------------------------------------------------------------ 실패 처리

def test_missing_input_raises_not_empty_output(tmp_path):
    with pytest.raises(VideoOpenError):
        VideoAnonymizer(detector=FakeDetector((320, 240))).process(
            str(tmp_path / "nope.mp4"), str(tmp_path / "out.mp4"),
        )
    assert not (tmp_path / "out.mp4").exists()


def test_bad_fourcc_raises(make_video, tmp_path):
    path, _, size = make_video(frames=5)
    with pytest.raises(VideoWriteError):
        VideoAnonymizer(detector=FakeDetector(size)).process(
            path, str(tmp_path / "o.mp4"), fourcc="ZZZZ",
        )


def test_output_directory_is_created(make_video, tmp_path):
    path, _, size = make_video(frames=5)
    out = str(tmp_path / "nested" / "deep" / "o.mp4")
    VideoAnonymizer(detector=FakeDetector(size)).process(path, out)
    assert os.path.exists(out)


def test_path_containing_mp4_in_directory_name(make_video, tmp_path):
    """`output.replace('.mp4', ...)` 회귀 테스트.

    전체 치환을 쓰면 디렉터리 이름의 '.mp4' 까지 바뀌어 존재하지 않는
    경로에 쓰려다 실패한다.
    """
    path, _, size = make_video(frames=5)
    weird = tmp_path / "clip.mp4.d"
    weird.mkdir()
    out = str(weird / "o.mp4")
    VideoAnonymizer(detector=FakeDetector(size)).process(path, out)
    assert os.path.exists(out)


def test_no_temp_files_left_behind(make_video, tmp_path):
    path, _, size = make_video(frames=8)
    outdir = tmp_path / "out"
    outdir.mkdir()
    VideoAnonymizer(detector=FakeDetector(size)).process(
        path, str(outdir / "o.mp4"),
    )
    assert os.listdir(outdir) == ["o.mp4"], "중간 산출물이 남았다"


def test_concurrent_outputs_do_not_collide(make_video, tmp_path):
    """임시 파일이 출력 경로 기준 고정 이름이면 동시 작업이 서로를 덮어쓴다."""
    import threading

    path, frames, size = make_video(frames=12)
    outdir = tmp_path / "out"
    outdir.mkdir()
    errors = []

    def run(n):
        try:
            VideoAnonymizer(detector=FakeDetector(size)).process(
                path, str(outdir / f"o{n}.mp4"),
            )
        except Exception as e:            # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert sorted(os.listdir(outdir)) == [f"o{n}.mp4" for n in range(4)]


# ----------------------------------------------------------- 진행률 / 취소

def test_progress_callback_is_called(make_video, tmp_path):
    path, frames, size = make_video(frames=20)
    seen = []
    VideoAnonymizer(detector=FakeDetector(size)).process(
        path, str(tmp_path / "o.mp4"),
        progress=lambda stage, done, total: seen.append(stage),
    )
    assert "detect" in seen and "render" in seen


def test_cancellation_stops_and_leaves_no_output(make_video, tmp_path):
    path, _, size = make_video(frames=40)
    out = tmp_path / "o.mp4"
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(Cancelled):
        VideoAnonymizer(detector=FakeDetector(size)).process(
            path, str(out), batch_size=1, should_cancel=should_cancel,
        )
    assert not out.exists()


# ---------------------------------------------------------------- 오디오

def test_video_without_audio_reports_no_audio(make_video, tmp_path):
    path, _, size = make_video(frames=8)
    res = VideoAnonymizer(detector=FakeDetector(size)).process(
        path, str(tmp_path / "o.mp4"),
    )
    assert res.audio in ("no-audio", "ffmpeg-missing")
    assert os.path.exists(res.output)


def test_audio_is_preserved(make_video, tmp_path):
    path, _, size = make_video(frames=20, with_audio=True)
    out = str(tmp_path / "o.mp4")
    res = VideoAnonymizer(detector=FakeDetector(size)).process(path, out)
    assert res.audio == "ok"

    import subprocess
    probe_out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", out],
        capture_output=True, text=True,
    ).stdout
    assert "audio" in probe_out


def test_keep_audio_false_skips_mux(make_video, tmp_path):
    path, _, size = make_video(frames=12, with_audio=True)
    res = VideoAnonymizer(detector=FakeDetector(size)).process(
        path, str(tmp_path / "o.mp4"), keep_audio=False,
    )
    assert res.audio == "disabled"
    assert os.path.exists(res.output)


# ------------------------------------------------------------------- 기타

def test_detector_and_kwargs_are_mutually_exclusive():
    with pytest.raises(TypeError):
        VideoAnonymizer(detector=FakeDetector((10, 10)), device="cpu")


def test_pipeline_importable_without_torch():
    """검출기를 주입해 쓰는 경로는 torch 없이도 동작해야 한다.

    CI 에서 2GB 짜리 torch 를 받지 않고도 회귀를 잡을 수 있는 근거다.
    """
    import sys
    assert "torch" not in sys.modules, "파이프라인만 쓰는데 torch 가 로드됐다"
