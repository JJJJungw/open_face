"""입력 코덱 정규화 테스트.

OpenCV 의 FFmpeg 빌드는 ffmpeg 본체보다 코덱 지원이 좁다. AV1 은 파일을 **열기는
열면서 한 프레임도 못 뽑는다**(실측: OpenCV 4.13, isOpened=True / 디코딩 0프레임).
코덱 이름 목록으로 판단하면 빌드마다 어긋나므로, 실제로 한 프레임을 뽑아 보고
안 되면 H.264 로 옮겨 담는다.
"""

import os
import shutil
import subprocess

import pytest

from conftest import FakeDetector, face_rect, read_frames, region_is_obscured

from face_anonymizer import VideoAnonymizer
from face_anonymizer.core import ingest

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg 없음")


def has_encoder(name):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                       capture_output=True, text=True)
    return name in r.stdout


def to_av1(src, dst):
    for enc, extra in (("libsvtav1", ["-preset", "10"]),
                       ("libaom-av1", ["-cpu-used", "8"])):
        if not has_encoder(enc):
            continue
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                            "-c:v", enc, "-crf", "40", *extra, str(dst)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return str(dst)
    pytest.skip("AV1 인코더가 없다")


def codec_of(path):
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()


def test_h264_input_is_not_transcoded(tmp_path, make_video):
    """대부분의 입력은 여기서 아무 비용도 내지 않아야 한다."""
    src, n, size = make_video(frames=8)
    h264 = tmp_path / "h264.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                    "-c:v", "libx264", "-crf", "23", str(h264)], check=True)

    path, info = ingest.ensure_decodable(str(h264), str(tmp_path / "w"))

    assert path == str(h264)
    assert info["transcoded"] is False
    assert info["source_codec"] == "h264"


def test_av1_input_is_transcoded(tmp_path, make_video):
    src, n, size = make_video(frames=12)
    av1 = to_av1(src, tmp_path / "in.av1.mp4")
    assert codec_of(av1) == "av1"

    path, info = ingest.ensure_decodable(av1, str(tmp_path / "w"))

    assert info["source_codec"] == "av1"
    if not info["transcoded"]:
        pytest.skip("이 OpenCV 빌드는 AV1 을 직접 읽는다")
    assert path != av1
    assert codec_of(path) == "h264"
    assert ingest.opencv_can_decode(path)


def test_av1_runs_end_to_end(tmp_path, make_video):
    """AV1 이 들어와도 결과물이 나오고 얼굴이 가려진다."""
    src, n, size = make_video(frames=20)
    av1 = to_av1(src, tmp_path / "in.av1.mp4")
    out = tmp_path / "out.mp4"

    res = VideoAnonymizer(detector=FakeDetector(size)).process(
        av1, str(out), batch_size=8, keep_audio=False)

    assert res.frames == n
    assert res.source_codec == "av1"
    assert codec_of(out) == "h264", "출력은 H.264 고정이다"
    frames = read_frames(str(out))
    leaked = [i for i, f in enumerate(frames)
              if not region_is_obscured(f, face_rect(i, *size))]
    assert not leaked, f"원본 얼굴이 남은 프레임: {leaked}"


def test_unreadable_input_fails_permanently(tmp_path):
    """깨진 입력은 재시도해도 같다 — VideoOpenError 계열로 던진다."""
    from face_anonymizer.core.pipeline import VideoOpenError
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video" * 100)

    with pytest.raises(VideoOpenError):
        ingest.ensure_decodable(str(broken), str(tmp_path / "w"))


def test_missing_input_is_reported_as_video_open_error(tmp_path):
    from face_anonymizer.core.pipeline import VideoOpenError
    with pytest.raises(VideoOpenError):
        ingest.ensure_decodable(str(tmp_path / "nope.mp4"), str(tmp_path / "w"))


def test_isopened_alone_is_not_trusted(tmp_path, make_video):
    """isOpened() 는 AV1 에서 True 를 주고도 read() 가 실패한다."""
    src, n, size = make_video(frames=8)
    av1 = to_av1(src, tmp_path / "in.av1.mp4")
    import cv2
    cap = cv2.VideoCapture(av1)
    opened = cap.isOpened()
    cap.release()
    if opened and not ingest.opencv_can_decode(av1):
        assert True                       # 정확히 이 상황을 막는 게 이 모듈이다
    else:
        pytest.skip("이 빌드에서는 재현되지 않는다")


# ── 전사 진행률 ──────────────────────────────────────────────────────────────
#
# 긴 영상은 전사만 수십 초가 걸린다. 그동안 화면이 '준비 0%' 로 멈춰 있어서
# 두 번째 작업부터 멈춘 것처럼 보였다. 실제로는 돌고 있었다.

def test_transcode_reports_progress(tmp_path, make_video):
    src, n, _size = make_video(name="src.mp4", frames=30)
    seen = []
    dst = str(tmp_path / "out.mp4")

    ingest.transcode(src, dst, progress=lambda d, t: seen.append((d, t)))

    assert seen, "진행률 보고가 한 번도 없었다"
    done = [d for d, _t in seen]
    assert done == sorted(done)                 # 뒤로 가지 않는다
    assert seen[-1][0] == seen[-1][1]           # 마지막 한 칸을 남기지 않는다
    assert seen[-1][1] > 0


def test_expected_frames_counts_the_source(make_video):
    src, n, _size = make_video(name="counted.mp4", frames=24)
    got = ingest.expected_frames(src)
    assert abs(got - n) <= 2, (got, n)


def test_transcode_works_without_a_progress_callback(tmp_path, make_video):
    src, _n, _size = make_video(name="quiet.mp4", frames=8)
    dst = str(tmp_path / "quiet_out.mp4")
    assert ingest.transcode(src, dst) == dst
    assert os.path.getsize(dst) > 0


# ── 입력 하드웨어 디코딩 ─────────────────────────────────────────────────────

def test_hwaccel_never_keeps_frames_on_gpu(monkeypatch):
    """**픽셀 경로를 바꾸면 안 된다.**

    `-hwaccel_output_format cuda` 를 붙이면 PCIe 왕복이 없어 더 빠른데, 실측에서
    검출된 프레임이 768 → 713 으로 7% 줄었다. 색 범위 처리가 달라져 중간 파일의
    픽셀이 미세하게 바뀌기 때문이다. 1초를 벌자고 얼굴 55프레임을 놓칠 수는 없다.
    """
    monkeypatch.setattr(ingest, "_hwaccel", True)
    for enc in ("h264_nvenc", "libx264"):
        args = ingest.hwaccel_args(enc)
        assert args == ["-hwaccel", "cuda"]
        assert "-hwaccel_output_format" not in args


def test_hwaccel_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(ingest, "_hwaccel", True)
    monkeypatch.setenv("FA_HWACCEL", "0")
    assert ingest.hwaccel_args("h264_nvenc") == []


def test_gpu_decode_failure_falls_back_to_cpu(monkeypatch, tmp_path):
    """이 파일에서만 GPU 디코딩이 안 될 수 있다. 그때 작업이 죽으면 안 된다."""
    monkeypatch.setattr(ingest, "_hwaccel", True)
    monkeypatch.setattr(ingest, "pick_encoder", lambda: ("h264_nvenc", "-cq", ()))
    monkeypatch.setattr(ingest, "expected_frames", lambda p: 10)
    dst = tmp_path / "out.mp4"
    seen = []

    def fake_run(cmd, total, progress, timeout):
        seen.append(cmd)
        if "-hwaccel" in cmd:
            return 1, "cuda decode failed"      # 첫 시도는 GPU — 실패시킨다
        dst.write_bytes(b"x" * 10)              # 두 번째는 CPU — 성공
        return 0, ""

    monkeypatch.setattr(ingest, "_run_with_progress", fake_run)
    monkeypatch.setattr(ingest, "opencv_can_decode", lambda p: True)
    ingest.transcode("in.mp4", str(dst))

    assert len(seen) == 2
    assert "-hwaccel" in seen[0] and "-hwaccel" not in seen[1]
