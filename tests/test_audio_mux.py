"""오디오 합성 회귀 테스트.

여기서 지키는 불변식은 하나다 — **익명화된 프레임은 어떤 경로로도 사라지지
않는다.** 오디오 합성은 부가 기능이고, 실패하면 무음으로라도 전부 남아야 한다.

과거 ``-shortest`` 를 쓰던 구현은 오디오가 영상보다 짧으면 영상을 잘라냈고,
ffmpeg 리턴코드가 0 이라 성공으로 보고됐다. 아래 첫 테스트가 그 회귀다.
"""

import os
import shutil
import subprocess

import pytest

from conftest import FakeDetector, read_frames

from face_anonymizer import VideoAnonymizer
from face_anonymizer import pipeline as P

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe 없음",
)


def with_audio(src, dst, seconds):
    """src 영상에 지정한 길이의 사인파 오디오를 붙인다 (영상 길이는 그대로)."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src,
         "-f", "lavfi", "-t", str(seconds), "-i", "sine=frequency=440:sample_rate=44100",
         "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", dst],
        check=True, capture_output=True)
    return dst


def run(inp, out, size, **kw):
    a = VideoAnonymizer(detector=FakeDetector(size))
    return a.process(str(inp), str(out), batch_size=8, **kw)


def test_short_audio_does_not_truncate_video(tmp_path, make_video):
    """오디오가 영상보다 훨씬 짧아도 익명화 프레임은 한 장도 잃지 않는다.

    이전 구현: 2.0초 영상 + 0.5초 오디오 -> 0.5초 결과물, audio='ok'.
    """
    src, n, size = make_video(frames=30, fps=15.0)          # 2.0초
    inp = with_audio(src, str(tmp_path / "in_audio.mp4"), 0.5)
    out = tmp_path / "out.mp4"

    res = run(inp, out, size)

    assert res.frames == n
    assert len(read_frames(str(out))) == n, "합성 과정에서 프레임이 잘렸다"
    got, _ = P.video_frame_count(str(out))
    assert got == n, f"파일에 실제로 들어 있는 프레임이 {got}/{n}"


def test_matching_audio_is_muxed(tmp_path, make_video):
    """길이가 맞는 오디오는 정상적으로 합성되고 결과물에 남는다."""
    src, n, size = make_video(frames=30, fps=15.0)
    inp = with_audio(src, str(tmp_path / "in_audio.mp4"), 2.0)
    out = tmp_path / "out.mp4"

    res = run(inp, out, size)

    assert res.audio == "ok"
    assert P.has_audio(str(out)) is True
    assert len(read_frames(str(out))) == n


def test_no_audio_source_reports_no_audio(tmp_path, make_video):
    src, n, size = make_video(frames=20)
    out = tmp_path / "out.mp4"

    res = run(src, out, size)

    assert res.audio == "no-audio"
    assert len(read_frames(str(out))) == n


def test_keep_audio_false(tmp_path, make_video):
    src, n, size = make_video(frames=20, fps=15.0)
    inp = with_audio(src, str(tmp_path / "in_audio.mp4"), 2.0)
    out = tmp_path / "out.mp4"

    res = run(inp, out, size, keep_audio=False)

    assert res.audio == "disabled"
    assert len(read_frames(str(out))) == n


def test_ffmpeg_missing_still_outputs_video(tmp_path, make_video, monkeypatch):
    src, n, size = make_video(frames=20, fps=15.0)
    inp = with_audio(src, str(tmp_path / "in_audio.mp4"), 2.0)
    out = tmp_path / "out.mp4"
    monkeypatch.setattr(P.shutil, "which", lambda _: None)

    res = run(inp, out, size)

    assert res.audio == "ffmpeg-missing"
    assert len(read_frames(str(out))) == n


def test_timeout_falls_back_to_silent_video(tmp_path, make_video, monkeypatch):
    """ffmpeg 가 매달려도 결과물은 나온다 (서버 워커가 영구 정지하면 안 된다)."""
    src, n, size = make_video(frames=20, fps=15.0)
    inp = with_audio(src, str(tmp_path / "in_audio.mp4"), 2.0)
    out = tmp_path / "out.mp4"

    real = P._run
    monkeypatch.setattr(P, "_run",
                        lambda cmd, timeout=P.FFMPEG_TIMEOUT:
                        None if cmd[0] == "ffmpeg" else real(cmd, timeout))

    res = run(inp, out, size)

    assert res.audio == "ffmpeg-timeout"
    assert len(read_frames(str(out))) == n


def test_frame_loss_in_muxed_output_is_rejected(tmp_path, make_video, monkeypatch):
    """합성 결과 프레임이 모자라면 합성을 버리고 무음본을 쓴다."""
    src, n, size = make_video(frames=20, fps=15.0)
    inp = with_audio(src, str(tmp_path / "in_audio.mp4"), 2.0)
    out = tmp_path / "out.mp4"
    monkeypatch.setattr(P, "video_frame_count", lambda p, t=None: (n - 5, 1.0))

    res = run(inp, out, size)

    assert res.audio.startswith("frame-loss:")
    assert len(read_frames(str(out))) == n, "무음본으로 폴백했으면 전부 남아야 한다"
    assert P.has_audio(str(out)) is False


def test_no_leftover_temp_files(tmp_path, make_video):
    """중간 산출물이 출력 디렉터리에 남지 않는다."""
    src, n, size = make_video(frames=20, fps=15.0)
    inp = with_audio(src, str(tmp_path / "in_audio.mp4"), 0.5)
    outdir = tmp_path / "out"
    outdir.mkdir()
    out = outdir / "out.mp4"

    run(inp, out, size)

    assert sorted(os.listdir(outdir)) == ["out.mp4"]
