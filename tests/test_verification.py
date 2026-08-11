"""디코딩 완결성 · 검출 신뢰도 검증 회귀 테스트.

두 사고 유형을 막는다.

* 디코딩이 중간에 끊겨도 성공하던 것 — 영상 뒷부분이 결과물에 통째로 없는데
  예외가 안 났다. 1·2차 패스 프레임 수 비교는 둘 다 같은 지점에서 끊기면
  통과하므로 이걸 못 잡는다.
* 검출 0건도 정상 성공이던 것 — 가중치 손상·회전 영상·잘못된 imgsz 등 원인이
  무엇이든 결과가 똑같이 조용했다.
"""

import pytest

from conftest import FakeDetector

from face_anonymizer import VideoAnonymizer, probe
from face_anonymizer import pipeline as P


class NoFaceDetector:
    """아무것도 못 찾는 검출기 (가중치 손상 / 회전 영상 상황 재현)."""

    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        return [[] for _ in frames]


def run(src, out, size, detector=None, **kw):
    a = VideoAnonymizer(detector=detector or FakeDetector(size))
    kw.setdefault("keep_audio", False)
    return a.process(str(src), str(out), batch_size=8, **kw)


# ── 디코딩 완결성 ────────────────────────────────────────────────────────────

def test_probe_uses_video_stream_duration(make_video):
    """기대 프레임 수는 비디오 스트림 길이 x fps 로 잡는다."""
    src, n, _ = make_video(frames=30)
    info = probe(src)
    assert info.count_source in ("duration", "container")
    assert info.frame_count == n


def test_truncated_decode_fails(tmp_path, make_video, monkeypatch):
    """디코딩이 절반만 되면 실패해야 한다 (예전에는 조용히 성공했다)."""
    src, n, size = make_video(frames=30)
    real = P.probe
    monkeypatch.setattr(
        P, "probe",
        lambda p: P.VideoInfo(**{**vars(real(p)), "frame_count": n * 2,
                                 "count_source": "packets"}))

    with pytest.raises(P.DecodeIncompleteError) as e:
        run(src, tmp_path / "out.mp4", size)
    assert f"{n}/{n * 2}" in str(e.value)


def test_allow_partial_downgrades_to_warning(tmp_path, make_video, monkeypatch):
    src, n, size = make_video(frames=30)
    real = P.probe
    monkeypatch.setattr(
        P, "probe",
        lambda p: P.VideoInfo(**{**vars(real(p)), "frame_count": n * 2,
                                 "count_source": "packets"}))

    res = run(src, tmp_path / "out.mp4", size, allow_partial=True)
    assert any(w.startswith("decode-partial") for w in res.warnings)


def test_small_count_mismatch_is_tolerated(tmp_path, make_video, monkeypatch):
    """마지막 GOP 처리 차이로 한두 장 어긋나는 건 정상으로 본다."""
    src, n, size = make_video(frames=30)
    real = P.probe
    monkeypatch.setattr(
        P, "probe",
        lambda p: P.VideoInfo(**{**vars(real(p)), "frame_count": n + 2,
                                 "count_source": "packets"}))

    res = run(src, tmp_path / "out.mp4", size)
    assert not [w for w in res.warnings if w.startswith("decode")]


def test_unknown_count_skips_check(tmp_path, make_video, monkeypatch):
    """프레임 수를 모르면 검사를 건너뛰되 그 사실을 남긴다."""
    src, n, size = make_video(frames=20)
    real = P.probe
    monkeypatch.setattr(
        P, "probe",
        lambda p: P.VideoInfo(**{**vars(real(p)), "frame_count": 0,
                                 "count_source": "unknown"}))

    res = run(src, tmp_path / "out.mp4", size)
    assert "decode-unverified" in res.warnings


# ── 검출 신뢰도 ──────────────────────────────────────────────────────────────

def test_zero_detections_is_flagged_not_silent(tmp_path, make_video):
    """검출 0건은 실패는 아니지만 반드시 드러나야 한다."""
    src, n, size = make_video(frames=20)
    res = run(src, tmp_path / "out.mp4", size, detector=NoFaceDetector())

    assert res.raw_boxes == 0
    assert "no-detections" in res.warnings
    assert res.detection_rate == 0.0


def test_min_detection_rate_can_force_failure(tmp_path, make_video):
    """얼굴이 반드시 있는 영상을 도는 파이프라인은 실패로 만들 수 있다."""
    src, n, size = make_video(frames=20)
    with pytest.raises(P.DetectionSanityError):
        run(src, tmp_path / "out.mp4", size, detector=NoFaceDetector(),
            min_detection_rate=0.5)


def test_normal_run_has_no_warnings(tmp_path, make_video):
    src, n, size = make_video(frames=20)
    res = run(src, tmp_path / "out.mp4", size)

    assert res.warnings == ()
    assert res.detection_rate == 1.0
    assert res.detected_frames == n


def test_detection_rate_counts_frames_not_boxes(tmp_path, make_video):
    """미검출 프레임이 있으면 비율에 그대로 반영된다."""
    src, n, size = make_video(frames=20)
    res = run(src, tmp_path / "out.mp4", size,
              detector=FakeDetector(size, miss_frames={0, 1, 2, 3}))

    assert res.detected_frames == n - 4
    assert res.detection_rate == pytest.approx((n - 4) / n)


# ── 실제 파일에서 오탐이 나지 않는가 ─────────────────────────────────────────
#
# 검증이 지나치게 빡빡하면 정상 영상을 거부한다. 서비스에서는 그게 더 큰 사고라,
# 아래 두 형태는 반드시 통과해야 한다.

import shutil as _shutil
import subprocess as _sp

ffmpeg_only = pytest.mark.skipif(
    not (_shutil.which("ffmpeg") and _shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe 없음")


@ffmpeg_only
def test_trimmed_video_is_not_rejected(tmp_path, make_video):
    """앞부분을 잘라낸 영상(edit list)을 거부하면 안 된다.

    컨테이너에 패킷은 그대로 남고 재생 대상만 줄어든다. 실측: 앞 0.5초를 자른
    파일이 패킷 30개 / 실제 디코딩 22프레임. 패킷 수를 기준으로 삼으면 멀쩡한
    영상이 '8장 누락'으로 보인다. 아이폰·편집 앱을 거친 영상 상당수가 이 형태다.
    """
    src, n, size = make_video(frames=30, fps=15.0)
    h264 = tmp_path / "src264.mp4"
    trimmed = tmp_path / "trimmed.mp4"
    _sp.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264",
             "-g", "250", "-crf", "23", "-preset", "veryfast", str(h264)],
            check=True)
    _sp.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(h264),
             "-c", "copy", str(trimmed)], check=True)

    res = run(trimmed, tmp_path / "out.mp4", size)

    assert not [w for w in res.warnings if w.startswith("decode")], res.warnings


@ffmpeg_only
def test_audio_longer_than_video_is_not_rejected(tmp_path, make_video):
    """오디오가 영상보다 길어도 거부하면 안 된다.

    format.duration 은 모든 스트림 중 가장 긴 값이라, 그걸 기준으로 삼으면
    영상 1.33초 + 오디오 3초 파일에서 20프레임을 45프레임으로 계산한다.
    마이크가 늦게 끊긴 녹화물에서 흔하다.
    """
    src, n, size = make_video(name="clip.mp4", frames=20, fps=15.0)
    withaudio = tmp_path / "with_audio.mp4"
    _sp.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-f", "lavfi", "-t", "3.0", "-i", "sine=frequency=440",
             "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
             str(withaudio)], check=True)

    res = run(withaudio, tmp_path / "out.mp4", size, keep_audio=True)

    assert not [w for w in res.warnings if w.startswith("decode")], res.warnings
    assert res.frames == n


@ffmpeg_only
def test_moderate_gap_warns_but_does_not_fail(tmp_path, make_video, monkeypatch):
    """애매한 차이는 실패가 아니라 경고다 (정상 영상 거부 방지)."""
    src, n, size = make_video(frames=100)
    real = P.probe
    monkeypatch.setattr(                      # 10% 부족하게 보이도록
        P, "probe",
        lambda p: P.VideoInfo(**{**vars(real(p)),
                                 "frame_count": int(n / 0.9),
                                 "count_source": "duration"}))

    res = run(src, tmp_path / "out.mp4", size)
    assert any(w.startswith("decode-short") for w in res.warnings)


@ffmpeg_only
def test_output_is_h264(tmp_path, make_video):
    """다운로드 결과물은 H.264 여야 한다 (mp4v 는 같은 화질에 약 9.5배)."""
    if P.pick_encoder() is None:
        pytest.skip("H.264 인코더 없음")
    src, n, size = make_video(frames=20)
    out = tmp_path / "out.mp4"

    run(src, out, size)

    codec = _sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                     str(out)], capture_output=True, text=True).stdout.strip()
    assert codec == "h264", f"코덱이 {codec}"


# ── 비트레이트 상한 ─────────────────────────────────────────────────────────
#
# 실측(1080p AV1 632 kbps): 상한을 원본 그대로 걸어 639 kbps 로 뽑으니 평평한
# 면이 전부 블록으로 깨졌다. 같은 원본을 상한 없이 뽑으면 2.16 Mbps 다.

def test_bitrate_cap_accounts_for_source_codec(monkeypatch):
    """AV1 632 kbps 를 H.264 632 kbps 상한으로 받으면 안 된다."""
    monkeypatch.setattr(P, "video_bitrate", lambda p, t=None: 632_561)

    monkeypatch.setattr(P, "video_codec", lambda p, t=None: "av1")
    av1 = P.bitrate_cap("x.mp4", 1.0)

    monkeypatch.setattr(P, "video_codec", lambda p, t=None: "h264")
    h264 = P.bitrate_cap("x.mp4", 1.0)

    assert h264 == 632_561              # 같은 코덱이면 원본 그대로
    assert av1 > 1_200_000              # AV1 이면 넉넉하게
    assert av1 == pytest.approx(h264 * 2.0)


def test_bitrate_cap_is_off_when_ratio_is_zero(monkeypatch):
    monkeypatch.setattr(P, "video_bitrate", lambda p, t=None: 632_561)
    monkeypatch.setattr(P, "video_codec", lambda p, t=None: "av1")
    assert P.bitrate_cap("x.mp4", 0) is None


def test_unknown_codec_falls_back_to_plain_ratio(monkeypatch):
    monkeypatch.setattr(P, "video_bitrate", lambda p, t=None: 1_000_000)
    monkeypatch.setattr(P, "video_codec", lambda p, t=None: "weirdcodec")
    assert P.bitrate_cap("x.mp4", 1.0) == 1_000_000
