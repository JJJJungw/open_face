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
    return a.process(str(src), str(out), batch_size=8, keep_audio=False, **kw)


# ── 디코딩 완결성 ────────────────────────────────────────────────────────────

def test_probe_prefers_packet_count(make_video):
    """ffprobe 로 실제 패킷을 세면 컨테이너 추정값보다 믿을 수 있다."""
    src, n, _ = make_video(frames=30)
    info = probe(src)
    assert info.count_source in ("packets", "container")
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
