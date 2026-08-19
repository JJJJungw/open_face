"""추적 문턱 회귀 테스트.

ByteTrack 은 활성화 임계값을 넘는 검출로만 트랙을 만들고, supervision 은 그
위에 0.1 을 더한 det_thresh 를 쓴다. 기본값으로 두면 실질 문턱이 0.35 가 되어
--conf 0.25 로 통과시킨 검출의 상당수가 트랙 없이 버려지고, 그러면 gap 보간도
linger 도 돌지 않는다.

간헐적으로 놓치는 얼굴이 곧 경계선 신뢰도 얼굴이므로, 보간이 가장 필요한
대상군에서만 안전망이 꺼지는 형태였다. 아래가 그 회귀다.
"""

import pytest

from conftest import FakeDetector, face_rect, read_frames, region_is_obscured

from face_anonymizer import VideoAnonymizer
from face_anonymizer.core.tracking import (
    TRACK_SCORE_FLOOR,
    interpolate,
    make_tracker,
    track_scores,
    track_video_boxes,
)


def boxes_at(score, n=20, w=320, h=240):
    """n 프레임 연속 검출. 전부 같은 점수."""
    return [[(*map(float, face_rect(i, w, h)), score)] for i in range(n)]


@pytest.mark.parametrize("conf", [0.15, 0.20, 0.25, 0.30, 0.40])
def test_tracks_are_created_at_pipeline_conf(conf):
    """conf 를 겨우 넘긴 검출도 트랙이 만들어져야 한다.

    이전 구현은 conf 0.30 에서 20프레임 연속 검출에도 트랙 0개였다.
    """
    dets = boxes_at(conf + 0.01)
    hist = track_video_boxes(dets, fps=30.0, conf=conf)

    assert hist, f"conf={conf}: 연속 검출인데 트랙이 하나도 없다"
    covered = max(len(h) for h in hist.values())
    assert covered >= 15, f"conf={conf}: 20프레임 중 {covered}프레임만 추적됨"


@pytest.mark.parametrize("conf", [0.15, 0.25, 0.40])
def test_effective_threshold_stays_below_score_floor(conf):
    """실질 문턱이 정규화 하한을 넘으면 트랙이 만들어지지 않는다."""
    t = make_tracker(fps=30.0, conf=conf)
    for attr in ("det_thresh", "track_activation_threshold", "track_thresh"):
        v = getattr(t, attr, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            assert v < TRACK_SCORE_FLOOR, f"{attr}={v} 가 하한 {TRACK_SCORE_FLOOR} 이상"


@pytest.mark.parametrize("conf", [0.15, 0.25, 0.40])
def test_track_scores_preserve_order_and_floor(conf):
    """정규화는 순위를 보존하고, conf 를 겨우 넘긴 값도 하한 이상으로 올린다."""
    raw = [conf + 1e-6, conf + 0.1, 0.5, 0.99]
    out = track_scores(raw, conf)
    assert list(out) == sorted(out), "순위가 뒤집혔다"
    assert out[0] >= TRACK_SCORE_FLOOR - 1e-9
    assert out[-1] <= 1.0 + 1e-9


def test_track_scores_untouched_when_conf_is_none():
    raw = [0.11, 0.42, 0.93]
    assert list(track_scores(raw, None)) == pytest.approx(raw)


def test_matching_survives_kalman_drift_at_low_conf():
    """정규화 없이는 IoU x 점수 > 0.2 를 못 넘겨 트랙이 한 프레임 만에 죽었다."""
    hist = track_video_boxes(boxes_at(0.26, n=25), fps=30.0, conf=0.25)
    assert hist
    assert max(len(h) for h in hist.values()) >= 20


def test_default_conf_none_keeps_library_default():
    """conf 를 안 주면 supervision 기본값을 건드리지 않는다."""
    t = make_tracker(fps=30.0, conf=None)
    assert getattr(t, "det_thresh", 0.35) == pytest.approx(0.35)


def test_gap_is_interpolated_for_low_confidence_track():
    """저신뢰 검출에서도 끊긴 구간이 실제로 메워진다."""
    dets = boxes_at(0.26, n=20)
    for i in (8, 9, 10):
        dets[i] = []                       # 검출기가 3프레임을 놓쳤다

    hist = track_video_boxes(dets, fps=30.0, conf=0.25)
    frame_dets = {i: [b[:4] for b in raw] for i, raw in enumerate(dets)}
    frame_dets = {k: list(v) for k, v in frame_dets.items()}

    from collections import defaultdict
    dd = defaultdict(list, frame_dets)
    _, added = interpolate(dd, hist, 20, linger=5)

    assert added > 0, "트랙은 생겼는데 보간이 하나도 채우지 못했다"
    for i in (8, 9, 10):
        assert dd[i], f"프레임 {i} 가 비어 있다 — 얼굴이 그대로 노출된다"


def test_pipeline_no_leak_with_low_confidence_detector(tmp_path, make_video):
    """전 구간: 저신뢰 검출 + 중간 미검출에서도 원본 얼굴이 남지 않는다."""
    src, n, size = make_video(frames=30)
    out = tmp_path / "out.mp4"

    det = FakeDetector(size, miss_frames={12, 13, 14}, score=0.27)
    res = VideoAnonymizer(detector=det).process(
        str(src), str(out), conf=0.25, batch_size=8, keep_audio=False)

    assert res.filled_boxes > 0, "보간이 전혀 돌지 않았다 (트랙 미생성 의심)"
    frames = read_frames(str(out))
    leaked = [i for i, f in enumerate(frames)
              if not region_is_obscured(f, face_rect(i, *size))]
    assert not leaked, f"원본 얼굴이 남은 프레임: {leaked}"
