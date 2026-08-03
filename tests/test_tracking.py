"""추적/보간 테스트.

보간은 이 프로젝트의 프라이버시 주장을 지탱하는 핵심 로직이다.
"검출이 몇 프레임 끊겨도 얼굴이 노출되지 않는다"는 말이 참인지 여기서 따진다.
"""

from collections import defaultdict

import pytest

from face_anonymizer.tracking import interpolate, track_video_boxes


def test_interpolate_fills_gap_linearly():
    frame_dets = defaultdict(list)
    hist = {0: (0.0, 0.0, 10.0, 10.0), 4: (40.0, 0.0, 50.0, 10.0)}

    interpolate(frame_dets, {1: hist}, total_frames=10, linger=0)

    # 1~3 프레임이 등간격으로 채워져야 한다
    assert [round(frame_dets[f][0][0]) for f in (1, 2, 3)] == [10, 20, 30]
    assert 0 not in frame_dets and 4 not in frame_dets   # 관측 프레임은 건드리지 않음


def test_interpolate_reports_added_count():
    frame_dets = defaultdict(list)
    _, added = interpolate(
        frame_dets, {1: {0: (0, 0, 10, 10), 5: (0, 0, 10, 10)}},
        total_frames=20, linger=3,
    )
    assert added == 4 + 3          # gap 4개 + linger 3개


def test_interpolate_no_gap_adds_nothing():
    frame_dets = defaultdict(list)
    hist = {0: (0, 0, 10, 10), 1: (1, 1, 11, 11), 2: (2, 2, 12, 12)}
    _, added = interpolate(frame_dets, {1: hist}, total_frames=10, linger=0)
    assert added == 0 and len(frame_dets) == 0


def test_linger_extends_past_last_observation():
    frame_dets = defaultdict(list)
    interpolate(frame_dets, {1: {3: (0, 0, 10, 10)}}, total_frames=100, linger=5)
    assert sorted(frame_dets) == [4, 5, 6, 7, 8]


def test_linger_respects_total_frames():
    """영상 끝을 넘어가는 프레임 번호를 만들면 렌더 단계에서 조용히 버려져
    디버깅이 어려워진다. 애초에 넘지 않아야 한다."""
    frame_dets = defaultdict(list)
    interpolate(frame_dets, {1: {8: (0, 0, 10, 10)}}, total_frames=10, linger=5)
    assert sorted(frame_dets) == [9]


def test_interpolate_respects_total_frames_in_gap():
    frame_dets = defaultdict(list)
    interpolate(
        frame_dets, {1: {0: (0, 0, 10, 10), 20: (0, 0, 10, 10)}},
        total_frames=5, linger=0,
    )
    assert max(frame_dets) < 5


def test_multiple_tracks_are_independent():
    frame_dets = defaultdict(list)
    hist_a = {0: (0, 0, 10, 10), 3: (30, 0, 40, 10)}
    hist_b = {0: (0, 50, 10, 60), 3: (30, 50, 40, 60)}
    interpolate(frame_dets, {1: hist_a, 2: hist_b}, total_frames=10, linger=0)
    assert all(len(frame_dets[f]) == 2 for f in (1, 2))


# ------------------------------------------------------------- ByteTrack 연동

def _straight_line(n=20, step=5):
    """등속으로 움직이는 얼굴 하나."""
    return [[(float(i * step), 50.0, float(i * step + 40), 100.0, 0.9)]
            for i in range(n)]


def test_track_video_boxes_keeps_single_identity():
    hist = track_video_boxes(_straight_line(), fps=30.0)
    assert len(hist) == 1, f"등속 단일 객체인데 트랙이 {len(hist)}개로 쪼개졌다"


def test_track_video_boxes_maps_custom_frame_indices():
    """프레임 스킵을 쓰면 추적 스텝 번호 != 원본 프레임 번호다.

    이 매핑이 어긋나면 보간 박스가 엉뚱한 프레임에 찍혀 노출이 생긴다.
    """
    dets = _straight_line(n=6)
    idx = [0, 3, 6, 9, 12, 15]
    hist = track_video_boxes(dets, fps=10.0, frame_indices=idx)
    observed = sorted(next(iter(hist.values())))
    assert set(observed).issubset(set(idx))
    assert max(observed) > 5, "원본 프레임 번호가 아니라 스텝 번호로 기록됐다"


def test_track_video_boxes_handles_empty_frames():
    dets = _straight_line(n=5) + [[]] * 3 + _straight_line(n=5)
    hist = track_video_boxes(dets, fps=30.0)     # 예외 없이 돌아야 한다
    assert isinstance(hist, dict)


def test_track_video_boxes_validates_index_length():
    with pytest.raises(ValueError):
        track_video_boxes(_straight_line(n=3), frame_indices=[0, 1])


def test_linger_box_grows_with_time_since_last_observation():
    """마지막 관측 이후에는 대상 위치를 알 수 없다. 정지한 박스를 유지하면
    움직이는 얼굴이 박스 밖으로 빠져나가므로, 경과 프레임에 비례해 넓어져야 한다."""
    frame_dets = defaultdict(list)
    hist = {i: (i * 10.0, 0.0, i * 10.0 + 20, 20.0) for i in range(5)}   # 10px/frame
    interpolate(frame_dets, {1: hist}, total_frames=20, linger=3)

    widths = [frame_dets[f][0][2] - frame_dets[f][0][0] for f in (5, 6, 7)]
    assert widths[0] < widths[1] < widths[2]
    assert widths[0] == pytest.approx(20 + 2 * 10)      # 1프레임 경과 x 10px/frame


def test_linger_box_does_not_grow_for_static_track():
    """정지한 대상까지 박스를 키우면 배경만 쓸데없이 가린다."""
    frame_dets = defaultdict(list)
    hist = {i: (0.0, 0.0, 20.0, 20.0) for i in range(5)}
    interpolate(frame_dets, {1: hist}, total_frames=20, linger=3)
    assert all(frame_dets[f][0] == (0.0, 0.0, 20.0, 20.0) for f in (5, 6, 7))


def test_recent_speed_of_single_observation_is_zero():
    from face_anonymizer.tracking import recent_speed
    assert recent_speed({3: (0, 0, 10, 10)}, [3]) == (0.0, 0.0)
