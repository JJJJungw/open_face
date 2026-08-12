"""ByteTrack 추적 + 트랙 보간.

프레임 단위 검출은 특정 프레임에서 얼굴을 순간적으로 놓칠 수 있고, 블러/모자이크
파이프라인에서 그건 곧 프라이버시 누출이다. 그래서:
  1. supervision ByteTrack 으로 프레임 간 트랙을 연결하고,
  2. 트랙이 관측된 프레임들 사이의 빈 구간을 선형 보간으로 메꾸며,
  3. 마지막 관측 이후 linger 프레임 동안 박스를 유지한다.

검출은 모든 프레임에서 돌린다. 프레임을 건너뛰면 건너뛴 구간의 커버리지를
보장할 수 없기 때문이다. 처리량은 배치 추론과 FP16 으로 올린다.
"""

from collections import defaultdict

import numpy as np
import supervision as sv


# ── 추적 문턱 ───────────────────────────────────────────────────────────────
#
# ByteTrack 이 저신뢰 검출을 버리는 경로는 두 군데다. 둘 다 막아야 한다.
#
# 1. **활성화 문턱** — supervision 은 det_thresh = 활성화 임계값 + 0.1 을 쓴다.
#    기본값 0.25 면 실질 문턱이 0.35 라, --conf 0.25 로 통과시킨 검출 중
#    0.25~0.35 구간은 트랙이 아예 생기지 않는다.
#
# 2. **매칭 비용에 점수가 곱해진다** (matching.fuse_score). 비용이
#    ``1 - IoU × 점수`` 이고 ``< minimum_matching_threshold(0.8)`` 일 때만
#    매칭되므로, 실질 조건은 ``IoU × 점수 > 0.2`` 다. 점수가 0.26 이면 IoU 가
#    0.77 을 넘어야 하는데 칼만 예측 오차만으로도 그 밑으로 떨어진다.
#    실측: 20프레임 연속 검출(점수 0.26)에 트랙이 **1프레임만** 유지됐다.
#
# 1번만 고치면 트랙이 생겼다가 다음 프레임에 죽는다. 그래서 추적에 넘기는
# 점수를 정규화한다 — 자세한 이유는 track_scores() 참고.
TRACK_SCORE_FLOOR = 0.7      # 정규화 후 최저 점수
TRACK_ACTIVATION = 0.5       # det_thresh = 0.6 < 0.7 이라 여유가 있다


def track_scores(scores, conf):
    """추적용 점수 정규화. [conf, 1] -> [TRACK_SCORE_FLOOR, 1] 선형 사상.

    파이프라인이 이미 ``conf`` 로 한 번 걸렀으므로, 통과한 검출은 전부 추적할
    가치가 있는 것들이다. 트래커에게 필요한 건 점수의 **절대값**이 아니라
    **상대 순위**뿐이다(누구를 먼저 매칭할지). 그런데 fuse_score 가 절대값을
    비용에 곱하는 탓에, 임계값을 낮게 잡을수록 트랙이 유지되지 않는다.

    순위를 보존한 채 하한만 끌어올려서 매칭이 점수 크기 때문에 깨지지 않게 한다.
    더 확신 있는 검출이 여전히 우선 매칭되는 성질은 그대로다.

    **이 점수는 추적 전용이다.** 익명화 대상 박스는 원본 검출을 그대로 쓰므로,
    여기서 점수를 올린다고 해서 없던 얼굴이 가려지거나 하지는 않는다.
    """
    s = np.asarray(scores, dtype=float)
    if conf is None:
        return s
    span = max(1e-6, 1.0 - float(conf))
    t = np.clip((s - float(conf)) / span, 0.0, 1.0)
    return TRACK_SCORE_FLOOR + (1.0 - TRACK_SCORE_FLOOR) * t


def make_tracker(fps=30.0, conf=None):
    """정규화된 점수(track_scores)를 전제로 한 ByteTrack 인스턴스.

    ``conf=None`` 이면 supervision 기본값을 그대로 둔다.
    """
    kw = {"frame_rate": max(1, int(round(fps)))}
    tracker = None
    if conf is not None:
        # 0.22 즈음 track_thresh -> track_activation_threshold 로 이름이 바뀌었다.
        # 지원 범위(>=0.18,<0.30)에 둘 다 있어 순서대로 시도한다.
        for name in ("track_activation_threshold", "track_thresh"):
            try:
                tracker = sv.ByteTrack(**kw, **{name: TRACK_ACTIVATION})
                break
            except TypeError:
                continue
    if tracker is None:
        tracker = sv.ByteTrack(**kw)

    if conf is not None:
        # 인자 이름이 또 바뀌거나 내부 공식(+0.1)이 달라져도 실질 문턱이
        # 정규화 하한을 넘지 않게 못 박는다. 조용히 되돌아가는 종류의 회귀다.
        cap = TRACK_SCORE_FLOOR - 1e-6
        for attr in ("det_thresh", "track_activation_threshold", "track_thresh"):
            v = getattr(tracker, attr, None)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > cap:
                setattr(tracker, attr, cap)
    return tracker


def track_video_boxes(detections_per_frame, fps=30.0, conf=None):
    """프레임별 검출을 ByteTrack 에 순차 투입해 트랙 이력을 만든다.

    Parameters
    ----------
    detections_per_frame : list[list[tuple]]
        각 프레임의 [(x1,y1,x2,y2,score), ...]. 프레임 순서대로.
    fps : float
        ByteTrack frame_rate 힌트.
    conf : float | None
        파이프라인 검출 임계값. 추적 문턱과 점수 정규화의 기준이 된다.

    Returns
    -------
    track_hist : dict[int, dict[int, tuple]]
        track_hist[track_id][frame_idx] = (x1,y1,x2,y2)
    """
    tracker = make_tracker(fps, conf)
    track_hist = defaultdict(dict)

    for frame_idx, raw in enumerate(detections_per_frame):
        if raw:
            dets = sv.Detections(
                xyxy=np.array([b[:4] for b in raw], dtype=float),
                confidence=track_scores([b[4] for b in raw], conf),
                class_id=np.zeros(len(raw), dtype=int),
            )
        else:
            dets = sv.Detections.empty()
        tracked = tracker.update_with_detections(dets)
        if tracked.tracker_id is not None:
            for box, tid in zip(tracked.xyxy, tracked.tracker_id):
                if tid is None:
                    continue
                track_hist[int(tid)][int(frame_idx)] = tuple(float(v) for v in box)
    return track_hist


def recent_speed(hist, frames, lookback=5):
    """최근 관측 구간의 프레임당 평균 이동량 (|dx|, |dy|).

    linger 구간에서 박스를 얼마나 키울지 정하는 데 쓴다.
    """
    pts = frames[-(lookback + 1):]
    if len(pts) < 2:
        return 0.0, 0.0
    a, b = pts[0], pts[-1]
    dt = b - a
    if dt <= 0:
        return 0.0, 0.0
    ba, bb = hist[a], hist[b]
    cax, cay = (ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2
    cbx, cby = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    return abs(cbx - cax) / dt, abs(cby - cay) / dt


def interpolate(frame_dets, track_hist, total_frames, linger=5):
    """track_hist 기반으로 끊긴 프레임의 박스를 frame_dets 에 채워 넣는다.

    두 가지를 채운다.

    * **관측 사이의 gap** — 앞뒤 관측이 모두 있으므로 선형 보간으로 정확히 메운다.
      검출기가 몇 프레임을 놓쳐도 여기서 덮인다.
    * **마지막 관측 이후 linger 구간** — 뒤쪽 관측이 없어 위치를 알 수 없다.
      마지막 박스를 그대로 유지하면 움직이는 대상이 박스 밖으로 빠져나가므로,
      최근 이동 속도에 비례해 시간이 지날수록 박스를 키운다. 방향을 추정해
      틀리는 것보다 양쪽으로 넓히는 편이 누출 관점에서 안전하다.

    frame_dets : dict[int, list[tuple]] — 프레임별 익명화 대상 박스 (in-place 수정)

    Returns
    -------
    (frame_dets, added) — added 는 새로 채워 넣은 박스 개수.
    """
    added = 0
    for hist in track_hist.values():
        frames = sorted(hist.keys())
        if not frames:
            continue
        # 관측 사이 gap 선형 보간
        for a, b in zip(frames, frames[1:]):
            if b - a <= 1:
                continue
            xa, xb = hist[a], hist[b]
            for f in range(a + 1, b):
                if f >= total_frames:
                    break
                t = (f - a) / (b - a)
                frame_dets[f].append(
                    tuple(xa[k] + (xb[k] - xa[k]) * t for k in range(4))
                )
                added += 1

        # 마지막 관측 이후 linger 유지 (불확실성만큼 박스를 키운다)
        last = frames[-1]
        x1, y1, x2, y2 = hist[last]
        vx, vy = recent_speed(hist, frames)
        for f in range(last + 1, min(total_frames, last + 1 + linger)):
            k = f - last
            gx, gy = vx * k, vy * k
            frame_dets[f].append((x1 - gx, y1 - gy, x2 + gx, y2 + gy))
            added += 1
    return frame_dets, added
