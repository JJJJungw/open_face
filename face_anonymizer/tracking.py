"""ByteTrack 추적 + 트랙 보간.

프레임 단위 검출은 특정 프레임에서 얼굴을 순간적으로 놓칠 수 있고, 블러/모자이크
파이프라인에서 그건 곧 프라이버시 누출이다. 그래서:
  1. supervision ByteTrack 으로 프레임 간 트랙을 연결하고,
  2. 트랙이 관측된 프레임들 사이의 빈 구간을 선형 보간으로 메꾸며,
  3. 마지막 관측 이후 linger 프레임 동안 박스를 유지한다.

프레임 스킵(detect_every > 1) 을 쓰면 검출하지 않은 프레임은 애초에 관측이
없으므로, 2번 보간이 그 구간을 덮는 유일한 안전장치가 된다. 그래서 스킵을
켜면서 보간을 끄는 조합은 파이프라인에서 막아 둔다.
"""

from collections import defaultdict

import numpy as np
import supervision as sv


def track_video_boxes(detections_per_step, fps=30.0, frame_indices=None):
    """검출 결과를 ByteTrack 에 순차 투입해 트랙 이력을 만든다.

    Parameters
    ----------
    detections_per_step : list[list[tuple]]
        각 추적 스텝의 [(x1,y1,x2,y2,score), ...]. 시간 순서대로.
    fps : float
        ByteTrack frame_rate 힌트. 프레임 스킵을 쓰면 실제 검출 주기
        (원본 fps / detect_every) 를 넘겨야 트랙 유지 시간이 맞는다.
    frame_indices : list[int] | None
        각 스텝이 대응하는 원본 프레임 번호. None 이면 0,1,2,... 로 간주.
        프레임 스킵을 쓸 때 결과를 원본 프레임 좌표계로 되돌리는 데 쓴다.

    Returns
    -------
    track_hist : dict[int, dict[int, tuple]]
        track_hist[track_id][frame_idx] = (x1,y1,x2,y2)
    """
    if frame_indices is None:
        frame_indices = range(len(detections_per_step))
    elif len(frame_indices) != len(detections_per_step):
        raise ValueError("frame_indices length must match detections_per_step")

    tracker = sv.ByteTrack(frame_rate=max(1, int(round(fps))))
    track_hist = defaultdict(dict)

    for frame_idx, raw in zip(frame_indices, detections_per_step):
        if raw:
            dets = sv.Detections(
                xyxy=np.array([b[:4] for b in raw], dtype=float),
                confidence=np.array([b[4] for b in raw], dtype=float),
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
      프레임 스킵(detect_every)으로 생긴 구멍도 여기서 덮인다.
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
