"""ByteTrack 추적 + 트랙 보간.

프레임 단위 검출은 특정 프레임에서 얼굴을 순간적으로 놓칠 수 있고, 블러/모자이크
파이프라인에서 그건 곧 프라이버시 누출이다. 그래서:
  1. supervision ByteTrack 으로 프레임 간 트랙을 연결하고,
  2. 트랙이 관측된 프레임들 사이의 빈 구간을 선형 보간으로 메꾸며,
  3. 마지막 관측 이후 linger 프레임 동안 박스를 유지한다.
"""

from collections import defaultdict

import numpy as np
import supervision as sv


def track_video_boxes(detections_per_frame, fps=30.0):
    """프레임별 raw 검출을 ByteTrack 에 순차 투입해 트랙 이력을 만든다.

    Parameters
    ----------
    detections_per_frame : list[list[tuple]]
        각 프레임의 [(x1,y1,x2,y2,score), ...]. 프레임 순서대로.
    fps : float
        ByteTrack frame_rate 힌트.

    Returns
    -------
    track_hist : dict[int, dict[int, tuple]]
        track_hist[track_id][frame_idx] = (x1,y1,x2,y2)
    """
    tracker = sv.ByteTrack(frame_rate=int(round(fps)))
    track_hist = defaultdict(dict)

    for idx, raw in enumerate(detections_per_frame):
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
                track_hist[int(tid)][idx] = tuple(float(v) for v in box)
    return track_hist


def interpolate(frame_dets, track_hist, total_frames, linger=5):
    """track_hist 기반으로 끊긴 프레임의 박스를 frame_dets 에 채워 넣는다.

    frame_dets : dict[int, list[tuple]] — 프레임별 익명화 대상 박스 (in-place 수정)
    """
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
                t = (f - a) / (b - a)
                frame_dets[f].append(tuple(xa[k] + (xb[k] - xa[k]) * t for k in range(4)))
        # 마지막 관측 이후 linger 유지
        last = frames[-1]
        for f in range(last + 1, min(total_frames, last + 1 + linger)):
            frame_dets[f].append(hist[last])
    return frame_dets
