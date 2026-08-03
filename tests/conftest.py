"""테스트 공용 픽스처.

핵심: **가중치도 torch 도 없이** 파이프라인 전 구간을 돌린다. 검출기를
`VideoAnonymizer(detector=...)` 로 주입할 수 있으므로, 얼굴 위치를 우리가 아는
합성 영상 + 그 위치를 돌려주는 가짜 검출기면 배선 검증에 충분하다.

모델 정확도(WIDERFace 성능)는 여기서 보지 않는다. 가중치가 필요한 별도 관심사다.
"""

import cv2
import numpy as np
import pytest

FACE_COLOR = (40, 60, 220)      # BGR
BG_COLOR = (30, 30, 30)


def face_rect(i, w, h, box_w=48, box_h=60):
    """프레임 i 에서 '얼굴'이 있어야 할 위치. 좌->우로 등속 이동."""
    x1 = 10 + int((w - box_w - 20) * (i % 30) / 29.0)
    y1 = h // 2 - box_h // 2
    return x1, y1, x1 + box_w, y1 + box_h


@pytest.fixture
def make_video(tmp_path):
    def _make(name="in.mp4", frames=30, w=320, h=240, fps=15.0):
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
        assert writer.isOpened(), "테스트 환경에 mp4v 인코더가 없다"
        for i in range(frames):
            frame = np.full((h, w, 3), BG_COLOR, np.uint8)
            frame[:, :, 1] = (np.arange(w, dtype=np.uint8)[None, :] * 3) % 255
            x1, y1, x2, y2 = face_rect(i, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), FACE_COLOR, -1)
            # 얼굴 안쪽 고주파 패턴 — 익명화가 실제로 지웠는지 판정하는 근거
            for k in range(y1, y2, 4):
                cv2.line(frame, (x1, k), (x2, k), (250, 250, 250), 1)
            writer.write(frame)
        writer.release()
        return str(path), frames, (w, h)

    return _make


class FakeDetector:
    """얼굴 위치를 이미 아는 가짜 검출기.

    miss_frames 에 넣은 프레임은 일부러 놓친다 — 보간이 그 구멍을 메우는지
    보는 데 쓴다.
    """

    def __init__(self, size, miss_frames=()):
        self.w, self.h = size
        self.miss = set(miss_frames)
        self.batch_sizes = []
        self._n = 0

    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        self.batch_sizes.append(len(frames))
        out = []
        for _ in frames:
            i, self._n = self._n, self._n + 1
            if i in self.miss:
                out.append([])
            else:
                out.append([(*map(float, face_rect(i, self.w, self.h)), 0.9)])
        return out


# 합성 얼굴은 4px 간격 줄무늬가 있어 라플라시안 분산이 ~25,000 이다.
# 모자이크 270 / 블러 12 / 박스 0 이므로 2,000 이면 여유 있게 구분한다.
def region_is_obscured(frame, box, threshold=2000.0):
    """평균 색이 아니라 **고주파 디테일**로 판정한다.

    모자이크는 영역 평균 색을 유지하므로 표준편차로는 가려졌는지 알 수 없다.
    """
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    roi = frame[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return True
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


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
