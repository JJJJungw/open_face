"""테스트 공용 픽스처.

여기서 중요한 건 **가중치도 torch 도 없이 파이프라인 전 구간을 돌린다**는 점이다.
검출기는 `VideoAnonymizer(detector=...)` 로 주입할 수 있으므로, 얼굴 위치를 우리가
아는 합성 영상 + 그 위치를 그대로 돌려주는 가짜 검출기를 쓰면
"검출 → 추적 → 보간 → 렌더 → 오디오" 배선이 맞는지 결정적으로 검증할 수 있다.

실제 모델 정확도(WIDERFace 성능)는 여기서 검증하지 않는다. 그건 가중치가 필요한
별도 관심사이고, 파이프라인 회귀와는 분리하는 편이 CI 를 가볍게 유지한다.
"""

import os
import subprocess

import cv2
import numpy as np
import pytest

FACE_COLOR = (40, 60, 220)      # BGR — 배경과 확실히 구분되는 색
BG_COLOR = (30, 30, 30)


def face_rect(i, w, h, box_w=48, box_h=60):
    """프레임 i 에서 '얼굴'이 있어야 할 위치. 좌->우로 등속 이동."""
    span = w - box_w - 20
    x1 = 10 + int(span * (i % 30) / 29.0)
    y1 = h // 2 - box_h // 2
    return x1, y1, x1 + box_w, y1 + box_h


@pytest.fixture
def make_video(tmp_path):
    """합성 영상을 만들어 (경로, 프레임수, 크기) 를 돌려주는 팩토리."""

    def _make(name="in.mp4", frames=45, w=320, h=240, fps=15.0, with_audio=False):
        path = tmp_path / name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        assert writer.isOpened(), "테스트 환경에 mp4v 인코더가 없다"
        for i in range(frames):
            frame = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)
            # 배경에 노이즈를 넣어 '모자이크 후 균일해짐' 을 검증할 수 있게 한다
            frame[:, :, 1] = (np.arange(w, dtype=np.uint8)[None, :] * 3) % 255
            x1, y1, x2, y2 = face_rect(i, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), FACE_COLOR, -1)
            # 얼굴 안쪽에 고주파 패턴 → 모자이크/블러가 실제로 지웠는지 판정용
            for k in range(y1, y2, 4):
                cv2.line(frame, (x1, k), (x2, k), (250, 250, 250), 1)
            writer.write(frame)
        writer.release()
        assert path.exists() and path.stat().st_size > 0

        if with_audio:
            path = _add_silent_audio(path, frames / fps)
        return str(path), frames, (w, h)

    return _make


def _add_silent_audio(path, seconds):
    """ffmpeg 가 있으면 무음 트랙을 붙인 사본을 만든다. 없으면 원본 그대로."""
    import shutil

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg 가 없어 오디오 경로를 테스트할 수 없다")
    out = path.with_name("with_audio.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-f", "lavfi", "-t", f"{seconds:.3f}",
         "-i", "anullsrc=r=44100:cl=mono", "-c:v", "copy", "-c:a", "aac",
         "-shortest", str(out)],
        capture_output=True, check=True,
    )
    return out


class FakeDetector:
    """얼굴 위치를 이미 아는 가짜 검출기.

    Parameters
    ----------
    miss_frames : set[int]
        일부러 검출에 실패할 프레임 번호. 보간이 이 구멍을 메우는지 보는 데 쓴다.
    """

    def __init__(self, size, miss_frames=(), score=0.9):
        self.w, self.h = size
        self.miss = set(miss_frames)
        self.score = score
        self.calls = 0
        self.batch_sizes = []
        self._frame_no = 0

    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        self.calls += 1
        self.batch_sizes.append(len(frames))
        out = []
        for _ in frames:
            i = self._frame_no
            self._frame_no += 1
            if i in self.miss:
                out.append([])
            else:
                x1, y1, x2, y2 = face_rect(i, self.w, self.h)
                out.append([(float(x1), float(y1), float(x2), float(y2), self.score)])
        return out

    def detect(self, frame, **kw):
        return self.detect_batch([frame], **kw)[0]


class IndexedFakeDetector(FakeDetector):
    """프레임 스킵 테스트용 — 실제 프레임 번호를 픽셀에서 읽지 않고,
    호출 순서가 아니라 지정한 인덱스 목록을 따라간다."""

    def __init__(self, size, frame_indices, miss_frames=(), score=0.9):
        super().__init__(size, miss_frames, score)
        self._indices = list(frame_indices)
        self._pos = 0

    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        self.calls += 1
        self.batch_sizes.append(len(frames))
        out = []
        for _ in frames:
            i = self._indices[self._pos]
            self._pos += 1
            if i in self.miss:
                out.append([])
            else:
                x1, y1, x2, y2 = face_rect(i, self.w, self.h)
                out.append([(float(x1), float(y1), float(x2), float(y2), self.score)])
        return out


@pytest.fixture
def fake_detector():
    return FakeDetector


# 합성 얼굴은 4px 간격 흰 줄무늬가 들어 있어 라플라시안 분산이 ~25,000 이다.
# 실제 측정값: 원본 25,000 / 모자이크(scale=0.05) 270 / 약한 모자이크(0.15) 880 /
# 블러 12 / 박스 0. 2,000 이면 어느 방식이든 여유 있게 구분한다.
DETAIL_THRESHOLD = 2000.0


def detail_score(frame, box):
    """box 영역의 고주파 성분 양 (라플라시안 분산).

    평균 색이나 표준편차가 아니라 **디테일**을 재는 게 핵심이다. 모자이크는
    영역의 평균 색을 유지하므로 std 로는 익명화 여부를 가릴 수 없지만,
    사람을 식별하게 하는 건 그 안의 고주파 디테일이다.
    """
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    roi = frame[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def region_is_obscured(frame, box, threshold=DETAIL_THRESHOLD):
    """box 영역의 식별 가능한 디테일이 지워졌는지."""
    return detail_score(frame, box) < threshold
