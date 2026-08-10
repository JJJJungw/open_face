"""세로 촬영 영상 회전 처리.

폰 세로 영상은 픽셀이 가로로 저장되고 회전 메타데이터가 붙는다. 누운 프레임에
검출을 돌리면 얼굴을 거의 못 잡는데 크기 검사는 통과해 조용히 새어 나간다.
"""

import shutil
import subprocess

import cv2
import numpy as np
import pytest

from face_anonymizer import VideoAnonymizer, probe
from face_anonymizer import pipeline as P

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg 없음")


class NoFace:
    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        return [[] for _ in frames]


class Always:
    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        return [[(10.0, 10.0, 40.0, 40.0, 0.9)] for _ in frames]


def rotated(src, dst, deg):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", str(deg),
                    "-i", str(src), "-c", "copy", str(dst)], check=True)
    return dst


def dims(path):
    c = cv2.VideoCapture(str(path))
    c.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    ok, f = c.read()
    c.release()
    assert ok
    return f.shape[1], f.shape[0]


def test_rotate_frame_matches_opencv_auto_rotation(tmp_path, make_video):
    """META 각도 -> cv2.rotate 코드 매핑이 OpenCV 자동회전과 픽셀 단위로 같아야 한다.

    여기가 틀리면 프레임이 엉뚱하게 돌아가고, 그대로 검출을 돌린다.
    """
    src, n, _ = make_video(frames=3)
    for deg in (90, 180, 270):
        f = rotated(src, tmp_path / f"r{deg}.mp4", deg)
        cap = cv2.VideoCapture(str(f))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        meta = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
        ok, auto_frame = cap.read()
        cap.release()
        if not meta:
            pytest.skip("이 OpenCV 빌드는 회전 메타데이터를 노출하지 않는다")

        cap = cv2.VideoCapture(str(f))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        ok2, raw = cap.read()
        cap.release()
        if raw.shape == auto_frame.shape and np.array_equal(raw, auto_frame):
            continue                     # 자동회전을 끌 수 없는 빌드
        assert np.array_equal(P.rotate_frame(raw, meta), auto_frame), \
            f"{deg}도 파일: META={meta} 매핑이 자동회전 결과와 다르다"


def test_probe_reports_display_dimensions(tmp_path, make_video):
    """세로 영상은 회전 후 크기로 보고돼야 한다."""
    src, n, (w, h) = make_video(frames=6)
    f = rotated(src, tmp_path / "r90.mp4", 90)
    info = probe(str(f))
    assert (info.width, info.height) == dims(f)
    if info.meta_rotation in (90, 270):
        assert (info.width, info.height) == (h, w)


def test_rotated_video_output_is_upright(tmp_path, make_video):
    """세로 영상을 처리해도 프레임 수와 방향이 유지된다."""
    src, n, (w, h) = make_video(frames=12)
    f = rotated(src, tmp_path / "r90.mp4", 90)
    out = tmp_path / "out.mp4"

    res = VideoAnonymizer(detector=NoFace()).process(
        str(f), str(out), batch_size=4, keep_audio=False)

    assert res.frames == n
    assert dims(out) == dims(f)


def test_manual_rotate(tmp_path, make_video):
    """--rotate 는 메타데이터 위에 추가로 적용되고 출력 크기에 반영된다."""
    src, n, (w, h) = make_video(frames=12)
    out = tmp_path / "out.mp4"

    res = VideoAnonymizer(detector=Always()).process(
        str(src), str(out), batch_size=8, keep_audio=False, rotate=90)

    assert res.frames == n
    assert dims(out) == (h, w)
