"""영상 얼굴 비식별화 파이프라인.

흐름: 검출(YOLO-FaceV2) → 추적(ByteTrack) → 보간 → 익명화 렌더 → 오디오 합성.

두 번 훑는 구조(1차 검출/추적, 2차 렌더)라 시간은 좀 걸리지만 프레임을 메모리에
쌓지 않고(박스 좌표만 보관) 순간 누출을 최대한 막는다.
"""

import os
import subprocess
from collections import defaultdict

import cv2

from .anonymize import apply as anonymize_apply
from .anonymize import pad_box
from .detector import FaceDetector
from .tracking import interpolate, track_video_boxes


def _has_audio(path):
    """ffprobe 로 오디오 스트림 유무 확인."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        return bool(r.stdout.strip())
    except FileNotFoundError:
        return False


def _mux_audio(noaudio_path, original, output):
    """원본 오디오를 익명화 영상에 합성. 실패/무음이면 무오디오본을 결과로.

    핵심: 무오디오본은 이미 실제 경로에 저장돼 있으므로 ffmpeg 가 어떻게 되든
    결과물이 사라지지 않는다.
    """
    if not _has_audio(original):
        os.replace(noaudio_path, output)
        return output, "no-audio"
    cmd = ["ffmpeg", "-y", "-i", noaudio_path, "-i", original,
           "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # 무오디오본은 남겨 두고 stderr 를 알려 준다
        return noaudio_path, f"ffmpeg-failed: {p.stderr[-500:]}"
    os.remove(noaudio_path)
    return output, "ok"


class VideoAnonymizer:
    def __init__(self, detector: FaceDetector = None, **detector_kwargs):
        self.detector = detector or FaceDetector(**detector_kwargs)

    def process(self, input_path, output_path,
                method="mosaic", imgsz=960, conf=0.25, iou=0.45,
                pad=0.15, mosaic_scale=0.06, linger=5, interp=True,
                verbose=True):
        det = self.detector

        # ---- 1차: 검출 + 프레임별 raw 박스 수집 ----
        if verbose:
            print("[1/3] detecting + tracking...")
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        per_frame_raw = []
        frame_dets = defaultdict(list)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw = det.detect(frame, imgsz=imgsz, conf=conf, iou=iou)
            per_frame_raw.append(raw)
            for b in raw:
                frame_dets[idx].append(tuple(b[:4]))   # 원본 검출은 무조건 익명화
            idx += 1
        cap.release()
        total = idx

        # ---- 추적 + 보간 ----
        track_hist = track_video_boxes(per_frame_raw, fps=fps)
        if interp:
            if verbose:
                print("[2/3] interpolating tracks (leak prevention)...")
            interpolate(frame_dets, track_hist, total, linger=linger)

        # ---- 2차: 익명화 렌더 (무오디오본을 실제 경로에 저장) ----
        if verbose:
            print(f"[3/3] rendering ({method})...")
        # str.replace 는 전체 치환이라 경로 중간에 ".mp4" 가 들어간 디렉터리가 있으면
        # 엉뚱한 경로가 나온다. 확장자만 안전하게 분리해서 접미사를 붙인다.
        stem, ext = os.path.splitext(output_path)
        noaudio = stem + "_noaudio" + (ext or ".mp4")
        cap = cv2.VideoCapture(input_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(noaudio, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        i = 0
        kw = {"scale": mosaic_scale} if method == "mosaic" else {}
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for box in frame_dets.get(i, []):
                pb = pad_box(box, pad, w, h)
                anonymize_apply(frame, pb, method=method, **kw)
            writer.write(frame)
            i += 1
        cap.release()
        writer.release()

        # ---- 오디오 합성 ----
        final, status = _mux_audio(noaudio, input_path, output_path)
        if verbose:
            print(f"done: {final} | exists: {os.path.exists(final)} | audio: {status}")
        return final
