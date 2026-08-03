"""영상 얼굴 비식별화 파이프라인.

흐름: 검출(YOLO-FaceV2) → 추적(ByteTrack) → 보간 → 익명화 렌더 → 오디오 합성.

영상을 두 번 훑는다. 1차에서 검출/추적만 하고 박스 좌표만 들고 있다가, 2차에서
다시 읽으며 렌더한다. 프레임을 메모리에 쌓지 않는 대신 디스크 I/O 를 두 배 쓴다.

가장 위험한 실패는 "빈 결과물이 성공으로 보고되는 것"이라, 조용히 넘어갈 수 있는
지점에는 전부 예외를 세워 뒀다. 반대로 오디오 합성 실패는 결과물을 버릴 이유가
아니므로 영상은 남기고 사유만 알린다.
"""

import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass

import cv2

from .anonymize import METHODS
from .anonymize import apply as anonymize_apply
from .anonymize import pad_box
from .tracking import interpolate, track_video_boxes

log = logging.getLogger(__name__)

DEFAULT_FPS = 30.0


class VideoOpenError(RuntimeError):
    """입력 영상을 열 수 없음 (경로 오류, 손상, 미지원 코덱)."""


class VideoWriteError(RuntimeError):
    """출력을 신뢰할 수 없음 (인코더 실패, 크기/프레임 수 불일치)."""


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int      # 컨테이너 메타값. 부정확할 수 있어 진행률 추정에만 쓴다.


@dataclass
class Result:
    output: str
    frames: int           # 렌더한 프레임 수
    raw_boxes: int        # 모델이 실제로 검출한 박스 수
    filled_boxes: int     # 보간/linger 로 채워 넣은 박스 수
    method: str
    audio: str            # 'ok' | 'no-audio' | 'disabled' | 'ffmpeg-...'
    video: VideoInfo = None


def sane_fps(value, default=DEFAULT_FPS):
    """컨테이너가 알려 준 fps 를 신뢰할 수 있는 값으로 정규화.

    ``cap.get(CAP_PROP_FPS)`` 는 0 뿐 아니라 NaN 을 돌려주기도 한다.
    ``fps or 30.0`` 으로는 NaN 이 truthy 라 그대로 통과해서, 이후 VideoWriter 가
    조용히 깨진 파일을 만든다.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) and 0 < v <= 1000 else default


def probe(path):
    """영상 메타데이터. 열 수 없으면 VideoOpenError."""
    if not os.path.exists(path):
        raise VideoOpenError(f"input does not exist: {path}")
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoOpenError(f"cannot open video (unsupported or corrupt): {path}")
        fps = sane_fps(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise VideoOpenError(f"video reports invalid frame size {w}x{h}: {path}")
    return VideoInfo(fps=fps, width=w, height=h, frame_count=max(0, n))


def _mux_audio(noaudio, original, output, keep_audio=True):
    """원본 오디오를 익명화 영상에 합성.

    어떤 경로로 실패하든 익명화된 영상은 반드시 ``output`` 에 남긴다.
    "오디오가 없어서 결과물이 통째로 없다" 는 최악의 트레이드오프다.
    """
    def fallback(reason):
        shutil.move(noaudio, output)
        return reason

    if not keep_audio:
        return fallback("disabled")
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        log.warning("ffmpeg/ffprobe 없음 — 오디오 없이 출력한다")
        return fallback("ffmpeg-missing")

    probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0", original]
    p = subprocess.run(probe_cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return fallback(f"ffprobe-failed: {p.stderr[-200:].strip()}")
    if not p.stdout.strip():
        return fallback("no-audio")

    cmd = ["ffmpeg", "-y", "-i", noaudio, "-i", original,
           "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(output):
        log.warning("ffmpeg 실패 (%s) — 오디오 없이 출력한다", p.returncode)
        return fallback(f"ffmpeg-failed: {p.stderr[-200:].strip()}")
    os.remove(noaudio)
    return "ok"


class VideoAnonymizer:
    """영상 얼굴 비식별화기.

    검출기는 주입할 수 있다. 테스트에서는 가짜 검출기를 넣어 torch/가중치 없이
    전 구간을 검증한다. 검출기는 다음만 만족하면 된다::

        detect_batch(frames, imgsz=..., conf=..., iou=...)
            -> list[list[(x1, y1, x2, y2, score)]]
    """

    def __init__(self, detector=None, **detector_kwargs):
        if detector is None:
            # torch 는 검출기를 실제로 쓸 때만 필요하다 — 지연 임포트.
            from .detector import FaceDetector
            detector = FaceDetector(**detector_kwargs)
        elif detector_kwargs:
            raise TypeError("detector 를 직접 주면 detector_kwargs 는 쓸 수 없다")
        self.detector = detector

    def process(self, input_path, output_path, method="mosaic", imgsz=960,
                conf=0.25, iou=0.45, pad=0.15, mosaic_scale=0.06, linger=5,
                interp=True, batch_size=1, keep_audio=True, progress=None):
        """영상 한 편을 익명화하고 Result 를 돌려준다.

        progress : callable(stage, done, total) | None
        """
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}. choose one of {list(METHODS)}")
        # 음수 pad 는 박스를 뒤집고 mosaic_scale>=1 은 축소를 없앤다. 둘 다
        # 익명화 함수가 x2<=x1 에서 조용히 return 해서 픽셀을 안 건드리는데,
        # 파이프라인은 "박스 N개 처리 완료" 라고 보고한다.
        if pad < 0:
            raise ValueError(f"pad 는 음수일 수 없다 (박스가 뒤집힌다): {pad}")
        if not 0 < mosaic_scale < 1:
            raise ValueError(f"mosaic_scale 은 0~1 사이여야 한다: {mosaic_scale}")
        batch_size = max(1, int(batch_size))

        info = probe(input_path)
        outdir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(outdir, exist_ok=True)

        def report(stage, done):
            if progress is not None:
                progress(stage, done, max(info.frame_count, done))

        # ---- 1차: 검출 ----
        log.info("[1/3] detecting: %s (%dx%d @%.2ffps, batch=%d)",
                 os.path.basename(input_path), info.width, info.height,
                 info.fps, batch_size)
        per_frame, raw_boxes, total = self._detect(
            input_path, imgsz, conf, iou, batch_size, report)
        if total == 0:
            raise VideoOpenError(f"no frames decoded from {input_path}")

        # ---- 2차 준비: 추적 + 보간 ----
        frame_dets = defaultdict(list)
        for i, raw in enumerate(per_frame):
            for b in raw:
                frame_dets[i].append(tuple(b[:4]))   # 원본 검출은 무조건 익명화

        filled = 0
        if interp:
            log.info("[2/3] tracking + interpolating (누출 방지)")
            tracks = track_video_boxes(per_frame, fps=info.fps)
            _, filled = interpolate(frame_dets, tracks, total, linger=linger)
            log.info("      tracks=%d filled_boxes=%d", len(tracks), filled)
        elif linger:
            log.warning("interp=False 라 linger=%d 는 적용되지 않는다", linger)

        # ---- 3차: 렌더 + 오디오 ----
        log.info("[3/3] rendering (%s)", method)
        # 중간 산출물은 출력 파일 옆 임시 디렉터리에. 같은 폴더를 노리는
        # 동시 작업끼리 서로를 덮어쓰지 않게 한다.
        tmpdir = tempfile.mkdtemp(prefix=".anon-", dir=outdir)
        try:
            ext = os.path.splitext(output_path)[1] or ".mp4"
            noaudio = os.path.join(tmpdir, "noaudio" + ext)
            rendered = self._render(input_path, noaudio, info, frame_dets,
                                    method, pad, mosaic_scale, total, report)
            status = _mux_audio(noaudio, input_path, output_path, keep_audio)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        log.info("done: %s | frames=%d boxes=%d(+%d) audio=%s",
                 output_path, rendered, raw_boxes, filled, status)
        return Result(output=output_path, frames=rendered, raw_boxes=raw_boxes,
                      filled_boxes=filled, method=method, audio=status, video=info)

    # ------------------------------------------------------------------ #

    def _detect(self, path, imgsz, conf, iou, batch_size, report):
        """1차 패스 — 프레임을 훑으며 배치 검출. 프레임은 보관하지 않는다."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise VideoOpenError(f"cannot open video: {path}")

        per_frame, pending = [], []
        raw_boxes = 0

        def flush():
            nonlocal raw_boxes
            if not pending:
                return
            results = self.detector.detect_batch(
                pending, imgsz=imgsz, conf=conf, iou=iou)
            if len(results) != len(pending):
                raise RuntimeError(
                    f"detector 가 {len(pending)}장에 {len(results)}개 결과를 돌려줬다")
            for dets in results:
                per_frame.append(list(dets))
                raw_boxes += len(dets)
            pending.clear()

        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pending.append(frame)
                if len(pending) >= batch_size:
                    flush()
                if idx % 30 == 0:
                    report("detect", idx)
                idx += 1
            flush()
        finally:
            cap.release()
        report("detect", idx)
        return per_frame, raw_boxes, idx

    def _render(self, path, out_path, info, frame_dets, method, pad,
                mosaic_scale, expected, report):
        """2차 패스 — 박스를 얹어 다시 쓴다."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise VideoOpenError(f"cannot reopen video: {path}")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, (info.width, info.height))
        if not writer.isOpened():
            cap.release()
            writer.release()
            raise VideoWriteError(
                f"인코더를 열 수 없다 ({info.width}x{info.height} @{info.fps}). "
                "opencv 빌드에 mp4v 코덱이 없을 수 있다.")

        kw = {"scale": mosaic_scale} if method == "mosaic" else {}
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # VideoWriter.write() 는 크기가 다르면 경고만 찍고 조용히
                # 아무것도 안 쓴다 — "N 프레임 완료" 를 보고하면서 재생 불가
                # 파일이 남는다.
                fh, fw = frame.shape[:2]
                if (fw, fh) != (info.width, info.height):
                    raise VideoWriteError(
                        f"frame {i} 디코딩 크기 {fw}x{fh} 가 컨테이너 메타 "
                        f"{info.width}x{info.height} 와 다르다")
                if i % 60 == 0:
                    report("render", i)
                for box in frame_dets.get(i, ()):
                    pb = pad_box(box, pad, info.width, info.height)
                    anonymize_apply(frame, pb, method=method, **kw)
                writer.write(frame)
                i += 1
        finally:
            cap.release()
            writer.release()
        report("render", i)

        if i != expected:
            # 프레임 수가 어긋나면 박스가 엉뚱한 프레임에 찍힌다. 더 많이
            # 디코딩된 경우 남는 프레임에는 박스가 없어 그대로 노출된다.
            raise VideoWriteError(f"프레임 수 불일치: 1차 {expected} vs 2차 {i}")
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise VideoWriteError(f"인코더가 빈 파일을 만들었다: {out_path}")
        return i
