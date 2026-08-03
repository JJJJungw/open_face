"""영상 얼굴 비식별화 파이프라인.

흐름: 검출(YOLO-FaceV2) → 추적(ByteTrack) → 보간 → 익명화 렌더 → 오디오 합성.

두 번 훑는 구조(1차 검출/추적, 2차 렌더)라 시간은 좀 걸리지만 프레임을 메모리에
쌓지 않고(박스 좌표만 보관) 순간 누출을 최대한 막는다.

서빙을 염두에 둔 설계 원칙:

* **조용히 실패하지 않는다.** 영상을 못 열거나 인코더를 못 잡으면 빈 파일을
  내놓는 대신 예외를 던진다. 결과물이 비어 있는데 성공으로 보고되는 것이
  비식별화 파이프라인에서는 가장 위험하다.
* **결과물은 최대한 살린다.** 오디오 합성(ffmpeg)이 실패하거나 ffmpeg 가 아예
  없어도 익명화된 영상 자체는 출력 경로에 남기고, 무슨 일이 있었는지는
  ``Result.audio`` 로 알린다.
* **동시 실행 안전.** 중간 산출물은 출력 파일 옆의 임시 디렉터리에 만들어
  같은 출력 경로를 노리는 두 작업이 서로를 덮어쓰지 않게 한다.
* **취소와 진행률.** 서버에 붙일 수 있도록 ``progress`` / ``should_cancel``
  훅을 받는다.
"""

import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

import cv2

from .anonymize import METHODS
from .anonymize import apply as anonymize_apply
from .anonymize import pad_box
from .tracking import interpolate, track_video_boxes

log = logging.getLogger(__name__)

DEFAULT_FPS = 30.0
MAX_SANE_FPS = 1000.0


class VideoOpenError(RuntimeError):
    """입력 영상을 열 수 없음 (경로 오류, 손상, 미지원 코덱)."""


class VideoWriteError(RuntimeError):
    """출력 인코더를 잡을 수 없음 (fourcc 미지원 등)."""


class Cancelled(RuntimeError):
    """should_cancel() 이 True 를 반환해 중단됨."""


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int      # 컨테이너 메타값. 부정확할 수 있어 진행률 추정에만 쓴다.


@dataclass
class Result:
    """process() 결과. 서버 응답으로 그대로 직렬화할 수 있게 평평하게 유지."""

    output: str
    frames: int               # 실제로 렌더한 프레임 수
    detected_frames: int      # 검출을 돌린 프레임 수 (프레임 스킵 시 frames 보다 적음)
    raw_boxes: int            # 모델이 실제로 검출한 박스 수
    filled_boxes: int         # 보간/linger 로 채워 넣은 박스 수
    method: str
    audio: str                # 'ok' | 'no-audio' | 'disabled' | 'ffmpeg-missing' | 'ffmpeg-failed: ...'
    video: VideoInfo = field(default=None)

    @property
    def total_boxes(self):
        return self.raw_boxes + self.filled_boxes


# --------------------------------------------------------------------------- #
# 입력 점검
# --------------------------------------------------------------------------- #

def sane_fps(value, default=DEFAULT_FPS):
    """컨테이너가 알려 준 fps 를 신뢰할 수 있는 값으로 정규화.

    ``cap.get(CAP_PROP_FPS)`` 는 0 뿐 아니라 NaN 을 돌려주는 경우가 있다.
    ``fps or 30.0`` 로는 NaN 이 truthy 라 그대로 통과해서, 이후 VideoWriter 가
    조용히 깨진 파일을 만든다.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v) or v <= 0 or v > MAX_SANE_FPS:
        return default
    return v


def probe(path):
    """영상 메타데이터를 읽고 열 수 없으면 VideoOpenError."""
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


# --------------------------------------------------------------------------- #
# 오디오 합성
# --------------------------------------------------------------------------- #

def _ffmpeg_available():
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _has_audio(path):
    """ffprobe 로 오디오 스트림 유무 확인."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        return bool(r.stdout.strip())
    except (FileNotFoundError, OSError):
        return False


def _mux_audio(noaudio_path, original, output, keep_audio=True):
    """원본 오디오를 익명화 영상에 합성.

    어떤 경로로 실패하든 익명화된 영상은 반드시 ``output`` 에 남긴다.
    비식별화가 목적인 파이프라인에서 "오디오가 없어서 결과물이 통째로 없다" 는
    최악의 트레이드오프이기 때문이다.

    Returns
    -------
    status : str
    """
    def _fallback(reason):
        shutil.move(noaudio_path, output)
        return reason

    if not keep_audio:
        return _fallback("disabled")
    if not _ffmpeg_available():
        log.warning("ffmpeg/ffprobe not found — 오디오 없이 출력한다")
        return _fallback("ffmpeg-missing")
    if not _has_audio(original):
        return _fallback("no-audio")

    cmd = ["ffmpeg", "-y", "-i", noaudio_path, "-i", original,
           "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except (FileNotFoundError, OSError) as e:
        return _fallback(f"ffmpeg-missing: {e}")
    if p.returncode != 0 or not os.path.exists(output):
        log.warning("ffmpeg failed (%s) — 오디오 없이 출력한다", p.returncode)
        return _fallback(f"ffmpeg-failed: {p.stderr[-300:].strip()}")
    os.remove(noaudio_path)
    return "ok"


# --------------------------------------------------------------------------- #

class VideoAnonymizer:
    """영상 얼굴 비식별화기.

    검출기는 주입할 수 있다. 테스트에서는 가짜 검출기를 넣어 torch/가중치 없이
    파이프라인 전 구간을 검증하고, 서버에서는 프로세스당 하나를 만들어 재사용한다.

    검출기는 다음만 만족하면 된다::

        detect_batch(frames, imgsz=..., conf=..., iou=...)
            -> list[list[(x1, y1, x2, y2, score)]]
    """

    def __init__(self, detector=None, **detector_kwargs):
        if detector is None:
            # torch 는 검출기를 실제로 쓸 때만 필요하다. 지연 임포트해 두면
            # 테스트나 후처리 전용 사용에서 무거운 의존성을 건너뛸 수 있다.
            from .detector import FaceDetector
            detector = FaceDetector(**detector_kwargs)
        elif detector_kwargs:
            raise TypeError("detector 를 직접 주면 detector_kwargs 는 쓸 수 없다")
        self.detector = detector

    # ------------------------------------------------------------------ #

    def process(self, input_path, output_path, method="mosaic", imgsz=960,
                conf=0.25, iou=0.45, pad=0.15, mosaic_scale=0.06, linger=5,
                interp=True, detect_every=1, batch_size=1, keep_audio=True,
                fourcc="mp4v", progress=None, should_cancel=None, verbose=False):
        """영상 한 편을 익명화한다.

        Parameters
        ----------
        detect_every : int
            N 프레임마다 검출. 사이 구간은 추적 보간이 덮는다. GPU 시간을
            거의 선형으로 줄여 주지만, 보간을 끄면 스킵된 프레임이 통째로
            노출되므로 ``interp=False`` 와 함께 쓸 수 없다.
        batch_size : int
            한 번에 모델에 넣을 프레임 수. GPU 처리량에 직접 영향을 준다.
        progress : callable(stage: str, done: int, total: int) | None
        should_cancel : callable() -> bool | None
            주기적으로 호출해 True 면 Cancelled 를 던진다.

        Returns
        -------
        Result
        """
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}. choose one of {list(METHODS)}")
        detect_every = max(1, int(detect_every))
        batch_size = max(1, int(batch_size))
        if detect_every > 1 and not interp:
            raise ValueError(
                "detect_every > 1 은 트랙 보간이 있어야 안전하다. "
                "검출을 건너뛴 프레임이 그대로 노출되므로 interp=False 와 함께 쓸 수 없다."
            )

        info = probe(input_path)
        outdir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(outdir, exist_ok=True)

        def tick():
            if should_cancel is not None and should_cancel():
                raise Cancelled("cancelled by caller")

        def report(stage, done, total):
            if progress is not None:
                progress(stage, done, total)
            if verbose and total and done % max(1, total // 10) == 0:
                log.info("%s %d/%d", stage, done, total)

        # ---- 1차: 검출 (+ 프레임 스킵/배치) ----
        log.info("[1/3] detecting: %s (%dx%d @%.2ffps, detect_every=%d, batch=%d)",
                 os.path.basename(input_path), info.width, info.height,
                 info.fps, detect_every, batch_size)
        per_frame_raw, raw_boxes, detected_frames, total = self._detect_pass(
            input_path, info, imgsz, conf, iou, detect_every, batch_size,
            tick, report,
        )
        if total == 0:
            raise VideoOpenError(f"no frames decoded from {input_path}")

        # ---- 추적 + 보간 ----
        frame_dets = defaultdict(list)
        steps, step_idx = [], []
        for i, raw in enumerate(per_frame_raw):
            if raw is None:                       # 검출을 건너뛴 프레임
                continue
            steps.append(raw)
            step_idx.append(i)
            for b in raw:
                frame_dets[i].append(tuple(b[:4]))   # 원본 검출은 무조건 익명화

        filled = 0
        if interp:
            log.info("[2/3] tracking + interpolating (leak prevention)")
            tick()
            track_hist = track_video_boxes(
                steps, fps=info.fps / detect_every, frame_indices=step_idx
            )
            _, filled = interpolate(frame_dets, track_hist, total, linger=linger)
            log.info("      tracks=%d filled_boxes=%d", len(track_hist), filled)

        # ---- 2차: 익명화 렌더 ----
        log.info("[3/3] rendering (%s)", method)
        tmpdir = tempfile.mkdtemp(prefix=".anon-", dir=outdir)
        try:
            ext = os.path.splitext(output_path)[1] or ".mp4"
            noaudio = os.path.join(tmpdir, "noaudio" + ext)
            rendered = self._render_pass(
                input_path, noaudio, info, frame_dets, method, pad,
                mosaic_scale, fourcc, tick, report,
            )
            status = _mux_audio(noaudio, input_path, output_path,
                                keep_audio=keep_audio)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        result = Result(
            output=output_path, frames=rendered, detected_frames=detected_frames,
            raw_boxes=raw_boxes, filled_boxes=filled, method=method,
            audio=status, video=info,
        )
        log.info("done: %s | frames=%d boxes=%d(+%d) audio=%s",
                 output_path, rendered, raw_boxes, filled, status)
        return result

    # ------------------------------------------------------------------ #

    def _detect_pass(self, input_path, info, imgsz, conf, iou,
                     detect_every, batch_size, tick, report):
        """1차 패스: 프레임을 훑으며 검출. 프레임 자체는 보관하지 않는다."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise VideoOpenError(f"cannot open video: {input_path}")

        per_frame_raw = []            # frame idx -> list | None(검출 안 함)
        pending_idx, pending_frames = [], []
        raw_boxes = detected_frames = 0
        total_hint = info.frame_count

        def flush():
            nonlocal raw_boxes, detected_frames
            if not pending_frames:
                return
            results = self.detector.detect_batch(
                pending_frames, imgsz=imgsz, conf=conf, iou=iou
            )
            if len(results) != len(pending_idx):
                raise RuntimeError(
                    f"detector returned {len(results)} results for "
                    f"{len(pending_idx)} frames"
                )
            for li, dets in zip(pending_idx, results):
                per_frame_raw[li] = list(dets)
                raw_boxes += len(dets)
                detected_frames += 1
            pending_idx.clear()
            pending_frames.clear()

        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                per_frame_raw.append(None)
                if idx % detect_every == 0:
                    pending_idx.append(idx)
                    pending_frames.append(frame)
                    if len(pending_frames) >= batch_size:
                        tick()
                        flush()
                        report("detect", idx + 1, total_hint)
                idx += 1
            tick()
            flush()
        finally:
            cap.release()
        report("detect", idx, total_hint or idx)
        return per_frame_raw, raw_boxes, detected_frames, idx

    def _render_pass(self, input_path, out_path, info, frame_dets, method,
                     pad, mosaic_scale, fourcc, tick, report):
        """2차 패스: 박스를 얹어 다시 쓴다."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise VideoOpenError(f"cannot reopen video: {input_path}")
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*fourcc), info.fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            cap.release()
            raise VideoWriteError(
                f"cannot open encoder (fourcc={fourcc!r}, {info.width}x{info.height} "
                f"@{info.fps}). opencv 빌드에 해당 코덱이 없을 수 있다."
            )

        kw = {"scale": mosaic_scale} if method == "mosaic" else {}
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % 60 == 0:
                    tick()
                    report("render", i, info.frame_count)
                for box in frame_dets.get(i, ()):
                    pb = pad_box(box, pad, info.width, info.height)
                    anonymize_apply(frame, pb, method=method, **kw)
                writer.write(frame)
                i += 1
        finally:
            cap.release()
            writer.release()
        report("render", i, info.frame_count or i)
        if i == 0:
            raise VideoWriteError(f"rendered 0 frames from {input_path}")
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise VideoWriteError(f"encoder produced an empty file: {out_path}")
        return i
