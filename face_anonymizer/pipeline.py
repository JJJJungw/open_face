"""영상 얼굴 비식별화 파이프라인.

흐름: 검출(YOLO-FaceV2) → 추적(ByteTrack) → 보간 → 익명화 렌더 → 오디오 합성.

영상을 두 번 훑는다. 1차에서 검출/추적만 하고 박스 좌표만 들고 있다가, 2차에서
다시 읽으며 렌더한다. 프레임을 메모리에 쌓지 않는 대신 디스크 I/O 를 두 배 쓴다.

가장 위험한 실패는 "빈 결과물이 성공으로 보고되는 것"이라, 조용히 넘어갈 수 있는
지점에는 전부 예외를 세워 뒀다. 반대로 오디오 합성 실패는 결과물을 버릴 이유가
아니므로 영상은 남기고 사유만 알린다.
"""

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass

import cv2

from .anonymize import METHODS
from .anonymize import apply as anonymize_apply
from .anonymize import pad_box
from .tracking import interpolate, track_video_boxes

log = logging.getLogger(__name__)

DEFAULT_FPS = 30.0

# ffmpeg/ffprobe 가 멈췄을 때 기다릴 최대 시간. CLI 라면 Ctrl-C 로 벗어날 수
# 있지만 서버는 워커가 하나뿐이라, 한 건이 매달리면 큐 전체가 영구 정지하고
# health 는 계속 정상을 보고한다.
FFMPEG_TIMEOUT = float(os.environ.get("FA_FFMPEG_TIMEOUT", "600"))


class VideoOpenError(RuntimeError):
    """입력 영상을 열 수 없음 (경로 오류, 손상, 미지원 코덱)."""


class VideoWriteError(RuntimeError):
    """출력을 신뢰할 수 없음 (인코더 실패, 크기/프레임 수 불일치)."""


class DecodeIncompleteError(VideoOpenError):
    """디코딩이 영상 끝에 도달하기 전에 멈췄음 (손상된 파일, 디코더 문제)."""


class DetectionSanityError(RuntimeError):
    """검출률이 요구 수준에 못 미침 — 설정/입력이 잘못됐을 가능성이 크다."""


# 디코딩 누락 판정. 서비스에서는 **정상 영상을 거부하는 쪽이 더 큰 사고**라,
# 명백한 절단일 때만 실패시키고 그 사이는 경고로만 남긴다.
#
#   누락 <= WARN      : 정상 (컨테이너 메타 오차, 마지막 GOP 처리 차이)
#   WARN < 누락 < FAIL: 경고 ('decode-short')
#   누락 >= FAIL      : 실패 (DecodeIncompleteError)
DECODE_WARN_RATIO = 0.02
DECODE_FAIL_RATIO = 0.20
DECODE_TOLERANCE_MIN = 5      # 짧은 영상에서 비율만으로 판단하지 않게

# 출력 인코딩. mp4v(MPEG-4 Part 2)는 OpenCV VideoWriter 가 쓸 수 있는 사실상
# 유일한 코덱인데, 같은 화질에 H.264 대비 약 9.5배 크다(1280x720 실측).
# 어차피 ffmpeg 를 거치므로 그 단계에서 H.264 로 다시 뜬다.
DEFAULT_CRF = int(os.environ.get("FA_CRF", "23"))
ENCODER_CANDIDATES = (
    # (인코더, 품질 옵션 이름, 추가 옵션) — 앞에서부터 되는 것을 쓴다.
    ("h264_nvenc", "-cq", ("-preset", "p4", "-rc", "vbr")),   # NVIDIA GPU
    ("libx264", "-crf", ("-preset", "veryfast")),             # CPU
)

# 출력 비트레이트 상한 = 원본 비트레이트 x 이 값.
#
# CRF 만 쓰면 "목표 화질"로 인코딩하므로, 이미 많이 압축된 원본을 받으면
# 결과물이 원본보다 커진다(실측: 1.89 -> 2.96 Mbps, 파일 46MB -> 70MB).
# 비식별화 결과물에 원본 이상의 화질이 필요할 이유가 없고, 서비스에서는
# 다운로드 용량이 곧 비용이다. CRF 는 그대로 두고 상한만 걸어(capped CRF)
# 단순한 장면은 더 작게, 복잡한 장면도 원본을 넘지 않게 한다.
# 0 이면 상한 없음.
DEFAULT_BITRATE_RATIO = float(os.environ.get("FA_BITRATE_RATIO", "1.0"))


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int      # 가장 믿을 만한 프레임 수 (아래 count_source 참고)
    count_source: str = "container"   # 'packets' | 'container' | 'unknown'


@dataclass
class Timing:
    """단계별 소요 시간(초). 벽시계 기준."""
    detect: float = 0.0
    track: float = 0.0
    render: float = 0.0
    audio: float = 0.0
    total: float = 0.0


@dataclass
class Result:
    output: str
    frames: int           # 렌더한 프레임 수
    raw_boxes: int        # 모델이 실제로 검출한 박스 수
    filled_boxes: int     # 보간/linger 로 채워 넣은 박스 수
    method: str
    audio: str            # 'ok' | 'no-audio' | 'disabled' | 'ffmpeg-...'
    video: VideoInfo = None
    timing: Timing = None
    detected_frames: int = 0          # 검출이 하나라도 있었던 프레임 수
    warnings: tuple = ()              # 결과를 그대로 믿으면 안 되는 사유들

    @property
    def detection_rate(self):
        """검출이 잡힌 프레임 비율. 0 이면 원본이 그대로 나갔다는 뜻이다."""
        return self.detected_frames / self.frames if self.frames else 0.0

    @property
    def fps(self):
        """전체 처리 속도(프레임/초). 두 번 훑는 시간이 모두 포함된 실효값."""
        if not self.timing or self.timing.total <= 0:
            return 0.0
        return self.frames / self.timing.total

    @property
    def realtime_factor(self):
        """영상 길이 대비 배속. 1 이상이면 실시간보다 빠르게 처리한 것."""
        if not (self.video and self.timing) or self.timing.total <= 0:
            return 0.0
        return (self.frames / self.video.fps) / self.timing.total

    @property
    def detect_fps(self):
        """검출 단계만의 속도. GPU/배치 설정을 조절할 때 보는 값."""
        if not self.timing or self.timing.detect <= 0:
            return 0.0
        return self.frames / self.timing.detect


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

    # 기대 프레임 수는 **재생 길이 x fps** 로 잡는다.
    #
    # 패킷 수(-count_packets)를 쓰면 안 된다. 앞뒤를 잘라낸 영상은 컨테이너에
    # edit list 가 붙어 패킷은 그대로 남고 재생 대상만 줄어든다. 실측: 앞 0.7초를
    # 잘라낸 파일이 패킷 150개, 실제 디코딩 129프레임 — 멀쩡한 영상인데 21장이
    # 누락된 것처럼 보인다. 아이폰/편집 앱을 거친 영상 상당수가 이 형태다.
    # format.duration 은 edit list 가 반영된 값이라 이 함정을 피한다.
    count, source = max(0, n), ("container" if n > 0 else "unknown")
    if shutil.which("ffprobe"):
        dur = video_duration(path)
        if dur and fps > 0:
            count, source = int(round(dur * fps)), "duration"
    return VideoInfo(fps=fps, width=w, height=h,
                     frame_count=count, count_source=source)


def _unlink(path):
    """실패 경로 정리용. 지워지지 않아도 흐름을 막지 않는다."""
    try:
        os.remove(path)
    except OSError:
        pass


def _run(cmd, timeout=FFMPEG_TIMEOUT):
    """외부 명령 실행. 멈추면 죽인다. 타임아웃이면 None."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("%s 타임아웃 (%.0fs)", cmd[0], timeout)
        return None


def has_audio(path, timeout=FFMPEG_TIMEOUT):
    """오디오 스트림 유무. 판정 자체가 실패하면 None."""
    p = _run(["ffprobe", "-v", "error", "-select_streams", "a",
              "-show_entries", "stream=codec_type", "-of", "csv=p=0", path], timeout)
    if p is None or p.returncode != 0:
        return None
    return bool(p.stdout.strip())


def video_duration(path, timeout=FFMPEG_TIMEOUT):
    """**비디오 스트림**의 재생 길이(초).

    format.duration 을 쓰면 안 된다. 그건 모든 스트림 중 가장 긴 것이라,
    오디오가 영상보다 길면 프레임 수를 과대추정한다(실측: 영상 1.33초 +
    오디오 2.0초 파일에서 20프레임을 30프레임으로 계산해 정상 영상을 거부).
    마이크가 늦게 끊긴 녹화물에서 흔한 형태다.

    스트림 길이도 edit list 는 반영하므로, 앞뒤를 잘라낸 영상에서도 맞는다.
    """
    for entry, sel in (("stream=duration", ["-select_streams", "v:0"]),
                       ("format=duration", [])):
        p = _run(["ffprobe", "-v", "error", *sel, "-show_entries", entry,
                  "-of", "default=nk=1:nw=1", path], timeout)
        if p is None or p.returncode != 0:
            continue
        try:
            d = float(p.stdout.strip())
        except ValueError:
            continue
        if math.isfinite(d) and d > 0:
            return d
    return None


_encoder = None


def pick_encoder(timeout=60):
    """쓸 수 있는 H.264 인코더를 한 번만 골라 캐시한다.

    NVENC 는 ffmpeg 빌드에 이름이 있어도 드라이버/GPU 가 없으면 실행 시점에
    실패한다. 그래서 목록 확인이 아니라 **아주 작은 실제 인코딩**으로 판정한다.
    30분짜리를 돌리다 실패하는 것보다 0.2초를 쓰는 편이 낫다.

    Returns (encoder, quality_flag, extra_opts) | None
    """
    global _encoder
    if _encoder is not None:
        return _encoder or None
    forced = os.environ.get("FA_ENCODER")
    cands = [c for c in ENCODER_CANDIDATES if not forced or c[0] == forced]
    for enc, qflag, extra in cands:
        p = _run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                  "-i", "testsrc=duration=0.1:size=64x64:rate=10",
                  "-c:v", enc, *extra, "-f", "null", "-"], timeout)
        if p is not None and p.returncode == 0:
            log.info("출력 인코더: %s", enc)
            _encoder = (enc, qflag, extra)
            return _encoder
    log.warning("쓸 수 있는 H.264 인코더가 없다 — mp4v 원본을 그대로 내보낸다")
    _encoder = ()
    return None


def video_bitrate(path, timeout=FFMPEG_TIMEOUT):
    """원본 비디오 비트레이트(bps).

    스트림 값이 없는 컨테이너가 흔해서 파일 크기/길이로도 물러선다(오디오가
    섞여 약간 과대추정되지만 상한 용도로는 충분하다).
    """
    p = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=bit_rate", "-of", "default=nk=1:nw=1",
              path], timeout)
    if p is not None and p.returncode == 0:
        try:
            v = int(p.stdout.strip())
            if v > 0:
                return v
        except ValueError:
            pass
    dur = video_duration(path, timeout)
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if dur and size:
        return int(size * 8 / dur)
    return None


def video_frame_count(path, timeout=FFMPEG_TIMEOUT):
    """파일에 실제로 들어 있는 비디오 프레임 수와 길이.

    ``-count_packets`` 는 컨테이너 인덱스만 읽고 디코딩은 하지 않으므로 싸다.
    컨테이너가 선언한 ``nb_frames`` 메타값과 달리 실제 패킷을 센 값이라,
    "메타는 600인데 실제로는 300장" 같은 상태를 잡아낼 수 있다.

    Returns (frames, duration). 셀 수 없으면 (None, None).
    """
    p = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
              "-show_entries", "stream=nb_read_packets,duration",
              "-of", "json", path], timeout)
    if p is None or p.returncode != 0:
        return None, None
    try:
        st = json.loads(p.stdout)["streams"][0]
        n = int(st["nb_read_packets"])
    except (ValueError, KeyError, IndexError, TypeError):
        return None, None
    try:
        d = float(st["duration"])
    except (KeyError, ValueError, TypeError):
        d = None
    return n, d


def finalize_output(noaudio, original, output, keep_audio=True,
                    expected_frames=None, crf=DEFAULT_CRF,
                    bitrate_ratio=DEFAULT_BITRATE_RATIO, timeout=FFMPEG_TIMEOUT):
    """중간 산출물을 최종 결과물로 만든다 — H.264 재인코딩 + 오디오 합성 + 검증.

    네 가지를 못 박는다.

    1. **익명화된 프레임을 한 장도 잃지 않는다.** 예전에는 ``-shortest`` 를 썼는데
       이건 짧은 쪽에 맞춰 자른다 — 오디오가 영상보다 짧으면 잘리는 건 **영상**
       이다. ffmpeg 는 리턴코드 0 을 주고 파일도 멀쩡해 보여서, 20초/600프레임
       결과물이 10초/300프레임으로 잘린 채 "ok" 로 보고됐다. 이제 프레임 수를
       세어 원본과 다르면 결과를 버린다.
    2. **검증 전에는 무음본을 지우지 않는다.** 작업은 임시 파일에 하고, 통과했을
       때만 출력 경로로 옮긴다. 어떤 경로로 실패하든 익명화된 영상은 반드시
       ``output`` 에 남는다 — 오디오나 코덱 때문에 결과물이 통째로 없는 게 최악이다.
    3. **H.264 로 다시 뜬다.** OpenCV VideoWriter 가 쓸 수 있는 mp4v 는 같은
       화질에 H.264 대비 약 9.5배 크다(1280x720 실측). 다운로드 대역폭과 대기
       시간이 그만큼 늘고, 재생 호환성도 떨어진다. GPU 가 있으면 NVENC 라
       비용도 거의 없다.
    4. **매달리지 않는다.** 서버는 워커가 하나라 한 건이 매달리면 큐 전체가 정지한다.

    반환값: 'ok' | 'no-audio' | 'disabled' | 'ffmpeg-missing' | 'ffprobe-failed'
            | 'ffmpeg-timeout' | 'ffmpeg-failed: ...' | 'verify-failed'
            | 'frame-loss: got/expected'
    """
    def fallback(reason):
        shutil.move(noaudio, output)
        return reason

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        log.warning("ffmpeg/ffprobe 없음 — 원본 코덱 그대로, 오디오 없이 출력한다")
        return fallback("ffmpeg-missing")

    status = "ok"
    if not keep_audio:
        status = "disabled"
    else:
        audio = has_audio(original, timeout)
        if audio is None:
            status = "ffprobe-failed"
        elif not audio:
            status = "no-audio"

    enc = pick_encoder()
    if enc is None:
        return fallback("ffmpeg-missing")
    encoder, qflag, extra = enc

    root, ext = os.path.splitext(noaudio)
    out_tmp = root + ".final" + (ext or ".mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", noaudio]
    if status == "ok":
        cmd += ["-i", original, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac"]
    else:
        cmd += ["-map", "0:v:0", "-an"]
    # -shortest 없음(위 1번). +faststart 는 moov 를 앞으로 옮겨 부분 다운로드
    # 상태에서도 재생이 시작되게 한다.
    cmd += ["-c:v", encoder, qflag, str(crf), *extra, "-pix_fmt", "yuv420p"]
    if bitrate_ratio:
        src_bps = video_bitrate(original, timeout)
        if src_bps:
            cap = int(src_bps * bitrate_ratio)
            cmd += ["-maxrate", str(cap), "-bufsize", str(cap * 2)]
            log.info("비트레이트 상한 %.2f Mbps (원본 %.2f Mbps)",
                     cap / 1e6, src_bps / 1e6)
    cmd += ["-movflags", "+faststart", out_tmp]

    p = _run(cmd, timeout)
    if p is None:
        _unlink(out_tmp)
        return fallback("ffmpeg-timeout")
    if p.returncode != 0 or not os.path.exists(out_tmp) or os.path.getsize(out_tmp) == 0:
        _unlink(out_tmp)
        log.warning("ffmpeg 실패 (%s) — 원본 코덱 그대로 출력한다", p.returncode)
        return fallback(f"ffmpeg-failed: {p.stderr[-200:].strip()}")

    if expected_frames is not None:
        got, _dur = video_frame_count(out_tmp, timeout)
        if got is None:
            _unlink(out_tmp)
            log.warning("결과물을 검증할 수 없다 — 무음본을 쓴다")
            return fallback("verify-failed")
        if got != expected_frames:
            _unlink(out_tmp)
            log.warning("결과물이 %d/%d 프레임 — 버리고 무음본을 쓴다",
                        got, expected_frames)
            return fallback(f"frame-loss: {got}/{expected_frames}")

    shutil.move(out_tmp, output)
    os.remove(noaudio)
    return status


def check_decode_complete(decoded, info, allow_partial=False):
    """디코딩이 영상 끝까지 갔는지 확인.

    ``cap.read()`` 가 False 를 돌려주는 건 "스트림 끝" 과 "디코드 실패" 둘 다다.
    구분하지 않고 break 하면, 손상된 GOP 나 부분 읽기에서 **영상 뒷부분이 결과물에
    통째로 없는데 정상 종료**한다. 1·2차 패스의 프레임 수를 비교하는 검사도
    둘 다 같은 지점에서 끊기면 통과하므로 이걸 못 잡는다.

    실측: 컨테이너에 600프레임이 선언된 파일에서 241프레임만 렌더되고 성공 반환.
    게다가 CLI 는 잘린 프레임 수로 길이를 계산해 출력하므로 원본이 얼마나 길었는지
    조차 화면에서 사라진다.

    Returns 경고 문자열 리스트.
    """
    if info.count_source == "unknown" or info.frame_count <= 0:
        log.warning("프레임 수를 확인할 수 없어 디코딩 완결성 검사를 건너뛴다")
        return ["decode-unverified"]

    expected = info.frame_count
    missing = expected - decoded
    if missing <= max(DECODE_TOLERANCE_MIN, expected * DECODE_WARN_RATIO):
        return []

    ratio = missing / expected
    detail = (f"{decoded}/{expected} 프레임 ({missing}장 누락, "
              f"{ratio:.1%}, 출처={info.count_source})")
    if ratio < DECODE_FAIL_RATIO:
        # 이 구간은 편집된 파일의 메타 오차일 수도, 진짜 손실일 수도 있다.
        # 서비스에서 정상 영상을 거부하는 대가가 더 크므로 통과시키되 남긴다.
        log.warning("디코딩 프레임 수가 예상보다 적다: %s", detail)
        return [f"decode-short: {decoded}/{expected}"]

    msg = (f"디코딩이 중간에 끊겼다: {detail}. 뒷부분이 결과물에서 통째로 "
           f"빠진다 — 손상된 파일이거나 디코더 문제다.")
    if allow_partial:
        log.warning("%s (allow_partial 이라 계속 진행한다)", msg)
        return [f"decode-partial: {decoded}/{expected}"]
    raise DecodeIncompleteError(
        msg + " 의도한 것이면 allow_partial=True (CLI: --allow-partial).")


def check_detections(raw_boxes, detected_frames, total, min_rate=None):
    """검출 결과가 신뢰할 만한지 확인.

    검출 0건은 예외를 던지지 않는다 — 얼굴이 없는 영상은 정당하게 0 이다.
    하지만 가중치 손상, 회전된 영상, 잘못된 imgsz, HDR 톤매핑 실패처럼 **설정이
    틀린 경우도 결과가 똑같이 0** 이고, 그때 원본이 그대로 출력된다. 원인이
    무엇이든 결과가 조용한 게 문제이므로, 판단은 호출자에게 넘기되 사실은
    반드시 드러낸다.

    ``min_rate`` 를 주면 그 미만일 때 실패시킨다 (얼굴이 반드시 있는 영상을
    처리하는 파이프라인용).

    Returns 경고 문자열 리스트.
    """
    rate = detected_frames / total if total else 0.0
    warnings = []
    if raw_boxes == 0:
        warnings.append("no-detections")
        log.error("검출 0건 — 원본이 그대로 출력된다. conf/imgsz/가중치/영상 "
                  "회전을 확인하라.")
    elif rate < 0.01:
        warnings.append(f"low-detection-rate: {rate:.2%}")
        log.warning("검출률 %.2f%% (%d/%d 프레임) — 비정상적으로 낮다",
                    rate * 100, detected_frames, total)
    if min_rate is not None and rate < min_rate:
        raise DetectionSanityError(
            f"검출률 {rate:.2%} 가 요구치 {min_rate:.2%} 에 못 미친다 "
            f"({detected_frames}/{total} 프레임, 박스 {raw_boxes}개)")
    return warnings


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
                interp=True, batch_size=1, keep_audio=True, progress=None,
                allow_partial=False, min_detection_rate=None, crf=DEFAULT_CRF,
                bitrate_ratio=DEFAULT_BITRATE_RATIO):
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
        t_start = time.perf_counter()

        def report(stage, done):
            if progress is not None:
                progress(stage, done, max(info.frame_count, done))

        # ---- 1차: 검출 ----
        log.info("[1/3] detecting: %s (%dx%d @%.2ffps, batch=%d)",
                 os.path.basename(input_path), info.width, info.height,
                 info.fps, batch_size)
        t0 = time.perf_counter()
        per_frame, raw_boxes, detected_frames, total = self._detect(
            input_path, imgsz, conf, iou, batch_size, report)
        t_detect = time.perf_counter() - t0
        if total == 0:
            raise VideoOpenError(f"no frames decoded from {input_path}")
        warnings = check_decode_complete(total, info, allow_partial)
        warnings += check_detections(raw_boxes, detected_frames, total,
                                     min_detection_rate)
        log.info("      %d frames, %d boxes (%d프레임에서 검출) — %.1fs (%.1f fps)",
                 total, raw_boxes, detected_frames, t_detect,
                 total / t_detect if t_detect else 0)

        # ---- 2차 준비: 추적 + 보간 ----
        frame_dets = defaultdict(list)
        for i, raw in enumerate(per_frame):
            for b in raw:
                frame_dets[i].append(tuple(b[:4]))   # 원본 검출은 무조건 익명화

        filled, t_track = 0, 0.0
        if interp:
            log.info("[2/3] tracking + interpolating (누출 방지)")
            t0 = time.perf_counter()
            tracks = track_video_boxes(per_frame, fps=info.fps, conf=conf)
            _, filled = interpolate(frame_dets, tracks, total, linger=linger)
            t_track = time.perf_counter() - t0
            log.info("      tracks=%d filled_boxes=%d — %.1fs",
                     len(tracks), filled, t_track)
        elif linger:
            log.warning("interp=False 라 linger=%d 는 적용되지 않는다", linger)

        # ---- 3차: 렌더 + 오디오 ----
        log.info("[3/3] rendering (%s) + 인코딩", method)
        # 중간 산출물은 출력 파일 옆 임시 디렉터리에. 같은 폴더를 노리는
        # 동시 작업끼리 서로를 덮어쓰지 않게 한다.
        tmpdir = tempfile.mkdtemp(prefix=".anon-", dir=outdir)
        try:
            ext = os.path.splitext(output_path)[1] or ".mp4"
            noaudio = os.path.join(tmpdir, "noaudio" + ext)
            t0 = time.perf_counter()
            rendered = self._render(input_path, noaudio, info, frame_dets,
                                    method, pad, mosaic_scale, total, report)
            t_render = time.perf_counter() - t0
            log.info("      %d frames — %.1fs (%.1f fps)", rendered, t_render,
                     rendered / t_render if t_render else 0)
            t0 = time.perf_counter()
            status = finalize_output(noaudio, input_path, output_path, keep_audio,
                                     expected_frames=rendered, crf=crf,
                                     bitrate_ratio=bitrate_ratio)
            t_audio = time.perf_counter() - t0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        timing = Timing(detect=t_detect, track=t_track, render=t_render,
                        audio=t_audio, total=time.perf_counter() - t_start)
        if status not in ("ok", "no-audio", "disabled"):
            warnings.append(f"audio: {status}")
        result = Result(output=output_path, frames=rendered, raw_boxes=raw_boxes,
                        filled_boxes=filled, method=method, audio=status,
                        video=info, timing=timing,
                        detected_frames=detected_frames,
                        warnings=tuple(warnings))
        log.info("done: %s | frames=%d boxes=%d(+%d) audio=%s",
                 output_path, rendered, raw_boxes, filled, status)
        log.info("      %.1fs 소요 · %.1f fps · 실시간 대비 %.2fx "
                 "(검출 %.1fs / 추적 %.1fs / 렌더 %.1fs / 오디오 %.1fs)",
                 timing.total, result.fps, result.realtime_factor,
                 timing.detect, timing.track, timing.render, timing.audio)
        return result

    # ------------------------------------------------------------------ #

    def _detect(self, path, imgsz, conf, iou, batch_size, report):
        """1차 패스 — 프레임을 훑으며 배치 검출. 프레임은 보관하지 않는다."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise VideoOpenError(f"cannot open video: {path}")

        per_frame, pending = [], []
        raw_boxes = detected_frames = 0

        def flush():
            nonlocal raw_boxes, detected_frames
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
                detected_frames += bool(dets)
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
        return per_frame, raw_boxes, detected_frames, idx

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
