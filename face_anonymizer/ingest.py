"""입력 정규화.

**OpenCV 가 읽을 수 있는 형태로 만들어 놓고 파이프라인에 넘긴다.**

OpenCV 의 FFmpeg 빌드는 코덱 지원이 ffmpeg 본체보다 좁다. 특히 AV1 은 파일을
**열기는 열면서 한 프레임도 못 뽑는다**(실측: OpenCV 4.13 에서 isOpened=True,
디코딩 0프레임, "Your platform doesn't support hardware accelerated AV1 decoding").
반면 ffmpeg 는 libdav1d 로 잘 읽는다.

그래서 코덱 이름으로 판단하지 않고 **실제로 한 프레임을 뽑아 본다.** 목록으로
관리하면 빌드마다 다르고 새 코덱이 나올 때마다 어긋난다. 못 뽑으면 ffmpeg 로
H.264 로 옮겨 담고 그 파일을 파이프라인에 준다.

전사(transcode)는 검출 전에 일어나므로 화질이 떨어지면 검출률이 같이 떨어진다.
그래서 CRF 를 낮게(고화질) 잡는다 — 이 파일은 중간 산출물이라 용량이 커도 되고
작업이 끝나면 지워진다. 최종 결과물의 화질/용량은 별도로 관리한다.

오디오는 담지 않는다. 최종 합성은 **원본** 에서 가져오므로 여기서 옮길 이유가 없다.
"""

import logging
import os

import cv2

from .pipeline import FFMPEG_TIMEOUT, VideoOpenError, _run, pick_encoder

log = logging.getLogger(__name__)

# 중간 산출물이라 고화질로 뜬다. 검출 전에 화질을 깎으면 검출률이 같이 떨어진다.
INGEST_CRF = int(os.environ.get("FA_INGEST_CRF", "16"))


class TranscodeError(VideoOpenError):
    """입력을 읽을 수 있는 형태로 만들지 못했다.

    VideoOpenError 를 상속한다 — 같은 파일로 다시 시도해도 결과가 같으므로
    서버가 재시도하지 않고 바로 실패로 남겨야 한다.
    """


def probe_codec(path):
    """비디오 코덱 이름. 알 수 없으면 빈 문자열."""
    p = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=codec_name", "-of", "default=nk=1:nw=1",
              path])
    return p.stdout.strip() if p is not None and p.returncode == 0 else ""


def opencv_can_decode(path):
    """OpenCV 가 실제로 프레임을 뽑을 수 있는가.

    ``isOpened()`` 는 믿을 수 없다 — AV1 에서 True 를 돌려주고도 read() 가
    계속 실패한다. 한 장을 실제로 받아 봐야 안다.
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None and frame.size)
    finally:
        cap.release()


def transcode(src, dst, crf=INGEST_CRF, timeout=None):
    """ffmpeg 로 H.264 로 옮겨 담는다 (영상만)."""
    enc = pick_encoder()
    if enc is None:
        raise TranscodeError("쓸 수 있는 H.264 인코더가 없다")
    encoder, qflag, extra = enc
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src,
           "-map", "0:v:0", "-an",
           "-c:v", encoder, qflag, str(crf), *extra,
           "-pix_fmt", "yuv420p", dst]
    p = _run(cmd, timeout or FFMPEG_TIMEOUT)
    if p is None:
        raise TranscodeError("ffmpeg 타임아웃")
    if p.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise TranscodeError(f"ffmpeg 실패 ({p.returncode}): "
                             f"{(p.stderr or '')[-200:].strip()}")
    if not opencv_can_decode(dst):
        raise TranscodeError("옮겨 담은 파일도 읽을 수 없다")
    return dst


def ensure_decodable(path, workdir, crf=INGEST_CRF):
    """파이프라인에 넘길 경로를 돌려준다.

    Returns (경로, 정보). 정보에는 ``source_codec``, ``transcoded``(bool) 이 담긴다.
    원본을 그대로 쓸 수 있으면 전사하지 않는다 — 대부분의 입력(H.264)은 여기서
    아무 비용도 내지 않는다.
    """
    if not os.path.exists(path):
        raise VideoOpenError(f"input does not exist: {path}")
    codec = probe_codec(path)
    if opencv_can_decode(path):
        return path, {"source_codec": codec, "transcoded": False}

    log.info("OpenCV 가 읽지 못한다 (codec=%s) — H.264 로 옮겨 담는다", codec or "?")
    os.makedirs(workdir, exist_ok=True)
    dst = os.path.join(workdir, "decodable.mp4")
    try:
        transcode(path, dst, crf)
    except TranscodeError as e:
        raise TranscodeError(
            f"입력을 읽을 수 없다 (codec={codec or '알 수 없음'}): {e}") from e
    log.info("전사 완료: %s -> h264", codec or "?")
    return dst, {"source_codec": codec, "transcoded": True}
