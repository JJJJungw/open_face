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
import subprocess
import time

import cv2

from .pipeline import (FFMPEG_TIMEOUT, VideoOpenError, _run, pick_encoder,
                       video_duration)

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


def expected_frames(path):
    """전사 진행률의 분모. 알 수 없으면 0.

    nb_frames 는 컨테이너에 따라 없다. 그때는 길이 x 프레임률로 센다 —
    진행률 표시용이라 정확할 필요는 없다.
    """
    p = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=nb_frames,r_frame_rate",
              "-of", "default=nk=1:nw=1", path])
    if p is None or p.returncode != 0:
        return 0
    lines = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    fps = 0.0
    for x in lines:
        if x.isdigit() and int(x) > 0:
            return int(x)
        if "/" in x:
            try:
                n, d = x.split("/")
                fps = float(n) / float(d) if float(d) else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
    dur = video_duration(path)
    return int(round(dur * fps)) if dur and fps else 0


def _run_with_progress(cmd, total, progress, timeout):
    """ffmpeg 를 돌리면서 frame= 을 읽어 진행률을 보고한다.

    ``-progress pipe:1`` 은 frame/fps/progress 를 key=value 로 흘려 준다.
    stderr 로 나오는 통계는 파싱하기 나쁘고 -v error 로 막아 두었다.
    """
    cmd = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    try:
        for line in proc.stdout:
            if line.startswith("frame=") and progress is not None and total:
                try:
                    n = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                progress(min(n, total), total)
            if time.time() - t0 > timeout:
                proc.kill()
                return None
    finally:
        try:
            proc.stdout.close()
        except Exception:                           # noqa: BLE001
            pass
    err = proc.stderr.read()
    proc.stderr.close()
    proc.wait()
    if progress is not None and total:
        progress(total, total)                      # 마지막 한 칸을 남기지 않는다
    return proc.returncode, err


_hwaccel = None


def hwaccel_args(encoder=None):
    """입력 디코딩을 GPU 로 넘기는 인자. 못 쓰면 빈 목록.

    **전사는 디코딩과 인코딩 둘 다 한다.** 출력은 진작 NVENC 로 가고 있었는데
    입력은 CPU 로 풀고 있었다 — AV1 원본에서 이 구간이 한 편의 3분의 1이었다
    (docs/issues/010).

    **``-hwaccel_output_format cuda`` 는 쓰지 않는다.** 프레임을 GPU 에 둔 채
    NVENC 로 넘기면 PCIe 왕복이 없어 더 빠른데, 실측에서 **검출된 프레임이
    768 → 713 으로 7% 줄었다.** 디코더가 내놓은 NV12 가 CPU 를 거쳐
    ``-pix_fmt yuv420p`` 로 정규화되는 경로와 색 범위 처리가 달라져 중간 파일의
    픽셀이 미세하게 바뀌고, 검출은 그 차이를 그대로 받는다.

    **1초를 벌자고 얼굴 55프레임을 놓칠 수는 없다.** 비식별화에서 속도와 검출률이
    부딪히면 검출률이 이긴다. 디코딩만 GPU 로 넘기고 픽셀 경로는 예전 그대로 둔다.

    되는지는 목록이 아니라 **실제로 한 번 돌려서** 판정한다. 빌드에 이름이 있어도
    드라이버·GPU 세대에 따라 실행 시점에 실패한다(pick_encoder 와 같은 이유).
    """
    global _hwaccel
    if os.environ.get("FA_HWACCEL", "1").strip().lower() in ("0", "false", "no"):
        return []
    if _hwaccel is None:
        p = _probe(["ffmpeg", "-v", "error", "-hwaccel", "cuda",
                    "-f", "lavfi", "-i", "testsrc=duration=0.1:size=64x64:rate=10",
                    "-f", "null", "-"])
        _hwaccel = bool(p)
        log.info("입력 하드웨어 디코딩: %s", "cuda" if _hwaccel else "없음 (CPU)")
    if not _hwaccel:
        return []
    return ["-hwaccel", "cuda"]


def _probe(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return False
    return p.returncode == 0


def transcode(src, dst, crf=INGEST_CRF, timeout=None, progress=None):
    """ffmpeg 로 H.264 로 옮겨 담는다 (영상만).

    ``progress`` 는 ``callable(done, total)``. 긴 영상은 전사만 수십 초가
    걸리는데, 그동안 화면이 '준비 0%' 로 멈춰 있으면 멈춘 것으로 보인다.
    """
    enc = pick_encoder()
    if enc is None:
        raise TranscodeError("쓸 수 있는 H.264 인코더가 없습니다")
    encoder, qflag, extra = enc
    total = expected_frames(src)
    hw = hwaccel_args(encoder)
    # GPU 디코딩이 이 파일에서만 실패할 수 있다(코덱 조합·프로파일). 그때는 CPU 로
    # 한 번 더 해 본다 — 빠르게 하려다 못 하게 만들면 안 된다.
    for args in ([hw, []] if hw else [[]]):
        cmd = ["ffmpeg", "-y", "-v", "error", *args, "-i", src,
               "-map", "0:v:0", "-an",
               "-c:v", encoder, qflag, str(crf), *extra,
               "-pix_fmt", "yuv420p", dst]
        res = _run_with_progress(cmd, total, progress,
                                 timeout or FFMPEG_TIMEOUT)
        if res is None:
            raise TranscodeError("ffmpeg 가 제한 시간 안에 끝나지 않았습니다")
        rc, err = res
        ok = rc == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
        if ok:
            break
        if args:
            log.warning("GPU 디코딩으로 전사 실패 — CPU 로 다시 한다: %s",
                        (err or "")[-200:].strip())
    if not ok:
        raise TranscodeError(f"ffmpeg 가 실패했습니다 (종료 코드 {rc}): {(err or '')[-200:].strip()}")
    if not opencv_can_decode(dst):
        raise TranscodeError("변환한 파일도 읽지 못했습니다")
    return dst


def ensure_decodable(path, workdir, crf=INGEST_CRF, progress=None):
    """파이프라인에 넘길 경로를 돌려준다.

    Returns (경로, 정보). 정보에는 ``source_codec``, ``transcoded``(bool) 이 담긴다.
    원본을 그대로 쓸 수 있으면 전사하지 않는다 — 대부분의 입력(H.264)은 여기서
    아무 비용도 내지 않는다.
    """
    if not os.path.exists(path):
        raise VideoOpenError(f"입력 파일이 없습니다: {path}")
    codec = probe_codec(path)
    if opencv_can_decode(path):
        return path, {"source_codec": codec, "transcoded": False}

    log.info("OpenCV 가 읽지 못한다 (codec=%s) — H.264 로 옮겨 담는다", codec or "?")
    os.makedirs(workdir, exist_ok=True)
    dst = os.path.join(workdir, "decodable.mp4")
    try:
        transcode(path, dst, crf, progress=progress)
    except TranscodeError as e:
        raise TranscodeError(
            f"입력 영상을 읽지 못했습니다 (코덱 {codec or '알 수 없음'}): {e}") from e
    log.info("전사 완료: %s -> h264", codec or "?")
    return dst, {"source_codec": codec, "transcoded": True}
