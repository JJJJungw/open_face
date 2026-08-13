"""서명된 URL 로 주고받기 — **자격 증명이 없는** 전송 경로.

`s3.py` 와 목적이 다르다. 저쪽은 우리 버킷에 우리 열쇠로 붙는다(단독 운영·웹
화면). 여기는 **열쇠 없이** 남이 열어 준 문 하나로만 드나든다.

RebornStudio 의 무DB 잡 프로토콜이 이 방식이다. 잡 페이로드에 `input_url`
(presigned GET)과 `put_url`(presigned PUT)이 들어 있고, 워커는 버킷 이름도
리전도 자격 증명도 모른다. 그래서 이미지에 비밀이 하나도 안 들어가고, 워커를
다른 클러스터·다른 계정으로 옮겨도 설정이 없다.

실패는 **일시(transient)와 영구로 1차 분류만** 한다. 판정은 잡을 준 쪽이
내린다 — 재시도 횟수와 상한을 아는 것은 그쪽이다.

    403  presigned URL 만료일 수 있다 → 일시 (재발급받으면 된다)
    408·425·429·5xx                  → 일시
    그 밖의 4xx                       → 영구 (URL 이 틀렸거나 정책이 막는다)
"""

import logging
import os

log = logging.getLogger(__name__)

# 403 이 여기 있는 이유: presign 만료와 권한 거부를 응답만으로 구분할 수 없다.
# 만료라면 재큐잉이 새 URL 을 주므로 살아나고, 진짜 권한 문제라면 재시도 상한에서
# 걸린다. 반대로 놓으면(영구 취급) 만료 한 번에 작업이 죽는다.
TRANSIENT_STATUS = {403, 408, 425, 429, 500, 502, 503, 504}

CONNECT_TIMEOUT = float(os.environ.get("FA_HTTP_CONNECT_TIMEOUT", 30))
READ_TIMEOUT = float(os.environ.get("FA_HTTP_READ_TIMEOUT", 600))
CHUNK = 1024 * 1024


class TransferError(RuntimeError):
    """전송 실패. ``transient`` 가 재시도 가치가 있는지를 말한다."""

    def __init__(self, message, *, transient):
        super().__init__(message)
        self.transient = transient


def _httpx():
    """지연 임포트. 단독 운영에서는 httpx 가 없어도 된다."""
    try:
        import httpx
    except ImportError as e:                        # pragma: no cover
        raise TransferError(
            "서명된 URL 전송에는 httpx 가 필요합니다 "
            "(pip install -r requirements-worker.txt)", transient=False) from e
    return httpx


def _timeout(write=False):
    httpx = _httpx()
    return httpx.Timeout(CONNECT_TIMEOUT,
                         read=READ_TIMEOUT,
                         write=READ_TIMEOUT if write else None)


def fetch(url, dest, callback=None):
    """서명된 GET 으로 ``dest`` 에 받는다. 받은 바이트 수를 돌려준다.

    **스트리밍으로 받는다.** 영상은 수백 MB 라 메모리에 통째로 올리면 워커가
    죽는다. ``callback(chunk_bytes)`` 로 진행을 흘려보내면 호출자가 하트비트나
    취소에 쓴다 — s3.py 의 전송 콜백과 같은 모양이다(docs/issues/004).
    """
    httpx = _httpx()
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    try:
        with httpx.stream("GET", url, timeout=_timeout(), follow_redirects=True) as r:
            if r.status_code >= 400:
                raise TransferError(f"내려받기 실패 HTTP {r.status_code}",
                                    transient=r.status_code in TRANSIENT_STATUS)
            seen = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(CHUNK):
                    f.write(chunk)
                    seen += len(chunk)
                    if callback is not None:
                        callback(len(chunk))
    except httpx.HTTPError as e:
        raise TransferError(f"내려받기 전송 오류: {type(e).__name__}",
                            transient=True) from e
    log.info("내려받기 완료: %s (%.1f MB)", dest, seen / 1e6)
    return seen


def put(url, path, content_type="video/mp4"):
    """서명된 PUT 으로 ``path`` 를 올린다.

    파일 객체를 그대로 넘긴다 — httpx 가 청크로 보낸다. 여기서도 통째로 읽으면
    큰 결과물에서 메모리를 두 배로 쓴다.
    """
    httpx = _httpx()
    size = os.path.getsize(path)
    try:
        with open(path, "rb") as f:
            r = httpx.put(url, content=f, timeout=_timeout(write=True),
                          headers={"Content-Type": content_type,
                                   "Content-Length": str(size)})
    except httpx.HTTPError as e:
        raise TransferError(f"올리기 전송 오류: {type(e).__name__}",
                            transient=True) from e
    if r.status_code >= 400:
        raise TransferError(f"올리기 실패 HTTP {r.status_code}",
                            transient=r.status_code in TRANSIENT_STATUS)
    log.info("올리기 완료: %.1f MB", size / 1e6)
    return size
