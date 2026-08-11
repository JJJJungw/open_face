"""오류 정의와 핸들러.

응답 형식은 **RFC 9457 Problem Details** 를 따른다(2023년 7월, RFC 7807 을 대체).
``application/problem+json`` 으로 이런 모양이 나간다::

    {
      "type": "/problems/queue-full",
      "title": "대기열이 가득 찼다",
      "status": 429,
      "detail": "대기 중 10건 (상한 10)",
      "code": "queue_full",
      "instance": "/api/jobs",
      "retryable": true
    }

**왜 문자열이 아니라 코드인가.** 호출하는 쪽(오케스트레이터)은 재시도할지, 다른
인스턴스로 보낼지, 사람을 불러야 할지를 정해야 한다. 한국어 문장을 파싱해서
정할 수는 없다. ``code`` 는 안정된 식별자고 ``retryable`` 은 그 판단을 서버가
대신 내려 준 것이다. ``detail`` 은 사람이 읽는 용도이므로 문구가 바뀔 수 있다.

오류 목록은 CATALOG 한곳에 있고 ``GET /api/problems`` 로 그대로 노출된다.
"""

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)

MEDIA_TYPE = "application/problem+json"
TYPE_BASE = "/problems/"


class Problem:
    """오류 한 종류의 정의."""

    __slots__ = ("code", "status", "title", "hint", "retryable")

    def __init__(self, code, status, title, hint="", retryable=False):
        self.code = code
        self.status = status
        self.title = title
        self.hint = hint            # 사용자가 다음에 뭘 하면 되는지
        self.retryable = retryable  # 같은 요청을 그대로 다시 보내면 될 수도 있는가

    @property
    def type_uri(self):
        return TYPE_BASE + self.code.replace("_", "-")

    def __call__(self, detail="", **extra):
        return ProblemError(self, detail, **extra)

    def as_dict(self):
        return {"code": self.code, "status": self.status, "title": self.title,
                "hint": self.hint, "retryable": self.retryable,
                "type": self.type_uri}


class ProblemError(Exception):
    """던지면 problem+json 으로 나가는 예외."""

    def __init__(self, problem, detail="", **extra):
        super().__init__(f"{problem.code}: {detail}" if detail else problem.code)
        self.problem = problem
        self.detail = detail
        self.extra = extra

    def body(self, instance=None):
        d = {"type": self.problem.type_uri, "title": self.problem.title,
             "status": self.problem.status, "code": self.problem.code,
             "retryable": self.problem.retryable}
        if self.detail:
            d["detail"] = self.detail
        if self.problem.hint:
            d["hint"] = self.problem.hint
        if instance:
            d["instance"] = instance
        d.update(self.extra)
        return d


def _p(code, status, title, hint="", retryable=False):
    p = Problem(code, status, title, hint, retryable)
    CATALOG[code] = p
    return p


CATALOG = {}

# ── 요청이 잘못된 경우 (400 계열) ──────────────────────────────────────────
INVALID_INPUT = _p(
    "invalid_input", 400, "요청 값이 잘못됐다",
    "보낸 파라미터를 확인하라. 값 범위는 GET /api/defaults 참고.")
MISSING_INPUT = _p(
    "missing_input", 400, "입력이 없다",
    "file 로 업로드하거나 s3_key 를 주어라.")
CONFLICTING_INPUT = _p(
    "conflicting_input", 400, "입력이 둘이다",
    "file 과 s3_key 중 하나만 보내라.")
INVALID_KEY = _p(
    "invalid_key", 400, "s3_key 가 잘못됐다",
    "상대 경로만 허용한다. '..' 이나 앞의 '/' 는 쓸 수 없다.")
UNSUPPORTED_MEDIA = _p(
    "unsupported_media", 415, "지원하지 않는 형식이다",
    "mp4 · mov · mkv · avi · webm · m4v 만 처리한다.")
EMPTY_FILE = _p("empty_file", 400, "빈 파일이다")
PAYLOAD_TOO_LARGE = _p(
    "payload_too_large", 413, "업로드 상한을 넘었다",
    "FA_MAX_UPLOAD_MB 를 올리거나 파일을 나눠라.")
BATCH_EMPTY = _p("batch_empty", 400, "처리할 항목이 없다")
BATCH_TOO_LARGE = _p(
    "batch_too_large", 400, "한 번에 넣을 수 있는 개수를 넘었다",
    "나눠서 보내라. 상한은 FA_BATCH_MAX.")

# ── 서비스 상태 (4xx/5xx) ──────────────────────────────────────────────────
NOT_READY = _p(
    "not_ready", 503, "모델이 아직 준비되지 않았다",
    "기동 중이면 잠시 후 다시. 계속 이러면 model_error 를 확인하라.",
    retryable=True)
MODEL_LOAD_FAILED = _p(
    "model_load_failed", 503, "모델을 올리지 못했다",
    "가중치 경로와 GPU 상태를 확인하라. python setup_weights.py 를 돌렸는가?")
QUEUE_FULL = _p(
    "queue_full", 429, "대기열이 가득 찼다",
    "Retry-After 뒤에 다시 보내거나 다른 인스턴스로 보내라.", retryable=True)
INSUFFICIENT_STORAGE = _p(
    "insufficient_storage", 507, "디스크 여유가 부족하다",
    "완료된 작업을 삭제하거나 볼륨을 늘려라.", retryable=True)

# ── 작업 (404/409/410) ─────────────────────────────────────────────────────
JOB_NOT_FOUND = _p(
    "job_not_found", 404, "그런 작업이 없다",
    "id 를 확인하라. 보관 기간이 지나 정리됐을 수도 있다.")
JOB_NOT_FINISHED = _p(
    "job_not_finished", 409, "아직 끝나지 않았다",
    "status 가 done 이 된 뒤에 받아라.", retryable=True)
JOB_FAILED = _p(
    "job_failed", 409, "실패한 작업이다",
    "error.code 로 원인을 확인하라.")
RESULT_EXPIRED = _p(
    "result_expired", 410, "결과물이 더 이상 없다",
    "보관 기간이 지났다. 다시 처리해야 한다.")
JOB_NOT_CANCELLABLE = _p(
    "job_not_cancellable", 409, "취소할 수 없는 상태다",
    "이미 끝났거나 실패한 작업이다.")

# ── S3 (404/502) ──────────────────────────────────────────────────────────
S3_NOT_CONFIGURED = _p(
    "s3_not_configured", 404, "S3 가 설정되지 않았다",
    "FA_S3_BUCKET 을 설정하고 다시 띄워라. 직접 업로드는 그대로 쓸 수 있다.")
S3_OBJECT_NOT_FOUND = _p(
    "s3_object_not_found", 404, "S3 에 그 객체가 없다",
    "키를 확인하라. 대소문자와 프리픽스가 정확해야 한다.")
S3_ACCESS_DENIED = _p(
    "s3_access_denied", 502, "S3 접근이 거부됐다",
    "인스턴스 역할이나 자격 증명의 s3:GetObject / s3:PutObject 권한을 확인하라.")
S3_UPSTREAM = _p(
    "s3_upstream", 502, "S3 호출이 실패했다",
    "리전과 네트워크를 확인하라.", retryable=True)

# ── 처리 실패 (작업 error.code 로 쓰인다) ──────────────────────────────────
VIDEO_UNREADABLE = _p(
    "video_unreadable", 422, "영상을 열 수 없다",
    "파일이 손상됐거나 지원하지 않는 코덱이다.")
DECODE_INCOMPLETE = _p(
    "decode_incomplete", 422, "디코딩이 중간에 끊겼다",
    "손상된 파일일 수 있다. 의도한 것이면 allow_partial 로 보내라.")
ENCODE_FAILED = _p(
    "encode_failed", 500, "출력을 만들지 못했다",
    "인코더(mp4v/libx264/NVENC)와 디스크를 확인하라.")
NO_DETECTIONS = _p(
    "no_detections", 422, "얼굴이 하나도 검출되지 않았다",
    "conf 를 낮추거나 imgsz 를 올려라. 영상이 누워 있는지도 확인하라.")
GPU_OUT_OF_MEMORY = _p(
    "gpu_out_of_memory", 503, "GPU 메모리가 부족하다",
    "batch_size 나 imgsz 를 낮춰라.", retryable=True)
FFMPEG_MISSING = _p(
    "ffmpeg_missing", 500, "ffmpeg 를 찾을 수 없다",
    "컨테이너에 ffmpeg 를 설치하라.")
CANCELLED = _p("cancelled", 499, "사용자가 취소했다")
INTERNAL = _p(
    "internal", 500, "내부 오류",
    "서버 로그를 확인하라.", retryable=True)


def classify(exc):
    """예외를 Problem 으로 옮긴다.

    작업이 실패했을 때 '무엇 때문인지' 를 코드로 남기기 위한 것이다. 파이프라인은
    자기 예외를 던지고, 여기서 한 번에 대응시킨다 — 예외 종류가 늘어날 때
    고칠 곳이 한 군데여야 한다.
    """
    from . import s3 as s3mod
    from .pipeline import (
        DecodeIncompleteError,
        DetectionSanityError,
        VideoOpenError,
        VideoWriteError,
    )

    if isinstance(exc, ProblemError):
        return exc.problem
    if isinstance(exc, DecodeIncompleteError):
        return DECODE_INCOMPLETE
    if isinstance(exc, VideoOpenError):
        return VIDEO_UNREADABLE
    if isinstance(exc, VideoWriteError):
        return ENCODE_FAILED
    if isinstance(exc, DetectionSanityError):
        return NO_DETECTIONS
    if isinstance(exc, s3mod.S3Error):
        return getattr(exc, "problem", None) or S3_UPSTREAM
    if isinstance(exc, FileNotFoundError):
        return VIDEO_UNREADABLE
    if isinstance(exc, ValueError):
        return INVALID_INPUT
    name = type(exc).__name__
    if "OutOfMemory" in name or "CUDA out of memory" in str(exc):
        return GPU_OUT_OF_MEMORY
    return INTERNAL


def job_error(exc):
    """작업 실패를 응답에 담을 형태로."""
    p = classify(exc)
    return {"code": p.code, "title": p.title, "detail": str(exc),
            "hint": p.hint, "retryable": p.retryable}


# ── 핸들러 ────────────────────────────────────────────────────────────────

def _response(err, request):
    body = err.body(instance=str(request.url.path) if request else None)
    headers = {}
    if err.problem.retryable and err.problem.status in (429, 503, 507):
        headers["Retry-After"] = str(err.extra.get("retry_after", 30))
    return JSONResponse(body, status_code=err.problem.status,
                        media_type=MEDIA_TYPE, headers=headers)


def install(app):
    """앱에 핸들러를 건다.

    모든 오류가 같은 형식으로 나가야 호출하는 쪽이 분기 하나로 처리한다.
    FastAPI 기본 핸들러는 {"detail": "..."} 만 주므로 전부 갈아 끼운다.
    """

    @app.exception_handler(ProblemError)
    async def _problem(request: Request, exc: ProblemError):
        if exc.problem.status >= 500:
            log.error("%s: %s", exc.problem.code, exc.detail)
        return _response(exc, request)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        # 우리가 안 감싼 것들(404 라우팅, 405 등)도 같은 형식으로.
        p = CATALOG.get("internal") if exc.status_code >= 500 else None
        err = ProblemError(
            p or Problem("http_error", exc.status_code,
                         str(exc.detail) or "요청을 처리할 수 없다"),
            "" if p is None else str(exc.detail))
        return _response(err, request)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        # 어느 필드가 왜 틀렸는지 그대로 넘긴다. "invalid_input" 만 주면
        # 호출자가 무엇을 고쳐야 할지 알 수 없다.
        errors = [{"field": ".".join(str(x) for x in e.get("loc", [])[1:]),
                   "detail": e.get("msg", "")} for e in exc.errors()]
        return _response(INVALID_INPUT("요청 형식이 올바르지 않다", errors=errors),
                         request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("처리되지 않은 오류")
        p = classify(exc)
        return _response(ProblemError(p, f"{type(exc).__name__}: {exc}"), request)

    return app
