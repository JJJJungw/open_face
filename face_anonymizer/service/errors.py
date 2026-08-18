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

# 문구 규칙: 사용자에게 보이는 title/hint 는 존댓말로, 무엇이 잘못됐는지와
# 다음에 무엇을 하면 되는지만 담는다. 명령조("확인하라", "낮춰라")는 쓰지
# 않는다 — 오류를 만난 사람은 이미 곤란한 상태고, 여기서 다그칠 이유가 없다.
# (코드 주석은 그대로 평서체다. 읽는 사람이 다르다.)

# ── 요청이 잘못된 경우 (400 계열) ──────────────────────────────────────────
INVALID_INPUT = _p(
    "invalid_input", 400, "요청 값이 올바르지 않습니다",
    "보내신 파라미터를 확인해 주세요. 값의 범위는 GET /api/defaults 에서 볼 수 있습니다.")
MISSING_INPUT = _p(
    "missing_input", 400, "처리할 입력이 없습니다",
    "파일을 업로드하시거나 S3 키를 함께 보내 주세요.")
CONFLICTING_INPUT = _p(
    "conflicting_input", 400, "입력이 두 가지로 들어왔습니다",
    "업로드 파일과 S3 선택 중 하나만 보내 주세요.")
INVALID_KEY = _p(
    "invalid_key", 400, "S3 키 형식이 올바르지 않습니다",
    "상대 경로만 사용할 수 있습니다. '..' 이나 맨 앞의 '/' 는 넣을 수 없습니다.")
UNSUPPORTED_MEDIA = _p(
    "unsupported_media", 415, "지원하지 않는 파일 형식입니다",
    "mp4 · mov · mkv · avi · webm · m4v 를 처리할 수 있습니다.")
EMPTY_FILE = _p("empty_file", 400, "파일이 비어 있습니다")
PAYLOAD_TOO_LARGE = _p(
    "payload_too_large", 413, "업로드 용량 상한을 넘었습니다",
    "파일을 나눠서 보내시거나, 서버의 FA_MAX_UPLOAD_MB 값을 올리면 됩니다.")
BATCH_EMPTY = _p("batch_empty", 400, "처리할 항목이 없습니다")
ALREADY_PROCESSED = _p(
    "already_processed", 409, "이미 비식별화된 영상입니다",
    "다시 처리하시려면 '처리된 건 건너뛰기' 를 끄고 보내 주세요 "
    "(API 는 skip_processed=false).")
BATCH_TOO_LARGE = _p(
    "batch_too_large", 400, "한 번에 넣을 수 있는 개수를 넘었습니다",
    "나눠서 보내 주세요. 상한은 서버의 FA_BATCH_MAX 로 조정할 수 있습니다.")

# ── 서비스 상태 (4xx/5xx) ──────────────────────────────────────────────────
NOT_READY = _p(
    "not_ready", 503, "모델이 아직 준비되지 않았습니다",
    "기동 중이라면 잠시 후 다시 시도해 주세요. 계속 같은 응답이면 "
    "GET /api/status 의 model_error 를 확인해 보시면 됩니다.",
    retryable=True)
MODEL_LOAD_FAILED = _p(
    "model_load_failed", 503, "모델을 불러오지 못했습니다",
    "가중치와 GPU 상태를 확인해 주세요. 가중치는 S3(FA_S3_WEIGHTS_KEY)에서 "
    "받아 오며, 버킷이 설정되어 있지 않으면 python setup_weights.py 로 "
    "직접 준비할 수 있습니다.")
QUEUE_FULL = _p(
    "queue_full", 429, "대기열이 가득 찼습니다",
    "Retry-After 에 적힌 시간 뒤에 다시 보내 주세요.", retryable=True)
INSUFFICIENT_STORAGE = _p(
    "insufficient_storage", 507, "디스크 여유 공간이 부족합니다",
    "완료된 작업을 정리하시거나 볼륨을 늘리시면 다시 받을 수 있습니다.",
    retryable=True)

# ── 작업 (404/409/410) ─────────────────────────────────────────────────────
JOB_NOT_FOUND = _p(
    "job_not_found", 404, "해당 작업을 찾을 수 없습니다",
    "작업 id 를 확인해 주세요. 보관 기간이 지나 정리됐을 수도 있습니다.")
JOB_NOT_FINISHED = _p(
    "job_not_finished", 409, "작업이 아직 끝나지 않았습니다",
    "상태가 done 이 된 뒤에 받으실 수 있습니다.", retryable=True)
JOB_FAILED = _p(
    "job_failed", 409, "실패한 작업입니다",
    "작업 상세의 error.code 에서 원인을 보실 수 있습니다.")
RESULT_EXPIRED = _p(
    "result_expired", 410, "결과물이 남아 있지 않습니다",
    "보관 기간이 지났습니다. 다시 처리하시면 새로 만들어집니다.")
JOB_NOT_CANCELLABLE = _p(
    "job_not_cancellable", 409, "취소할 수 없는 상태입니다",
    "이미 끝났거나 실패한 작업입니다.")
JOB_NOT_IN_REVIEW = _p(
    "job_not_in_review", 409, "검수 대기 중인 작업이 아닙니다",
    "상태가 review 인 작업만 승인하거나 반려할 수 있습니다.")
REVIEW_ACTION_INVALID = _p(
    "review_action_invalid", 400, "알 수 없는 검수 판정입니다",
    "approve(승인) 또는 reject(반려) 중 하나여야 합니다.")

# ── S3 (404/502) ──────────────────────────────────────────────────────────
S3_NOT_CONFIGURED = _p(
    "s3_not_configured", 404, "S3 가 설정되어 있지 않습니다",
    "FA_S3_BUCKET 을 설정하고 서버를 다시 띄우면 됩니다. "
    "직접 업로드는 설정 없이도 쓰실 수 있습니다.")
S3_OBJECT_NOT_FOUND = _p(
    "s3_object_not_found", 404, "S3 에서 해당 파일을 찾지 못했습니다",
    "키를 확인해 주세요. 대소문자와 폴더 경로가 정확해야 합니다.")
S3_ACCESS_DENIED = _p(
    "s3_access_denied", 502, "S3 접근 권한이 없습니다",
    "인스턴스 역할이나 자격 증명에 s3:GetObject · s3:PutObject 권한이 "
    "있는지 확인해 주세요.")
S3_UPSTREAM = _p(
    "s3_upstream", 502, "S3 호출에 실패했습니다",
    "리전 설정과 네트워크 상태를 확인해 주세요.", retryable=True)

# ── 처리 실패 (작업 error.code 로 쓰인다) ──────────────────────────────────
VIDEO_UNREADABLE = _p(
    "video_unreadable", 422, "영상을 열 수 없습니다",
    "파일이 손상됐거나 지원하지 않는 코덱일 수 있습니다.")
DECODE_INCOMPLETE = _p(
    "decode_incomplete", 422, "디코딩이 중간에 끊겼습니다",
    "손상된 파일일 수 있습니다. 일부만 처리해도 괜찮다면 "
    "allow_partial 을 켜고 보내 주세요.")
ENCODE_FAILED = _p(
    "encode_failed", 500, "결과물을 만들지 못했습니다",
    "인코더(NVENC · libx264)와 디스크 여유를 확인해 주세요.")
NO_DETECTIONS = _p(
    "no_detections", 422, "얼굴이 하나도 검출되지 않았습니다",
    "conf 를 낮추거나 imgsz 를 올려서 다시 시도해 보세요. "
    "영상이 눕혀져 있는 경우에도 이렇게 나올 수 있습니다.")
GPU_OUT_OF_MEMORY = _p(
    "gpu_out_of_memory", 503, "GPU 메모리가 부족합니다",
    "batch_size 나 imgsz 를 낮추면 통과할 수 있습니다.", retryable=True)
FFMPEG_MISSING = _p(
    "ffmpeg_missing", 500, "ffmpeg 를 찾을 수 없습니다",
    "서버에 ffmpeg 를 설치해 주세요.")
CANCELLED = _p("cancelled", 499, "사용자가 취소했습니다")
INTERNAL = _p(
    "internal", 500, "서버 내부 오류가 발생했습니다",
    "잠시 후 다시 시도해 주세요. 계속되면 서버 로그를 확인해 주세요.",
    retryable=True)


def classify(exc):
    """예외를 Problem 으로 옮긴다.

    작업이 실패했을 때 '무엇 때문인지' 를 코드로 남기기 위한 것이다. 파이프라인은
    자기 예외를 던지고, 여기서 한 번에 대응시킨다 — 예외 종류가 늘어날 때
    고칠 곳이 한 군데여야 한다.
    """
    from ..storage import s3 as s3mod
    from ..storage.weights import WeightsUnavailable
    from ..core.pipeline import (
        DecodeIncompleteError,
        DetectionSanityError,
        VideoOpenError,
        VideoWriteError,
    )

    if isinstance(exc, ProblemError):
        return exc.problem
    if isinstance(exc, WeightsUnavailable):
        return MODEL_LOAD_FAILED
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
