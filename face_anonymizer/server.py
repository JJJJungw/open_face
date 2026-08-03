"""FastAPI 비식별화 서버.

    uvicorn face_anonymizer.server:app --host 0.0.0.0 --port 8000

설계 의도
---------
* **모델은 프로세스당 1개.** 기동 시 한 번 로드하고 warmup 까지 돌려 첫 요청만
  느려지는 일을 없앤다. ``FA_EAGER_LOAD=0`` 이면 첫 요청 때 지연 로드한다.
* **GPU 작업은 직렬 큐.** 워커 스레드 하나가 큐를 비운다. 요청마다 스레드를
  띄우면 VRAM 이 터지고 배치 추론 이득도 사라진다. 처리량은 워커 수가 아니라
  ``FA_BATCH_SIZE`` / ``FA_DETECT_EVERY`` 로 올린다.
* **업로드 → 작업 → 다운로드 비동기.** 영상 처리는 초 단위가 아니라 분 단위라
  HTTP 요청 하나로 붙들고 있으면 프록시 타임아웃에 걸린다. 202 로 job_id 를
  먼저 주고 폴링하게 한다.
* **작업 격리.** 작업마다 별도 디렉터리를 쓰고 끝나면 지운다. 업로드 파일명은
  절대 경로 조립에 쓰지 않는다(경로 탈출 방지).

이 모듈은 인증을 하지 않는다. 외부에 노출한다면 앞단에 리버스 프록시를 두고
인증·업로드 크기 제한·레이트 리밋을 거는 것을 전제로 한다.
"""

import contextlib
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .anonymize import METHODS
from .pipeline import Cancelled, VideoAnonymizer, VideoOpenError, VideoWriteError

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 설정 (환경변수)
# --------------------------------------------------------------------------- #

WORKDIR = os.environ.get("FA_WORKDIR", os.path.join(tempfile.gettempdir(), "face-anon"))
MAX_UPLOAD_MB = int(os.environ.get("FA_MAX_UPLOAD_MB", "512"))
JOB_TTL_SEC = int(os.environ.get("FA_JOB_TTL_SEC", str(60 * 60)))
QUEUE_MAX = int(os.environ.get("FA_QUEUE_MAX", "32"))
EAGER_LOAD = os.environ.get("FA_EAGER_LOAD", "1") != "0"

DEFAULTS = {
    "method": os.environ.get("FA_METHOD", "mosaic"),
    "imgsz": int(os.environ.get("FA_IMGSZ", "960")),
    "conf": float(os.environ.get("FA_CONF", "0.25")),
    "pad": float(os.environ.get("FA_PAD", "0.15")),
    "mosaic_scale": float(os.environ.get("FA_MOSAIC_SCALE", "0.06")),
    "linger": int(os.environ.get("FA_LINGER", "5")),
    "detect_every": int(os.environ.get("FA_DETECT_EVERY", "1")),
    "batch_size": int(os.environ.get("FA_BATCH_SIZE", "8")),
}

CHUNK = 1 << 20


class Status(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


@dataclass
class Job:
    id: str
    status: Status = Status.queued
    stage: str = ""
    progress: float = 0.0
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None
    error: Optional[str] = None
    result: Optional[dict] = None
    params: dict = field(default_factory=dict)
    _dir: str = ""
    _input: str = ""
    _output: str = ""
    _cancel: bool = False

    def public(self):
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d["status"] = self.status.value
        return d


# --------------------------------------------------------------------------- #
# 모델 · 작업 큐
# --------------------------------------------------------------------------- #

_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue(maxsize=QUEUE_MAX)
_anonymizer: Optional[VideoAnonymizer] = None
_model_lock = threading.Lock()
_model_error: Optional[str] = None


def get_anonymizer():
    """프로세스 전역 VideoAnonymizer 를 반환(최초 1회 로드)."""
    global _anonymizer, _model_error
    if _anonymizer is not None:
        return _anonymizer
    with _model_lock:
        if _anonymizer is None:
            from .detector import FaceDetector
            log.info("loading detector ...")
            det = FaceDetector(
                imgsz=DEFAULTS["imgsz"],
                device=os.environ.get("FA_DEVICE") or None,
            )
            det.warmup(batch_size=DEFAULTS["batch_size"])
            _anonymizer = VideoAnonymizer(detector=det)
            _model_error = None
            log.info("detector ready")
    return _anonymizer


def _run_job(job: Job):
    anonymizer = get_anonymizer()

    def progress(stage, done, total):
        job.stage = stage
        job.progress = round(min(1.0, done / total), 4) if total else 0.0

    res = anonymizer.process(
        job._input, job._output,
        progress=progress,
        should_cancel=lambda: job._cancel,
        **job.params,
    )
    payload = asdict(res)
    payload["video"] = asdict(res.video) if res.video else None
    payload["output"] = os.path.basename(res.output)   # 서버 내부 경로는 숨긴다
    return payload


def _worker():
    while True:
        job_id = _queue.get()
        job = _jobs.get(job_id)
        if job is None:
            _queue.task_done()
            continue
        if job._cancel:
            job.status, job.finished = Status.cancelled, time.time()
            _queue.task_done()
            continue
        job.status = Status.running
        try:
            job.result = _run_job(job)
            job.status, job.progress = Status.done, 1.0
        except Cancelled:
            job.status = Status.cancelled
        except (VideoOpenError, VideoWriteError) as e:
            job.status, job.error = Status.failed, str(e)
            log.warning("job %s failed: %s", job.id, e)
        except Exception as e:                              # noqa: BLE001
            job.status, job.error = Status.failed, f"{type(e).__name__}: {e}"
            log.exception("job %s crashed", job.id)
        finally:
            job.finished = time.time()
            _queue.task_done()


def _reaper():
    """TTL 이 지난 작업 디렉터리를 정리한다. 없으면 디스크가 찬다."""
    while True:
        time.sleep(60)
        now = time.time()
        for job in list(_jobs.values()):
            if job.finished and now - job.finished > JOB_TTL_SEC:
                shutil.rmtree(job._dir, ignore_errors=True)
                with _jobs_lock:
                    _jobs.pop(job.id, None)


# --------------------------------------------------------------------------- #
# 앱
# --------------------------------------------------------------------------- #

def _preload():
    """기동 시 모델을 미리 올린다. 실패해도 서버는 뜨고 /healthz 가 이유를 알린다."""
    global _model_error
    try:
        get_anonymizer()
    except Exception as e:                                  # noqa: BLE001
        _model_error = f"{type(e).__name__}: {e}"
        log.error("detector preload failed: %s", _model_error)


@contextlib.asynccontextmanager
async def lifespan(app):
    os.makedirs(WORKDIR, exist_ok=True)
    threading.Thread(target=_worker, name="fa-worker", daemon=True).start()
    threading.Thread(target=_reaper, name="fa-reaper", daemon=True).start()
    if EAGER_LOAD:
        threading.Thread(target=_preload, name="fa-preload", daemon=True).start()
    yield


app = FastAPI(
    title="face-anonymizer",
    version="0.2.0",
    description="YOLO-FaceV2 + ByteTrack 기반 영상 얼굴 비식별화 API",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz():
    """로드밸런서용. 모델이 아직 안 올라왔거나 실패했으면 503."""
    ready = _anonymizer is not None
    body = {
        "ready": ready,
        "model_error": _model_error,
        "queued": _queue.qsize(),
        "jobs": len(_jobs),
    }
    return JSONResponse(body, status_code=200 if ready or not EAGER_LOAD else 503)


@app.post("/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(..., description="입력 영상"),
    method: str = Form(DEFAULTS["method"]),
    imgsz: int = Form(DEFAULTS["imgsz"]),
    conf: float = Form(DEFAULTS["conf"]),
    pad: float = Form(DEFAULTS["pad"]),
    mosaic_scale: float = Form(DEFAULTS["mosaic_scale"]),
    linger: int = Form(DEFAULTS["linger"]),
    detect_every: int = Form(DEFAULTS["detect_every"]),
    batch_size: int = Form(DEFAULTS["batch_size"]),
    keep_audio: bool = Form(True),
):
    if method not in METHODS:
        raise HTTPException(400, f"unknown method: {method}. choose {list(METHODS)}")
    if detect_every > 1 and linger < 1:
        raise HTTPException(400, "detect_every > 1 이면 linger 를 1 이상으로 두는 것이 안전하다")

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(WORKDIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # 업로드 파일명은 신뢰하지 않는다. 확장자만 취해 고정 이름으로 저장.
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        ext = ".mp4"
    in_path = os.path.join(job_dir, "input" + ext)

    written = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    try:
        with open(in_path, "wb") as f:
            while chunk := await file.read(CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_MB}MB")
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    if written == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, "empty upload")

    job = Job(
        id=job_id,
        params=dict(method=method, imgsz=imgsz, conf=conf, pad=pad,
                    mosaic_scale=mosaic_scale, linger=linger,
                    detect_every=detect_every, batch_size=batch_size,
                    keep_audio=keep_audio),
    )
    job._dir = job_dir
    job._input = in_path
    job._output = os.path.join(job_dir, "output" + ext)

    with _jobs_lock:
        _jobs[job_id] = job
    try:
        _queue.put_nowait(job_id)
    except queue.Full:
        with _jobs_lock:
            _jobs.pop(job_id, None)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(503, "queue is full, retry later")

    return {"job_id": job_id, "status": job.status.value,
            "position": _queue.qsize(), "bytes": written}


def _get(job_id) -> Job:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    return _get(job_id).public()


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = _get(job_id)
    if job.status is not Status.done:
        raise HTTPException(409, f"job is {job.status.value}, not done")
    if not os.path.exists(job._output):
        raise HTTPException(410, "result expired")
    return FileResponse(job._output, media_type="video/mp4",
                        filename=f"{job_id}_anon{os.path.splitext(job._output)[1]}")


@app.delete("/jobs/{job_id}", status_code=202)
def cancel_job(job_id: str, background: BackgroundTasks):
    job = _get(job_id)
    job._cancel = True
    if job.status in (Status.done, Status.failed, Status.cancelled):
        background.add_task(shutil.rmtree, job._dir, ignore_errors=True)
        with _jobs_lock:
            _jobs.pop(job_id, None)
        return {"job_id": job_id, "status": "deleted"}
    return {"job_id": job_id, "status": "cancelling"}


@app.get("/jobs")
def list_jobs(limit: int = 50):
    items = sorted(_jobs.values(), key=lambda j: j.created, reverse=True)
    return {"jobs": [j.public() for j in items[:limit]], "queued": _queue.qsize()}
