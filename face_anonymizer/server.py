"""HTTP API + 웹 UI.

영상을 올리면 작업 큐에 넣고, 진행률을 폴링으로 보여 주고, 끝나면 내려받게
하는 최소 구성이다.

설계상 못 박아 둔 것 두 가지.

1. **추론은 한 번에 하나만.** GPU 는 한 장뿐이고 검출기도 하나만 올린다.
   요청마다 스레드를 띄우면 VRAM 이 터지거나 서로 느려지기만 한다. 워커
   스레드를 하나만 두고 나머지는 큐에서 대기시킨다 — 총 처리량은 오히려 는다.
2. **작업 파일은 작업별 디렉터리에.** 원본과 결과가 섞이지 않고, 삭제가
   디렉터리 하나 지우는 것으로 끝난다.

환경 변수
    FA_DEVICE          'cuda:0' | 'cpu'    (기본: 자동)
    FA_IMGSZ           검출기 기본 해상도  (기본: 1280)
    FA_JOBS_DIR        작업 디렉터리       (기본: ./jobs)
    FA_MAX_UPLOAD_MB   업로드 상한         (기본: 2048)
    FA_JOB_TTL_MIN     완료 후 자동 삭제   (기본: 120, 0이면 안 지움)

실행
    uvicorn face_anonymizer.server:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .anonymize import METHODS
from .pipeline import VideoOpenError, VideoWriteError
from .webui import INDEX_HTML

log = logging.getLogger(__name__)

DEVICE = os.environ.get("FA_DEVICE") or None
IMGSZ = int(os.environ.get("FA_IMGSZ", 1280))
JOBS_DIR = os.path.abspath(os.environ.get("FA_JOBS_DIR", "jobs"))
MAX_BYTES = int(os.environ.get("FA_MAX_UPLOAD_MB", 2048)) * 1024 * 1024
JOB_TTL = int(os.environ.get("FA_JOB_TTL_MIN", 120)) * 60

CHUNK = 1 << 20
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# 추론 직렬화. max_workers=1 이 이 서버의 동시성 정책 전부다.
_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anon")
_JOBS = {}
_LOCK = threading.Lock()

_anonymizer = None
_anon_lock = threading.Lock()


def get_anonymizer():
    """검출기 싱글턴. 첫 요청 때 한 번만 올린다 (모델 로드가 수 초 걸린다)."""
    global _anonymizer
    with _anon_lock:
        if _anonymizer is None:
            from .pipeline import VideoAnonymizer
            log.info("loading detector (device=%s imgsz=%d)", DEVICE, IMGSZ)
            _anonymizer = VideoAnonymizer(device=DEVICE, imgsz=IMGSZ)
        return _anonymizer


@dataclass
class Job:
    id: str
    name: str
    params: dict
    workdir: str
    status: str = "queued"        # queued | running | done | error
    stage: str = ""               # detect | render
    done: int = 0
    total: int = 0
    error: str = ""
    output: str = ""
    result: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    stage_t0: float = 0.0


def snapshot(j):
    """폴링 응답 한 건. 락 안에서 호출한다."""
    pct = int(100 * min(1.0, j.done / j.total)) if j.total else 0
    elapsed = time.time() - j.stage_t0 if j.stage_t0 else 0.0
    fps = j.done / elapsed if elapsed > 0 else 0.0
    eta = (j.total - j.done) / fps if fps > 0 and j.done < j.total else 0.0
    # 검출과 렌더가 각각 영상 전체를 한 번씩 훑으므로 절반씩 배분한다.
    overall = pct // 2 + (50 if j.stage == "render" else 0)
    if j.status == "done":
        overall = 100
    return {
        "id": j.id, "name": j.name, "status": j.status, "stage": j.stage,
        "percent": pct, "overall": overall, "fps": round(fps, 1),
        "eta": round(eta), "error": j.error, "result": j.result,
        "queued_ahead": sum(1 for o in _JOBS.values()
                            if o.status == "queued" and o.created < j.created),
    }


def _sweep():
    """TTL 지난 작업 정리. 새 작업이 들어올 때마다 한 번씩 훑는다."""
    if not JOB_TTL:
        return
    now = time.time()
    for jid, j in list(_JOBS.items()):
        if j.finished and now - j.finished > JOB_TTL:
            shutil.rmtree(j.workdir, ignore_errors=True)
            _JOBS.pop(jid, None)


def _run(job_id):
    with _LOCK:
        j = _JOBS.get(job_id)
        if j is None:
            return
        j.status, j.stage_t0 = "running", time.time()
        params, workdir, name = dict(j.params), j.workdir, j.name

    def progress(stage, done, total):
        with _LOCK:
            if j.stage != stage:
                j.stage, j.stage_t0 = stage, time.time()
            j.done, j.total = done, total

    src = os.path.join(workdir, "input" + os.path.splitext(name)[1])
    dst = os.path.join(workdir, os.path.splitext(name)[0] + "_anon.mp4")
    try:
        res = get_anonymizer().process(src, dst, progress=progress, **params)
        with _LOCK:
            j.status, j.output, j.finished = "done", res.output, time.time()
            j.result = {
                "frames": res.frames, "raw_boxes": res.raw_boxes,
                "filled_boxes": res.filled_boxes, "method": res.method,
                "audio": res.audio,
                "fps": round(res.fps, 1),
                "detect_fps": round(res.detect_fps, 1),
                "realtime_factor": round(res.realtime_factor, 2),
                # 짧은 클립에서 소수점 첫째 자리로 반올림하면 단계 시간이
                # 전부 0.0 이 되어 어디가 느린지 안 보인다.
                "seconds": round(res.timing.total, 2),
                "timing": {"detect": round(res.timing.detect, 2),
                           "track": round(res.timing.track, 2),
                           "render": round(res.timing.render, 2),
                           "audio": round(res.timing.audio, 2)},
                "video": {"width": res.video.width, "height": res.video.height,
                          "fps": round(res.video.fps, 2)},
            }
    except (VideoOpenError, VideoWriteError, ValueError, FileNotFoundError) as e:
        with _LOCK:
            j.status, j.error, j.finished = "error", str(e), time.time()
    except Exception as e:                      # noqa: BLE001 — 워커가 조용히 죽으면 안 된다
        log.exception("job %s failed", job_id)
        with _LOCK:
            j.status = "error"
            j.error = f"{type(e).__name__}: {e}"
            j.finished = time.time()


app = FastAPI(title="face-anonymizer", version="0.2.0")


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/api/health")
def health():
    loaded = _anonymizer is not None
    info = {"status": "ok", "model_loaded": loaded,
            "device": DEVICE or "auto", "imgsz": IMGSZ,
            "methods": list(METHODS)}
    if loaded:
        # 검출기는 주입 가능하므로 FaceDetector 의 속성이 있다고 단정하지 않는다.
        d = _anonymizer.detector
        info.update(device=str(getattr(d, "device", "?")),
                    half=getattr(d, "half", None),
                    stride=getattr(d, "stride", None),
                    detector=type(d).__name__)
    with _LOCK:
        info["jobs"] = {"total": len(_JOBS),
                        "running": sum(1 for j in _JOBS.values() if j.status == "running"),
                        "queued": sum(1 for j in _JOBS.values() if j.status == "queued")}
    return info


@app.post("/api/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    method: str = Form("mosaic"),
    conf: float = Form(0.25),
    imgsz: int = Form(1280),
    batch_size: int = Form(16),
    pad: float = Form(0.15),
    mosaic_scale: float = Form(0.06),
    linger: int = Form(5),
    interp: bool = Form(True),
    keep_audio: bool = Form(True),
):
    if method not in METHODS:
        raise HTTPException(400, f"unknown method: {method}")
    if not 0 < conf < 1:
        raise HTTPException(400, "conf 는 0~1 사이여야 한다")
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"지원하지 않는 확장자: {ext or '(없음)'}")
    # imgsz 는 stride 배수여야 한다. 클라이언트가 아무 값이나 보내도 여기서 맞춘다.
    imgsz = max(32, round(imgsz / 32) * 32)

    with _LOCK:
        _sweep()
    jid = uuid.uuid4().hex[:12]
    workdir = os.path.join(JOBS_DIR, jid)
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, "input" + ext)

    size = 0
    try:
        with open(src, "wb") as f:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, f"업로드 상한 초과 ({MAX_BYTES // 1048576} MB)")
                f.write(chunk)
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    if size == 0:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(400, "빈 파일")

    job = Job(id=jid, name=os.path.basename(file.filename), workdir=workdir,
              params=dict(method=method, conf=conf, imgsz=imgsz,
                          batch_size=batch_size, pad=pad,
                          mosaic_scale=mosaic_scale, linger=linger,
                          interp=interp, keep_audio=keep_audio))
    with _LOCK:
        _JOBS[jid] = job
        snap = snapshot(job)
    _EXEC.submit(_run, jid)
    return snap


@app.get("/api/jobs")
def list_jobs():
    with _LOCK:
        return [snapshot(j) for j in sorted(_JOBS.values(),
                                            key=lambda x: -x.created)]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    with _LOCK:
        j = _JOBS.get(jid)
        if j is None:
            raise HTTPException(404, "no such job")
        return snapshot(j)


@app.get("/api/jobs/{jid}/download")
def download(jid: str):
    with _LOCK:
        j = _JOBS.get(jid)
        if j is None:
            raise HTTPException(404, "no such job")
        if j.status != "done":
            raise HTTPException(409, f"아직 준비되지 않았다 (status={j.status})")
        path, name = j.output, os.path.splitext(j.name)[0] + "_anon.mp4"
    return FileResponse(path, media_type="video/mp4", filename=name)


@app.delete("/api/jobs/{jid}", status_code=204)
def delete_job(jid: str):
    with _LOCK:
        j = _JOBS.pop(jid, None)
    if j is None:
        raise HTTPException(404, "no such job")
    if j.status in ("queued", "running"):
        _JOBS[jid] = j
        raise HTTPException(409, "진행 중인 작업은 삭제할 수 없다")
    shutil.rmtree(j.workdir, ignore_errors=True)
