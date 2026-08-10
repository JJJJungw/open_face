"""HTTP API + 웹 UI.

영상을 올리면 작업 큐에 넣고, 진행률을 폴링으로 보여 주고, 끝나면 내려받게
하는 최소 구성이다.

설계상 못 박아 둔 것.

1. **추론은 한 번에 하나만.** GPU 는 한 장뿐이고 검출기도 하나만 올린다.
   요청마다 스레드를 띄우면 VRAM 이 터지거나 서로 느려지기만 한다. 워커
   스레드를 하나만 두고 나머지는 큐에서 대기시킨다 — 총 처리량은 오히려 는다.
   프로세스를 여러 개 띄워도(`--workers N`) 파일 락으로 직렬화한다.
2. **작업 상태는 디스크에.** 작업별 디렉터리에 ``job.json`` 을 둔다.
   전역 dict 에만 두면 (a) 재시작 시 전부 사라져 폴링 중인 클라이언트가 404 를
   받고, (b) ``--workers 2`` 로 띄우는 순간 업로드는 A 프로세스, 폴링은 B
   프로세스로 가서 계속 404 가 난다. 디스크를 거치면 둘 다 해결된다.
3. **작업 파일은 작업별 디렉터리에.** 원본과 결과가 섞이지 않고, 삭제가
   디렉터리 하나 지우는 것으로 끝난다.

환경 변수
    FA_DEVICE          'cuda:0' | 'cpu'    (기본: 자동)
    FA_IMGSZ           검출기 기본 해상도  (기본: 1280)
    FA_JOBS_DIR        작업 디렉터리       (기본: ./jobs)
    FA_MAX_UPLOAD_MB   업로드 상한         (기본: 2048)
    FA_JOB_TTL_MIN     완료 후 자동 삭제   (기본: 120, 0이면 안 지움)
    FA_SWEEP_SEC       정리 주기           (기본: 300)

실행
    uvicorn face_anonymizer.server:app --host 0.0.0.0 --port 8000
"""

import errno
import json
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, fields

try:
    import fcntl                      # POSIX 전용. 없으면 프로세스 간 직렬화 생략.
except ImportError:                   # pragma: no cover
    fcntl = None

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
SWEEP_SEC = int(os.environ.get("FA_SWEEP_SEC", 300))
STATE_FILE = "job.json"
GPU_LOCK_FILE = ".gpu.lock"
PROGRESS_FLUSH_SEC = 0.5      # 진행률을 디스크에 쓰는 최소 간격

CHUNK = 1 << 20
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# 추론 직렬화. max_workers=1 이 이 서버의 동시성 정책 전부다.
_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anon")
_JOBS = {}
_LOCK = threading.Lock()

_anonymizer = None
_anon_lock = threading.Lock()
_sweeper = None


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


def _state_path(jid):
    return os.path.join(JOBS_DIR, jid, STATE_FILE)


def save_job(j, force=True):
    """작업 상태를 디스크에 쓴다.

    진행률은 초당 수십 번 갱신되므로 ``force=False`` 로 호출해 간격을 둔다.
    쓰기는 임시 파일 + rename 으로 원자적으로 한다 — 다른 프로세스가 읽는
    중에 반쪽짜리 JSON 을 보면 안 된다.
    """
    now = time.time()
    if not force and now - getattr(j, "_flushed", 0.0) < PROGRESS_FLUSH_SEC:
        return
    path = _state_path(j.id)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in asdict(j).items()
                       if not k.startswith("_")}, f, ensure_ascii=False)
        os.replace(tmp, path)
        j._flushed = now
    except OSError as e:
        log.warning("작업 상태를 쓰지 못했다 (%s): %s", j.id, e)


def load_job_file(jid):
    """디스크에서 작업 상태를 읽는다. 없거나 깨졌으면 None."""
    try:
        with open(_state_path(jid), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    known = {f.name for f in fields(Job)}
    try:
        return Job(**{k: v for k, v in data.items() if k in known})
    except TypeError:
        return None


def find_job(jid):
    """작업 조회. 이 프로세스가 돌리는 중이면 메모리 값이 최신이다.

    다른 프로세스(``--workers N``)가 만든 작업은 메모리에 없으므로 디스크에서
    읽는다. 이게 없으면 업로드와 폴링이 다른 워커로 갈 때 계속 404 가 난다.
    """
    if not jid or "/" in jid or "\\" in jid or jid.startswith("."):
        return None
    with _LOCK:
        j = _JOBS.get(jid)
    return j if j is not None else load_job_file(jid)


def all_jobs():
    """메모리 + 디스크 병합 목록. 메모리 쪽이 우선(진행 중인 값이 최신)."""
    with _LOCK:
        merged = dict(_JOBS)
    try:
        entries = os.listdir(JOBS_DIR)
    except OSError:
        entries = []
    for jid in entries:
        if jid in merged or not os.path.isdir(os.path.join(JOBS_DIR, jid)):
            continue
        j = load_job_file(jid)
        if j is not None:
            merged[jid] = j
    return sorted(merged.values(), key=lambda x: -x.created)


def snapshot(j, queued_ahead=0):
    """폴링 응답 한 건."""
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
        "queued_ahead": queued_ahead,
    }


def sweep():
    """TTL 지난 작업 정리.

    예전에는 새 작업이 들어올 때만 돌아서, 업로드가 끊기면 디스크가 영원히
    안 비워졌다. 지금은 백그라운드 스레드가 주기적으로 돈다.
    """
    if not JOB_TTL:
        return 0
    now, removed = time.time(), 0
    for j in all_jobs():
        if j.finished and now - j.finished > JOB_TTL:
            shutil.rmtree(j.workdir or os.path.join(JOBS_DIR, j.id),
                          ignore_errors=True)
            with _LOCK:
                _JOBS.pop(j.id, None)
            removed += 1
    if removed:
        log.info("TTL 정리: %d건", removed)
    return removed


def _sweep_loop():
    while True:
        time.sleep(SWEEP_SEC)
        try:
            sweep()
        except Exception:                       # noqa: BLE001 — 청소가 서버를 죽이면 안 된다
            log.exception("정리 중 오류")


def recover_orphans():
    """재시작 시, 중단된 채 남은 작업을 정리한다.

    프로세스가 죽으면 queued/running 상태 파일이 그대로 남는다. 그대로 두면
    클라이언트는 영원히 '처리 중' 을 폴링한다. 시작할 때 한 번 훑어 실패로
    표시한다 (이 프로세스가 방금 만든 작업은 메모리에 있으므로 건드리지 않는다).
    """
    n = 0
    for j in all_jobs():
        with _LOCK:
            live = j.id in _JOBS
        if live or j.status not in ("queued", "running"):
            continue
        j.status = "error"
        j.error = "서버가 재시작되어 작업이 중단됐다. 다시 올려 주세요."
        j.finished = time.time()
        save_job(j)
        n += 1
    if n:
        log.warning("중단된 작업 %d건을 실패로 표시했다", n)
    return n


class gpu_lock:
    """프로세스 간 추론 직렬화.

    스레드 풀(max_workers=1)은 한 프로세스 안에서만 유효하다. ``--workers N``
    으로 띄우면 N개가 동시에 GPU 를 쓰려 해서 VRAM 이 터진다. 작업 디렉터리에
    잠금 파일을 두고 그 위에서 직렬화한다. fcntl 이 없는 플랫폼에서는 아무것도
    하지 않는다(그 경우 단일 프로세스로 운영해야 한다).
    """

    def __init__(self, path):
        self.path = path
        self.fh = None

    def __enter__(self):
        if fcntl is None:
            return self
        try:
            self.fh = open(self.path, "w")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EROFS):
                raise
            log.warning("GPU 잠금 파일을 쓸 수 없다 (%s) — 직렬화 없이 진행한다", e)
            self.fh = None
        return self

    def __exit__(self, *exc):
        if self.fh is not None:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_UN)
            finally:
                self.fh.close()
                self.fh = None
        return False


def _run(job_id):
    with _LOCK:
        j = _JOBS.get(job_id)
        if j is None:
            return
        j.status, j.stage_t0 = "running", time.time()
        params, workdir, name = dict(j.params), j.workdir, j.name
    save_job(j)

    def progress(stage, done, total):
        with _LOCK:
            if j.stage != stage:
                j.stage, j.stage_t0 = stage, time.time()
            j.done, j.total = done, total
        save_job(j, force=False)      # 폴링용 — 간격을 두고 흘려 쓴다

    src = os.path.join(workdir, "input" + os.path.splitext(name)[1])
    dst = os.path.join(workdir, os.path.splitext(name)[0] + "_anon.mp4")
    try:
        # 프로세스가 여러 개여도 GPU 는 한 번에 하나만 쓴다.
        with gpu_lock(os.path.join(JOBS_DIR, GPU_LOCK_FILE)):
            res = get_anonymizer().process(src, dst, progress=progress, **params)
        with _LOCK:
            j.status, j.output, j.finished = "done", res.output, time.time()
            j.result = {
                "frames": res.frames, "raw_boxes": res.raw_boxes,
                "filled_boxes": res.filled_boxes, "method": res.method,
                "audio": res.audio,
                # 결과를 그대로 믿으면 안 되는 사유. UI 가 배너로 띄운다.
                "warnings": list(res.warnings),
                "detected_frames": res.detected_frames,
                "detection_rate": round(res.detection_rate, 4),
                "fps": round(res.fps, 1),
                "detect_fps": round(res.detect_fps, 1),
                "realtime_factor": round(res.realtime_factor, 2),
                # 짧은 클립에서 소수점 첫째 자리로 반올림하면 단계 시간이
                # 전부 0.0 이 되어 어디가 느린지 안 보인다.
                "seconds": round(res.timing.total, 3),
                "timing": {"detect": round(res.timing.detect, 3),
                           "track": round(res.timing.track, 3),
                           "render": round(res.timing.render, 3),
                           "audio": round(res.timing.audio, 3)},
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
    save_job(j)


@asynccontextmanager
async def lifespan(_app):
    global _sweeper
    os.makedirs(JOBS_DIR, exist_ok=True)
    recover_orphans()
    if SWEEP_SEC > 0 and _sweeper is None:
        _sweeper = threading.Thread(target=_sweep_loop, daemon=True,
                                    name="sweeper")
        _sweeper.start()
    yield


app = FastAPI(title="face-anonymizer", version="0.2.0", lifespan=lifespan)


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
    jobs = all_jobs()
    info["jobs"] = {"total": len(jobs),
                    "running": sum(1 for j in jobs if j.status == "running"),
                    "queued": sum(1 for j in jobs if j.status == "queued")}
    info["pid"] = os.getpid()
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
    save_job(job)
    _EXEC.submit(_run, jid)
    return snapshot(job, queued_ahead=_queued_ahead(job))


def _queued_ahead(job, jobs=None):
    jobs = jobs if jobs is not None else all_jobs()
    return sum(1 for o in jobs
               if o.status == "queued" and o.created < job.created)


@app.get("/api/jobs")
def list_jobs():
    jobs = all_jobs()
    return [snapshot(j, _queued_ahead(j, jobs)) for j in jobs]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = find_job(jid)
    if j is None:
        raise HTTPException(404, "no such job")
    return snapshot(j, _queued_ahead(j))


@app.get("/api/jobs/{jid}/download")
def download(jid: str):
    j = find_job(jid)
    if j is None:
        raise HTTPException(404, "no such job")
    if j.status != "done":
        raise HTTPException(409, f"아직 준비되지 않았다 (status={j.status})")
    if not j.output or not os.path.exists(j.output):
        raise HTTPException(410, "결과물이 더 이상 없다 (보관 기간 만료)")
    name = os.path.splitext(j.name)[0] + "_anon.mp4"
    return FileResponse(j.output, media_type="video/mp4", filename=name)


@app.delete("/api/jobs/{jid}", status_code=204)
def delete_job(jid: str):
    j = find_job(jid)
    if j is None:
        raise HTTPException(404, "no such job")
    if j.status in ("queued", "running"):
        raise HTTPException(409, "진행 중인 작업은 삭제할 수 없다")
    with _LOCK:
        _JOBS.pop(jid, None)
    shutil.rmtree(j.workdir or os.path.join(JOBS_DIR, jid), ignore_errors=True)
