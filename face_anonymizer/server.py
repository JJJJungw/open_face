"""HTTP API + 웹 UI.

영상을 올리면 작업 큐에 넣고, 진행률을 폴링으로 보여 주고, 끝나면 내려받게
하는 최소 구성이다.

설계상 못 박아 둔 것.

1. **한 번에 한 편.** 추론은 워커 스레드 하나가 순차로 돌린다(GPU 한 장에
   검출기 하나). 프로세스를 여러 개 띄워도(`--workers N`) 파일 락으로 GPU 를
   직렬화한다.

   대기열은 개수로 막지 않는다. 전체 수행처럼 한꺼번에 수백 건을 넣는 사용이
   정상이기 때문이다. 대신 **디스크 여유 공간**으로 막는다(507) — 대기 중인
   작업은 입력 파일을 디스크에 들고 있으므로 진짜 제약은 거기다.

   상태는 ``queued``(대기) -> ``running``(수행중) -> ``done``(완료) 이고,
   실패하면 일시적 오류에 한해 ``FA_MAX_ATTEMPTS`` 회까지 다시 큐에 넣은 뒤
   그래도 안 되면 ``failed``(실패) 로 남긴다.
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
    FA_PRELOAD         기동 시 모델 로드   (기본: 1)

처리 파라미터 기본값 (JOB_DEFAULTS)
    FA_METHOD mosaic · FA_CONF 0.25 · FA_BATCH_SIZE 32 · FA_PAD 0.15
    FA_MOSAIC_SCALE 0.06 · FA_LINGER 5 · FA_INTERP 1 · FA_KEEP_AUDIO 1
    FA_CRF 23 · FA_BITRATE_RATIO 1.0
    (imgsz 는 FA_IMGSZ 를 검출기와 공유한다)
    FA_QUEUE_MAX       대기열 개수 상한    (기본: 0 = 무제한)
    FA_MIN_FREE_MB     최소 여유 디스크    (기본: 2048, 미달이면 507)
    FA_LIST_LIMIT      목록 기본 개수      (기본: 100)

S3 설정은 face_anonymizer/s3.py 참고 (FA_S3_BUCKET 등). 버킷이 설정돼 있으면
입력을 S3 에서 내려받고 결과물을 다시 올린다.
    FA_MAX_ATTEMPTS    일시적 오류 재시도  (기본: 3)

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

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from . import naming
from . import s3 as s3mod
from .anonymize import METHODS
from .pipeline import (
    DEFAULT_BITRATE_RATIO,
    DEFAULT_CRF,
    VideoOpenError,
    VideoWriteError,
)
from .webui import INDEX_HTML

log = logging.getLogger(__name__)

DEVICE = os.environ.get("FA_DEVICE") or None
IMGSZ = int(os.environ.get("FA_IMGSZ", 1280))
JOBS_DIR = os.path.abspath(os.environ.get("FA_JOBS_DIR", "jobs"))
MAX_BYTES = int(os.environ.get("FA_MAX_UPLOAD_MB", 2048)) * 1024 * 1024
JOB_TTL = int(os.environ.get("FA_JOB_TTL_MIN", 120)) * 60
SWEEP_SEC = int(os.environ.get("FA_SWEEP_SEC", 300))
PRELOAD = os.environ.get("FA_PRELOAD", "1") not in ("0", "false", "False")
RETRY_AFTER = int(os.environ.get("FA_RETRY_AFTER", 30))
# 대기열은 기본적으로 개수로 제한하지 않는다. 전체 수행처럼 한꺼번에 수백 건을
# 넣는 사용이 정상이고, 개수는 애초에 잘못된 기준이다 — 10건이 50MB 짜리면
# 아무것도 아니고 2GB 짜리면 이미 위험하다. 진짜 제약은 디스크다(MIN_FREE_MB).
QUEUE_MAX = int(os.environ.get("FA_QUEUE_MAX", 0))          # 0 = 무제한
MIN_FREE_MB = int(os.environ.get("FA_MIN_FREE_MB", 2048))   # 0 = 검사 안 함
LIST_LIMIT = int(os.environ.get("FA_LIST_LIMIT", 100))
MAX_ATTEMPTS = int(os.environ.get("FA_MAX_ATTEMPTS", 3))

# 다시 시도해도 결과가 같은 오류들. 깨진 파일이나 잘못된 인자를 세 번 돌리는 건
# 그냥 낭비이고, 그동안 뒤에 쌓인 정상 작업이 밀린다.
PERMANENT_ERRORS = (VideoOpenError, VideoWriteError, ValueError, FileNotFoundError)
STATE_FILE = "job.json"
GPU_LOCK_FILE = ".gpu.lock"
PROGRESS_FLUSH_SEC = 0.5      # 진행률을 디스크에 쓰는 최소 간격

CHUNK = 1 << 20
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _bool_env(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() not in ("0", "false", "no")


# 처리 파라미터 기본값.
#
# **호출하는 쪽은 입력만 주면 된다.** 튜닝된 값은 서비스가 들고 있어야지,
# 호출자마다 들고 다니면 어느 설정으로 처리됐는지가 호출 지점마다 달라진다.
# 운영 중 조정은 환경 변수로 하고, 필요할 때만 요청에서 개별 항목을 덮는다.
#
# imgsz 는 검출기와 같은 값을 쓴다(FA_IMGSZ). 둘이 어긋나면 워밍업한 커널과
# 실제 추론이 달라진다.
JOB_DEFAULTS = {
    "method": os.environ.get("FA_METHOD", "mosaic"),
    "conf": float(os.environ.get("FA_CONF", "0.25")),
    "imgsz": IMGSZ,
    "batch_size": int(os.environ.get("FA_BATCH_SIZE", "32")),
    "pad": float(os.environ.get("FA_PAD", "0.15")),
    "mosaic_scale": float(os.environ.get("FA_MOSAIC_SCALE", "0.06")),
    "linger": int(os.environ.get("FA_LINGER", "5")),
    "interp": _bool_env("FA_INTERP", True),
    "keep_audio": _bool_env("FA_KEEP_AUDIO", True),
    "crf": DEFAULT_CRF,
    "bitrate_ratio": DEFAULT_BITRATE_RATIO,
}

# 추론 직렬화. max_workers=1 이 이 서버의 동시성 정책 전부다.
_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anon")
_JOBS = {}
_LOCK = threading.Lock()

_anonymizer = None
_anon_lock = threading.Lock()
_sweeper = None
_model_error = None      # 기동 시 모델 로드 실패 사유
_current = None          # 이 프로세스가 지금 붙잡고 있는 작업 id


def get_anonymizer():
    """검출기 싱글턴.

    기본적으로 기동 시(lifespan) 미리 올린다. 첫 요청 때 로드하면 헬스체크는
    이미 통과한 상태라, 오케스트레이터가 보낸 첫 요청이 모델 로딩 수십 초를
    기다리게 된다.
    """
    global _anonymizer
    with _anon_lock:
        if _anonymizer is None:
            from .pipeline import VideoAnonymizer
            log.info("검출기 로드 중 (device=%s imgsz=%d)", DEVICE, IMGSZ)
            _anonymizer = VideoAnonymizer(device=DEVICE, imgsz=IMGSZ)
            log.info("검출기 준비 완료")
        return _anonymizer


def is_ready():
    """추론을 받을 수 있는 상태인가."""
    return _anonymizer is not None and _model_error is None


def is_busy():
    """지금 추론을 돌리고 있는가."""
    with _LOCK:
        return any(j.status == "running" for j in _JOBS.values())


def free_mb():
    """작업 디렉터리가 놓일 볼륨의 여유 공간(MB).

    첫 작업 전에는 JOBS_DIR 이 아직 없을 수 있다. 그때 None 을 돌려주면 디스크
    검사가 조용히 건너뛰어지므로, 존재하는 상위 경로까지 올라가서 잰다.
    """
    path = os.path.abspath(JOBS_DIR)
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent
    try:
        return shutil.disk_usage(path).free // (1024 * 1024)
    except OSError:
        return None


def queue_depth():
    """대기 중인 작업 수."""
    with _LOCK:
        return sum(1 for j in _JOBS.values() if j.status == "queued")


@dataclass
class Job:
    id: str
    name: str
    params: dict
    workdir: str
    status: str = "queued"        # queued(대기) | running(수행중) | done(완료) | failed(실패)
    attempts: int = 0             # 시도 횟수 (재시도 포함)
    s3_key: str = ""              # S3 입력 키 (업로드면 빈 문자열)
    s3_output: str = ""           # S3 결과물 키
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
        "attempts": j.attempts, "max_attempts": MAX_ATTEMPTS,
        "s3_key": j.s3_key, "s3_output": j.s3_output,
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
        j.status = "failed"
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


def _fail_or_retry(j, exc, permanent):
    """실패 처리. 일시적 오류면 다시 큐에 넣는다.

    같은 입력으로 같은 결과가 나올 오류(깨진 파일, 잘못된 인자)는 재시도하지
    않는다 — 세 번 돌려도 결과가 같고 그동안 뒤에 쌓인 정상 작업이 밀린다.
    """
    msg = f"{type(exc).__name__}: {exc}"
    retryable = not permanent and j.attempts < MAX_ATTEMPTS
    with _LOCK:
        j.error = msg
        if retryable:
            j.status, j.done, j.total, j.stage = "queued", 0, 0, ""
        else:
            j.status, j.finished = "failed", time.time()
    save_job(j)
    if retryable:
        log.warning("작업 %s 실패 (%d/%d회) — 다시 시도한다: %s",
                    j.id, j.attempts, MAX_ATTEMPTS, msg)
        _EXEC.submit(_run, j.id)
    else:
        log.error("작업 %s 실패 (%d회 시도, %s): %s", j.id, j.attempts,
                  "재시도 불가" if permanent else "재시도 소진", msg)


def _run(job_id):
    with _LOCK:
        j = _JOBS.get(job_id)
        if j is None:
            return
        j.status, j.stage_t0 = "running", time.time()
        j.attempts += 1
        params, workdir, name = dict(j.params), j.workdir, j.name
    save_job(j)

    def progress(stage, done, total):
        with _LOCK:
            if j.stage != stage:
                j.stage, j.stage_t0 = stage, time.time()
            j.done, j.total = done, total
        save_job(j, force=False)      # 폴링용 — 간격을 두고 흘려 쓴다

    src = os.path.join(workdir, "input" + os.path.splitext(name)[1])
    dst = os.path.join(workdir, naming.output_name(name))
    try:
        if j.s3_key and not os.path.exists(src):
            store = s3mod.get_store()
            if store is None:
                raise s3mod.S3Error("S3 가 설정되지 않았다")
            log.info("S3 에서 내려받는다: %s", j.s3_key)
            store.download(j.s3_key, src)
        # 프로세스가 여러 개여도 GPU 는 한 번에 하나만 쓴다.
        with gpu_lock(os.path.join(JOBS_DIR, GPU_LOCK_FILE)):
            res = get_anonymizer().process(src, dst, progress=progress, **params)
        if j.s3_key:
            store = s3mod.get_store()
            key = store.output_key(j.s3_key)
            log.info("S3 에 올린다: %s", key)
            store.upload(res.output, key)
            with _LOCK:
                j.s3_output = key
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
                "s3_key": j.s3_key, "s3_output": j.s3_output,
            }
            j.error = ""
        save_job(j)
    except PERMANENT_ERRORS as e:
        _fail_or_retry(j, e, permanent=True)
    except Exception as e:                      # noqa: BLE001 — 워커가 조용히 죽으면 안 된다
        log.exception("작업 %s 실패", job_id)
        _fail_or_retry(j, e, permanent=False)


@asynccontextmanager
async def lifespan(_app):
    global _sweeper, _model_error
    os.makedirs(JOBS_DIR, exist_ok=True)
    recover_orphans()
    if SWEEP_SEC > 0 and _sweeper is None:
        _sweeper = threading.Thread(target=_sweep_loop, daemon=True,
                                    name="sweeper")
        _sweeper.start()
    if PRELOAD:
        # 여기서 죽이지 않고 사유를 남긴다. 크래시 루프로 재시작하면 로그가
        # 흘러가서 왜 안 뜨는지 알기 어렵다. health 가 503 으로 이유를 알려준다.
        try:
            get_anonymizer()
        except Exception as e:                  # noqa: BLE001
            _model_error = f"{type(e).__name__}: {e}"
            log.exception("모델 로드 실패 — 준비되지 않은 상태로 뜬다")
    yield


app = FastAPI(title="face-anonymizer", version="0.2.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/api/status")
def status():
    """오케스트레이터용 최소 응답. 디스크를 훑지 않는다."""
    return {"ready": is_ready(), "busy": is_busy(),
            "queued": queue_depth(), "free_mb": free_mb(),
            "model_error": _model_error}


@app.get("/api/health")
def health(response: Response):
    ready = is_ready()
    if not ready:
        # 준비 전에는 컨테이너를 healthy 로 보이게 하면 안 된다. 오케스트레이터가
        # 트래픽을 보내고, 그 요청이 모델 로딩을 통째로 기다리게 된다.
        response.status_code = 503
    info = {"status": "ok" if ready else "not-ready",
            "ready": ready, "busy": is_busy(), "queued": queue_depth(),
            "free_mb": free_mb(),
            "model_loaded": _anonymizer is not None, "model_error": _model_error,
            "device": DEVICE or "auto", "imgsz": IMGSZ,
            "methods": list(METHODS)}
    loaded = _anonymizer is not None
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


@app.get("/api/defaults")
def defaults():
    """서비스가 쓰는 처리 파라미터 기본값.

    UI 가 컨트롤 초깃값을 여기서 받아 간다 — 화면에 값을 박아 두면 서버 설정을
    바꿔도 화면은 옛 값을 보내서 둘이 조용히 어긋난다.
    """
    return dict(JOB_DEFAULTS)


@app.get("/api/s3/objects")
def s3_objects(prefix: str = ""):
    """버킷을 한 단계씩 나열한다 (S3 콘솔과 같은 방식).

    설정 전이면 404. UI 는 그때 안내만 띄우고 직접 업로드로 쓴다.
    """
    store = s3mod.get_store()
    if store is None:
        raise HTTPException(404, "S3 가 설정되지 않았다 (FA_S3_BUCKET)")
    if ".." in prefix:
        raise HTTPException(400, "잘못된 prefix")
    try:
        folders, objects = store.list(prefix)
    except s3mod.S3Error as e:
        raise HTTPException(502, str(e)) from e
    done = store.processed_keys()
    for o in objects:
        o["processed"] = store.output_key(o["key"]) in done
    return {"bucket": store.bucket, "prefix": prefix or store.root_prefix,
            "folders": folders, "objects": objects,
            "output_prefix": store.output_prefix}


@app.post("/api/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(None),
    s3_key: str = Form(""),
    # 전부 선택 사항이다. 안 보내면 서비스 기본값(JOB_DEFAULTS)을 쓴다 —
    # 호출하는 쪽은 입력만 주면 된다.
    method: str = Form(None),
    conf: float = Form(None),
    imgsz: int = Form(None),
    batch_size: int = Form(None),
    pad: float = Form(None),
    mosaic_scale: float = Form(None),
    linger: int = Form(None),
    interp: bool = Form(None),
    keep_audio: bool = Form(None),
    crf: int = Form(None),
    bitrate_ratio: float = Form(None),
):
    given = {"method": method, "conf": conf, "imgsz": imgsz,
             "batch_size": batch_size, "pad": pad, "mosaic_scale": mosaic_scale,
             "linger": linger, "interp": interp, "keep_audio": keep_audio,
             "crf": crf, "bitrate_ratio": bitrate_ratio}
    params = {**JOB_DEFAULTS, **{k: v for k, v in given.items() if v is not None}}
    method, conf, imgsz = params["method"], params["conf"], params["imgsz"]
    # 입력은 둘 중 하나다 — 업로드한 파일이거나 S3 키.
    if bool(s3_key) == bool(file is not None and file.filename):
        raise HTTPException(400, "file 또는 s3_key 중 하나만 보내라")
    if method not in METHODS:
        raise HTTPException(400, f"unknown method: {method}")
    if not 0 < conf < 1:
        raise HTTPException(400, "conf 는 0~1 사이여야 한다")

    if s3_key:
        if s3mod.get_store() is None:
            raise HTTPException(404, "S3 가 설정되지 않았다 (FA_S3_BUCKET)")
        if ".." in s3_key or s3_key.startswith("/"):
            raise HTTPException(400, "잘못된 s3_key")
        name = os.path.basename(s3_key)
    else:
        name = os.path.basename(file.filename)
    ext = os.path.splitext(name)[1].lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"지원하지 않는 확장자: {ext or '(없음)'}")
    # stride 배수 맞추기는 검출기가 한다(geometry.snap_to_stride). 여기서
    # 또 계산하면 규칙이 두 벌이 되고 실제로 서로 달랐다(round vs ceil).
    imgsz = max(320, min(int(imgsz), 2048))
    params["imgsz"] = imgsz

    if not is_ready():
        raise HTTPException(503, f"모델이 준비되지 않았다: {_model_error or '로딩 중'}")
    # 대기 중인 작업은 입력 파일을 들고 있다. 디스크가 차면 업로드가 중간에
    # 깨지거나 처리 중인 작업의 출력까지 같이 망가진다.
    free = free_mb()
    if MIN_FREE_MB and free is not None and free < MIN_FREE_MB:
        raise HTTPException(507, f"디스크 여유 부족 ({free}MB < {MIN_FREE_MB}MB)",
                            headers={"Retry-After": str(RETRY_AFTER)})
    jid = uuid.uuid4().hex[:12]
    workdir = os.path.join(JOBS_DIR, jid)
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, "input" + ext)

    # S3 입력은 워커가 내려받는다. 접수 요청을 붙들고 수백 MB 를 받으면
    # 클라이언트가 그동안 응답을 기다리게 된다.
    if not s3_key:
        size = 0
        try:
            with open(src, "wb") as f:
                while True:
                    chunk = await file.read(CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413,
                                            f"업로드 상한 초과 ({MAX_BYTES // 1048576} MB)")
                    f.write(chunk)
        except Exception:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        if size == 0:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(400, "빈 파일")

    job = Job(id=jid, name=name, workdir=workdir, s3_key=s3_key,
              params=params)
    global _current
    with _LOCK:
        # 대기열 길이 확인과 등록이 같은 락 안에 있어야 동시 요청이 상한을 넘지 않는다.
        if QUEUE_MAX and sum(1 for o in _JOBS.values()
                             if o.status == "queued") >= QUEUE_MAX:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(429, f"대기열이 가득 찼다 ({QUEUE_MAX}건)",
                                headers={"Retry-After": str(RETRY_AFTER)})
        _JOBS[jid] = job
        _current = jid
    save_job(job)
    # 응답 스냅샷은 제출 **전에** 뜬다. 제출 후에 뜨면 워커가 이미 시작해
    # status 가 running 으로 보일 수 있다 (경합).
    snap = snapshot(job, queued_ahead=_queued_ahead(job))
    _EXEC.submit(_run, jid)
    return snap


def _queued_ahead(job):
    """앞에 몇 건 대기 중인가. **메모리만** 본다.

    폴링 경로라 여기서 디스크를 훑으면 안 된다. 대기 중인 작업은 이 프로세스의
    워커가 들고 있으므로 메모리에 있고, 없으면(재시작 후 남은 기록) 어차피
    대기 중이 아니다.
    """
    with _LOCK:
        return sum(1 for o in _JOBS.values()
                   if o.status == "queued" and o.created < job.created)


@app.get("/api/jobs")
def list_jobs(limit: int = LIST_LIMIT, status: str = None):
    """작업 목록. 최신순, 기본 100건.

    대기 순번은 한 번에 계산한다 — 작업마다 전체를 다시 훑으면 O(N^2) 이고,
    전체 수행으로 수백 건을 넣으면 목록 한 번에 수십만 번 반복하게 된다.
    """
    jobs = all_jobs()
    if status:
        jobs = [j for j in jobs if j.status == status]
    order = sorted((j for j in jobs if j.status == "queued"),
                   key=lambda x: x.created)
    ahead = {j.id: i for i, j in enumerate(order)}
    return [snapshot(j, ahead.get(j.id, 0)) for j in jobs[:max(0, limit)]]


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
    name = naming.output_name(j.name)
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
