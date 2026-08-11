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
    FA_BATCH_MAX       한 번에 넣을 개수   (기본: 0 = 무제한)
    FA_FAILED_TTL_MIN  실패 보관           (기본: 0 = 안 지움)
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

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from . import errors, naming
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
# 한 번에 넣을 개수도 막지 않는다. 폴더 하나에 수천 건이 들어 있는 게 정상이고,
# 상한에 걸리면 사용자가 폴더를 손으로 쪼개야 한다 — 그게 훨씬 나쁘다.
# S3 입력은 대기 중에 디스크를 쓰지 않는다(내려받기는 _run 에서 한다). 그래서
# 대기열이 길어도 드는 건 작업 디렉터리와 job.json 뿐이다.
BATCH_MAX = int(os.environ.get("FA_BATCH_MAX", 0))          # 0 = 무제한
# 실패/취소 작업은 기본적으로 지우지 않는다. 배치로 수백 건 돌린 뒤 몇 건이
# 실패했을 때, 입력과 사유가 남아 있어야 원인을 볼 수 있다.
FAILED_TTL = int(os.environ.get("FA_FAILED_TTL_MIN", 0)) * 60
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
    # queued(대기) | running(수행중) | done(완료) | failed(실패) | cancelled(취소)
    status: str = "queued"
    attempts: int = 0             # 시도 횟수 (재시도 포함)
    cancel: bool = False          # 취소 요청 표시. 진행 콜백이 보고 중단한다
    s3_key: str = ""              # S3 입력 키 (업로드면 빈 문자열)
    s3_output: str = ""           # S3 결과물 키
    stage: str = ""               # detect | render
    done: int = 0
    total: int = 0
    error: dict = field(default_factory=dict)
    output: str = ""
    result: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    started: float = 0.0          # 실제로 돌기 시작한 시각 (대기 시간 제외)
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
    # 전사(transcode)는 그 앞 단계라 자기 게이지를 따로 채운다 — 검출이
    # 시작되면 0 부터 다시 오른다.
    if j.stage == "transcode":
        overall = pct
    else:
        overall = pct // 2 + (50 if j.stage == "render" else 0)
    if j.status == "done":
        overall = 100

    # 이 작업 하나가 끝나기까지 남은 시간. eta 는 **현재 단계**만 보므로
    # 검출 중이면 렌더가 통째로 빠져 절반으로 나온다. 대기열 전체 예상을
    # 세우려면 작업 한 건의 총 소요가 필요해서 진행률로 되짚는다.
    job_elapsed = time.time() - j.started if j.started else 0.0
    job_eta = (job_elapsed * (100 - overall) / overall
               if j.status == "running" and overall > 0
               and j.stage in ("detect", "render") else 0.0)

    return {
        "id": j.id, "name": j.name, "status": j.status, "stage": j.stage,
        "percent": pct, "overall": overall, "fps": round(fps, 1),
        "eta": round(eta), "job_eta": round(job_eta),
        "job_elapsed": round(job_elapsed), "error": j.error, "result": j.result,
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
        # 실패·취소는 원인을 보려면 남아 있어야 한다.
        ttl = FAILED_TTL if j.status in ("failed", "cancelled") else JOB_TTL
        if not ttl:
            continue
        if j.finished and now - j.finished > ttl:
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
        j.error = {"code": "interrupted", "title": "서버 재시작으로 중단됨",
                   "detail": "처리 중 프로세스가 종료됐다", "hint": "다시 제출하라",
                   "retryable": True}
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


class JobCancelled(Exception):
    """진행 콜백이 취소 요청을 보고 던진다."""


def _fail_or_retry(j, exc, permanent):
    """실패 처리. 일시적 오류면 다시 큐에 넣는다.

    같은 입력으로 같은 결과가 나올 오류(깨진 파일, 잘못된 인자)는 재시도하지
    않는다 — 세 번 돌려도 결과가 같고 그동안 뒤에 쌓인 정상 작업이 밀린다.
    """
    info = errors.job_error(exc)
    retryable = not permanent and info["retryable"] and j.attempts < MAX_ATTEMPTS
    msg = info["detail"]
    # 어디서 넘어졌는지, 그리고 왜 다시 시도했는지/안 했는지를 오류에 같이
    # 남긴다. 사유만 있고 이 둘이 없으면 "3회 시도" 라는 숫자를 어떻게 읽어야
    # 할지 알 수 없다.
    info["stage"] = j.stage or ("download" if j.s3_key and not j.done else "")
    info["policy"] = ("permanent" if permanent
                      else "exhausted" if not retryable else "retrying")
    with _LOCK:
        j.error = info
        if retryable:
            j.status, j.done, j.total, j.stage = "queued", 0, 0, ""
        else:
            j.status, j.finished = "failed", time.time()
    save_job(j)
    if retryable:
        log.warning("작업 %s 실패 [%s] (%d/%d회) — 다시 시도한다: %s",
                    j.id, info["code"], j.attempts, MAX_ATTEMPTS, msg)
        _EXEC.submit(_run, j.id)
    else:
        log.error("작업 %s 실패 [%s] (%d회 시도, %s): %s", j.id, info["code"],
                  j.attempts, "재시도 불가" if permanent else "재시도 소진", msg)


def _run(job_id):
    with _LOCK:
        j = _JOBS.get(job_id)
        if j is None:
            return
        if j.cancel or j.status == "cancelled":
            # 대기 중에 취소된 건 아예 시작하지 않는다.
            j.status, j.finished = "cancelled", j.finished or time.time()
            save_job(j)
            return
        j.status, j.stage_t0 = "running", time.time()
        j.started = time.time()
        j.attempts += 1
        params, workdir, name = dict(j.params), j.workdir, j.name
    save_job(j)

    def progress(stage, done, total):
        # 취소는 여기서만 끊을 수 있다. 파이프라인이 프레임마다 부르는 유일한
        # 지점이라, 예외를 던지면 다음 프레임으로 넘어가지 않고 빠져나온다.
        if j.cancel:
            raise JobCancelled()
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
                "source_codec": res.source_codec, "transcoded": res.transcoded,
                "timing": {"ingest": round(res.timing.ingest, 3),
                           "detect": round(res.timing.detect, 3),
                           "track": round(res.timing.track, 3),
                           "render": round(res.timing.render, 3),
                           "audio": round(res.timing.audio, 3)},
                "video": {"width": res.video.width, "height": res.video.height,
                          "fps": round(res.video.fps, 2)},
                "s3_key": j.s3_key, "s3_output": j.s3_output,
            }
            j.error = {}
        save_job(j)
    except JobCancelled:
        with _LOCK:
            j.status, j.finished = "cancelled", time.time()
            j.error = {"code": errors.CANCELLED.code,
                       "title": errors.CANCELLED.title, "detail": "",
                       "hint": "", "retryable": False}
        save_job(j)
        log.info("작업 %s 취소됨", job_id)
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


app = FastAPI(title="face-anonymizer", version="0.3.0", lifespan=lifespan)
errors.install(app)


@app.get("/api/problems")
def problems():
    """이 서비스가 낼 수 있는 오류 목록.

    호출하는 쪽이 code 별 대응(재시도/전환/사람 호출)을 미리 짜 둘 수 있게
    한곳에서 노출한다.
    """
    return {"problems": [p.as_dict() for p in errors.CATALOG.values()]}


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
        raise errors.S3_NOT_CONFIGURED()
    if ".." in prefix:
        raise errors.INVALID_KEY(prefix)
    try:
        folders, objects = store.list(prefix)
    except s3mod.S3Error as e:
        raise (e.problem or errors.S3_UPSTREAM)(str(e)) from e
    done = store.processed_keys()
    for o in objects:
        o["processed"] = store.output_key(o["key"]) in done
    return {"bucket": store.bucket, "prefix": prefix or store.root_prefix,
            "folders": folders, "objects": objects,
            "output_prefix": store.output_prefix}


def _queued_ahead(job):
    """앞에 몇 건 대기 중인가. **메모리만** 본다.

    폴링 경로라 여기서 디스크를 훑으면 안 된다. 대기 중인 작업은 이 프로세스의
    워커가 들고 있으므로 메모리에 있고, 없으면(재시작 후 남은 기록) 어차피
    대기 중이 아니다.
    """
    with _LOCK:
        return sum(1 for o in _JOBS.values()
                   if o.status == "queued" and o.created < job.created)


def _coerce(key, value):
    """폼은 전부 문자열로 온다. 기본값의 타입에 맞춰 되돌린다."""
    ref = JOB_DEFAULTS.get(key)
    if isinstance(ref, bool):
        return str(value).strip().lower() in ("1", "true", "on", "yes")
    try:
        if isinstance(ref, int):
            return int(float(value))
        if isinstance(ref, float):
            return float(value)
    except (TypeError, ValueError) as e:
        raise errors.INVALID_INPUT(f"{key} 값이 숫자가 아니다: {value!r}",
                                   field=key) from e
    return value


def check_admission():
    """받을 수 있는 상태인지. 못 받으면 ProblemError."""
    if not is_ready():
        raise (errors.MODEL_LOAD_FAILED(_model_error) if _model_error
               else errors.NOT_READY("기동 중이다"))
    free = free_mb()
    if MIN_FREE_MB and free is not None and free < MIN_FREE_MB:
        raise errors.INSUFFICIENT_STORAGE(f"{free}MB < {MIN_FREE_MB}MB",
                                          free_mb=free, retry_after=RETRY_AFTER)


def resolve_params(given):
    """보낸 것만 덮고 나머지는 서비스 기본값. 검증도 여기서."""
    params = {**JOB_DEFAULTS,
              **{k: v for k, v in given.items() if v is not None}}
    if params["method"] not in METHODS:
        raise errors.INVALID_INPUT(f"모르는 방식: {params['method']}",
                                   field="method", allowed=list(METHODS))
    if not 0 < params["conf"] < 1:
        raise errors.INVALID_INPUT(
            f"conf 는 0~1 사이여야 한다 (받은 값 {params['conf']})", field="conf")
    params["imgsz"] = max(320, min(int(params["imgsz"]), 2048))
    return params


def check_video_name(name):
    ext = os.path.splitext(name)[1].lower()
    if ext not in VIDEO_EXT:
        raise errors.UNSUPPORTED_MEDIA(f"확장자 {ext or '(없음)'}",
                                       supported=sorted(VIDEO_EXT))
    return ext


def check_s3_key(key):
    if s3mod.get_store() is None:
        raise errors.S3_NOT_CONFIGURED()
    if ".." in key or key.startswith("/"):
        raise errors.INVALID_KEY(key)
    return check_video_name(os.path.basename(key))


def new_job_id():
    return uuid.uuid4().hex[:12]


def enqueue(name, params, s3_key="", jid=None, workdir=None):
    """작업을 등록하고 워커에 넘긴다. 대기열 상한도 여기서 본다.

    ``jid`` 와 ``workdir`` 은 함께 온다 — 업로드 경로는 파일을 받아야 해서
    디렉터리를 먼저 만든다. 둘이 어긋나면 상태 파일이 엉뚱한 곳에 쓰인다.
    """
    global _current
    jid = jid or new_job_id()
    workdir = workdir or os.path.join(JOBS_DIR, jid)
    os.makedirs(workdir, exist_ok=True)
    job = Job(id=jid, name=name, workdir=workdir, s3_key=s3_key, params=params)
    with _LOCK:
        # 대기열 확인과 등록이 같은 락 안에 있어야 동시 요청이 상한을 넘지 않는다.
        if QUEUE_MAX and sum(1 for o in _JOBS.values()
                             if o.status == "queued") >= QUEUE_MAX:
            shutil.rmtree(workdir, ignore_errors=True)
            raise errors.QUEUE_FULL(f"대기 {QUEUE_MAX}건", retry_after=RETRY_AFTER)
        _JOBS[jid] = job
        _current = jid
    save_job(job)
    snap = snapshot(job, queued_ahead=_queued_ahead(job))
    _EXEC.submit(_run, jid)
    return job, snap


@app.post("/api/jobs", status_code=202)
async def create_jobs(request: Request):
    """**제출은 여기 하나다.** 한 건이든 여러 건이든 폴더든 같은 요청, 같은 응답.

    진입점을 나누면 클라이언트가 경우마다 분기해야 하고, 화면에도 버튼이 그만큼
    늘어난다. 입력이 무엇이냐만 다르고 나머지는 전부 같다.

    받는 형태::

        multipart/form-data   file=@clip.mp4                     # 업로드 한 건
        application/json      {"s3_keys": ["a.mp4", "b.mp4"]}    # 고른 파일들
        application/json      {"s3_prefix": "kbs/"}              # 폴더 하나
        application/json      {"s3_prefix": ["kbs/", "mbc/"]}    # 폴더 여럿

    **파일과 폴더는 같이 보낼 수 있다.** 화면에서 파일 두 개와 폴더 하나를
    한꺼번에 체크하는 게 자연스럽기 때문이다. 펼친 결과가 겹치면 한 번만
    넣는다. 업로드(``file``)만 S3 선택과 같이 못 보낸다 — 올라오는 바이트와
    버킷의 키는 아예 다른 경로다.

    옵션은 JSON 이면 ``params``, multipart 면 폼 필드로 준다. 안 주면 서비스
    기본값(GET /api/defaults).

    폴더 제출은 ``recursive``(하위 폴더까지, 기본 false)와
    ``skip_processed``(이미 결과물이 있는 건 건너뛰기, 기본 false)를 받는다.
    이름이 ``_deid`` 로 끝나는 결과물은 어느 쪽이든 입력에서 뺀다.

    응답은 항상 같다::

        {"accepted": [{"id", "name", "s3_key"}],
         "rejected": [{"s3_key", "error": {...}}],
         "queued": 3}

    **한 건이 거절돼도 나머지는 받는다.** 수백 건에서 키 하나가 오타라고 전체를
    되돌리면 호출하는 쪽이 무엇이 들어갔는지 알 수 없다.
    """
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip()
    upload = None
    keys, prefixes, recursive, skip_processed = [], [], False, False

    if ctype == "application/json":
        try:
            body = await request.json()
        except Exception as e:                      # noqa: BLE001
            raise errors.INVALID_INPUT("JSON 을 읽을 수 없다") from e
        if not isinstance(body, dict):
            raise errors.INVALID_INPUT("객체를 보내라")
        keys = body.get("s3_keys") or []
        prefixes = body.get("s3_prefix") or body.get("s3_prefixes") or []
        if isinstance(prefixes, str):               # 한 개는 문자열로도 받는다
            prefixes = [prefixes]
        if not isinstance(prefixes, list):
            raise errors.INVALID_INPUT("s3_prefix 는 문자열이나 배열이어야 한다",
                                       field="s3_prefix")
        recursive = bool(body.get("recursive"))
        skip_processed = bool(body.get("skip_processed"))
        given = body.get("params") or {}
        if not isinstance(keys, list):
            raise errors.INVALID_INPUT("s3_keys 는 배열이어야 한다", field="s3_keys")
    else:
        form = await request.form()
        upload = form.get("file")
        if isinstance(upload, str):                 # 파일이 아니라 문자열이면 무시
            upload = None
        keys = [v for v in form.getlist("s3_keys") if v]
        one = form.get("s3_key")
        if one:
            keys.append(one)
        prefixes = [v for v in form.getlist("s3_prefix") if v]
        recursive = str(form.get("recursive", "")).lower() in ("1", "true", "on")
        skip_processed = str(form.get("skip_processed", "")).lower() in (
            "1", "true", "on")
        given = {k: form.get(k) for k in JOB_DEFAULTS if form.get(k) is not None}
        given = {k: _coerce(k, v) for k, v in given.items()}

    uploaded = upload is not None and bool(getattr(upload, "filename", ""))
    if not uploaded and not keys and not prefixes:
        raise errors.MISSING_INPUT()
    if uploaded and (keys or prefixes):
        raise errors.CONFLICTING_INPUT(
            "업로드(file)와 S3 선택은 같이 보낼 수 없다")

    params = resolve_params(given)

    # 폴더는 여기서 펼친다. 클라이언트가 목록을 먼저 받아 오게 하면 그 사이에
    # 파일이 늘거나 줄 수 있고, 왕복도 한 번 더 든다.
    if prefixes:
        store = s3mod.get_store()
        if store is None:
            raise errors.S3_NOT_CONFIGURED()
        expanded = []
        for prefix in prefixes:
            if ".." in prefix:
                raise errors.INVALID_KEY(prefix)
            try:
                objs = (store.list_all(prefix) if recursive
                        else store.list(prefix)[1])
            except s3mod.S3Error as e:
                raise (e.problem or errors.S3_UPSTREAM)(str(e)) from e
            done = store.processed_keys() if skip_processed else set()
            expanded += [
                o["key"] for o in objs
                if os.path.splitext(o["key"])[1].lower() in VIDEO_EXT
                and not naming.is_output(o["key"])
                and (not skip_processed or store.output_key(o["key"]) not in done)]
        if not expanded and not keys:
            raise errors.BATCH_EMPTY(
                f"{' · '.join(prefixes)} 에 처리할 영상이 없다")
        keys = keys + expanded

    # 폴더를 펼친 결과가 따로 고른 파일과 겹칠 수 있다. 순서는 유지한다 —
    # 화면에 보인 차례대로 큐에 들어가야 진행 상황이 읽힌다.
    keys = list(dict.fromkeys(keys))

    if BATCH_MAX and len(keys) > BATCH_MAX:
        raise errors.BATCH_TOO_LARGE(f"{len(keys)}건 (상한 {BATCH_MAX})",
                                     limit=BATCH_MAX)
    check_admission()

    accepted, rejected = [], []

    if uploaded:
        name = os.path.basename(upload.filename)
        ext = check_video_name(name)
        jid = new_job_id()
        workdir = os.path.join(JOBS_DIR, jid)
        os.makedirs(workdir, exist_ok=True)
        src = os.path.join(workdir, "input" + ext)
        size = 0
        try:
            with open(src, "wb") as f:
                while True:
                    chunk = await upload.read(CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise errors.PAYLOAD_TOO_LARGE(
                            f"상한 {MAX_BYTES // 1048576} MB")
                    f.write(chunk)
        except Exception:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        if size == 0:
            shutil.rmtree(workdir, ignore_errors=True)
            raise errors.EMPTY_FILE()
        _job, snap = enqueue(name, params, jid=jid, workdir=workdir)
        accepted.append({"id": snap["id"], "name": name, "s3_key": None})
    else:
        for key in keys:
            try:
                if not isinstance(key, str):
                    raise errors.INVALID_KEY(str(key))
                check_s3_key(key)
                name = os.path.basename(key)
                _job, snap = enqueue(name, dict(params), s3_key=key)
                accepted.append({"id": snap["id"], "name": name, "s3_key": key})
            except errors.ProblemError as e:
                rejected.append({"s3_key": key, "error": e.body()})

    if not accepted and rejected:
        # 하나도 못 받았으면 202 를 줄 수 없다. 단건 제출이면 그 사유가 곧
        # 응답 코드가 되고(예: 415), 여러 건이면 항목별 사유를 함께 준다.
        codes = {r["error"].get("code") for r in rejected}
        problem = (errors.CATALOG.get(codes.pop()) if len(codes) == 1
                   else errors.INVALID_INPUT)
        raise (problem or errors.INVALID_INPUT)(
            rejected[0]["error"].get("detail", "") if len(rejected) == 1
            else f"{len(rejected)}건 전부 거절됐다",
            rejected=rejected)
    return {"accepted": accepted, "rejected": rejected, "queued": queue_depth()}


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
        raise errors.JOB_NOT_FOUND(jid)
    return snapshot(j, _queued_ahead(j))


@app.get("/api/jobs/{jid}/download")
def download(jid: str):
    j = find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    if j.status != "done":
        raise (errors.JOB_FAILED(j.error.get("detail", "") if isinstance(j.error, dict) else j.error)
           if j.status == "failed" else errors.JOB_NOT_FINISHED(f"status={j.status}"))
    if not j.output or not os.path.exists(j.output):
        # 로컬 사본은 정리됐어도 S3 원본은 남아 있다.
        store = s3mod.get_store()
        if j.s3_output and store is not None:
            return RedirectResponse(store.presigned_url(j.s3_output),
                                    status_code=302)
        raise errors.RESULT_EXPIRED(jid)
    name = naming.output_name(j.name)
    return FileResponse(j.output, media_type="video/mp4", filename=name)


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    """대기 중이면 즉시, 수행 중이면 다음 진행 보고에서 끊는다.

    잘못 넣은 배치를 끝날 때까지 기다릴 이유가 없다. 수행 중인 작업은 진행
    콜백에서만 안전하게 끊을 수 있어(프레임 경계) 표시만 남기고 워커가 처리한다.
    """
    j = find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    if j.status in ("done", "failed", "cancelled"):
        raise errors.JOB_NOT_CANCELLABLE(f"status={j.status}", status=j.status)
    with _LOCK:
        j.cancel = True
        if j.status == "queued":
            j.status, j.finished = "cancelled", time.time()
    save_job(j)
    return snapshot(j, _queued_ahead(j))


@app.get("/api/jobs/{jid}/result")
def job_result(jid: str):
    """결과물 받는 방법을 알려준다.

    S3 작업이면 **presigned URL** 을 준다 — GPU 서버가 파일 전송까지 떠안을
    이유가 없고, 로컬 사본이 보관 기간에 정리돼도 S3 원본은 남아 있다.
    """
    j = find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    if j.status != "done":
        raise (errors.JOB_FAILED(j.error.get("detail", ""))
               if j.status == "failed"
               else errors.JOB_NOT_FINISHED(f"status={j.status}"))
    out = {"id": j.id, "name": naming.output_name(j.name),
           "s3_key": j.s3_output or None}
    store = s3mod.get_store()
    if j.s3_output and store is not None:
        out["download_url"] = store.presigned_url(j.s3_output)
        out["expires_in"] = s3mod.URL_TTL
        out["via"] = "s3"
    else:
        out["download_url"] = f"/api/jobs/{j.id}/download"
        out["via"] = "server"
    return out


@app.delete("/api/jobs/{jid}", status_code=204)
def delete_job(jid: str):
    j = find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    if j.status in ("queued", "running"):
        raise errors.JOB_NOT_CANCELLABLE(
            "진행 중이다. /cancel 로 먼저 취소하라", status=j.status)
    with _LOCK:
        _JOBS.pop(jid, None)
    shutil.rmtree(j.workdir or os.path.join(JOBS_DIR, jid), ignore_errors=True)
