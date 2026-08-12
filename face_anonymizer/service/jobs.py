"""작업 상태 — 만들고, 디스크에 남기고, 다시 찾고, 정리한다.

**작업 상태는 디스크에 있다.** 전역 dict 에만 두면 (a) 재시작 시 전부 사라져
폴링 중인 클라이언트가 404 를 받고, (b) ``--workers 2`` 로 띄우는 순간 업로드는
A 프로세스, 폴링은 B 프로세스로 가서 계속 404 가 난다. 작업별 디렉터리에
``job.json`` 을 두면 둘 다 해결된다.

메모리의 ``JOBS`` 는 그 캐시다. 이 프로세스가 만든 작업은 여기 있고, 다른
프로세스가 만든 것은 디스크에서 읽는다.

**작업 파일은 작업별 디렉터리에.** 원본과 결과가 섞이지 않고, 삭제가 디렉터리
하나 지우는 것으로 끝난다.
"""

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field, fields

from . import config

log = logging.getLogger(__name__)

# 이 프로세스가 만든 작업들. 진짜 원장은 디스크이고 이건 캐시다.
JOBS = {}
LOCK = threading.Lock()


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


def state_path(jid):
    return os.path.join(config.JOBS_DIR, jid, config.STATE_FILE)


def save_job(j, force=True):
    """작업 상태를 디스크에 쓴다.

    진행률은 초당 수십 번 갱신되므로 ``force=False`` 로 호출해 간격을 둔다.
    쓰기는 임시 파일 + rename 으로 원자적으로 한다 — 다른 프로세스가 읽는
    중에 반쪽짜리 JSON 을 보면 안 된다.
    """
    now = time.time()
    if not force and now - getattr(j, "_flushed", 0.0) < config.PROGRESS_FLUSH_SEC:
        return
    path = state_path(j.id)
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
        with open(state_path(jid), encoding="utf-8") as f:
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
    with LOCK:
        j = JOBS.get(jid)
    return j if j is not None else load_job_file(jid)


def all_jobs():
    """메모리 + 디스크 병합 목록. 메모리 쪽이 우선(진행 중인 값이 최신)."""
    with LOCK:
        merged = dict(JOBS)
    try:
        entries = os.listdir(config.JOBS_DIR)
    except OSError:
        entries = []
    for jid in entries:
        if jid in merged or not os.path.isdir(os.path.join(config.JOBS_DIR, jid)):
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
        "attempts": j.attempts, "max_attempts": config.MAX_ATTEMPTS,
        "s3_key": j.s3_key, "s3_output": j.s3_output,
        "queued_ahead": queued_ahead,
    }


def drop_media(j, why=""):
    """작업 폴더에서 영상 파일만 지우고 ``job.json`` 은 남긴다.

    원인을 보는 데 필요한 것은 사유·단계·시도 횟수이고 그건 전부 job.json 에
    있다. 200MB 짜리 영상을 몇 KB 짜리 기록 때문에 붙들고 있을 이유가 없다.
    S3 작업이면 원본도 결과물도 버킷에 있으므로 잃는 것이 없다.
    """
    workdir = j.workdir or os.path.join(config.JOBS_DIR, j.id)
    freed = 0
    for name in os.listdir(workdir) if os.path.isdir(workdir) else []:
        if name == config.STATE_FILE:
            continue
        path = os.path.join(workdir, name)
        try:
            if os.path.isdir(path):
                freed += _dir_size(path)
                shutil.rmtree(path, ignore_errors=True)
            else:
                freed += os.path.getsize(path)
                os.remove(path)
        except OSError as e:                        # 지우기 실패로 작업을 망치지 않는다
            log.warning("정리 실패 %s: %s", path, e)
    if freed:
        log.info("작업 %s 로컬 파일 정리 %.1f MB%s", j.id, freed / 1e6,
                 f" ({why})" if why else "")
    return freed


def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def sweep_temp():
    """죽은 프로세스가 남긴 임시 디렉터리를 치운다.

    파이프라인은 작업 폴더 안에 ``.anon-*`` 를 만들고 finally 에서 지운다.
    프로세스가 강제로 죽으면 그 finally 가 안 돈다 — 서버를 재시작할 때마다
    하나씩 쌓인다.
    """
    removed = 0
    for jid in os.listdir(config.JOBS_DIR) if os.path.isdir(config.JOBS_DIR) else []:
        d = os.path.join(config.JOBS_DIR, jid)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.startswith(".anon-"):
                continue
            with LOCK:
                live = jid in JOBS and JOBS[jid].status == "running"
            if live:                                # 지금 쓰고 있는 것은 건드리지 않는다
                continue
            shutil.rmtree(os.path.join(d, name), ignore_errors=True)
            removed += 1
    if removed:
        log.info("남은 임시 디렉터리 %d개 정리", removed)
    return removed


def sweep():
    """TTL 지난 작업 정리.

    예전에는 새 작업이 들어올 때만 돌아서, 업로드가 끊기면 디스크가 영원히
    안 비워졌다. 지금은 백그라운드 스레드가 주기적으로 돈다.
    """
    sweep_temp()
    if not config.JOB_TTL:
        return 0
    now, removed = time.time(), 0
    for j in all_jobs():
        # 실패·취소는 원인을 보려면 남아 있어야 한다.
        ttl = config.FAILED_TTL if j.status in ("failed", "cancelled") else config.JOB_TTL
        if not ttl:
            continue
        if j.finished and now - j.finished > ttl:
            shutil.rmtree(j.workdir or os.path.join(config.JOBS_DIR, j.id),
                          ignore_errors=True)
            with LOCK:
                JOBS.pop(j.id, None)
            removed += 1
    if removed:
        log.info("TTL 정리: %d건", removed)
    return removed


def sweep_loop():
    while True:
        time.sleep(config.SWEEP_SEC)
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
        with LOCK:
            live = j.id in JOBS
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


def free_mb():
    """작업 디렉터리가 놓일 볼륨의 여유 공간(MB).

    첫 작업 전에는 config.JOBS_DIR 이 아직 없을 수 있다. 그때 None 을 돌려주면 디스크
    검사가 조용히 건너뛰어지므로, 존재하는 상위 경로까지 올라가서 잰다.
    """
    path = os.path.abspath(config.JOBS_DIR)
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
    with LOCK:
        return sum(1 for j in JOBS.values() if j.status == "queued")



def queued_ahead_of(job):
    """앞에 몇 건 대기 중인가. **메모리만** 본다.

    폴링 경로라 여기서 디스크를 훑으면 안 된다. 대기 중인 작업은 이 프로세스의
    워커가 들고 있으므로 메모리에 있고, 없으면(재시작 후 남은 기록) 어차피
    대기 중이 아니다.
    """
    with LOCK:
        return sum(1 for o in JOBS.values()
                   if o.status == "queued" and o.created < job.created)
