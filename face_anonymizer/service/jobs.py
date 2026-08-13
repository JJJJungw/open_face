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

from .. import progress, timefmt
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
    # 이 시각 전에는 꺼내지 않는다. 재시도 백오프와 보류가 같이 쓴다.
    not_before: float = 0.0
    waiting: str = ""             # "" | retry | defer — 왜 기다리는가
    # 폴더 하나를 통으로 넣으면 그 묶음의 이름이 여기 붙는다. 사람이 묻는 단위가
    # 파일이 아니라 폴더인 경우가 많다 — "kbs 언제 시작해서 언제 끝났어?"
    batch: str = ""
    deferred_since: float = 0.0   # 보류가 시작된 시각 (상한 판정용)
    stage_t0: float = 0.0
    # 전체 진행률(0~100). **되감기지 않게** 지금까지의 최고치를 들고 다닌다.
    # 단계가 통째로 없을 수 있어서(h264 원본이면 전사가 없다) 계산값만으로는
    # 뒤로 갈 수 있는데, 뒤로 가는 진행바는 사용자가 고장으로 읽는다.
    overall: float = 0.0


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

    # 전체 진행률. **모든 단계를 한 자 위에 올린다**(progress.STAGE_SPAN).
    #
    # 예전에는 검출·렌더에만 50%씩 주고 전사·전송은 자기 게이지를 따로 채웠다.
    # 게이지가 0 부터 다시 오르는 것도 문제지만 진짜 사고는 **남은 시간**에서
    # 났다. 남은 시간은 "지금까지 걸린 시간 × (100-진행률)/진행률" 로 되짚는데,
    # 걸린 시간은 작업 시작부터 재고 진행률은 검출·렌더만 세니 기준이 어긋난다.
    # 전사에 25초를 쓰고 검출이 2% 를 채운 순간 25×98/2 = 1225초 — 40초짜리
    # 영상에 **20분**이 떴다. 실제로 그렇게 떴다.
    #
    # j.overall 을 들고 다니며 되감기를 막는다. 단계가 통째로 없을 수 있고
    # (h264 원본이면 전사가 없다) 저장·복원 사이에 값이 뒤집힐 수도 있다.
    overall = progress.overall(j.stage, j.done, j.total, floor=j.overall or 0.0)
    if j.status == "done":
        overall = 100.0
    j.overall = overall

    # 이 작업 하나가 끝나기까지 남은 시간. 위 진행률이 전 단계를 덮으므로
    # 이제 분자(작업 시작부터)와 분모(진행률)의 기준이 맞는다.
    job_elapsed = time.time() - j.started if j.started else 0.0
    job_eta = (progress.eta(job_elapsed, overall) or 0.0) \
        if j.status == "running" else 0.0

    # 왜 안 도는지 화면이 말할 수 있어야 한다. not_before 가 미래인 queued 는
    # 그냥 '대기' 가 아니라 '재시도 대기' 나 '보류' 다.
    wait_left = max(0, round(j.not_before - time.time())) if j.not_before else 0

    return {
        "id": j.id, "name": j.name, "status": j.status, "stage": j.stage,
        "waiting": j.waiting if wait_left else "", "wait_left": wait_left,
        "percent": pct, "overall": overall, "fps": round(fps, 1),
        "eta": round(eta), "job_eta": round(job_eta),
        "job_elapsed": round(job_elapsed), "error": j.error, "result": j.result,
        "attempts": j.attempts, "max_attempts": config.MAX_ATTEMPTS,
        "s3_key": j.s3_key, "s3_output": j.s3_output,
        "queued_ahead": queued_ahead, "batch": j.batch,
        # 언제 시작해서 언제 끝났나. 화면이 계산하지 않고 서버가 정해 준 문장을
        # 그대로 쓴다 — 로그와 화면이 다른 시각을 말하면 대조가 안 된다.
        "started_at": timefmt.iso(j.started),
        "finished_at": timefmt.iso(j.finished),
        "span": timefmt.span(j.started, j.finished),
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

    **``queued`` 와 ``running`` 은 성격이 다르다.**

    - ``running`` — 중간에 끊겼다. 결과물이 온전한지 알 수 없으므로 실패로 둔다.
    - ``queued`` — **아직 아무 일도 일어나지 않았다.** 입력도 그대로고 부작용도
      없다. 다시 큐에 넣으면 그만이다.

    예전에는 둘을 한 덩어리로 보고 전부 실패로 표시했다. 그래서 500건을 넣어
    두고 코드를 배포하려고 서버를 내렸다 올리면 남은 대기 건이 통째로 날아갔다
    (docs/issues/002).

    재큐한 작업은 ``JOBS`` 에 넣어 돌려준다 — 실제로 워커에 제출하는 것은
    호출하는 쪽 몫이다. 이 모듈이 워커를 임포트하면 의존이 순환한다.

    ``FA_RECOVER=0`` 이면 아무것도 하지 않는다. ``--workers N`` 으로 여러 개를
    띄울 때는 **한 프로세스만 켜 두어야 한다** — 그렇지 않으면 각자 같은 작업을
    재큐해 중복 처리한다.

    Returns 다시 돌려야 할 Job 목록 (오래된 순).
    """
    if not config.RECOVER:
        return []
    failed, resume = 0, []
    for j in sorted(all_jobs(), key=lambda x: x.created):
        with LOCK:
            live = j.id in JOBS
        if live:
            continue
        if j.status == "running":
            j.status = "failed"
            j.error = {"code": "interrupted", "title": "서버 재시작으로 중단됐습니다",
                       "detail": "처리 중 프로세스가 종료됐습니다",
                       "hint": "다시 제출해 주세요.", "retryable": True}
            j.finished = time.time()
            save_job(j)
            failed += 1
        elif j.status == "queued":
            # 시도 횟수는 올리지 않는다. 재시작은 이 작업이 실패한 것이 아니다 —
            # 여기서 세면 배포를 몇 번 하는 것만으로 재시도가 소진된다.
            j.stage, j.done, j.total, j.overall = "", 0, 0, 0.0
            with LOCK:
                JOBS[j.id] = j
            save_job(j)
            resume.append(j)
    if failed:
        log.warning("중단된 작업 %d건을 실패로 표시했다", failed)
    if resume:
        log.info("대기 중이던 작업 %d건을 다시 큐에 넣는다", len(resume))
    return resume


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


STATUSES = ("queued", "running", "done", "failed", "cancelled")


def counts():
    """상태별 건수. **세는 일과 보여 주는 일을 분리한다.**

    목록은 화면에 담을 만큼만 잘라서 준다(``FA_LIST_LIMIT``). 큐가 그보다 길면
    잘린 창 안에 대기만 남고 수행중·완료는 창 밖으로 밀려난다. 그 창에서
    숫자까지 세면 화면이 "유휴" 라고 말한다(docs/issues/006). 집계를 여기서
    따로 하면 창을 어떻게 잡든 숫자는 맞는다.

    메모리(``JOBS``)만 본다. 폴링이 1초에 한 번 넘게 도는 자리라 디스크를
    훑으면 안 된다. TTL 이 지나 메모리에서 빠진 작업은 화면의 관심 밖이다.
    """
    with LOCK:
        rows = list(JOBS.values())
    out = {k: 0 for k in STATUSES}
    for j in rows:
        out[j.status] = out.get(j.status, 0) + 1
    out["total"] = len(rows)
    # 화면의 기본 목록은 '아직 처리할 일' 이다. 페이지 수를 여기서 뽑는다.
    out["active"] = out["running"] + out["queued"]
    return out


def memory_jobs():
    """메모리에 있는 작업들. 폴링 경로에서 디스크를 훑지 않으려고 쓴다."""
    with LOCK:
        return list(JOBS.values())


def next_up(limit=5):
    """다음에 처리될 대기 작업 이름들. 큐 화면이 '무엇이 들어와 있나' 로 쓴다."""
    with LOCK:
        rows = [j for j in JOBS.values() if j.status == "queued"]
    rows.sort(key=lambda x: x.created)
    return [j.name for j in rows[:max(0, limit)]]


def running_snapshot():
    """지금 도는 작업 한 건. 없으면 ``None``.

    진행 상황 표시는 목록 필터와 무관해야 한다 — '완료' 탭을 눌렀다고 돌고
    있는 작업이 사라지면 안 된다. 그래서 목록에서 찾지 않고 여기서 준다.
    """
    with LOCK:
        j = next((x for x in JOBS.values() if x.status == "running"), None)
    return snapshot(j) if j is not None else None      # 락 밖에서 — 재진입 금지


def recent_stats(limit=5):
    """최근 완료 ``limit`` 건의 평균 처리 시간과 실시간 대비. 없으면 ``{}``.

    화면이 목록 안의 완료 카드를 뒤져서 계산하던 값이다. 목록이 잘리면 같이
    사라져 '기록 없음' 이 됐다(docs/issues/006).
    """
    with LOCK:
        rows = [j for j in JOBS.values()
                if j.status == "done" and isinstance(j.result, dict)
                and j.result.get("seconds")]
    rows.sort(key=lambda x: -(x.finished or 0))
    rows = rows[:max(1, limit)]
    if not rows:
        return {}
    secs = [j.result["seconds"] for j in rows]
    rf = [j.result["realtime_factor"] for j in rows
          if j.result.get("realtime_factor")]
    out = {"avg_seconds": round(sum(secs) / len(secs), 1), "samples": len(secs)}
    if rf:
        out["realtime_factor"] = round(sum(rf) / len(rf), 2)
    return out


def cancel_all():
    """진행중인 작업(수행중 + 대기)을 한 번에 취소한다.

    **대기는 즉시, 수행중은 표시만 한다.** 대기는 아직 아무것도 시작하지 않아서
    상태만 바꾸면 끝이다. 수행중은 프레임 경계에서만 안전하게 끊을 수 있어
    플래그를 세우고 워커가 다음 진행 보고에서 스스로 빠져나온다(docs/issues/004
    에서 만든 협조적 취소 지점을 그대로 쓴다).

    한 건씩 라우트를 500번 부르지 않는 이유가 여기 있다. 그 사이에도 워커는
    큐를 계속 꺼내므로, 앞에서 취소하는 동안 뒤에서 새 작업이 시작된다.
    **락 한 번 안에서 전부 표시**해야 그 틈이 없다.

    반환은 ``(취소한 대기 수, 표시한 수행중 수)``.
    """
    queued = running = 0
    with LOCK:
        rows = [j for j in JOBS.values() if j.status in ("queued", "running")]
        now = time.time()
        for j in rows:
            j.cancel = True
            if j.status == "queued":
                j.status, j.finished = "cancelled", now
                queued += 1
            else:
                running += 1
    for j in rows:
        save_job(j)          # 디스크 쓰기는 락 밖에서 — 폴링을 붙잡지 않는다
    return queued, running


def batches(rows=None):
    """폴더(배치)별 시작·종료·진척. **사람이 묻는 단위로 묶어 준다.**

    파일 하나하나의 시각은 카드에 있지만, "kbs 폴더 언제 시작해서 언제 끝났나" 는
    거기서 읽어 낼 수 없다. 시작은 그 묶음에서 **제일 먼저 시작한 작업**, 종료는
    **제일 늦게 끝난 작업**이다. 아직 안 끝난 게 하나라도 있으면 종료는 비운다 —
    "끝났다" 를 절반만 맞게 말하면 안 된다.
    """
    rows = rows if rows is not None else all_jobs()
    out = {}
    for j in rows:
        if not j.batch:
            continue
        b = out.setdefault(j.batch, {
            "batch": j.batch, "total": 0, "done": 0, "failed": 0,
            "cancelled": 0, "running": 0, "queued": 0,
            "started": 0.0, "finished": 0.0, "seconds": 0.0})
        b["total"] += 1
        if j.status in b:
            b[j.status] += 1
        if j.started and (not b["started"] or j.started < b["started"]):
            b["started"] = j.started
        if j.finished and j.finished > b["finished"]:
            b["finished"] = j.finished
        if isinstance(j.result, dict):
            b["seconds"] += j.result.get("seconds") or 0.0

    for b in out.values():
        left = b["total"] - b["done"] - b["failed"] - b["cancelled"]
        b["remain"] = left
        b["percent"] = round(100 * (b["total"] - left) / b["total"]) if b["total"] else 0
        if left:                      # 하나라도 안 끝났으면 종료 시각은 없다
            b["finished"] = 0.0
        b["elapsed"] = (b["finished"] - b["started"]) if (b["finished"] and b["started"]) else 0.0
        b["started_at"] = timefmt.iso(b["started"])
        b["finished_at"] = timefmt.iso(b["finished"])
        b["span"] = timefmt.span(b["started"], b["finished"])
    return sorted(out.values(), key=lambda x: -(x["started"] or 0))


def queued_ahead_of(job):
    """앞에 몇 건 대기 중인가. **메모리만** 본다.

    폴링 경로라 여기서 디스크를 훑으면 안 된다. 대기 중인 작업은 이 프로세스의
    워커가 들고 있으므로 메모리에 있고, 없으면(재시작 후 남은 기록) 어차피
    대기 중이 아니다.
    """
    with LOCK:
        return sum(1 for o in JOBS.values()
                   if o.status == "queued" and o.created < job.created)
