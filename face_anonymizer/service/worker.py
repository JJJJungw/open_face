"""워커 — 큐에서 한 건씩 꺼내 실제로 처리한다.

**한 번에 한 편.** 추론은 워커 스레드 하나가 순차로 돌린다(GPU 한 장에 검출기
하나). 프로세스를 여러 개 띄워도 파일 락으로 GPU 를 직렬화한다.

실패하면 일시적 오류에 한해 ``FA_MAX_ATTEMPTS`` 회까지 다시 큐에 넣고, 그래도
안 되면 ``failed`` 로 남긴다. 같은 입력으로 같은 결과가 나올 오류(깨진 파일,
잘못된 인자)는 재시도하지 않는다.

이 모듈이 HTTP 를 모른다는 점이 중요하다. 나중에 AWS Batch 같은 배치 실행기로
바꿀 때 갈아끼우는 자리가 여기다 — 라우트는 손대지 않는다.
"""

import errno
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

try:
    import fcntl                      # POSIX 전용. 없으면 프로세스 간 직렬화 생략.
except ImportError:                   # pragma: no cover
    fcntl = None

from ..storage import naming
from ..storage import s3 as s3mod
from . import config, errors, jobs

log = logging.getLogger(__name__)

# 추론 직렬화. max_workers=1 이 이 서버의 동시성 정책 전부다.
EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anon")

_anonymizer = None
_anon_lock = threading.Lock()
model_error = None       # 기동 시 모델 로드 실패 사유
current = None           # 이 프로세스가 지금 붙잡고 있는 작업 id


def get_anonymizer():
    """검출기 싱글턴.

    기본적으로 기동 시(lifespan) 미리 올린다. 첫 요청 때 로드하면 헬스체크는
    이미 통과한 상태라, 오케스트레이터가 보낸 첫 요청이 모델 로딩 수십 초를
    기다리게 된다.
    """
    global _anonymizer
    with _anon_lock:
        if _anonymizer is None:
            from ..core.paths import DEFAULT_WEIGHTS
            from ..core.pipeline import VideoAnonymizer
            from ..storage import weights as weights_store

            # 새 EC2 나 컨테이너에서는 여기서 처음 받는다. 이미 있으면 네트워크
            # 호출조차 없다. 검출기(core)가 S3 를 모르게 하려고 만드는 쪽에서
            # 갖춰 놓고 넘긴다.
            weights_store.ensure(DEFAULT_WEIGHTS)
            log.info("검출기 로드 중 (device=%s imgsz=%d)", config.DEVICE, config.IMGSZ)
            _anonymizer = VideoAnonymizer(device=config.DEVICE, imgsz=config.IMGSZ)
            log.info("검출기 준비 완료")
        return _anonymizer


def is_ready():
    """추론을 받을 수 있는 상태인가."""
    return _anonymizer is not None and model_error is None


def is_busy():
    """지금 추론을 돌리고 있는가."""
    with jobs.LOCK:
        return any(j.status == "running" for j in jobs.JOBS.values())




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


def fail_or_retry(j, exc, permanent):
    """실패 처리. 일시적 오류면 다시 큐에 넣는다.

    같은 입력으로 같은 결과가 나올 오류(깨진 파일, 잘못된 인자)는 재시도하지
    않는다 — 세 번 돌려도 결과가 같고 그동안 뒤에 쌓인 정상 작업이 밀린다.
    """
    info = errors.job_error(exc)
    retryable = not permanent and info["retryable"] and j.attempts < config.MAX_ATTEMPTS
    msg = info["detail"]
    # 어디서 넘어졌는지, 그리고 왜 다시 시도했는지/안 했는지를 오류에 같이
    # 남긴다. 사유만 있고 이 둘이 없으면 "3회 시도" 라는 숫자를 어떻게 읽어야
    # 할지 알 수 없다.
    info["stage"] = j.stage or ("download" if j.s3_key and not j.done else "")
    info["policy"] = ("permanent" if permanent
                      else "exhausted" if not retryable else "retrying")
    with jobs.LOCK:
        j.error = info
        if retryable:
            j.status, j.done, j.total, j.stage = "queued", 0, 0, ""
        else:
            j.status, j.finished = "failed", time.time()
    jobs.save_job(j)
    if retryable:
        log.warning("작업 %s 실패 [%s] (%d/%d회) — 다시 시도한다: %s",
                    j.id, info["code"], j.attempts, config.MAX_ATTEMPTS, msg)
        EXEC.submit(run, j.id)
    else:
        log.error("작업 %s 실패 [%s] (%d회 시도, %s): %s", j.id, info["code"],
                  j.attempts, "재시도 불가" if permanent else "재시도 소진", msg)
        # 원인 파악에 필요한 것은 사유·단계·시도 횟수이고 전부 job.json 에
        # 있다. S3 작업이면 원본도 버킷에 그대로다 — 200MB 를 붙들고 있을
        # 이유가 없다. 직접 업로드는 원본이 여기밖에 없어 남긴다.
        if j.s3_key:
            jobs.drop_media(j, "실패 — 기록만 남긴다")


def run(job_id):
    with jobs.LOCK:
        j = jobs.JOBS.get(job_id)
        if j is None:
            return
        if j.cancel or j.status == "cancelled":
            # 대기 중에 취소된 건 아예 시작하지 않는다.
            j.status, j.finished = "cancelled", j.finished or time.time()
            jobs.save_job(j)
            return
        j.status, j.stage_t0 = "running", time.time()
        j.started = time.time()
        j.attempts += 1
        params, workdir, name = dict(j.params), j.workdir, j.name
    jobs.save_job(j)

    # 제출 시점에만 보면 500건짜리 배치 도중에 디스크가 차는 것을 못 잡는다.
    # 부족하면 정리를 한 번 강제로 돌려 회수부터 시도한다 — TTL 지난 것이
    # 남아 있으면 그때 비워진다.
    if config.MIN_FREE_MB:
        free = jobs.free_mb()
        if free is not None and free < config.MIN_FREE_MB:
            log.warning("여유 %sMB — 정리를 먼저 돌린다", free)
            jobs.sweep()
            free = jobs.free_mb()
        if free is not None and free < config.MIN_FREE_MB:
            fail_or_retry(j, errors.INSUFFICIENT_STORAGE(
                f"남은 공간 {free}MB, 최소 {config.MIN_FREE_MB}MB 가 필요합니다"),
                permanent=False)
            return

    def progress(stage, done, total):
        # 취소는 여기서만 끊을 수 있다. 파이프라인이 프레임마다 부르는 유일한
        # 지점이라, 예외를 던지면 다음 프레임으로 넘어가지 않고 빠져나온다.
        if j.cancel:
            raise JobCancelled()
        with jobs.LOCK:
            if j.stage != stage:
                j.stage, j.stage_t0 = stage, time.time()
            j.done, j.total = done, total
        jobs.save_job(j, force=False)      # 폴링용 — 간격을 두고 흘려 쓴다

    src = os.path.join(workdir, "input" + os.path.splitext(name)[1])
    dst = os.path.join(workdir, naming.output_name(name))
    try:
        if j.s3_key and not os.path.exists(src):
            store = s3mod.get_store()
            if store is None:
                raise s3mod.S3Error("S3 가 설정되어 있지 않습니다")
            log.info("S3 에서 내려받는다: %s", j.s3_key)
            store.download(j.s3_key, src)
        # 프로세스가 여러 개여도 GPU 는 한 번에 하나만 쓴다.
        with gpu_lock(os.path.join(config.JOBS_DIR, config.GPU_LOCK_FILE)):
            res = get_anonymizer().process(src, dst, progress=progress, **params)
        if j.s3_key:
            store = s3mod.get_store()
            key = store.output_key(j.s3_key)
            log.info("S3 에 올린다: %s", key)
            store.upload(res.output, key)
            with jobs.LOCK:
                j.s3_output = key
        with jobs.LOCK:
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
        jobs.save_job(j)
        # 결과는 이미 버킷에 있다. 로컬 사본을 TTL 동안 들고 있으면 대량
        # 처리에서 디스크가 먼저 찬다(docs/issues/001). 다운로드 라우트가
        # "로컬에 없으면 S3 로 302" 이므로 잃는 것이 없다. 직접 업로드는
        # 로컬이 유일한 사본이라 건드리지 않는다.
        if j.s3_key and j.s3_output and not config.KEEP_LOCAL:
            jobs.drop_media(j, "S3 업로드 완료")
    except JobCancelled:
        with jobs.LOCK:
            j.status, j.finished = "cancelled", time.time()
            j.error = {"code": errors.CANCELLED.code,
                       "title": errors.CANCELLED.title, "detail": "",
                       "hint": "", "retryable": False}
        jobs.save_job(j)
        log.info("작업 %s 취소됨", job_id)
    except config.PERMANENT_ERRORS as e:
        fail_or_retry(j, e, permanent=True)
    except Exception as e:                      # noqa: BLE001 — 워커가 조용히 죽으면 안 된다
        log.exception("작업 %s 실패", job_id)
        fail_or_retry(j, e, permanent=False)


def resume_orphans():
    """재시작 뒤, 대기 중이던 작업을 다시 워커에 올린다.

    상태를 정리하는 것은 jobs 가, 실제로 제출하는 것은 여기가 한다. 순서는
    만들어진 순 그대로다 — 재시작했다고 순번이 뒤바뀌면 진행 상황이 안 읽힌다.
    """
    resumed = jobs.recover_orphans()
    for j in resumed:
        EXEC.submit(run, j.id)
    return len(resumed)


# ── 큐에 넣기 ────────────────────────────────────────────────────────────────

def new_job_id():
    return uuid.uuid4().hex[:12]


def enqueue(name, params, s3_key="", jid=None, workdir=None):
    """작업을 등록하고 워커에 넘긴다. 대기열 상한도 여기서 본다.

    ``jid`` 와 ``workdir`` 은 함께 온다 — 업로드 경로는 파일을 받아야 해서
    디렉터리를 먼저 만든다. 둘이 어긋나면 상태 파일이 엉뚱한 곳에 쓰인다.
    """
    global current
    jid = jid or new_job_id()
    workdir = workdir or os.path.join(config.JOBS_DIR, jid)
    os.makedirs(workdir, exist_ok=True)
    job = jobs.Job(id=jid, name=name, workdir=workdir, s3_key=s3_key, params=params)
    with jobs.LOCK:
        # 대기열 확인과 등록이 같은 락 안에 있어야 동시 요청이 상한을 넘지 않는다.
        if config.QUEUE_MAX and sum(1 for o in jobs.JOBS.values()
                             if o.status == "queued") >= config.QUEUE_MAX:
            shutil.rmtree(workdir, ignore_errors=True)
            raise errors.QUEUE_FULL(f"대기 중인 작업이 {config.QUEUE_MAX}건입니다", retry_after=config.RETRY_AFTER)
        jobs.JOBS[jid] = job
        current = jid
    jobs.save_job(job)
    snap = jobs.snapshot(job, queued_ahead=jobs.queued_ahead_of(job))
    EXEC.submit(run, jid)
    return job, snap
