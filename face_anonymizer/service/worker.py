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
import random
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
from .. import events, job_runner, timefmt
from . import config, errors, jobs

log = logging.getLogger(__name__)

# 추론 직렬화. max_workers=1 이 이 서버의 동시성 정책 전부다.
EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anon")

_anonymizer = None
_anon_lock = threading.Lock()
model_error = None       # 모델 로드 실패 사유
loading = False          # 지금 실제로 올리고 있는 중인가 (화면 문구용)
current = None           # 이 프로세스가 지금 붙잡고 있는 작업 id


def get_anonymizer():
    """검출기 싱글턴.

    기본적으로 기동 시(lifespan) 미리 올린다. 첫 요청 때 로드하면 헬스체크는
    이미 통과한 상태라, 오케스트레이터가 보낸 첫 요청이 모델 로딩 수십 초를
    기다리게 된다.
    """
    global _anonymizer, loading
    with _anon_lock:
        if _anonymizer is None:
            from ..core.paths import DEFAULT_WEIGHTS
            from ..core.pipeline import VideoAnonymizer
            from ..storage import weights as weights_store

            # 새 EC2 나 컨테이너에서는 여기서 처음 받는다. 이미 있으면 네트워크
            # 호출조차 없다. 검출기(core)가 S3 를 모르게 하려고 만드는 쪽에서
            # 갖춰 놓고 넘긴다.
            loading = True
            try:
                weights_store.ensure(DEFAULT_WEIGHTS)
                log.info("검출기 로드 중 (device=%s imgsz=%d)",
                         config.DEVICE, config.IMGSZ)
                _anonymizer = VideoAnonymizer(device=config.DEVICE,
                                              imgsz=config.IMGSZ)
                log.info("검출기 준비 완료")
            finally:
                loading = False
        return _anonymizer


def is_ready():
    """작업을 **받을 수 있는** 상태인가.

    모델이 올라와 있는가와 **다른 질문이다.** ``FA_PRELOAD=0`` 은 "첫 요청 때
    올린다" 는 뜻인데, 예전에는 이 함수가 모델 객체의 유무만 봤다. 그래서
    받는 문(``check_admission``)이 닫히고 → 작업이 안 들어오고 → 모델을 올릴
    계기가 영영 없고 → 화면은 "모델 올리는 중" 을 영원히 띄웠다. **실제로는
    아무것도 올리고 있지 않았다.** 지연 로딩이 아니라 그냥 죽은 서버였다.

    지금은 이렇게 답한다. 로드가 실패했으면 못 받는다. 이미 올라와 있으면
    받는다. 아직 안 올라왔어도 **지연 로딩 설정이면 받는다** — 실제 로드는
    워커 스레드가 첫 작업에서 한다.
    """
    if model_error is not None:
        return False
    return _anonymizer is not None or not config.PRELOAD


def model_status():
    """모델에 관한 사실들. **화면이 문구를 만들 재료다.**

    'ready' 하나로는 표현이 안 된다 — 지연 로딩이면 받을 수는 있는데 아직
    안 올라와 있다. 그 상태를 "올리는 중" 이라고 부르면 거짓말이 된다.
    """
    return {"loaded": _anonymizer is not None, "loading": loading,
            "error": model_error, "preload": bool(config.PRELOAD)}


def is_busy():
    """지금 추론을 돌리고 있는가."""
    with jobs.LOCK:
        return any(j.status == "running" for j in jobs.JOBS.values())




class gpu_lock:
    """프로세스 간 추론 직렬화.

    스레드 풀(max_workers=1)은 한 프로세스 안에서만 유효하다. 같은 작업 폴더를
    두 프로세스가 쓰는 것은 기동 때 거절하지만(``jobs.claim_jobs_dir``),
    **작업 폴더를 따로 준 두 서버가 같은 GPU 를 나눠 쓰는 것**은 여전히 가능하고
    막을 이유도 없다. 그때 둘이 동시에 추론하면 VRAM 이 터진다. 작업 디렉터리에
    잠금 파일을 두고 그 위에서 직렬화한다.

    **그래서 이 잠금은 작업 폴더가 다르면 서로를 못 본다.** 한 대에 두 서버를
    띄울 거면 ``FA_JOBS_DIR`` 은 나누되 이 파일은 공유해야 하는데, 지금은 그
    경로가 작업 폴더에 묶여 있다. 한 대 한 서버가 전제다(docs/issues/017).

    fcntl 이 없는 플랫폼에서는 아무것도 하지 않는다.
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


def schedule(jid, delay):
    """``delay`` 초 뒤에 워커에 올린다.

    **자는 게 아니라 예약한다.** 워커가 하나뿐이라 ``sleep`` 을 넣으면 뒤에
    쌓인 정상 작업까지 그동안 멈춘다. 재시도 대기가 큐 전체를 막는 것은 원래
    문제보다 나쁘다.

    타이머는 프로세스가 죽으면 사라진다. 그래도 유실되지 않는 이유는 그 작업이
    ``queued`` 로 남아 있어 **재시작 때 다시 큐에 들어가기** 때문이다
    (docs/issues/002).
    """
    if delay <= 0:
        EXEC.submit(run, jid)
        return
    t = threading.Timer(delay, lambda: EXEC.submit(run, jid))
    t.daemon = True                 # 대기 중인 재시도가 종료를 막지 않는다
    t.start()


def backoff_for(attempt):
    """``attempt`` 번 실패한 뒤 얼마나 기다릴지. 기본 5초 -> 30초 -> 60초.

    목록이 시도 횟수보다 짧으면 마지막 값을 계속 쓴다.

    지터를 섞는 이유는 여러 건이 같은 순간에 실패했을 때 회복하는 순간 한꺼번에
    몰리지 않게 하기 위해서다. 지금은 워커가 하나라 어차피 직렬화되어 효과가
    크지 않지만, 워커가 늘거나 배치 실행기로 갈 때 필요해진다.
    """
    delays = config.RETRY_DELAYS
    base = delays[min(max(0, attempt - 1), len(delays) - 1)]
    jitter = base * config.RETRY_JITTER
    return max(0.0, base + random.uniform(-jitter, jitter))


def defer(j, problem, why):
    """실패가 아니라 **아직 시작할 조건이 안 됐다.**

    시도 횟수를 쓰지 않는다 — 디스크를 세 번 확인했다고 포기할 일이 아니다.
    대신 상한을 둔다. 영구히 찬 디스크를 영원히 숨기면 안 된다.
    """
    now = time.time()
    with jobs.LOCK:
        if not j.deferred_since:
            j.deferred_since = now
        waited = now - j.deferred_since
        if waited > config.DEFER_MAX_SEC:
            j.status, j.finished = "failed", now
            j.error = problem(f"{why} — {round(waited / 60)}분째 기다렸습니다").body()
            j.error["policy"] = "deferred_too_long"
            j.waiting, j.not_before = "", 0.0
        else:
            j.status, j.waiting = "queued", "defer"
            j.not_before = now + config.DEFER_SEC
    jobs.save_job(j)
    if j.status == "failed":
        log.error("작업 %s 보류 상한 초과 (%.0f분): %s", j.id, waited / 60, why)
        return
    log.warning("작업 %s 보류 — %s (%.0f초 뒤 다시 확인)", j.id, why, config.DEFER_SEC)
    schedule(j.id, config.DEFER_SEC)


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
            # 진행률도 처음으로. 안 되돌리면 다시 시작한 작업이 화면에서
            # 60% 부터 출발한다 — 되감기를 막는 바닥값이 그대로 남아서다.
            j.status, j.done, j.total, j.stage = "queued", 0, 0, ""
            j.overall = 0.0
        else:
            j.status, j.finished = "failed", time.time()
    jobs.save_job(j)
    if retryable:
        delay = backoff_for(j.attempts)
        with jobs.LOCK:
            j.not_before, j.waiting = time.time() + delay, "retry"
        jobs.save_job(j)
        log.warning("작업 %s 실패 [%s] (%d/%d회) — %.0f초 뒤 다시 시도한다: %s",
                    j.id, info["code"], j.attempts, config.MAX_ATTEMPTS, delay, msg)
        events.emit("job.retry", job=j.id, name=j.name, batch=j.batch or None,
                    code=info["code"], attempts=j.attempts, delay_s=round(delay, 1),
                    detail=msg)
        schedule(j.id, delay)
    else:
        log.error("작업 %s 실패 [%s] (%d회 시도, %s): %s", j.id, info["code"],
                  j.attempts, "재시도 불가" if permanent else "재시도 소진", msg)
        # **transient 를 반드시 싣는다.** 저널 문장 생성기는 이 필드만 보고
        # "일시적/영구" 를 찍는데(events.decorate), 여기서 빠뜨리면 재시도를
        # 세 번 다 쓰고 죽은 일시적 오류까지 전부 "영구" 로 기록된다. MSA 경로는
        # 싣고 있었다 — 같은 이벤트가 경로마다 다른 필드를 갖고 있던 것이다.
        events.emit("job.failed", job=j.id, name=j.name, batch=j.batch or None,
                    code=info["code"], stage=info.get("stage") or None,
                    policy=info.get("policy"), attempts=j.attempts, detail=msg,
                    transient=bool(info.get("retryable")))
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

    # 아직 예약 시각이 안 됐으면(재시도 대기·보류) 여기서 물러난다. 타이머가
    # 겹쳐 도는 경우를 막는다.
    if j.not_before and time.time() < j.not_before:
        return

    # 시작할 조건을 **시도 횟수를 쓰기 전에** 본다. 디스크 부족은 이 작업이
    # 실패한 것이 아니므로 재시도를 소모하면 안 된다(docs/issues/003).
    #
    # 제출 시점에만 보면 500건짜리 배치 도중에 차는 것을 못 잡는다. 부족하면
    # 정리를 한 번 강제로 돌려 회수부터 시도한다.
    if config.MIN_FREE_MB:
        free = jobs.free_mb()
        if free is not None and free < config.MIN_FREE_MB:
            log.warning("여유 %sMB — 정리를 먼저 돌린다", free)
            jobs.sweep()
            free = jobs.free_mb()
        if free is not None and free < config.MIN_FREE_MB:
            defer(j, errors.INSUFFICIENT_STORAGE,
                  f"남은 공간 {free}MB, 최소 {config.MIN_FREE_MB}MB 가 필요합니다")
            return

    with jobs.LOCK:
        j.status, j.stage_t0 = "running", time.time()
        j.started = time.time()
        j.attempts += 1
        j.not_before, j.waiting, j.deferred_since = 0.0, "", 0.0
        params, workdir, name = dict(j.params), j.workdir, j.name
        log.info("▶ 시작  %s%s  [%s]", j.name,
                 f"  ({j.batch})" if j.batch else "", timefmt.stamp(j.started))
    jobs.save_job(j)
    events.emit("job.started", job=j.id, name=j.name, batch=j.batch or None,
                attempt=j.attempts, s3_key=j.s3_key or None)

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

    def transfer(stage, total):
        """전송 진행률을 보고하고, 그 김에 취소도 확인한다.

        boto3 가 청크마다 부른다. 이 콜백이 없으면 큰 파일을 받는 동안
        취소가 안 먹고(docs/issues/004) 진행률도 멈춰 보인다.
        """
        seen = [0]

        def cb(chunk):
            # 취소를 **먼저** 본다. 여기서 JobCancelled 를 던지면 전송 계층이
            # 그걸 S3 오류로 감싸서, 사용자가 취소한 것이 "S3 호출 실패" 로
            # 보고된다. 저장소가 아는 신호로 던지고 밖에서 되돌린다.
            if j.cancel:
                raise s3mod.TransferAborted()
            seen[0] += chunk
            progress(stage, min(seen[0], total) if total else seen[0], total)
        return cb

    # **로컬 파일 이름은 우리 사정이다.** 예전에는 결과물만 원본 이름을 그대로
    # 썼는데(입력은 이미 input.<ext> 였다), 그러면 남이 지은 이름의 길이·특수
    # 문자·정규화가 전부 우리 문제가 된다. 실제로 맥에서 올라온 한글 이름이
    # 자모로 분리돼 저장돼(NFD) 298바이트가 됐고, ext4 한계 255를 넘겨 터졌다.
    #
    # 예쁜 이름이 필요한 자리는 **버킷 키와 내려받기 파일명**뿐이고 둘 다
    # `j.name` 에서 따로 만든다. 여기서까지 쓸 이유가 없다.
    src = os.path.join(workdir, "input" + os.path.splitext(name)[1])
    dst = os.path.join(workdir, "output" + naming.DEFAULT_EXT)
    # **작업 하나는 처음부터 끝까지 같은 저장소를 본다.**
    #
    # 예전에는 내려받을 때와 올릴 때 각각 get_store() 를 불렀다. 그 사이에
    # 누가 화면에서 저장소를 바꾸면(연결 해제·다른 버킷) 이 영상은 A 에서
    # 받아서 B 에 올라간다 — 조용히, 오류 하나 없이. 비식별화한 결과물이
    # 엉뚱한 버킷에 떨어지는 것은 이 서비스에서 제일 나쁜 결말이다.
    # 끊긴 뒤라면 get_store() 가 None 이라 올리는 자리에서 AttributeError 가
    # 나고, 그건 '알 수 없는 오류' 로 분류돼 GPU 파이프라인을 네 번 더 돌린다.
    store = s3mod.get_store() if j.s3_key else None
    if j.s3_key and store is None:
        raise s3mod.S3Error("S3 가 설정되어 있지 않습니다")
    try:
        if j.s3_key and not os.path.exists(src):
            log.info("S3 에서 내려받는다: %s", j.s3_key)
            store.download(j.s3_key, src,
                           callback=transfer("download", store.size_of(j.s3_key)))
            # 콜백에서 던진 예외를 전송 계층이 삼키는 경우를 대비한 이중 확인.
            if j.cancel:
                raise JobCancelled()
        # 프로세스가 여러 개여도 GPU 는 한 번에 하나만 쓴다.
        with gpu_lock(os.path.join(config.JOBS_DIR, config.GPU_LOCK_FILE)):
            # **메모리가 부족하면 배치를 낮춰 다시 해 본다.** 예전에는 이
            # 회복이 MSA 경로에만 있어서, 같은 OOM 이 저쪽에서는 살아나고
            # 우리 서버에서는 그냥 실패했다. 두 경로가 같은 GPU 를 쓴다.
            def lowered(b):
                log.warning("GPU 메모리 부족 — 배치를 %d 로 낮춘다  %s", b, j.name)
            res = job_runner.process_with_oom_retry(
                get_anonymizer(), src, dst, params,
                progress=progress, note=lowered)
        if j.s3_key:
            key = store.output_key(j.s3_key)
            log.info("S3 에 올린다: %s", key)
            store.upload(res.output, key,
                         callback=transfer("upload", os.path.getsize(res.output)))
            # **올린 다음에는 먼저 기록한다.** 취소 확인이 앞에 있으면, 업로드가
            # 끝난 직후에 취소가 들어왔을 때 결과물은 버킷에 있는데 그걸
            # 가리키는 기록이 없다. 그 파일은 아무도 못 찾고, 다시 제출하면
            # '이미 처리됨' 으로 거절된다 — 취소했다면서.
            with jobs.LOCK:
                j.s3_output = key
            if j.cancel:
                raise JobCancelled()
        # 사람이 봐야 하는 사유가 있으면 **완료로 넘기지 않는다.**
        # 딱지만 붙이고 done 으로 두면 결국 완료 목록에 섞여 그대로 납품된다.
        # 얼굴이 하나도 안 잡힌 영상은 원본이 그대로 나간 것이므로, 그게 정당한
        # 0 인지 설정이 틀려서 0 인지 사람이 한 번 봐야 한다(docs/issues/008).
        needs = job_runner.review_of(res.warnings)
        with jobs.LOCK:
            j.review = needs
            j.status = "review" if needs else "done"
            j.output, j.finished = res.output, time.time()
            log.info("%s  %s%s  %s", "⚑ 검수 필요" if needs else "■ 완료", j.name,
                     f"  ({j.batch})" if j.batch else "",
                     timefmt.span(j.started, j.finished))
            if needs:
                for item in needs:
                    log.warning("검수 사유  %s — %s", j.name, item["message"])
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
        # **결과가 채워진 뒤에** 찍는다. 락 안에서 찍으면 프레임 수도 검출률도
        # 아직 비어 있어서, 정작 근거로 쓸 값이 하나도 안 남는다.
        r = j.result
        # **저널의 `seconds` 는 벽시계다.** 예전에는 여기만 파이프라인 시간을
        # 넣어서, 같은 칸에 API 경로는 처리 시간을, MSA 경로는 내려받기·올리기까지
        # 포함한 전체 시간을 찍고 있었다. 같은 이름으로 다른 것을 재면 나중에
        # 둘을 나란히 놓고 비교하는 순간 틀린 결론이 나온다. 파이프라인 시간은
        # `pipeline_s` 로 따로 남긴다.
        wall = round((j.finished or time.time()) - j.started, 3) if j.started else None
        events.emit("job.finished", job=j.id, name=j.name,
                    batch=j.batch or None, seconds=wall,
                    pipeline_s=r.get("seconds"),
                    frames=r.get("frames"),
                    detected_frames=r.get("detected_frames"),
                    detection_rate=r.get("detection_rate"),
                    realtime_factor=r.get("realtime_factor"),
                    warnings=list(r.get("warnings") or ()),
                    # **검수 여부를 여기 안 넣으면 CSV 의 '검수 필요' 칸이
                    # 영영 빈다.** 검수로 넘어간 것도 완료 줄은 똑같이 생겨서,
                    # 나중에 기록만 보고는 구분할 방법이 없어진다.
                    review_needed=bool(j.review),
                    review=[i["code"] for i in j.review] or None,
                    source_codec=r.get("source_codec"),
                    transcoded=r.get("transcoded"),
                    timing=r.get("timing"), attempts=j.attempts,
                    started_at=timefmt.iso(j.started),
                    finished_at=timefmt.iso(j.finished),
                    span=timefmt.span(j.started, j.finished))
        if j.review:
            events.emit("job.review", job=j.id, name=j.name, batch=j.batch or None,
                        codes=[i["code"] for i in j.review],
                        detail=" / ".join(i["detail"] for i in j.review))
        # 결과는 이미 버킷에 있다. 로컬 사본을 TTL 동안 들고 있으면 대량
        # 처리에서 디스크가 먼저 찬다(docs/issues/001). 다운로드 라우트가
        # "로컬에 없으면 S3 로 302" 이므로 잃는 것이 없다. 직접 업로드는
        # 로컬이 유일한 사본이라 건드리지 않는다.
        # 검수 대기라고 로컬을 붙들지 않는다. 다운로드 라우트가 "로컬에 없으면
        # S3 로 302" 라서 검수하는 사람은 그대로 볼 수 있다. 붙들고 있으면
        # 검수가 밀린 만큼 디스크가 차고, 결국 새 작업 제출이 거부된다
        # (docs/issues/001 이 이 형태로 되살아난다).
        #
        # 직접 업로드분은 원래대로 남긴다 — 로컬이 유일한 사본이다.
        if j.s3_key and j.s3_output and not config.KEEP_LOCAL:
            jobs.drop_media(j, "S3 업로드 완료")
    except (JobCancelled, s3mod.TransferAborted):
        with jobs.LOCK:
            j.status, j.finished = "cancelled", time.time()
            j.error = {"code": errors.CANCELLED.code,
                       "title": errors.CANCELLED.title, "detail": "",
                       "hint": "", "retryable": False}
        jobs.save_job(j)
        # **취소도 영상은 버린다.** 실패 경로와 같은 이유다 — 기록은 job.json 에
        # 다 있고 원본은 버킷에 그대로 있다. 여기가 빠져 있어서 취소한 작업의
        # 입력 영상이 서버 디스크에 **영구히** 남았다(실측: 원본의 100%).
        # `cancelled` 는 TTL 정리 대상도 아니라(FA_FAILED_TTL_MIN=0) 시간이
        # 지나도 안 없어진다. 취소 한 번에 원본 하나씩 쌓이고, 그러다 여유가
        # FA_MIN_FREE_MB 밑으로 가면 새 작업이 507 로 거절되기 시작한다.
        jobs.drop_media(j, "취소 — 기록만 남긴다")
        log.info("작업 %s 취소됨", job_id)
        events.emit("job.cancelled", job=j.id, name=j.name, batch=j.batch or None)
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


def enqueue(name, params, s3_key="", jid=None, workdir=None, batch=""):
    """작업을 등록하고 워커에 넘긴다. 대기열 상한도 여기서 본다.

    ``jid`` 와 ``workdir`` 은 함께 온다 — 업로드 경로는 파일을 받아야 해서
    디렉터리를 먼저 만든다. 둘이 어긋나면 상태 파일이 엉뚱한 곳에 쓰인다.
    """
    global current
    jid = jid or new_job_id()
    workdir = workdir or os.path.join(config.JOBS_DIR, jid)
    os.makedirs(workdir, exist_ok=True)
    job = jobs.Job(id=jid, name=name, workdir=workdir, s3_key=s3_key,
                   params=params, batch=batch)
    events.emit("job.queued", job=jid, name=name, batch=batch or None,
                s3_key=s3_key or None)
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
