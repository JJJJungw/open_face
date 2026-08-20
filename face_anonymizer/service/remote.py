"""저쪽이 우리를 부르는 문 — HTTP 로 들어오는 잡 하나.

**새 로직이 없다.** 큐 경로(`msa/`)가 브로커에서 꺼내 부르던 `job_runner.run_job`
을 여기서도 그대로 부른다. 러너는 자기가 어느 문으로 들어왔는지 모른다 —
계약은 페이로드고 전송은 선택이다(docs/integration/rebornstudio.md §0).

    저쪽 build_jobs ──HTTP POST──▶ remote.submit ──▶ job_runner.run_job
                                        │                    │
                     GET /{id} ◀── 진행·결과 ◀───────────────┘

## 왜 큐를 하나 더 만들지 않는가

바쁘면 **거절한다**(429). 우리가 대기열을 만들면 큐가 두 개가 된다 — 저쪽은
이미 리스·펜싱·회수·재시도 상한을 갖고 있고, 두 곳이 각자 판단하면 같은 영상이
몇 배로 돈다. 이 문서의 §2-1 이 큐 경로에 대해 적은 것과 같은 이유다. 우리는
"지금 못 받는다" 만 정확히 말하고, 언제 다시 줄지는 저쪽이 정한다.

## 상태는 메모리에만 둔다

이 기록은 **폴링에 답하기 위한 것**이지 작업의 정본이 아니다. 정본은 저쪽 DB 의
`deident_status` 다. 그래서 재시작하면 사라져도 되고, 사라지면 저쪽 리스가
만료돼 그 영상은 다시 배달된다 — 이미 그 경우를 전제한 설계다.

환경 변수
    FA_REMOTE_TOKEN         이 문을 여는 열쇠. 비면 루프백에서만 열린다
    FA_REMOTE_MAX_INFLIGHT  동시에 받을 잡 수 (기본 1 — GPU 는 한 장이다)
    FA_REMOTE_TTL_SEC       끝난 기록을 얼마나 들고 있나 (기본 3600)
"""

import hmac
import logging
import os
import threading
import time
import uuid

from .. import events, gpu, timefmt
from ..env import flag as _bool_env

log = logging.getLogger(__name__)


# GPU 여유의 정본은 `gpu.fields()` 하나다 — 두 경로가 같은 이름으로 남겨야
# 나중에 나란히 놓고 볼 수 있다(같은 걸 두 곳에 두면 갈라진다).
_vram = gpu.fields

# 이 문을 여는 열쇠. **비어 있으면 같은 기계 밖에서는 안 열린다.**
#
# 오픈소스로 받아 자기 노트북에서 돌리는 사람은 아무 설정 없이 바로 쓰고, 외부
# 주소에서 들어오면 거부한다. "설정 안 하면 안 열린다" 와 "설정 안 하면 활짝
# 열린다" 사이에서, 처음 받는 사람을 안 막으면서 안전한 쪽을 골랐다.
TOKEN = os.environ.get("FA_REMOTE_TOKEN", "").strip()
LOOPBACK = ("127.0.0.1", "::1", "localhost")

# **인증을 아예 끄는 스위치.** 컨테이너끼리 붙여 보는 단계에서는 열쇠가 곧
# 마찰이다 — 받아서 띄운 사람이 403 을 보고 "왜 안 되지" 부터 시작하게 된다.
#
# 그래서 명시적인 스위치로 둔다. 기본값이 아니고, 켜면 기동 로그가 크게 말한다.
# 조용히 열려 있는 것과 **켜 놓고 열려 있다고 말하는 것**은 다르다.
OPEN = _bool_env("FA_REMOTE_OPEN", False)

# GPU 는 한 장이다. 두 편을 동시에 올리면 둘 다 느려지거나 메모리에서 터진다.
MAX_INFLIGHT = int(os.environ.get("FA_REMOTE_MAX_INFLIGHT", 1))
# 끝난 기록 보관. 저쪽이 결과를 가져갈 시간만 있으면 된다.
TTL = float(os.environ.get("FA_REMOTE_TTL_SEC", 3600))

_LOCK = threading.Lock()
_JOBS = {}


class Busy(RuntimeError):
    """지금은 받을 수 없다. 대기열을 만들지 않는다는 뜻이다."""


class Job:
    """잡 하나의 상태. **폴링에 답하기 위한 것**이지 정본이 아니다."""

    __slots__ = ("id", "video_id", "status", "progress", "result", "problem",
                 "error", "transient", "stage", "created", "finished")

    def __init__(self, jid, video_id):
        self.id = jid
        self.video_id = video_id
        self.status = "running"
        self.progress = None
        self.result = self.problem = self.error = None
        self.transient = self.stage = None
        self.created = time.time()
        self.finished = None

    def view(self):
        """저쪽이 읽는 모양. 응답 스키마는 docs/integration §5 와 같다."""
        d = {"job_id": self.id, "video_id": self.video_id,
             "status": self.status, "progress": self.progress,
             "started_at": timefmt.iso(self.created)}
        if self.status == "done":
            d["result"] = self.result
        elif self.status == "failed":
            # `error` 는 우리 내부 문구다. 화면에는 problem 을 띄운다 —
            # 큐 경로가 완료 메시지에 싣는 것과 같은 짝이다.
            d.update(error=self.error, transient=self.transient,
                     stage=self.stage, problem=self.problem)
        if self.finished:
            d["finished_at"] = timefmt.iso(self.finished)
        return d


def announce():
    """기동 때 이 문이 어떤 상태인지 한 줄 남긴다.

    **열려 있다는 사실을 로그가 말해야 한다.** 나중에 "이거 인증 있었나" 를
    물었을 때 기억이 아니라 기록으로 답이 나와야 한다.
    """
    if OPEN:
        log.warning("잡 접수 문이 **인증 없이** 열려 있습니다 "
                    "(FA_REMOTE_OPEN=1). 시험용 설정입니다 — 공개된 망에 "
                    "띄우지 마세요.")
    elif TOKEN:
        log.info("잡 접수 문: 토큰 필요 (X-Deident-Token)")
    else:
        log.info("잡 접수 문: 같은 기계에서만 열림 "
                 "(FA_REMOTE_TOKEN 또는 FA_REMOTE_OPEN 으로 바꿉니다)")


def door_open(request):
    """이 요청을 받아도 되나. (되나, 사유)

    **토큰이 있으면 토큰만 본다.** 망으로 막는 것은 이 코드 밖의 일이고(보안
    그룹), 여기서 보는 것은 "이 요청에 열쇠가 붙어 있나" 하나다. 둘은 서로를
    대신하지 못한다 — IP 로만 막으면 저쪽 서버를 통해 우회로 들어오는 요청은
    출처가 정상이라 그냥 통과한다.
    """
    if OPEN:
        return True, ""
    if TOKEN:
        # **바이트로 견준다.** `hmac.compare_digest` 는 str 을 받으면 한쪽에
        # 아스키 밖 글자가 있을 때 TypeError 를 던진다 — 그러면 틀린 토큰을
        # 보낸 요청이 403 이 아니라 **500** 으로 돌아간다. 남이 보내는 값이라
        # 아스키라는 보장이 없고, 운영자가 한글 토큰을 넣을 수도 있다.
        sent = (request.headers.get("x-deident-token") or "").strip()
        if sent and hmac.compare_digest(sent.encode("utf-8", "replace"),
                                        TOKEN.encode("utf-8", "replace")):
            return True, ""
        return False, "토큰이 없거나 맞지 않습니다"
    host = (request.client.host if request.client else "") or ""
    if host in LOOPBACK:
        return True, ""
    return False, ("이 문은 같은 기계에서만 열려 있습니다. 밖에서 부르시려면 "
                   "FA_REMOTE_TOKEN 을 설정하고 X-Deident-Token 헤더로 같은 값을 "
                   "보내 주세요. 시험 중이라면 FA_REMOTE_OPEN=1 로 열 수 있습니다.")


def _sweep(now):
    """끝난 지 오래된 기록을 지운다. **호출자가 잠금을 들고 있어야 한다.**"""
    for jid in [k for k, j in _JOBS.items()
                if j.finished and now - j.finished > TTL]:
        del _JOBS[jid]


def inflight():
    with _LOCK:
        return sum(1 for j in _JOBS.values() if j.status == "running")


def get(jid):
    with _LOCK:
        return _JOBS.get(jid)


# 부르는 쪽이 쓰는 이름이 우리와 다를 수 있다. 넉넉히 본다 — 이름 하나가
# 어긋났을 때 나는 오류가 "input_url 이 필요합니다" 면 **무엇이 잘못됐는지가
# 안 드러난다.** 이름을 맞추는 것은 저쪽 코드를 고치는 일이라 시간이 걸리고,
# 그 사이에 붙지 못할 이유가 없다.
INPUT_KEYS = ("input_key", "input_path", "input", "source_key", "source")
OUTPUT_KEYS = ("put_key", "output_key", "output_path", "output")


def _first(d, names):
    for n in names:
        v = d.get(n)
        if v:
            return v
    return None


def _split_uri(value, default_bucket):
    """`s3://버킷/키` 도 받고 그냥 `키` 도 받는다. (버킷, 키)"""
    v = (value or "").strip()
    if v.startswith("s3://"):
        rest = v[len("s3://"):]
        bucket, _, key = rest.partition("/")
        return bucket or default_bucket, key
    return default_bucket, v.lstrip("/")


def resolve(job):
    """경로만 온 잡을 **서명된 URL 이 든 잡**으로 바꾼다.

    2026-08-20 결정: **클라우드 접근은 이쪽이 맡는다.** 저쪽은 분석할 파일의
    경로와 결과를 둘 경로만 넘기고, 자격 증명은 우리 `.env` 에 있다.

    바꾸는 자리를 **문 하나로 몰아 둔 것**이 요점이다. 여기서 서명해 두면
    러너(`job_runner.run_job`)는 예전과 똑같이 URL 두 개만 본다 — 저쪽이 직접
    서명해 보내던 방식과 코드가 한 줄도 갈라지지 않는다. 그래서 둘 다 받는다:
    ``input_url`` 이 오면 그대로 쓰고, ``input_key`` 가 오면 우리가 서명한다.

    받는 이름은 넉넉히 본다. 저쪽이 부르는 이름이 우리 것과 다를 수 있고, 그때
    나는 오류는 "input_url 이 필요합니다" 라서 **무엇이 잘못됐는지가 안 드러난다.**
    """
    if job.get("input_url") and all(t.get("put_url") for t in job.get("targets") or []):
        return job                                   # 이미 서명돼 있다

    from ..storage import s3 as s3mod
    store = s3mod.get_store()
    if store is None:
        raise ValueError(
            "경로로 받으려면 이 서버에 저장소가 설정돼 있어야 합니다 — "
            "FA_S3_BUCKET 과 자격 증명을 확인해 주세요 "
            "(GET /api/credentials/health 로 볼 수 있습니다)")

    out = dict(job)
    if not out.get("input_url"):
        src = _first(out, INPUT_KEYS)
        if not src:
            raise ValueError("input_url 또는 input_key 가 필요합니다")
        bucket, key = _split_uri(src, store.bucket)
        signer = store if bucket == store.bucket else store.for_bucket(bucket)
        out["input_url"] = signer.presigned_url(key)
        out.setdefault("input_key", key)

    targets = []
    for t in out.get("targets") or []:
        t = dict(t)
        if not t.get("put_url"):
            dst = _first(t, OUTPUT_KEYS)
            if not dst:
                raise ValueError(
                    f"타깃 '{t.get('label') or '이름없음'}' 에 "
                    "put_url 또는 output_key 가 필요합니다")
            bucket, key = _split_uri(dst, store.bucket)
            signer = store if bucket == store.bucket else store.for_bucket(bucket)
            ctype = t.get("content_type") or "video/mp4"
            t["put_url"] = signer.presigned_put(key, ctype)
            t["content_type"] = ctype
            t.setdefault("put_key", key)
        targets.append(t)
    out["targets"] = targets
    return out


def submit(job, runner=None):
    """잡 하나를 받아 스레드에서 돌린다. 받은 기록을 돌려준다.

    Raises:
        ValueError: 페이로드가 모양을 안 갖췄다 (400).
        Busy:       지금 처리 중인 것이 상한이다 (429).
    """
    if not isinstance(job, dict):
        raise ValueError("잡은 객체여야 합니다")
    # **입력부터 본다.** 무엇이 빠졌는지를 순서대로 말해 줘야, 보내는 쪽이
    # 한 번에 하나씩 고칠 수 있다.
    if not (job.get("input_url") or _first(job, INPUT_KEYS)):
        raise ValueError("input_url 또는 input_key 가 필요합니다")
    if not job.get("targets"):
        raise ValueError("targets 가 비어 있습니다")
    # **서명은 접수할 때 한다.** 스레드 안에서 하면 실패가 202 뒤에 숨어서,
    # 저쪽은 잡을 받았다고 믿고 폴링부터 시작한다 — 경로 오타 하나가 리스
    # 만료까지 안 드러난다.
    job = resolve(job)

    now = time.time()
    with _LOCK:
        _sweep(now)
        running = sum(1 for j in _JOBS.values() if j.status == "running")
        if running >= MAX_INFLIGHT:
            raise Busy(f"처리 중 {running}건 — 상한 {MAX_INFLIGHT}건")
        rec = Job(uuid.uuid4().hex, job.get("video_id"))
        _JOBS[rec.id] = rec

    events.emit("remote.job.received", job=rec.id, video_id=rec.video_id or "",
                **_vram())
    log.info("잡 접수(HTTP): job_id=%s video_id=%s · %s",
             rec.id, rec.video_id, gpu.line())
    threading.Thread(target=_run, args=(rec, job, runner), daemon=True,
                     name=f"remote-{rec.id[:8]}").start()
    return rec


def _run(rec, job, runner=None):
    """스레드 본체. **예외를 밖으로 던지지 않는다** — 던지면 스레드만 죽고
    기록은 영원히 `running` 으로 남는다. 그러면 저쪽은 리스가 만료될 때까지
    기다리게 되고, 우리 화면에는 아무 사유도 없다."""
    from ..job_runner import JobError

    if runner is None:
        from ..job_runner import run_job as runner

    def beat(snapshot):
        # 진행률에 GPU 여유를 얹는다. **도는 동안의 값이라야 쓸모가 있다** —
        # 터진 뒤에 재면 이미 다 반납된 뒤다.
        with _LOCK:
            rec.progress = {**snapshot, **_vram()}

    anonymizer = None
    try:
        from . import worker
        anonymizer = worker.get_anonymizer()
    except Exception:                               # noqa: BLE001
        # 러너가 알아서 만든다. 거기서도 실패하면 model 단계로 분류된다.
        anonymizer = None

    try:
        result = runner(job, on_heartbeat=beat, anonymizer=anonymizer)
    except JobError as e:
        _finish_failed(rec, e.as_dict(), str(e), e.transient, e.stage)
        return
    except Exception as e:                          # noqa: BLE001
        # 우리가 분류하지 못한 오류다. **일시로 본다** — 영구로 두면 우리 버그
        # 하나가 저쪽 재시도 상한을 통째로 태운다. 큐 경로와 같은 판단이다.
        log.exception("잡 처리 중 분류되지 않은 오류 job_id=%s", rec.id)
        p = JobError(f"{type(e).__name__}: {e}", transient=True,
                     stage="unknown").as_dict()
        _finish_failed(rec, p, f"{type(e).__name__}: {e}", True, "unknown")
        return

    with _LOCK:
        rec.status, rec.result, rec.finished = "done", result, time.time()
    events.emit("remote.job.finished", job=rec.id, video_id=rec.video_id or "",
                elapsed_s=result.get("elapsed_s"),
                review_needed=bool(result.get("review_needed")), **_vram())
    log.info("잡 완료(HTTP): job_id=%s %.1fs · %s",
             rec.id, result.get("elapsed_s") or 0, gpu.line())


def _finish_failed(rec, problem, error, transient, stage):
    with _LOCK:
        rec.status, rec.finished = "failed", time.time()
        rec.problem, rec.error = problem, error
        rec.transient, rec.stage = bool(transient), stage or "unknown"
    events.emit("remote.job.failed", job=rec.id, video_id=rec.video_id or "",
                stage=rec.stage, transient=rec.transient, error=error, **_vram())
    # **실패에는 GPU 여유를 반드시 남긴다.** OOM 이면 이 줄이 유일한 단서다.
    log.warning("잡 실패(HTTP): job_id=%s stage=%s transient=%s · %s — %s",
                rec.id, rec.stage, rec.transient, gpu.line(), error)


def reset():
    """시험용. 기록을 비운다."""
    with _LOCK:
        _JOBS.clear()
