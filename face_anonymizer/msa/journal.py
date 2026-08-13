"""큐 워커의 기록 — **API 쪽 worker.py 와 같은 질문에, 다른 구조로 답한다.**

왜 따로 두나
------------
목적은 같다. 로그가 답해야 하는 질문은 어느 경로든 똑같다.

    이 파일 언제 시작해서 언제 끝났나 (한국 시각) · 얼마나 걸렸나 ·
    얼굴은 몇 프레임에서 잡혔나 · 사람이 봐야 할 건이 있나 ·
    실패했다면 어느 단계에서, 다시 해 볼 만한 실패인가

구조는 다를 수밖에 없다. 세 가지가 근본적으로 다르기 때문이다.

**사는 곳이 다르다.** API 는 EC2 에 계속 떠 있어서 파일에 쌓고 나중에
``/api/events`` 로 되읽는다. MSA 는 컨테이너다 — 큐가 비면 KEDA 가 0대로
줄이고, 그때 **파일과 함께 기록이 사라진다.** 그래서 이쪽 기록은 stdout 으로도
같이 나간다(events.STDOUT). 되읽는 API 는 없다. 로그 수집기가 걷어 가는 것이
전부다.

**아는 것이 다르다.** API 는 큐를 자기가 들고 있어서 "앞에 몇 건 남았고 평균
몇 초니까 남은 시간은 얼마" 를 계산할 수 있다. MSA 는 **큐 깊이를 모른다** —
큐도 순번도 저쪽 것이다. 그래서 남은 시간을 우리가 만들지 않는다. 대신 우리만
아는 값(이 워커의 최근 평균 처리 시간)을 완료 보고에 실어 보내서, 큐 깊이를
아는 쪽이 곱하게 한다. 계산을 넘기는 게 아니라 **계산에 필요한 절반을 넘긴다.**

**묶음의 주인이 다르다.** 폴더 단위 시작/종료는 API 에서는 우리가 낸다
(jobs.batches). MSA 는 잡 하나에 영상 하나로 오고, 같은 폴더의 영상들이 여러
컨테이너로 흩어진다 — **한 컨테이너는 폴더 전체를 볼 수 없다.** 우리가 할 수
있는 건 잡이 들고 온 묶음 표시를 줄마다 같이 남겨서, 나중에 그 값으로 모을 수
있게 해 두는 것까지다. 집계 자체는 전부를 보는 쪽의 일이다.

이 파일은 그 셋을 한군데 모아 둔 곳이다. celery_app 은 흐름만 갖고,
"무엇을 어떤 문장으로 남기나" 는 여기서 정한다.
"""

import logging
import threading
import time

from .. import events, timefmt

log = logging.getLogger("msa")

# 묶음 표시를 어떤 이름으로 받을지 아직 저쪽과 합의되지 않았다. 흔한 이름을
# 모두 받아 둔다 — 이름이 정해지면 그 하나만 남기면 되고, 그 전에 저쪽이
# 무엇을 보내든 기록은 남는다. (docs/integration 의 요청 목록 참고)
BATCH_KEYS = ("batch_id", "batch", "group_id", "folder", "collection_id")

# 원본 파일명. 없으면 video_id 로 부른다 — 서명된 URL 에서 파일명을 뽑아내지
# 않는다. 거기엔 서명이 붙어 있고, 저널에 서명을 남기지 않는 것이 규칙이다.
NAME_KEYS = ("name", "filename", "source_name", "original_name")

# 평균을 낼 때 보는 최근 건수. 워커 수명 전체 평균을 쓰면 초반 콜드스타트가
# 계속 섞여서 남은 시간 추정이 실제보다 길게 나온다.
WINDOW = 20


def _first(job, keys):
    for k in keys:
        v = (job or {}).get(k)
        if v:
            return str(v)
    return None


def batch_of(job):
    return _first(job, BATCH_KEYS)


def name_of(job):
    return _first(job, NAME_KEYS) or (job or {}).get("video_id") or "?"


def label_of(job):
    t = ((job or {}).get("targets") or [{}])[0]
    return t.get("label")


class Stats:
    """이 컨테이너가 지금까지 무엇을 했나. **프로세스 안에서만 산다.**

    영속시키지 않는다. 컨테이너는 언제든 사라지고, 살아남아야 하는 집계는
    전부를 보는 쪽이 저널을 모아서 낸다. 여기 값의 쓸모는 딱 둘이다 —
    완료 보고에 실어 보낼 평균, 그리고 종료할 때 남기는 한 줄.
    """

    def __init__(self, window=WINDOW):
        self._lock = threading.Lock()
        self._recent = []
        self._window = window
        self.started_at = time.time()
        self.done = 0
        self.failed = 0
        self.review = 0

    def record(self, elapsed_s, ok=True, review=False):
        with self._lock:
            if ok:
                self.done += 1
                if elapsed_s:
                    self._recent.append(float(elapsed_s))
                    del self._recent[:-self._window]
            else:
                self.failed += 1
            if review:
                self.review += 1

    @property
    def avg(self):
        """최근 평균 처리 시간(초). 표본이 없으면 None — **0 을 주지 않는다.**

        0 을 주면 저쪽이 "남은 시간 0분" 이라고 표시한다. 모르는 것은 모른다고
        보내야 화면에 '계산 중' 이 뜬다.
        """
        with self._lock:
            if not self._recent:
                return None
            return round(sum(self._recent) / len(self._recent), 1)

    def summary(self):
        up = time.time() - self.started_at
        avg = self.avg
        return (f"{timefmt.dur(up)} 동안 성공 {self.done}건 · 실패 {self.failed}건"
                + (f" · 검수 {self.review}건" if self.review else "")
                + (f" · 평균 {avg}초" if avg else ""))


STATS = Stats()


def worker_up(queue, task, cold_s=None):
    """컨테이너가 일할 준비가 됐다.

    콜드스타트 시간을 남기는 이유: KEDA 가 0↔N 으로 오르내리므로, 잡이 큐에
    들어온 뒤 실제로 처리가 시작되기까지의 지연에 이 시간이 통째로 들어간다.
    나중에 "왜 대기가 기냐" 를 따질 때 여기가 첫 용의자다.
    """
    log.info("● 워커 준비  큐=%s%s", queue,
             f"  (기동 {cold_s:.1f}초)" if cold_s else "")
    events.emit("worker.ready", queue=queue, task=task,
                cold_s=round(cold_s, 1) if cold_s else None)


def worker_down(reason="shutdown"):
    """컨테이너가 내려간다. **마지막 한 줄이 이 컨테이너의 유일한 요약이다.**"""
    log.info("○ 워커 종료  %s  [%s]", STATS.summary(), reason)
    events.emit("worker.stopped", reason=reason, done=STATS.done,
                failed=STATS.failed, review=STATS.review,
                avg_elapsed_s=STATS.avg,
                uptime_s=round(time.time() - STATS.started_at, 1))


def job_started(job, started):
    """``▶ 시작  K_00297.mp4  (kbs)  [8월13일 01시04분]``

    문장 모양을 API 쪽(worker.py)과 일부러 똑같이 맞춘다. 같은 파이프라인이
    남긴 줄이니 두 경로의 로그를 한데 놓고 읽을 수 있어야 한다.
    """
    batch, name = batch_of(job), name_of(job)
    log.info("▶ 시작  %s%s  [%s]", name,
             f"  ({batch})" if batch else "", timefmt.stamp(started))
    events.emit("job.started", job=job.get("video_id"), name=name,
                batch=batch, label=label_of(job),
                targets=len(job.get("targets") or []),
                started_at=timefmt.iso(started))


def job_finished(job, result, started, finished):
    """``■ 완료  K_00297.mp4  8월13일 01시04분 ~ 01시05분 (40.7초)``

    저널 줄에는 근거가 될 값을 전부 싣는다 — 나중에 "이 파일 어떤 설정으로
    얼마나 걸려 처리했고 검출률이 얼마였나" 에 이 한 줄로 답해야 한다.
    """
    batch, name = batch_of(job), name_of(job)
    review = result.get("review") or []
    t = (result.get("targets") or [{}])[0]

    STATS.record(result.get("elapsed_s"), ok=True, review=bool(review))

    log.info("■ 완료  %s%s  %s%s", name, f"  ({batch})" if batch else "",
             timefmt.span(started, finished),
             f"  검출률 {t['detection_rate'] * 100:.1f}%"
             if t.get("detection_rate") is not None else "")
    if review:
        for item in review:
            log.warning("검수 필요  %s — %s", name, item.get("message", ""))

    events.emit("job.finished", job=job.get("video_id"), name=name,
                batch=batch, label=t.get("label") or label_of(job),
                seconds=result.get("elapsed_s"),
                started_at=timefmt.iso(started),
                finished_at=timefmt.iso(finished),
                span=timefmt.span(started, finished),
                frames=t.get("frames"), detected_frames=t.get("detected_frames"),
                detection_rate=t.get("detection_rate"),
                realtime_factor=t.get("realtime_factor"),
                source_codec=t.get("source_codec"),
                transcoded=t.get("transcoded"),
                timing=t.get("timing"), warnings=t.get("warnings"),
                review_needed=bool(review),
                review=[i.get("code") for i in review] or None,
                notices=[i.get("code") for i in (result.get("notices") or [])]
                or None,
                worker_avg_s=STATS.avg, worker_done=STATS.done)


def job_progress(job, progress):
    """하트비트마다 한 줄. **어디까지 갔다가 멎었는지**가 여기 남는다.

    완료 줄만 남기면 죽은 잡은 아무 흔적이 없다 — 저쪽 리스가 회수해서 다시
    돌리고 나면 "왜 처음에 실패했나" 를 물을 근거가 사라진다. 60초에 한 줄이라
    양은 문제가 되지 않는다.
    """
    events.emit("job.progress", job=job.get("video_id"), name=name_of(job),
                batch=batch_of(job), percent=progress.get("percent"),
                stage=progress.get("stage"), eta_s=progress.get("eta_s"),
                elapsed_s=progress.get("elapsed_s"))


def job_failed(job, stage, transient, detail, started=None):
    """실패도 시각과 함께. **얼마나 돌다 실패했는지**가 원인을 좁힌다 —
    3초 만이면 입력을 못 받은 것이고, 40초면 처리 중에 넘어진 것이다."""
    batch, name = batch_of(job), name_of(job)
    now = time.time()
    STATS.record(None, ok=False)
    log.warning("✕ 실패  %s%s  [%s] %s%s", name,
                f"  ({batch})" if batch else "", stage,
                "일시적 — 다시 시도해 볼 만함" if transient else "영구",
                f"  ({timefmt.dur(now - started)} 경과)" if started else "")
    events.emit("job.failed", job=job.get("video_id"), name=name, batch=batch,
                stage=stage, transient=transient, detail=detail,
                started_at=timefmt.iso(started) if started else None,
                failed_at=timefmt.iso(now),
                seconds=round(now - started, 2) if started else None)
