"""큐를 지켜보는 껍데기 — **인바운드 포트가 없는** 워커.

우리를 호출하는 사람은 없다. 이 프로세스가 Redis 큐를 지켜보다가 잡을 스스로
꺼내 간다(밀어 넣기가 아니라 당겨 오기). 그래서 로드밸런서도, 헬스체크 엔드포인트
도, API 인증도 필요 없고, **큐가 비면 컨테이너를 0대로 줄일 수 있다** — 저쪽 KEDA
설정이 `minReplicaCount: 0` 인 이유다.

이 파일이 하는 일은 넷뿐이고, 영상 처리는 한 줄도 없다.

    ① 저쪽 브로커에 붙어 우리 큐를 구독한다
    ② 잡이 오면 job_runner.run_job 에 넘긴다
    ③ 도는 동안 하트비트를 되돌려 보낸다 (리스 연장)
    ④ 끝나면 완료를 되돌려 보낸다 (토큰을 그대로 붙여서)

**상태를 갖지 않는다.** 순번·재시도 횟수·리스·중복 판정은 전부 잡을 준 쪽 DB 가
한다. 우리 service/worker.py 의 백오프(issues/003)와 재시작 복구(issues/002)는
이 경로에서 쓰지 않는다 — 같은 일을 두 곳에서 하면 서로를 방해한다.

실행::

    celery -A face_anonymizer.msa.celery_app worker \\
           -Q q.deidentify -c 1 --prefetch-multiplier 1 -l info

`-c 1` 과 `--prefetch-multiplier 1` 은 취향이 아니라 계약이다. GPU 는 한 장이고,
분 단위 작업을 쟁여 두면 컨테이너가 죽었을 때 쟁여 둔 것까지 같이 멎는다.
"""

import logging

from celery import Celery
from celery.signals import worker_process_init

from . import config

log = logging.getLogger(__name__)

app = Celery("face-anonymizer", broker=config.BROKER_URL)

app.conf.update(
    # 결과 백엔드가 없다. 결과는 완료 태스크로 되돌려 보내지, 우리가 저장하지 않는다.
    result_backend=None,
    task_ignore_result=True,
    # pickle 금지. 브로커를 통해 임의 객체가 실행되는 길을 열지 않는다.
    accept_content=["json"],
    task_serializer="json",
    # 긴 작업의 계약. 컨테이너가 죽으면 메시지는 브로커에 남아 다시 전달된다.
    # 재집행이 안전한 근거는 저쪽 리스+펜싱이다.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # Redis 에는 ack 가 없다 — acks_late 를 이 시간으로 흉내 낸다. 한 건 처리
    # 시간보다 짧으면 멀쩡히 도는 작업이 중복 배달된다(config 주석 참고).
    broker_transport_options={"visibility_timeout": config.VISIBILITY_TIMEOUT},
    # 우리가 되돌려 보내는 메시지도 자기 큐로 간다.
    task_routes={
        config.TASK_NAME: {"queue": config.QUEUE},
        config.HEARTBEAT_TASK: {"queue": config.CALLBACK_QUEUE},
        config.COMPLETE_TASK: {"queue": config.CALLBACK_QUEUE},
    },
    timezone="UTC",
)

_anonymizer = None


@worker_process_init.connect
def _preload(**_kw):
    """검출기를 미리 올린다. 첫 잡이 로딩 수십 초를 뒤집어쓰지 않게.

    가중치를 **잡 페이로드가 주는** 방식(FA_WEIGHTS_SOURCE=url)에서는 미리 올릴
    수 없다 — URL 이 첫 잡과 함께 오기 때문이다. 그 경우는 조용히 건너뛴다.
    """
    global _anonymizer
    if not config.PRELOAD:
        return
    from ..storage import weights as weights_store
    if weights_store.SOURCE == "url":
        log.info("가중치를 잡에서 받는 설정이라 미리 올리지 않는다")
        return
    try:
        _anonymizer = _build()
        log.info("검출기 준비 완료 — 첫 잡부터 바로 돈다")
    except Exception as e:                          # noqa: BLE001
        # 여기서 죽이면 컨테이너가 부팅 루프에 빠진다. 첫 잡에서 다시 시도하고,
        # 그때도 안 되면 그 잡이 '일시 실패' 로 저쪽에 보고된다.
        log.warning("검출기 예열 실패 — 첫 잡에서 다시 시도한다: %s", e)


def _build(job=None):
    """검출기를 만든다. 가중치 URL 은 **잡이 들고 온다**.

    ``FA_WEIGHTS_SOURCE=url`` 이면 가중치를 잡 페이로드가 준다. 여기서 그걸 안
    넘기면 그 방식이 통째로 못 쓰게 된다 — 스위치만 만들고 양쪽 경로를 다
    돌려 보지 않아서 놓쳤던 자리다.
    """
    from ..core.paths import DEFAULT_WEIGHTS
    from ..core.pipeline import VideoAnonymizer
    from ..storage import weights as weights_store

    weights_store.ensure(DEFAULT_WEIGHTS, url=(job or {}).get("weights_url"))
    return VideoAnonymizer()


def send(task, *, queue=None, **kwargs):
    """되돌려 보내기. 저쪽 태스크 이름은 config 가 안다."""
    app.send_task(task, kwargs=kwargs, queue=queue or config.CALLBACK_QUEUE)


@app.task(name=config.TASK_NAME, acks_late=True,
          reject_on_worker_lost=True, ignore_result=True)
def deidentify_one(job):
    """영상 1건 비식별화 — **무DB**.

    이 태스크는 저쪽 코드를 임포트하지 않는다. 페이로드 하나로 일하고, 하트비트와
    완료를 큐 메시지로 돌려보낸다. 그래서 이미지·언어·클러스터를 우리 마음대로
    고를 수 있다(저쪽 msa-boundaries M5 의 목적).

    예외를 밖으로 던지지 않는 것이 중요하다. 던지면 celery 가 이 메시지를 다시
    돌리는데, 저쪽은 이미 리스와 재시도 상한으로 그 판단을 하고 있다. 두 곳이
    각자 재시도하면 같은 영상이 몇 배로 돈다.
    """
    from ..job_runner import JobError, run_job

    global _anonymizer
    vid, token = job.get("video_id"), job.get("token")

    def beat(progress_s):
        send(config.HEARTBEAT_TASK, video_id=vid, token=token,
             progress_s=progress_s)

    try:
        if _anonymizer is None:
            _anonymizer = _build(job)
        result = run_job(job, on_heartbeat=beat, anonymizer=_anonymizer)
    except JobError as e:
        log.warning("잡 실패 video_id=%s [%s] transient=%s: %s",
                    vid, e.stage, e.transient, e)
        send(config.COMPLETE_TASK, video_id=vid, token=token, ok=False,
             error=str(e), transient=e.transient, stage=e.stage)
        return
    except Exception as e:                          # noqa: BLE001 — 조용히 죽으면 안 된다
        # 여기까지 온 것은 우리가 분류하지 못한 오류다. 일시로 본다 — 영구로
        # 두면 우리 버그 하나가 큐 전체를 상한까지 태운다.
        log.exception("잡 처리 중 분류되지 않은 오류 video_id=%s", vid)
        send(config.COMPLETE_TASK, video_id=vid, token=token, ok=False,
             error=f"{type(e).__name__}: {e}", transient=True, stage="unknown")
        return

    # 검수 딱지는 **성공과 함께** 간다. 실패로 보내면 저쪽이 재시도하는데, 다시
    # 돌려도 같은 결과다 — 필요한 것은 재시도가 아니라 사람의 확인이다.
    # 응답 모양이 바뀌어도 껍데기가 죽으면 안 된다 — 여기서 KeyError 가 나면
    # 작업은 끝났는데 완료 보고가 안 가서, 저쪽은 매달린 것으로 보고 회수한다.
    review = result.get("review") or []
    if review:
        log.warning("검수 필요 video_id=%s: %s", vid,
                    " / ".join(i.get("message", "") for i in review))
    log.info("잡 완료 video_id=%s %.1fs", vid, result.get("elapsed_s"))
    send(config.COMPLETE_TASK, video_id=vid, token=token, ok=True,
         elapsed_s=result.get("elapsed_s"), targets=result.get("targets") or [],
         review_needed=bool(review), review=review,
         notices=result.get("notices") or [])
