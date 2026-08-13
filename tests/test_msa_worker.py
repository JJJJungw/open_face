"""큐 껍데기 테스트 — 브로커 없이 계약만 본다.

Redis 도 저쪽 서버도 띄우지 않는다. 확인하는 것은 '무엇을 어떤 이름으로 되돌려
보내는가' 이고, 그게 이 껍데기가 지는 계약의 전부다.
"""

import pytest

pytest.importorskip("celery", reason="pip install -r requirements-worker.txt")

from face_anonymizer import job_runner                      # noqa: E402
from face_anonymizer.msa import celery_app as shell         # noqa: E402
from face_anonymizer.msa import config as mc                # noqa: E402


@pytest.fixture
def sent(monkeypatch):
    """되돌려 보낸 메시지를 모은다. 브로커에 붙지 않는다."""
    out = []
    monkeypatch.setattr(shell.app, "send_task",
                        lambda name, kwargs=None, queue=None:
                            out.append((name, kwargs or {}, queue)))
    monkeypatch.setattr(shell, "_anonymizer", object())     # 모델 로드 건너뛰기
    return out


JOB = {"video_id": "vid-1", "token": "tok-1", "input_url": "http://x/in.mp4",
       "targets": [{"label": "deid-720p", "put_url": "http://x/out.mp4"}]}


def test_success_reports_completion_with_the_same_token(sent, monkeypatch):
    """완료 보고에 **받았던 토큰을 그대로** 붙인다.

    저쪽은 이 토큰으로 펜싱을 판정한다. 토큰이 없거나 다르면 그 보고는 버려지고,
    작업은 리스가 만료될 때까지 매달린 것처럼 보인다.
    """
    monkeypatch.setattr(job_runner, "run_job",
                        lambda job, **kw: {"elapsed_s": 12.3, "targets": []})

    shell.deidentify_one(JOB)

    name, kw, queue = sent[-1]
    assert name == mc.COMPLETE_TASK and queue == mc.CALLBACK_QUEUE
    assert kw["video_id"] == "vid-1" and kw["token"] == "tok-1"
    assert kw["ok"] is True and kw["elapsed_s"] == 12.3


def test_failure_reports_the_transient_flag(sent, monkeypatch):
    """실패는 분류해서 넘긴다. 판정(재시도할지)은 저쪽이 한다."""
    def boom(job, **kw):
        raise job_runner.JobError("presign 만료", transient=True, stage="download")
    monkeypatch.setattr(job_runner, "run_job", boom)

    shell.deidentify_one(JOB)

    name, kw, _q = sent[-1]
    assert name == mc.COMPLETE_TASK
    assert kw["ok"] is False and kw["transient"] is True
    assert kw["stage"] == "download" and kw["token"] == "tok-1"


def test_unclassified_error_is_reported_as_transient(sent, monkeypatch):
    """우리가 분류하지 못한 오류를 영구로 두면, 버그 하나가 큐 전체를 태운다."""
    def boom(job, **kw):
        raise ZeroDivisionError("우리 버그")
    monkeypatch.setattr(job_runner, "run_job", boom)

    shell.deidentify_one(JOB)

    _n, kw, _q = sent[-1]
    assert kw["ok"] is False and kw["transient"] is True and kw["stage"] == "unknown"


def test_task_never_raises_so_celery_does_not_retry(sent, monkeypatch):
    """예외를 밖으로 던지면 celery 가 메시지를 다시 돌린다.

    저쪽도 리스로 같은 판단을 하고 있어서, 두 곳이 각자 재시도하면 같은 영상이
    몇 배로 돈다. 이 태스크는 무슨 일이 있어도 조용히 반환해야 한다.
    """
    for exc in (job_runner.JobError("x", transient=False), RuntimeError("y"),
                KeyError("z")):
        monkeypatch.setattr(job_runner, "run_job",
                            lambda job, _e=exc, **kw: (_ for _ in ()).throw(_e))
        shell.deidentify_one(JOB)          # 던지면 여기서 테스트가 깨진다
    assert all(kw["ok"] is False for _n, kw, _q in sent)


def test_heartbeat_carries_video_and_token(sent, monkeypatch):
    """하트비트는 리스 연장 신호다. 자기 토큰일 때만 연장되므로 같이 보낸다."""
    def fake_run(job, on_heartbeat=None, **kw):
        on_heartbeat(60.0)
        return {"elapsed_s": 1.0, "targets": []}
    monkeypatch.setattr(job_runner, "run_job", fake_run)

    shell.deidentify_one(JOB)

    beats = [(kw, q) for n, kw, q in sent if n == mc.HEARTBEAT_TASK]
    assert len(beats) == 1
    kw, queue = beats[0]
    assert kw["video_id"] == "vid-1" and kw["token"] == "tok-1"
    assert queue == mc.CALLBACK_QUEUE


def test_registered_under_the_name_the_producer_sends():
    """이름이 한 글자만 달라도 메시지는 큐에 남아 아무도 안 먹는다 — 오류도 안 난다."""
    assert mc.TASK_NAME in shell.app.tasks
    assert shell.app.conf.task_routes[mc.TASK_NAME]["queue"] == mc.QUEUE


def test_long_job_contract_is_set():
    """긴 작업의 계약 — 죽으면 재전달되고, 쟁여 두지 않고, 결과를 저장하지 않는다."""
    c = shell.app.conf
    assert c.task_acks_late is True
    assert c.task_reject_on_worker_lost is True
    assert c.worker_prefetch_multiplier == 1
    assert c.result_backend is None and c.task_ignore_result is True
    # pickle 을 허용하면 브로커를 통해 임의 객체가 실행된다.
    assert list(c.accept_content) == ["json"]
