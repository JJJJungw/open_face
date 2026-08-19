"""큐 워커의 기록 — **컨테이너라서 달라지는 부분만** 본다.

저널 스키마 자체(줄 모양, 비밀 필드 제외)는 events 쪽 테스트가 본다. 여기서
확인하는 것은 "API 로는 되는데 컨테이너에서는 안 되는 것" 세 가지다.
"""

import json
import time

import pytest

pytest.importorskip("celery", reason="pip install -r requirements/worker.txt")

from face_anonymizer import events, job_runner, timefmt      # noqa: E402
from face_anonymizer.msa import celery_app as shell          # noqa: E402
from face_anonymizer.msa import journal                      # noqa: E402


JOB = {"video_id": "vid-9", "token": "tok-9", "input_url": "http://x/in.mp4",
       "batch_id": "kbs-0813", "name": "K_00297.mp4",
       "targets": [{"label": "deid-720p", "put_url": "http://x/out.mp4"}]}

RESULT = {"elapsed_s": 40.7, "review": [], "notices": [],
          "targets": [{"label": "deid-720p", "frames": 1027,
                       "detected_frames": 768, "detection_rate": 0.7478,
                       "timing": {"detect": 13.6}}]}


@pytest.fixture
def sent(monkeypatch, tmp_path):
    out = []
    monkeypatch.setattr(shell.app, "send_task",
                        lambda name, kwargs=None, queue=None:
                            out.append((name, kwargs or {}, queue)))
    monkeypatch.setattr(shell, "_anonymizer", object())
    monkeypatch.setattr(events, "DIR", str(tmp_path / "ev"))
    monkeypatch.setattr(journal, "STATS", journal.Stats())
    return out


def test_batch_travels_on_every_line(sent, monkeypatch):
    """폴더 표시는 **줄마다** 붙어야 한다.

    한 컨테이너는 폴더 전체를 볼 수 없다 — 같은 폴더의 영상들이 여러 컨테이너로
    흩어지기 때문이다. 그래서 우리가 폴더 집계를 내는 대신, 나중에 누가 모을 수
    있도록 표시만 빠짐없이 남긴다. 한 줄이라도 빠지면 그 파일은 집계에서 샌다.
    """
    monkeypatch.setattr(job_runner, "run_job", lambda job, **kw: RESULT)

    shell.deidentify_one(JOB)

    rows = events.read(job="vid-9")
    assert rows and all(r.get("batch") == "kbs-0813" for r in rows)
    assert all(r.get("name") == "K_00297.mp4" for r in rows)


def test_batch_id_goes_back_in_the_completion_report(sent, monkeypatch):
    """묶음 표시를 완료 보고에도 되돌려 준다 — 집계는 저쪽이 한다."""
    monkeypatch.setattr(job_runner, "run_job", lambda job, **kw: RESULT)

    shell.deidentify_one(JOB)

    _n, kw, _q = sent[-1]
    assert kw["batch_id"] == "kbs-0813"


def test_times_are_sent_as_finished_strings(sent, monkeypatch):
    """시각은 **문자열까지 만들어서** 보낸다.

    epoch 만 넘기면 받는 쪽이 자기 타임존으로 찍는다. 컨테이너는 UTC 로 도는
    일이 많아서, 그러면 우리 로그와 저쪽 화면이 같은 작업을 놓고 9시간 다른
    시각을 말한다. 납품 근거로 쓸 기록이라 그건 곤란하다.
    """
    monkeypatch.setattr(job_runner, "run_job", lambda job, **kw: RESULT)

    shell.deidentify_one(JOB)

    _n, kw, _q = sent[-1]
    assert kw["started_at"].endswith(f"+0{int(timefmt.OFFSET_HOURS)}:00")
    assert "~" in kw["span"]
    fin = next(r for r in events.read(job="vid-9")
               if r["event"] == "job.finished")
    assert fin["started_at"] and fin["finished_at"] and fin["span"]


def test_average_is_reported_but_never_as_zero(sent, monkeypatch):
    """남은 시간의 **절반만** 우리가 안다.

    큐 깊이는 저쪽 것이라 남은 시간을 우리가 계산할 수 없다. 대신 우리만 아는
    평균을 실어 보내 저쪽이 곱하게 한다. 표본이 없을 때 0 을 보내면 저쪽 화면에
    "남은 시간 0분" 이 뜬다 — 모르는 것은 None 으로 보낸다.
    """
    assert journal.STATS.avg is None            # 아직 한 건도 안 돌았다

    monkeypatch.setattr(job_runner, "run_job", lambda job, **kw: RESULT)
    shell.deidentify_one(JOB)

    _n, kw, _q = sent[-1]
    assert kw["worker_avg_s"] == 40.7

    journal.STATS.record(20.7, ok=True)         # 두 건째
    assert journal.STATS.avg == 30.7


def test_failure_line_carries_how_long_it_ran(sent, monkeypatch):
    """실패는 **얼마나 돌다** 실패했는지가 원인을 좁힌다.

    3초 만이면 입력을 못 받은 것이고 40초면 처리 중에 넘어진 것이다. 단계
    이름만으로는 그 구분이 안 될 때가 있다.
    """
    def boom(job, **kw):
        time.sleep(0.01)
        raise job_runner.JobError("presign 만료", transient=True, stage="download")
    monkeypatch.setattr(job_runner, "run_job", boom)

    shell.deidentify_one(JOB)

    row = next(r for r in events.read(job="vid-9") if r["event"] == "job.failed")
    assert row["seconds"] > 0 and row["stage"] == "download"
    assert row["batch"] == "kbs-0813"


def test_journal_also_goes_to_stdout_in_msa_mode(sent, monkeypatch, capsys):
    """컨테이너는 사라진다 — **파일만으로는 기록이 안 남는다.**

    KEDA 가 큐가 비면 0대로 줄이고, 그때 컨테이너 안에 쌓아 둔 저널 파일도 같이
    지워진다. stdout 으로 나가야 로그 수집기가 걷어 간다. 이게 API 모드와
    갈리는 지점이라 껍데기가 import 될 때 켜 둔다.
    """
    assert events.STDOUT is True and events.MODE == "msa"
    monkeypatch.setattr(job_runner, "run_job", lambda job, **kw: RESULT)

    shell.deidentify_one(JOB)

    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()
             if x.startswith("{")]
    assert {r["event"] for r in lines} >= {"job.started", "job.finished"}
    assert all(r["mode"] == "msa" for r in lines)


def test_stdout_lines_still_hide_the_signed_urls(sent, monkeypatch, capsys):
    """stdout 은 수집기로 그대로 실려 간다 — 여기 서명이 섞이면 유출이다."""
    monkeypatch.setattr(job_runner, "run_job", lambda job, **kw: RESULT)

    shell.deidentify_one(JOB)

    out = capsys.readouterr().out
    assert "http://x/in.mp4" not in out and "tok-9" not in out


def test_shutdown_leaves_one_summary_line(sent, monkeypatch):
    """내려갈 때 한 줄. 이 컨테이너에 대해 남는 유일한 요약이다."""
    monkeypatch.setattr(job_runner, "run_job", lambda job, **kw: RESULT)
    shell.deidentify_one(JOB)

    shell._bye(signal="TERM")

    row = next(r for r in events.read(event="worker.stopped"))
    assert row["done"] == 1 and row["avg_elapsed_s"] == 40.7
    assert row["reason"] == "TERM"


def test_batch_key_name_is_not_yet_agreed_so_we_accept_several():
    """묶음 필드 이름을 저쪽이 아직 안 정했다.

    이름이 정해질 때까지 흔한 이름을 다 받아 둔다. 못 알아들으면 그 파일은
    폴더 집계에서 통째로 새는데, 그건 나중에 되돌릴 수 없다.
    """
    for key in ("batch_id", "batch", "group_id", "folder"):
        assert journal.batch_of({key: "kbs"}) == "kbs"
    assert journal.batch_of({}) is None         # 없으면 없는 대로 돈다


def test_name_falls_back_to_video_id_not_to_the_url():
    """파일명이 안 오면 video_id 로 부른다. **URL 에서 뽑지 않는다** —
    거기엔 서명이 붙어 있고, 저널에 서명을 남기지 않는 것이 규칙이다."""
    assert journal.name_of({"video_id": "vid-1",
                            "input_url": "http://x/secret.mp4?sig=..."}) == "vid-1"
