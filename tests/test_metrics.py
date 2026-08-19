"""운영 지표 테스트.

지표는 틀려도 조용하다 — 화면에 숫자가 뜨니까 맞는 것처럼 보인다. 그래서
'무엇을 세는가' 를 여기서 못 박는다.
"""

import time


from face_anonymizer.service import metrics
from face_anonymizer.storage import s3 as s3mod
from face_anonymizer.service.jobs import Job

from test_s3 import NOW, FakeS3Client            # noqa: E402


def job(jid, status, **kw):
    return Job(id=jid, name=f"{jid}.mp4", params={}, workdir="/tmp", status=status, **kw)


def test_latency_is_the_oldest_wait_not_the_average():
    """깊이만 보면 '1분에 빠지는 100건'과 '두 시간째 멈춘 3건'이 같아 보인다."""
    now = time.time()
    jobs = [job("a", "queued", created=now - 7200),
            job("b", "queued", created=now - 10),
            job("c", "queued", created=now - 5)]

    m = metrics.queue_metrics(jobs, now=now)

    assert m["depth"] == 3
    assert m["latency"] == 7200          # 평균(2405)이 아니라 최댓값


def test_throughput_counts_only_the_last_hour():
    now = time.time()
    jobs = [job("old", "done", finished=now - 4000),
            job("new1", "done", finished=now - 100),
            job("new2", "done", finished=now - 200)]

    m = metrics.queue_metrics(jobs, now=now)

    assert m["done"] == 3                # 누적은 셋
    assert m["throughput_1h"] == 2       # 최근 한 시간은 둘


def test_retried_counts_jobs_that_needed_more_than_one_attempt():
    jobs = [job("a", "done", attempts=1), job("b", "done", attempts=3),
            job("c", "failed", attempts=2)]
    assert metrics.queue_metrics(jobs)["retried"] == 2


def test_empty_queue_reports_zero_not_none():
    """화면이 '—' 와 0 을 구분해서 그리므로 None 을 흘리면 안 된다."""
    m = metrics.queue_metrics([])
    assert m["depth"] == 0 and m["latency"] == 0 and m["avg_seconds"] == 0


def make_store(objects):
    return s3mod.S3Store(bucket="b", client=FakeS3Client(objects),
                         output_prefix="v1/results/face/", root_prefix="v1/input/")


def test_folder_progress_compares_input_with_results():
    """진척률은 큐가 아니라 버킷 기준이다 — 서버가 재시작해도 값이 같아야 한다."""
    objs = {}
    for i in range(4):
        objs[f"v1/input/kbs/K_{i:05d}_00_0000000_0034342.mp4"] = (b"x", NOW)
    for i in range(3):
        objs[f"v1/results/face/kbs_deid/K_{i:05d}_00_0000000_0034342_deid.mp4"] = (b"x", NOW)
    objs["v1/input/mbc/M_00000_00_0000000_0034342.mp4"] = (b"x", NOW)
    objs["v1/input/kbs/notes.txt"] = (b"x", NOW)          # 영상 아님

    rows = metrics.folder_progress(make_store(objs), "v1/input/")

    by = {r["name"]: r for r in rows}
    assert by["kbs"]["total"] == 4 and by["kbs"]["done"] == 3
    assert by["kbs"]["remain"] == 1 and by["kbs"]["percent"] == 75
    assert by["mbc"]["total"] == 1 and by["mbc"]["done"] == 0
    assert rows[0]["name"] == "kbs"                        # 큰 폴더가 위로


def test_folder_progress_excludes_results_from_the_input_count():
    """결과물이 입력 폴더에 섞여 있어도 '해야 할 일' 로 세면 안 된다."""
    objs = {"v1/input/kbs/K_00000_00_0000000_0034342.mp4": (b"x", NOW),
            "v1/input/kbs/K_00001_00_0000000_0034342_deid.mp4": (b"x", NOW)}
    rows = metrics.folder_progress(make_store(objs), "v1/input/")
    assert rows[0]["total"] == 1


# ── 저널이 두 경로에서 같은 뜻이어야 한다 ──────────────────────────────────
#
# API 경로와 MSA 경로는 구조가 다르지만 **기록의 뜻은 같아야 한다.** 같은 칸에
# 다른 것을 재 놓으면 나중에 둘을 나란히 놓고 비교하는 순간 틀린 결론이 나온다.

def test_finished_row_carries_the_same_meaning_on_both_paths():
    """`seconds` 는 벽시계, `pipeline_s` 는 처리만 — 양쪽 다."""
    import inspect
    from face_anonymizer.service import worker
    from face_anonymizer.msa import journal

    api = inspect.getsource(worker.run)
    msa = inspect.getsource(journal.job_finished)
    for src, who in ((api, "api"), (msa, "msa")):
        assert "pipeline_s=" in src, who
        assert "review_needed=" in src, who


def test_review_needed_reaches_the_csv_column():
    """검수로 넘어간 것을 완료 줄만 보고 구분할 수 있어야 한다."""
    from face_anonymizer import events
    from face_anonymizer.service import server
    assert "review_needed" in events.LIST_FIELDS
    col = dict((n, g) for n, g in server.EXPORT_COLUMNS)
    assert col["검수 필요"]({"review_needed": True}) == "예"
    assert col["검수 필요"]({}) == ""
    assert col["처리(초)"]({"pipeline_s": 12.5}) == 12.5


def test_one_flag_parser_for_every_switch():
    """다섯 벌이던 시절 `FA_PRELOAD=off` 만 안 먹었다 — 목록이 달랐기 때문이다."""
    import os
    from face_anonymizer import env

    for word in ("0", "false", "FALSE", "no", "off", "OFF", " off ", ""):
        os.environ["FA_TEST_FLAG"] = word
        assert env.flag("FA_TEST_FLAG", True) is False, word
    for word in ("1", "true", "yes", "on", "아무거나"):
        os.environ["FA_TEST_FLAG"] = word
        assert env.flag("FA_TEST_FLAG", False) is True, word
    del os.environ["FA_TEST_FLAG"]
    assert env.flag("FA_TEST_FLAG", True) is True


def test_stage_names_come_from_one_table():
    """화면이 자기 단계 이름표를 들고 있으면 갈라진다 — 실제로 갈라져 있었다.

    MSA 경로는 "원본 받는 중" 을, 우리 화면은 "S3 내려받기" 를 썼다. 게다가
    화면 표에는 `track` 이 없어서 추적 중에는 '준비' 라고 떴고, 저장소를 고를
    수 있게 된 뒤로는 'S3' 라는 말 자체가 틀렸다.
    """
    import pathlib
    from face_anonymizer import progress

    # 모든 단계에 이름이 있어야 한다 — 하나라도 비면 화면이 '준비' 로 흐른다.
    for stage, _base, _w in progress.STAGE_SPAN:
        assert progress.label(stage), stage

    # 그리고 화면은 자기 표를 갖고 있으면 안 된다.
    html = (pathlib.Path(__file__).resolve().parent.parent
            / "face_anonymizer/service/static/index.html").read_text(encoding="utf-8")
    assert "STAGE_TEXT" not in html
    assert "stage_label" in html


def test_snapshot_carries_the_stage_name(tmp_path, monkeypatch):
    """화면이 쓰려면 스냅샷에 실려 나와야 한다."""
    from face_anonymizer.service import config, jobs as jobsmod
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    j = jobsmod.Job(id="x", name="a.mp4", params={}, workdir=str(tmp_path))
    j.status, j.stage = "running", "detect"
    snap = jobsmod.snapshot(j, 0)
    assert snap["stage"] == "detect"
    assert snap["stage_label"] == "얼굴 찾는 중"
