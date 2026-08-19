"""운영 지표 테스트.

지표는 틀려도 조용하다 — 화면에 숫자가 뜨니까 맞는 것처럼 보인다. 그래서
'무엇을 세는가' 를 여기서 못 박는다.
"""

import pathlib
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


# ── 같아야 하는 것이 갈라지지 않게 (docs/issues/023) ─────────────────────────

def test_metrics_does_not_lose_jobs_waiting_for_review():
    """**검수 대기는 처리가 끝난 것이다.** 남은 것은 사람의 확인이다.

    상태 목록을 metrics 가 따로 적어 두는 바람에 `review` 가 빠져 있었다.
    그래서 검출 0건으로 검수에 걸린 영상이 처리량·평균에서 통째로 사라졌다 —
    같은 화면의 `recent_stats()` 는 done+review 를 세므로 **두 평균이 서로
    다른 모집단**을 쓰고 있었다.
    """
    from face_anonymizer.service import jobs as jobsmod
    now = time.time()

    def job(status, seconds=None):
        return jobsmod.Job(id=status, name="a.mp4", params={}, workdir="",
                           status=status, finished=now - 10,
                           result={"seconds": seconds} if seconds else {})

    m = metrics.queue_metrics([job("done", 10), job("review", 20)], now=now)

    assert m["done"] == 1 and m["review"] == 1
    assert m["throughput_1h"] == 2, "검수 대기가 처리량에서 빠졌다"
    assert m["avg_seconds"] == 15, "검수 대기가 평균에서 빠졌다"


def test_metrics_covers_every_status_there_is():
    """상태를 하나 추가하면 여기도 따라와야 한다 — 손으로 적으면 안 따라온다."""
    from face_anonymizer.service import jobs as jobsmod
    src = (pathlib.Path(metrics.__file__)).read_text(encoding="utf-8")
    assert "jobsmod.STATUSES" in src or "jobs.STATUSES" in src, (
        "상태 목록을 metrics 안에 다시 적어 두면 언젠가 갈라진다")
    assert len(jobsmod.STATUSES) == 6


def test_every_thrown_stage_has_something_to_say():
    """`stage="oom"` 을 던지는데 문구 표에 없어서 '알 수 없는 오류' 가 나갔다.

    같은 상황을 API 경로는 "GPU 메모리가 부족합니다 / batch_size 나 imgsz 를
    낮추면 통과할 수 있습니다" 라고 정확히 말한다. 014 가 판정을 통일했는데
    문구 표만 안 따라온 것이다.
    """
    import re

    from face_anonymizer import job_runner
    src = pathlib.Path(job_runner.__file__).read_text(encoding="utf-8")
    thrown = set(re.findall(r'stage="([a-z]+)"', src)) - {""}
    missing = thrown - set(job_runner.STAGE_FACE)
    assert not missing, f"문구가 없는 단계: {sorted(missing)}"


def test_a_failure_says_whether_it_was_temporary(tmp_path, monkeypatch):
    """저널 문장은 `transient` 만 보고 '일시적/영구' 를 찍는다.

    API 경로가 이 필드를 안 실어서 **모든 실패가 '영구' 로 기록**됐다.
    재시도를 세 번 다 쓰고 죽은 일시적 오류까지.
    """
    import re

    from face_anonymizer import events
    from face_anonymizer.service import worker

    src = pathlib.Path(worker.__file__).read_text(encoding="utf-8")
    at = src.find('events.emit("job.failed"')
    assert at >= 0, "job.failed 를 내는 곳을 못 찾았다"
    # 괄호가 여러 겹이라 정규식으로 끝을 찾지 않는다 — 그 호출이 끝나는
    # 지점(다음 빈 줄이나 다음 문장)까지의 창을 그대로 본다.
    call = src[at:src.find("\n\n", at)]
    assert "transient" in call, "실패에 transient 가 빠졌다"
    # 목록·CSV 에서 사유별로 셀 수 있어야 한다.
    assert "code" in events.LIST_FIELDS
