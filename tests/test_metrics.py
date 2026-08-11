"""운영 지표 테스트.

지표는 틀려도 조용하다 — 화면에 숫자가 뜨니까 맞는 것처럼 보인다. 그래서
'무엇을 세는가' 를 여기서 못 박는다.
"""

import time

import pytest

from face_anonymizer import metrics
from face_anonymizer import s3 as s3mod
from face_anonymizer.server import Job

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
