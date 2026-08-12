"""HTTP API 테스트.

파이프라인 테스트와 같은 원칙 — 가짜 검출기를 주입해 torch/가중치 없이
업로드부터 다운로드까지 전 구간을 돈다. 서빙 의존성이 안 깔린 환경에서는
통째로 skip 한다 (코어만 쓰는 사람에게 fastapi 를 강요하지 않는다).
"""

import os
import time

import pytest

from conftest import FakeDetector, face_rect, region_is_obscured, read_frames

pytest.importorskip("fastapi", reason="pip install -r requirements-serve.txt")
pytest.importorskip("multipart", reason="pip install -r requirements-serve.txt")
pytest.importorskip("httpx", reason="pip install -r requirements-dev.txt")

from fastapi.testclient import TestClient           # noqa: E402

from face_anonymizer import VideoAnonymizer                # noqa: E402
from face_anonymizer.service import jobs as jobsmod        # noqa: E402
from face_anonymizer.service import config, server, worker          # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """작업 디렉터리와 전역 상태를 테스트마다 격리한다."""
    jobs = tmp_path / "jobs"
    monkeypatch.setattr(config, "JOBS_DIR", str(jobs))
    monkeypatch.setattr(jobsmod, "JOBS", {})
    monkeypatch.setattr(worker, "_anonymizer", None)
    monkeypatch.setattr(worker, "current", None)
    monkeypatch.setattr(worker, "model_error", None)

    def attach(size, miss_frames=()):
        anon = VideoAnonymizer(detector=FakeDetector(size, miss_frames))
        monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)
        monkeypatch.setattr(worker, "_anonymizer", anon)
        return anon

    c = TestClient(server.app)
    c.attach = attach
    return c


def wait(c, jid, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = c.get(f"/api/jobs/{jid}").json()
        if s["status"] in ("done", "failed", "cancelled"):
            return s
        time.sleep(0.02)
    raise AssertionError(f"작업이 {timeout}s 안에 끝나지 않았다")


def submit(c, path, **form):
    with open(path, "rb") as f:
        return c.post("/api/jobs",
                      files={"file": ("clip.mp4", f, "video/mp4")}, data=form)


def job_id(response):
    """제출 응답에서 작업 id. 한 건이든 여러 건이든 폴더든 응답 형태는 같다."""
    assert response.status_code == 202, response.text
    return response.json()["accepted"][0]["id"]


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>face-anonymizer</title>" in r.text


def test_health_survives_injected_detector(client):
    """검출기는 주입 가능하다 — health 가 FaceDetector 속성을 단정하면 안 된다."""
    client.attach((320, 240))
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["detector"] == "FakeDetector"


@pytest.mark.parametrize("files,data", [
    ({"file": ("a.mp4", b"x")}, {"method": "nope"}),      # 없는 방식
    pytest.param({"file": ("a.txt", b"x")}, {}, id="bad-ext", marks=[]),
    ({"file": ("a.mp4", b"")}, {}),                       # 빈 파일
    ({"file": ("a.mp4", b"x")}, {"conf": "1.5"}),         # 범위 밖 임계값
])
def test_rejects_bad_input(client, files, data):
    client.attach((320, 240))            # 준비된 서버 기준 (아니면 503 이 먼저)
    r = client.post("/api/jobs", files=files, data=data)
    # 형식이 안 맞는 건 415, 나머지는 400 — 둘 다 problem+json 이어야 한다
    assert r.status_code in (400, 415), r.text
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["code"]


def test_bad_upload_leaves_no_workdir(client, tmp_path):
    """거절된 업로드가 작업 디렉터리를 남기면 디스크가 조용히 찬다."""
    client.post("/api/jobs", files={"file": ("a.mp4", b"")})
    jobs = tmp_path / "jobs"
    assert not jobs.exists() or not list(jobs.iterdir())


def test_download_before_done_is_409(client, make_video):
    path, n, size = make_video(frames=6)
    client.attach(size)
    jid = job_id(submit(client, path))
    # 완료 전 상태를 강제로 만들어 둔다 (실제 진행 중 상태를 잡으려면 경쟁이 생긴다)
    jobsmod.JOBS[jid].status = "running"
    assert client.get(f"/api/jobs/{jid}/download").status_code == 409
    assert client.delete(f"/api/jobs/{jid}").status_code == 409


def test_imgsz_is_range_clamped(client, make_video):
    """서버는 범위만 본다. stride 배수 맞추기는 검출기 몫이다(규칙을 두 벌 두지 않는다)."""
    path, n, size = make_video(frames=6)
    client.attach(size)
    jid = job_id(submit(client, path, imgsz="99999"))
    assert jobsmod.JOBS[jid].params["imgsz"] == 2048
    wait(client, jid)


def test_full_lifecycle_and_no_leak(client, tmp_path, make_video):
    """업로드 → 처리 → 다운로드 → 삭제. 검출기가 놓친 프레임도 가려져야 한다."""
    path, n, size = make_video(frames=30)
    client.attach(size, miss_frames={7, 8})

    r = submit(client, path, method="mosaic", batch_size="8", keep_audio="false")
    assert r.status_code == 202
    jid = job_id(r)

    s = wait(client, jid)
    assert s["status"] == "done", s.get("error")
    res = s["result"]
    assert res["frames"] == n
    assert res["filled_boxes"] >= 2          # 놓친 두 프레임을 보간이 메웠다
    assert res["fps"] > 0 and res["seconds"] > 0
    # 단계 시간은 짧은 클립에서 반올림으로 0 이 될 수 있어 개별 값은 안 본다.
    # 각 단계가 반올림된 값이라 합이 총계를 몇 ms 넘길 수 있다.
    assert sum(res["timing"].values()) <= res["seconds"] + 0.01

    r = client.get(f"/api/jobs/{jid}/download")
    assert r.status_code == 200
    out = tmp_path / "out.mp4"
    out.write_bytes(r.content)

    frames = read_frames(str(out))
    assert len(frames) == n
    leaked = [i for i, f in enumerate(frames)
              if not region_is_obscured(f, face_rect(i, *size))]
    assert not leaked, f"원본 얼굴이 남은 프레임: {leaked}"

    assert client.delete(f"/api/jobs/{jid}").status_code == 204
    assert client.get(f"/api/jobs/{jid}").status_code == 404
    assert not (tmp_path / "jobs" / jid).exists()




# ── 상태 영속화 ──────────────────────────────────────────────────────────────
#
# 작업 상태를 전역 dict 에만 두면 (a) 재시작 시 전부 사라져 폴링 중인 클라이언트가
# 404 를 받고, (b) --workers 2 로 띄우면 업로드와 폴링이 다른 프로세스로 가서
# 계속 404 가 난다. 아래가 그 회귀다.

def test_job_state_is_written_to_disk(client, tmp_path, make_video):
    path, n, size = make_video(frames=10)
    client.attach(size)
    jid = job_id(submit(client, path))
    wait(client, jid)

    state = tmp_path / "jobs" / jid / "job.json"
    assert state.exists(), "작업 상태가 디스크에 없다"
    import json
    assert json.loads(state.read_text(encoding="utf-8"))["status"] == "done"


def test_survives_restart(client, tmp_path, make_video):
    """프로세스 메모리가 비어도 조회와 다운로드가 된다 (재시작 / 다른 워커)."""
    path, n, size = make_video(frames=10)
    client.attach(size)
    jid = job_id(submit(client, path))
    wait(client, jid)

    jobsmod.JOBS.clear()                       # 재시작 또는 다른 워커의 시야

    r = client.get(f"/api/jobs/{jid}")
    assert r.status_code == 200
    assert r.json()["status"] == "done"
    assert r.json()["result"]["frames"] == n
    assert client.get(f"/api/jobs/{jid}/download").status_code == 200
    assert any(j["id"] == jid for j in client.get("/api/jobs").json())


def test_orphaned_running_job_is_marked_failed(client, tmp_path, make_video):
    """죽은 프로세스가 남긴 running 상태를 영원히 폴링하게 두면 안 된다."""
    path, n, size = make_video(frames=10)
    client.attach(size)
    jid = job_id(submit(client, path))
    wait(client, jid)

    j = jobsmod.JOBS[jid]
    j.status, j.finished = "running", 0.0
    jobsmod.save_job(j)
    jobsmod.JOBS.clear()                       # 프로세스가 죽은 상태

    assert jobsmod.recover_orphans() == []      # 재큐할 것은 없다
    s = client.get(f"/api/jobs/{jid}").json()
    assert s["status"] == "failed"
    assert s["error"]["code"] == "interrupted"


# ── 재시작 시 대기열 복구 (docs/issues/002) ─────────────────────────────────
#
# queued 와 running 은 성격이 다르다. running 은 중간에 끊겨 결과를 믿을 수
# 없지만, queued 는 아직 아무 일도 일어나지 않았다. 예전에는 둘을 한 덩어리로
# 보고 전부 실패로 만들어서, 500건을 넣어 두고 배포하면 통째로 날아갔다.

def orphan(jid, status, created=0.0):
    """죽은 프로세스가 남긴 상태 파일을 만든다."""
    import os
    d = os.path.join(config.JOBS_DIR, jid)
    os.makedirs(d, exist_ok=True)
    j = jobsmod.Job(id=jid, name=f"{jid}.mp4", params={}, workdir=d,
                    status=status, created=created or time.time())
    jobsmod.save_job(j)
    return j


def test_queued_jobs_are_requeued_not_failed(client, tmp_path, monkeypatch):
    """아직 시작도 안 한 작업을 실패로 만들면, 배포할 때마다 큐가 날아간다."""
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(jobsmod, "JOBS", {})
    orphan("q1", "queued", created=1.0)
    orphan("q2", "queued", created=2.0)
    orphan("r1", "running", created=3.0)

    resumed = jobsmod.recover_orphans()

    assert [j.id for j in resumed] == ["q1", "q2"]          # 만들어진 순서 그대로
    assert jobsmod.find_job("q1").status == "queued"
    assert jobsmod.find_job("r1").status == "failed"        # 이건 중간에 끊겼다


def test_requeue_does_not_burn_a_retry(client, tmp_path, monkeypatch):
    """재시작은 그 작업이 실패한 것이 아니다. 배포 세 번에 재시도가 소진되면 안 된다."""
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(jobsmod, "JOBS", {})
    j = orphan("q1", "queued")
    j.attempts = 1
    jobsmod.save_job(j)

    jobsmod.recover_orphans()

    assert jobsmod.find_job("q1").attempts == 1


def test_recovery_can_be_turned_off(client, tmp_path, monkeypatch):
    """--workers N 이면 한 프로세스만 복구해야 중복 처리가 안 생긴다."""
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(jobsmod, "JOBS", {})
    monkeypatch.setattr(config, "RECOVER", False)
    orphan("q1", "queued")
    orphan("r1", "running")

    assert jobsmod.recover_orphans() == []
    assert jobsmod.find_job("r1").status == "running"       # 손대지 않는다


def test_resume_puts_them_back_on_the_worker(client, tmp_path, monkeypatch):
    """상태만 되돌리고 제출을 안 하면 영원히 대기다."""
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(jobsmod, "JOBS", {})
    submitted = []
    monkeypatch.setattr(worker.EXEC, "submit",
                        lambda fn, *a: submitted.append(a[0]))
    orphan("q1", "queued", created=1.0)
    orphan("q2", "queued", created=2.0)

    assert worker.resume_orphans() == 2
    assert submitted == ["q1", "q2"]


def test_sweep_removes_expired_jobs(client, tmp_path, make_video, monkeypatch):
    """정리가 새 업로드에만 의존하면 디스크가 안 비워진다."""
    path, n, size = make_video(frames=10)
    client.attach(size)
    jid = job_id(submit(client, path))
    wait(client, jid)

    monkeypatch.setattr(config, "JOB_TTL", 1)
    jobsmod.JOBS[jid].finished = time.time() - 10
    jobsmod.save_job(jobsmod.JOBS[jid])

    assert jobsmod.sweep() == 1
    assert not (tmp_path / "jobs" / jid).exists()
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_missing_output_reports_410_not_500(client, tmp_path, make_video):
    """보관 기간이 지나 파일만 사라진 경우를 구분해서 알린다."""
    path, n, size = make_video(frames=10)
    client.attach(size)
    jid = job_id(submit(client, path))
    wait(client, jid)

    os.remove(jobsmod.JOBS[jid].output)
    assert client.get(f"/api/jobs/{jid}/download").status_code == 410


@pytest.mark.parametrize("bad", ["../etc", "..", "a/b", ".hidden"])
def test_job_id_traversal_is_rejected(client, bad):
    assert client.get(f"/api/jobs/{bad}").status_code in (404, 400, 405)


# ── 한 번에 한 편 · 대기열 없음 ──────────────────────────────────────────────
#
# 큐는 바깥(오케스트레이터)에 둔다. 안에 쌓아 두면 바깥에서 이 인스턴스의 실제
# 부하를 알 수 없고, 재시도할지 다른 인스턴스로 보낼지 결정하지 못한다.

class SlowDetector:
    """작업이 확실히 진행 중인 상태를 만들기 위한 느린 검출기."""

    def __init__(self, delay=0.05):
        self.delay = delay

    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        time.sleep(self.delay)
        return [[] for _ in frames]


def test_queue_is_unlimited_by_default(client, make_video, monkeypatch):
    """전체 수행처럼 한꺼번에 여러 건 넣는 게 정상이다."""
    assert config.QUEUE_MAX == 0
    path, n, size = make_video(frames=6)
    client.attach(size)

    ids = [job_id(submit(client, path)) for _ in range(5)]
    assert len(ids) == 5
    for jid in ids:
        assert wait(client, jid, timeout=60)["status"] == "done"


def test_rejects_when_disk_is_low(client, make_video, monkeypatch):
    """대기 중인 작업은 입력 파일을 들고 있다. 진짜 제약은 개수가 아니라 디스크다."""
    path, n, size = make_video(frames=4)
    client.attach(size)
    monkeypatch.setattr(jobsmod, "free_mb", lambda: 10)
    monkeypatch.setattr(config, "MIN_FREE_MB", 2048)

    r = submit(client, path)
    assert r.status_code == 507
    assert r.headers.get("Retry-After")


def test_list_is_bounded_and_filterable(client, make_video):
    path, n, size = make_video(frames=4)
    client.attach(size)
    for _ in range(3):
        wait(client, job_id(submit(client, path)), timeout=60)

    assert len(client.get("/api/jobs?limit=2").json()) == 2
    done = client.get("/api/jobs?status=done").json()
    assert done and all(j["status"] == "done" for j in done)
    assert client.get("/api/jobs?status=failed").json() == []


def test_queue_max_still_works_when_set(client, make_video, monkeypatch):
    monkeypatch.setattr(config, "QUEUE_MAX", 2)
    path, n, size = make_video(frames=40)
    anon = VideoAnonymizer(detector=SlowDetector())
    monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(worker, "_anonymizer", anon)

    ids = []
    codes = []
    for _ in range(5):
        r = submit(client, path, batch_size="1")
        codes.append(r.status_code)
        if r.status_code == 202:
            ids.append(r.json()["accepted"][0]["id"])

    assert 202 in codes and 429 in codes, codes
    rejected = [c for c in codes if c == 429]
    assert rejected, "상한을 넘겨도 계속 받았다"
    for jid in ids:
        wait(client, jid, timeout=90)


def test_transient_failure_is_retried(client, make_video, monkeypatch):
    """일시적 오류는 다시 큐에 넣는다."""
    monkeypatch.setattr(config, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(config, "RETRY_DELAYS", (0.01,))   # 테스트는 기다리지 않는다
    path, n, size = make_video(frames=6)
    calls = {"n": 0}

    class Flaky:
        def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("CUDA out of memory")
            return [[] for _ in frames]

    anon = VideoAnonymizer(detector=Flaky())
    monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(worker, "_anonymizer", anon)

    jid = job_id(submit(client, path))
    s = wait(client, jid, timeout=60)

    assert s["status"] == "done", s.get("error")
    assert s["attempts"] == 3


def test_retries_are_exhausted_then_failed(client, make_video, monkeypatch):
    monkeypatch.setattr(config, "MAX_ATTEMPTS", 2)
    monkeypatch.setattr(config, "RETRY_DELAYS", (0.01,))
    path, n, size = make_video(frames=6)

    class AlwaysBroken:
        def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
            raise RuntimeError("일시적인 척하는 영구 오류")

    anon = VideoAnonymizer(detector=AlwaysBroken())
    monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(worker, "_anonymizer", anon)

    jid = job_id(submit(client, path))
    s = wait(client, jid, timeout=60)

    assert s["status"] == "failed"
    assert s["attempts"] == 2
    assert "일시적인 척하는" in s["error"]["detail"]


def test_permanent_error_is_not_retried(client, tmp_path, monkeypatch):
    """깨진 입력은 세 번 돌려도 결과가 같다 — 바로 실패로 둔다."""
    monkeypatch.setattr(config, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(config, "RETRY_DELAYS", (0.01,))
    client.attach((320, 240))
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video at all" * 100)

    jid = job_id(submit(client, broken))
    s = wait(client, jid, timeout=30)

    assert s["status"] == "failed"
    assert s["attempts"] == 1, "재시도하면 안 된다"


def test_accepts_again_after_finishing(client, make_video):
    path, n, size = make_video(frames=8)
    client.attach(size)

    wait(client, job_id(submit(client, path)))
    wait(client, job_id(submit(client, path)))


def test_status_endpoint_reports_ready_and_queue(client, make_video):
    path, n, size = make_video(frames=8)
    client.attach(size)

    s = client.get("/api/status").json()
    assert s["ready"] is True and s["busy"] is False
    assert s["queued"] == 0 and s["free_mb"] > 0

    jid = job_id(submit(client, path))
    wait(client, jid)
    assert client.get("/api/status").json()["busy"] is False


def test_health_is_503_before_model_is_ready(client):
    worker._anonymizer = None
    r = client.get("/api/health")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_upload_rejected_when_model_failed(client, make_video):
    path, n, size = make_video(frames=4)
    worker._anonymizer = None
    worker.model_error = "RuntimeError: 가중치 없음"

    r = submit(client, path)
    assert r.status_code == 503
    assert "가중치 없음" in r.json()["detail"]


# ── 기본값 ───────────────────────────────────────────────────────────────────
#
# 호출하는 쪽은 입력만 주면 된다. 튜닝된 값은 서비스가 들고 있어야지, 호출자마다
# 들고 다니면 어느 설정으로 처리됐는지가 호출 지점마다 달라진다.

def test_submit_without_any_parameter(client, make_video):
    path, n, size = make_video(frames=6)
    client.attach(size)

    with open(path, "rb") as f:
        r = client.post("/api/jobs", files={"file": ("clip.mp4", f, "video/mp4")})
    jid = job_id(r)

    assert jobsmod.JOBS[jid].params == server.JOB_DEFAULTS
    assert wait(client, jid)["status"] == "done"


def test_defaults_endpoint_matches_what_is_used(client):
    d = client.get("/api/defaults").json()
    assert d == server.JOB_DEFAULTS
    for k in ("method", "conf", "imgsz", "batch_size", "keep_audio"):
        assert k in d


def test_given_parameter_overrides_only_that_one(client, make_video):
    path, n, size = make_video(frames=6)
    client.attach(size)
    jid = job_id(submit(client, path, conf="0.4"))

    p = jobsmod.JOBS[jid].params
    assert p["conf"] == 0.4
    assert p["method"] == server.JOB_DEFAULTS["method"]
    assert p["batch_size"] == server.JOB_DEFAULTS["batch_size"]
    wait(client, jid)


def test_defaults_are_a_copy_not_the_live_dict(client, make_video):
    """작업이 기본값 dict 를 공유하면 한 건의 변경이 이후 전부에 번진다."""
    path, n, size = make_video(frames=6)
    client.attach(size)
    jid = job_id(submit(client, path))

    jobsmod.JOBS[jid].params["conf"] = 0.99
    assert server.JOB_DEFAULTS["conf"] != 0.99
    wait(client, jid)


# ── 오류 형식 (RFC 9457) ─────────────────────────────────────────────────────
#
# 호출하는 쪽은 재시도할지, 다른 인스턴스로 보낼지, 사람을 불러야 할지를 정해야
# 한다. 한국어 문장을 파싱해서 정할 수는 없다.

from face_anonymizer.service import errors            # noqa: E402


def test_errors_are_problem_json(client):
    client.attach((320, 240))
    r = client.post("/api/jobs", data={"s3_key": "x.mp4", "conf": "1.5"})

    assert r.headers["content-type"].startswith("application/problem+json")
    b = r.json()
    for k in ("type", "title", "status", "code", "retryable"):
        assert k in b, b
    assert b["status"] == r.status_code
    assert b["type"].startswith("/problems/")


def test_error_carries_actionable_fields(client):
    client.attach((320, 240))
    b = client.post("/api/jobs", data={"s3_key": "x.mp4",
                                       "method": "nope"}).json()
    assert b["code"] == "invalid_input"
    assert b["field"] == "method"
    assert "mosaic" in b["allowed"]
    assert b["hint"]


def test_retryable_errors_carry_retry_after(client, make_video, monkeypatch):
    monkeypatch.setattr(jobsmod, "free_mb", lambda: 1)
    monkeypatch.setattr(config, "MIN_FREE_MB", 2048)
    path, n, size = make_video(frames=4)
    client.attach(size)

    r = submit(client, path)
    assert r.status_code == 507
    assert r.json()["retryable"] is True
    assert r.headers.get("Retry-After")


def test_missing_and_conflicting_input_are_distinct(client, make_video):
    client.attach((320, 240))
    assert client.post("/api/jobs", data={}).json()["code"] == "missing_input"

    src, n, size = make_video(name="x.mp4", frames=4)
    with open(src, "rb") as f:
        b = client.post("/api/jobs", files={"file": ("x.mp4", f, "video/mp4")},
                        data={"s3_key": "a.mp4"}).json()
    assert b["code"] == "conflicting_input"


def test_problem_catalog_is_published(client):
    items = client.get("/api/problems").json()["problems"]
    codes = {p["code"] for p in items}
    for expected in ("queue_full", "not_ready", "job_not_found",
                     "s3_access_denied", "video_unreadable", "cancelled"):
        assert expected in codes
    assert all(p["type"] and p["title"] for p in items)


def test_unknown_route_is_also_problem_json(client):
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")


def test_job_failure_carries_a_code(client, make_video, monkeypatch):
    """실패 사유를 코드로 남긴다 — 문자열만 있으면 분기할 수 없다."""
    path, n, size = make_video(frames=6)

    class Broken:
        def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
            raise RuntimeError("CUDA out of memory")

    anon = VideoAnonymizer(detector=Broken())
    monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(worker, "_anonymizer", anon)
    monkeypatch.setattr(config, "MAX_ATTEMPTS", 1)

    jid = job_id(submit(client, path))
    s = wait(client, jid, timeout=60)

    assert s["status"] == "failed"
    assert s["error"]["code"] == "gpu_out_of_memory"
    assert s["error"]["hint"]


# ── 취소 ─────────────────────────────────────────────────────────────────────

def test_cancel_queued_job(client, make_video, monkeypatch):
    path, n, size = make_video(frames=40)
    anon = VideoAnonymizer(detector=SlowDetector(0.05))
    monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(worker, "_anonymizer", anon)

    first = job_id(submit(client, path, batch_size="1"))
    second = job_id(submit(client, path, batch_size="1"))

    r = client.post(f"/api/jobs/{second}/cancel")
    assert r.status_code == 200
    assert client.get(f"/api/jobs/{second}").json()["status"] == "cancelled"

    wait(client, first, timeout=90)
    assert client.get(f"/api/jobs/{second}").json()["status"] == "cancelled"


def test_cancel_running_job_stops_it(client, make_video, monkeypatch):
    """수행 중인 작업도 다음 진행 보고에서 끊긴다."""
    path, n, size = make_video(frames=60)
    anon = VideoAnonymizer(detector=SlowDetector(0.03))
    monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(worker, "_anonymizer", anon)

    jid = job_id(submit(client, path, batch_size="1"))
    for _ in range(200):                       # 실제로 돌기 시작할 때까지
        if client.get(f"/api/jobs/{jid}").json()["status"] == "running":
            break
        time.sleep(0.02)

    client.post(f"/api/jobs/{jid}/cancel")
    s = wait(client, jid, timeout=60)
    assert s["status"] == "cancelled"
    assert s["error"]["code"] == "cancelled"


def test_cannot_cancel_finished_job(client, make_video):
    path, n, size = make_video(frames=6)
    client.attach(size)
    jid = job_id(submit(client, path))
    wait(client, jid)

    r = client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 409
    assert r.json()["code"] == "job_not_cancellable"


# ── 보관 정책 ────────────────────────────────────────────────────────────────

def test_failed_jobs_survive_sweep(client, make_video, monkeypatch):
    """배치에서 몇 건 실패했을 때 원인을 볼 수 있어야 한다."""
    path, n, size = make_video(frames=6)

    class Broken:
        def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
            raise RuntimeError("망가짐")

    anon = VideoAnonymizer(detector=Broken())
    monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(worker, "_anonymizer", anon)
    monkeypatch.setattr(config, "MAX_ATTEMPTS", 1)
    monkeypatch.setattr(config, "JOB_TTL", 1)
    monkeypatch.setattr(config, "FAILED_TTL", 0)       # 기본값 = 안 지움

    jid = job_id(submit(client, path))
    wait(client, jid, timeout=60)
    jobsmod.JOBS[jid].finished = time.time() - 9999
    jobsmod.save_job(jobsmod.JOBS[jid])

    jobsmod.sweep()
    assert client.get(f"/api/jobs/{jid}").status_code == 200


# ── 로컬 디스크 정리 (docs/issues/001) ──────────────────────────────────────

def test_direct_upload_keeps_its_local_copy(client, make_video):
    """S3 를 안 쓰는 업로드는 로컬이 유일한 사본이다. 지우면 결과가 사라진다."""
    path, n, size = make_video(frames=6)
    client.attach(size)
    jid = job_id(submit(client, path))
    s = wait(client, jid)

    assert s["status"] == "done"
    left = sorted(os.listdir(jobsmod.JOBS[jid].workdir))
    assert left != ["job.json"], "업로드분까지 지우면 안 된다"
    assert client.get(f"/api/jobs/{jid}/download").status_code == 200


def test_sweep_removes_temp_dirs_left_by_a_killed_process(client, tmp_path, monkeypatch):
    """파이프라인의 .anon-* 는 finally 에서 지워진다. 강제 종료되면 남는다."""
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    d = tmp_path / "abc123"
    (d / ".anon-dead").mkdir(parents=True)
    (d / ".anon-dead" / "noaudio.mp4").write_bytes(b"x" * 100)
    (d / "job.json").write_text("{}", encoding="utf-8")

    assert jobsmod.sweep_temp() == 1
    assert not (d / ".anon-dead").exists()
    assert (d / "job.json").exists(), "기록까지 지우면 안 된다"


def test_sweep_does_not_touch_a_running_jobs_temp_dir(client, tmp_path, monkeypatch):
    """지금 쓰고 있는 임시 디렉터리를 지우면 그 작업이 깨진다."""
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    jid = "live0001"
    d = tmp_path / jid
    (d / ".anon-live").mkdir(parents=True)
    j = jobsmod.Job(id=jid, name="x.mp4", params={}, workdir=str(d), status="running")
    monkeypatch.setattr(jobsmod, "JOBS", {jid: j})

    assert jobsmod.sweep_temp() == 0
    assert (d / ".anon-live").exists()


def test_drop_media_keeps_the_record(tmp_path):
    d = tmp_path / "j1"
    d.mkdir()
    (d / "input.mp4").write_bytes(b"x" * 5000)
    (d / "out_deid.mp4").write_bytes(b"y" * 7000)
    (d / "job.json").write_text("{}", encoding="utf-8")
    j = jobsmod.Job(id="j1", name="x.mp4", params={}, workdir=str(d))

    freed = jobsmod.drop_media(j)

    assert freed == 12000
    assert sorted(os.listdir(d)) == ["job.json"]


# ── 재시도 백오프와 보류 (docs/issues/003) ──────────────────────────────────
#
# 세 번을 같은 순간에 시도하면 세 번 다 같은 세상을 본다. 시도는 했는데
# 기다리지는 않은 것이다. 실제로 리전을 틀리게 준 서버에서 attempts 가 첫
# 조회에 이미 3 이었다 — 연결 거부가 즉시 오니까 셋이 몇 밀리초 만에 끝났다.

def test_backoff_follows_the_configured_list():
    """5 -> 30 -> 60. 처음 1회 + 재시도 3회로 95초짜리 창을 덮는다."""
    assert config.RETRY_DELAYS == (5, 30, 60)
    assert config.MAX_ATTEMPTS == 4
    assert sum(config.RETRY_DELAYS) == 95

    for attempt, base in ((1, 5), (2, 30), (3, 60)):
        got = [worker.backoff_for(attempt) for _ in range(50)]
        assert all(base * 0.8 <= x <= base * 1.2 for x in got), (attempt, min(got), max(got))


def test_backoff_reuses_the_last_delay_when_attempts_exceed_the_list(monkeypatch):
    """목록이 짧아도 계산이 깨지면 안 된다."""
    monkeypatch.setattr(config, "RETRY_DELAYS", (10,))
    assert 8 <= worker.backoff_for(1) <= 12
    assert 8 <= worker.backoff_for(9) <= 12


def test_backoff_has_jitter():
    """여러 건이 같은 순간에 실패했을 때 회복 순간에 몰리지 않게 흩뜨린다."""
    got = {round(worker.backoff_for(1), 4) for _ in range(30)}
    assert len(got) > 25, "간격이 전부 같으면 지터가 없는 것이다"


def test_retry_is_scheduled_not_slept(client, make_video, monkeypatch):
    """자면 뒤에 쌓인 정상 작업까지 멈춘다. 예약이어야 한다."""
    monkeypatch.setattr(config, "MAX_ATTEMPTS", 2)
    delays = []
    monkeypatch.setattr(worker, "schedule",
                        lambda jid, delay: delays.append(delay))

    path, n, size = make_video(frames=6)
    client.attach(size)                         # 먼저 붙이고

    class Broken:
        def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
            raise RuntimeError("일시적인 척")
    anon = VideoAnonymizer(detector=Broken())
    monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)   # 덮어쓴다

    jid = job_id(submit(client, path))
    for _ in range(200):
        if delays:
            break
        time.sleep(0.02)

    assert delays and 4 <= delays[0] <= 6, delays   # 첫 간격 5초 ±20%
    s = client.get(f"/api/jobs/{jid}").json()
    assert s["waiting"] == "retry" and s["wait_left"] > 0


def test_defer_does_not_burn_an_attempt(tmp_path, monkeypatch):
    """디스크 부족은 이 작업이 실패한 것이 아니다. 세 번 확인했다고 포기할 일이 아니다."""
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(worker, "schedule", lambda jid, delay: None)
    j = jobsmod.Job(id="d1", name="x.mp4", params={}, workdir=str(tmp_path / "d1"))
    os.makedirs(j.workdir, exist_ok=True)
    j.attempts = 1

    worker.defer(j, errors.INSUFFICIENT_STORAGE, "남은 공간 10MB")

    assert j.attempts == 1                      # 그대로
    assert j.status == "queued"
    assert j.waiting == "defer"
    assert j.not_before > time.time()


def test_defer_gives_up_after_the_cap(tmp_path, monkeypatch):
    """영구히 찬 디스크를 영원히 숨기면 안 된다."""
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DEFER_MAX_SEC", 60)
    monkeypatch.setattr(worker, "schedule", lambda jid, delay: None)
    j = jobsmod.Job(id="d2", name="x.mp4", params={}, workdir=str(tmp_path / "d2"))
    os.makedirs(j.workdir, exist_ok=True)
    j.deferred_since = time.time() - 120        # 2분째 기다리는 중

    worker.defer(j, errors.INSUFFICIENT_STORAGE, "남은 공간 10MB")

    assert j.status == "failed"
    assert j.error["code"] == "insufficient_storage"
    assert j.error["policy"] == "deferred_too_long"


def test_starting_clears_the_wait_marks(client, make_video, monkeypatch):
    """보류가 풀렸는데 화면에 '보류' 가 남아 있으면 안 된다."""
    path, n, size = make_video(frames=6)
    client.attach(size)
    jid = job_id(submit(client, path))
    wait(client, jid)

    s = client.get(f"/api/jobs/{jid}").json()
    assert s["waiting"] == "" and s["wait_left"] == 0
