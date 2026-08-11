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

from face_anonymizer import VideoAnonymizer, server  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """작업 디렉터리와 전역 상태를 테스트마다 격리한다."""
    jobs = tmp_path / "jobs"
    monkeypatch.setattr(server, "JOBS_DIR", str(jobs))
    monkeypatch.setattr(server, "_JOBS", {})
    monkeypatch.setattr(server, "_anonymizer", None)
    monkeypatch.setattr(server, "_current", None)
    monkeypatch.setattr(server, "_model_error", None)

    def attach(size, miss_frames=()):
        anon = VideoAnonymizer(detector=FakeDetector(size, miss_frames))
        monkeypatch.setattr(server, "get_anonymizer", lambda: anon)
        monkeypatch.setattr(server, "_anonymizer", anon)
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
    server._JOBS[jid].status = "running"
    assert client.get(f"/api/jobs/{jid}/download").status_code == 409
    assert client.delete(f"/api/jobs/{jid}").status_code == 409


def test_imgsz_is_range_clamped(client, make_video):
    """서버는 범위만 본다. stride 배수 맞추기는 검출기 몫이다(규칙을 두 벌 두지 않는다)."""
    path, n, size = make_video(frames=6)
    client.attach(size)
    jid = job_id(submit(client, path, imgsz="99999"))
    assert server._JOBS[jid].params["imgsz"] == 2048
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

    server._JOBS.clear()                       # 재시작 또는 다른 워커의 시야

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

    j = server._JOBS[jid]
    j.status, j.finished = "running", 0.0
    server.save_job(j)
    server._JOBS.clear()                       # 프로세스가 죽은 상태

    assert server.recover_orphans() == 1
    s = client.get(f"/api/jobs/{jid}").json()
    assert s["status"] == "failed"
    assert s["error"]["code"] == "interrupted"


def test_sweep_removes_expired_jobs(client, tmp_path, make_video, monkeypatch):
    """정리가 새 업로드에만 의존하면 디스크가 안 비워진다."""
    path, n, size = make_video(frames=10)
    client.attach(size)
    jid = job_id(submit(client, path))
    wait(client, jid)

    monkeypatch.setattr(server, "JOB_TTL", 1)
    server._JOBS[jid].finished = time.time() - 10
    server.save_job(server._JOBS[jid])

    assert server.sweep() == 1
    assert not (tmp_path / "jobs" / jid).exists()
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_missing_output_reports_410_not_500(client, tmp_path, make_video):
    """보관 기간이 지나 파일만 사라진 경우를 구분해서 알린다."""
    path, n, size = make_video(frames=10)
    client.attach(size)
    jid = job_id(submit(client, path))
    wait(client, jid)

    os.remove(server._JOBS[jid].output)
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
    assert server.QUEUE_MAX == 0
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
    monkeypatch.setattr(server, "free_mb", lambda: 10)
    monkeypatch.setattr(server, "MIN_FREE_MB", 2048)

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
    monkeypatch.setattr(server, "QUEUE_MAX", 2)
    path, n, size = make_video(frames=40)
    anon = VideoAnonymizer(detector=SlowDetector())
    monkeypatch.setattr(server, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(server, "_anonymizer", anon)

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
    monkeypatch.setattr(server, "MAX_ATTEMPTS", 3)
    path, n, size = make_video(frames=6)
    calls = {"n": 0}

    class Flaky:
        def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("CUDA out of memory")
            return [[] for _ in frames]

    anon = VideoAnonymizer(detector=Flaky())
    monkeypatch.setattr(server, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(server, "_anonymizer", anon)

    jid = job_id(submit(client, path))
    s = wait(client, jid, timeout=60)

    assert s["status"] == "done", s.get("error")
    assert s["attempts"] == 3


def test_retries_are_exhausted_then_failed(client, make_video, monkeypatch):
    monkeypatch.setattr(server, "MAX_ATTEMPTS", 2)
    path, n, size = make_video(frames=6)

    class AlwaysBroken:
        def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
            raise RuntimeError("일시적인 척하는 영구 오류")

    anon = VideoAnonymizer(detector=AlwaysBroken())
    monkeypatch.setattr(server, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(server, "_anonymizer", anon)

    jid = job_id(submit(client, path))
    s = wait(client, jid, timeout=60)

    assert s["status"] == "failed"
    assert s["attempts"] == 2
    assert "일시적인 척하는" in s["error"]["detail"]


def test_permanent_error_is_not_retried(client, tmp_path, monkeypatch):
    """깨진 입력은 세 번 돌려도 결과가 같다 — 바로 실패로 둔다."""
    monkeypatch.setattr(server, "MAX_ATTEMPTS", 3)
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
    server._anonymizer = None
    r = client.get("/api/health")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_upload_rejected_when_model_failed(client, make_video):
    path, n, size = make_video(frames=4)
    server._anonymizer = None
    server._model_error = "RuntimeError: 가중치 없음"

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

    assert server._JOBS[jid].params == server.JOB_DEFAULTS
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

    p = server._JOBS[jid].params
    assert p["conf"] == 0.4
    assert p["method"] == server.JOB_DEFAULTS["method"]
    assert p["batch_size"] == server.JOB_DEFAULTS["batch_size"]
    wait(client, jid)


def test_defaults_are_a_copy_not_the_live_dict(client, make_video):
    """작업이 기본값 dict 를 공유하면 한 건의 변경이 이후 전부에 번진다."""
    path, n, size = make_video(frames=6)
    client.attach(size)
    jid = job_id(submit(client, path))

    server._JOBS[jid].params["conf"] = 0.99
    assert server.JOB_DEFAULTS["conf"] != 0.99
    wait(client, jid)


# ── 오류 형식 (RFC 9457) ─────────────────────────────────────────────────────
#
# 호출하는 쪽은 재시도할지, 다른 인스턴스로 보낼지, 사람을 불러야 할지를 정해야
# 한다. 한국어 문장을 파싱해서 정할 수는 없다.

from face_anonymizer import errors                    # noqa: E402


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
    monkeypatch.setattr(server, "free_mb", lambda: 1)
    monkeypatch.setattr(server, "MIN_FREE_MB", 2048)
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
    monkeypatch.setattr(server, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(server, "_anonymizer", anon)
    monkeypatch.setattr(server, "MAX_ATTEMPTS", 1)

    jid = job_id(submit(client, path))
    s = wait(client, jid, timeout=60)

    assert s["status"] == "failed"
    assert s["error"]["code"] == "gpu_out_of_memory"
    assert s["error"]["hint"]


# ── 취소 ─────────────────────────────────────────────────────────────────────

def test_cancel_queued_job(client, make_video, monkeypatch):
    path, n, size = make_video(frames=40)
    anon = VideoAnonymizer(detector=SlowDetector(0.05))
    monkeypatch.setattr(server, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(server, "_anonymizer", anon)

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
    monkeypatch.setattr(server, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(server, "_anonymizer", anon)

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
    monkeypatch.setattr(server, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(server, "_anonymizer", anon)
    monkeypatch.setattr(server, "MAX_ATTEMPTS", 1)
    monkeypatch.setattr(server, "JOB_TTL", 1)
    monkeypatch.setattr(server, "FAILED_TTL", 0)       # 기본값 = 안 지움

    jid = job_id(submit(client, path))
    wait(client, jid, timeout=60)
    server._JOBS[jid].finished = time.time() - 9999
    server.save_job(server._JOBS[jid])

    server.sweep()
    assert client.get(f"/api/jobs/{jid}").status_code == 200
