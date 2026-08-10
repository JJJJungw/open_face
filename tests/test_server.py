"""HTTP API 테스트.

파이프라인 테스트와 같은 원칙 — 가짜 검출기를 주입해 torch/가중치 없이
업로드부터 다운로드까지 전 구간을 돈다. 서빙 의존성이 안 깔린 환경에서는
통째로 skip 한다 (코어만 쓰는 사람에게 fastapi 를 강요하지 않는다).
"""

import os
import time

import pytest

from conftest import FakeDetector, face_rect, region_is_obscured, read_frames

pytest.importorskip("fastapi", reason="requirements-serve.txt 미설치")
pytest.importorskip("httpx", reason="requirements-serve.txt 미설치")
pytest.importorskip("multipart", reason="python-multipart 미설치")

from fastapi.testclient import TestClient           # noqa: E402

from face_anonymizer import VideoAnonymizer, server  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """작업 디렉터리와 전역 상태를 테스트마다 격리한다."""
    jobs = tmp_path / "jobs"
    monkeypatch.setattr(server, "JOBS_DIR", str(jobs))
    monkeypatch.setattr(server, "_JOBS", {})
    monkeypatch.setattr(server, "_anonymizer", None)

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
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.02)
    raise AssertionError(f"작업이 {timeout}s 안에 끝나지 않았다")


def submit(c, path, **form):
    with open(path, "rb") as f:
        return c.post("/api/jobs",
                      files={"file": ("clip.mp4", f, "video/mp4")}, data=form)


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
    ({"file": ("a.txt", b"x")}, {}),                      # 영상 아닌 확장자
    ({"file": ("a.mp4", b"")}, {}),                       # 빈 파일
    ({"file": ("a.mp4", b"x")}, {"conf": "1.5"}),         # 범위 밖 임계값
])
def test_rejects_bad_input(client, files, data):
    assert client.post("/api/jobs", files=files, data=data).status_code == 400


def test_bad_upload_leaves_no_workdir(client, tmp_path):
    """거절된 업로드가 작업 디렉터리를 남기면 디스크가 조용히 찬다."""
    client.post("/api/jobs", files={"file": ("a.mp4", b"")})
    jobs = tmp_path / "jobs"
    assert not jobs.exists() or not list(jobs.iterdir())


def test_download_before_done_is_409(client, make_video):
    path, n, size = make_video(frames=6)
    client.attach(size)
    jid = submit(client, path).json()["id"]
    # 완료 전 상태를 강제로 만들어 둔다 (실제 진행 중 상태를 잡으려면 경쟁이 생긴다)
    server._JOBS[jid].status = "running"
    assert client.get(f"/api/jobs/{jid}/download").status_code == 409
    assert client.delete(f"/api/jobs/{jid}").status_code == 409


def test_imgsz_is_snapped_to_stride_multiple(client, make_video):
    """클라이언트가 아무 값이나 보내도 32 배수로 맞춰야 forward 가 안 깨진다."""
    path, n, size = make_video(frames=6)
    client.attach(size)
    jid = submit(client, path, imgsz="1000").json()["id"]
    assert server._JOBS[jid].params["imgsz"] == 992
    wait(client, jid)


def test_full_lifecycle_and_no_leak(client, tmp_path, make_video):
    """업로드 → 처리 → 다운로드 → 삭제. 검출기가 놓친 프레임도 가려져야 한다."""
    path, n, size = make_video(frames=30)
    client.attach(size, miss_frames={7, 8})

    r = submit(client, path, method="mosaic", batch_size="8", keep_audio="false")
    assert r.status_code == 202
    jid = r.json()["id"]
    assert r.json()["status"] == "queued"

    s = wait(client, jid)
    assert s["status"] == "done", s.get("error")
    res = s["result"]
    assert res["frames"] == n
    assert res["filled_boxes"] >= 2          # 놓친 두 프레임을 보간이 메웠다
    assert res["fps"] > 0 and res["seconds"] > 0
    # 단계 시간은 짧은 클립에서 반올림으로 0 이 될 수 있다. 합계로 검증한다.
    assert sum(res["timing"].values()) <= res["seconds"] + 1e-6

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


def test_jobs_are_serialized(client, make_video):
    """GPU 는 하나뿐이다 — 동시에 두 건이 추론에 들어가면 안 된다."""
    path, n, size = make_video(frames=12)
    client.attach(size)
    ids = [submit(client, path).json()["id"] for _ in range(3)]
    for jid in ids:
        assert wait(client, jid)["status"] == "done"
    assert server._EXEC._max_workers == 1


# ── 상태 영속화 ──────────────────────────────────────────────────────────────
#
# 작업 상태를 전역 dict 에만 두면 (a) 재시작 시 전부 사라져 폴링 중인 클라이언트가
# 404 를 받고, (b) --workers 2 로 띄우면 업로드와 폴링이 다른 프로세스로 가서
# 계속 404 가 난다. 아래가 그 회귀다.

def test_job_state_is_written_to_disk(client, tmp_path, make_video):
    path, n, size = make_video(frames=10)
    client.attach(size)
    jid = submit(client, path).json()["id"]
    wait(client, jid)

    state = tmp_path / "jobs" / jid / "job.json"
    assert state.exists(), "작업 상태가 디스크에 없다"
    import json
    assert json.loads(state.read_text(encoding="utf-8"))["status"] == "done"


def test_survives_restart(client, tmp_path, make_video):
    """프로세스 메모리가 비어도 조회와 다운로드가 된다 (재시작 / 다른 워커)."""
    path, n, size = make_video(frames=10)
    client.attach(size)
    jid = submit(client, path).json()["id"]
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
    jid = submit(client, path).json()["id"]
    wait(client, jid)

    j = server._JOBS[jid]
    j.status, j.finished = "running", 0.0
    server.save_job(j)
    server._JOBS.clear()                       # 프로세스가 죽은 상태

    assert server.recover_orphans() == 1
    s = client.get(f"/api/jobs/{jid}").json()
    assert s["status"] == "error" and "재시작" in s["error"]


def test_sweep_removes_expired_jobs(client, tmp_path, make_video, monkeypatch):
    """정리가 새 업로드에만 의존하면 디스크가 안 비워진다."""
    path, n, size = make_video(frames=10)
    client.attach(size)
    jid = submit(client, path).json()["id"]
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
    jid = submit(client, path).json()["id"]
    wait(client, jid)

    os.remove(server._JOBS[jid].output)
    assert client.get(f"/api/jobs/{jid}/download").status_code == 410


@pytest.mark.parametrize("bad", ["../etc", "..", "a/b", ".hidden"])
def test_job_id_traversal_is_rejected(client, bad):
    assert client.get(f"/api/jobs/{bad}").status_code in (404, 400, 405)
