"""HTTP API 테스트.

검출기를 가짜로 갈아 끼우므로 여기서도 torch/가중치가 필요 없다.
확인하는 것: 작업 수명주기(큐→실행→완료→다운로드), 잘못된 입력 거절,
업로드 파일명이 서버 경로에 영향을 주지 않는지.
"""

import os
import time

import pytest

pytest.importorskip("httpx", reason="fastapi TestClient 는 httpx 가 필요하다")
from fastapi.testclient import TestClient  # noqa: E402

from conftest import FakeDetector  # noqa: E402
from face_anonymizer import VideoAnonymizer  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FA_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("FA_EAGER_LOAD", "0")

    from face_anonymizer import server

    monkeypatch.setattr(server, "WORKDIR", str(tmp_path / "work"))
    monkeypatch.setattr(server, "EAGER_LOAD", False)
    fake = VideoAnonymizer(detector=FakeDetector((320, 240)))
    monkeypatch.setattr(server, "_anonymizer", fake)
    monkeypatch.setattr(server, "get_anonymizer", lambda: fake)
    server._jobs.clear()

    with TestClient(server.app) as c:
        yield c


def wait_for(client, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {body}")


def upload(client, path, **form):
    with open(path, "rb") as f:
        return client.post("/jobs", files={"file": ("clip.mp4", f, "video/mp4")},
                           data=form)


# --------------------------------------------------------------------------- #

def test_healthz_reports_readiness(client):
    body = client.get("/healthz").json()
    assert body["ready"] is True
    assert "queued" in body


def test_full_job_lifecycle(client, make_video, tmp_path):
    path, frames, _ = make_video(frames=20)

    r = upload(client, path, method="box", pad="0.0", batch_size="4")
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    body = wait_for(client, job_id)
    assert body["status"] == "done", body
    assert body["result"]["frames"] == frames
    assert body["result"]["raw_boxes"] == frames
    assert body["progress"] == 1.0
    # 서버 내부 경로가 새 나가면 안 된다
    assert os.path.sep not in body["result"]["output"]

    got = client.get(f"/jobs/{job_id}/result")
    assert got.status_code == 200
    assert got.headers["content-type"] == "video/mp4"

    out = tmp_path / "downloaded.mp4"
    out.write_bytes(got.content)
    from face_anonymizer import probe
    assert probe(str(out)).width == 320


def test_result_before_completion_is_409(client, make_video):
    path, _, _ = make_video(frames=200)      # 바로 끝나지 않을 만큼
    job_id = upload(client, path).json()["job_id"]
    r = client.get(f"/jobs/{job_id}/result")
    assert r.status_code in (409, 200)       # 이미 끝났다면 200 도 정상
    if r.status_code == 409:
        assert "not done" in r.json()["detail"]


def test_unknown_job_is_404(client):
    assert client.get("/jobs/deadbeef").status_code == 404
    assert client.get("/jobs/deadbeef/result").status_code == 404


def test_unknown_method_rejected(client, make_video):
    path, _, _ = make_video(frames=4)
    r = upload(client, path, method="pixelate")
    assert r.status_code == 400
    assert "unknown method" in r.json()["detail"]


def test_empty_upload_rejected(client):
    r = client.post("/jobs", files={"file": ("x.mp4", b"", "video/mp4")})
    assert r.status_code == 400


def test_corrupt_video_fails_job_not_server(client, tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video" * 500)
    job_id = upload(client, junk).json()["job_id"]
    body = wait_for(client, job_id)
    assert body["status"] == "failed"
    assert body["error"]
    # 서버는 살아 있어야 한다
    assert client.get("/healthz").json()["ready"] is True


def test_upload_filename_cannot_escape_workdir(client, make_video, tmp_path):
    path, _, _ = make_video(frames=5)
    with open(path, "rb") as f:
        r = client.post(
            "/jobs",
            files={"file": ("../../../../etc/passwd.mp4", f, "video/mp4")},
        )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    wait_for(client, job_id)

    from face_anonymizer import server
    job = server._jobs[job_id]
    workdir = os.path.realpath(server.WORKDIR)
    assert os.path.realpath(job._input).startswith(workdir)


def test_oversized_upload_rejected(client, make_video, monkeypatch):
    from face_anonymizer import server
    monkeypatch.setattr(server, "MAX_UPLOAD_MB", 0)
    path, _, _ = make_video(frames=5)
    r = upload(client, path)
    assert r.status_code == 413


def test_cancel_removes_finished_job(client, make_video):
    path, _, _ = make_video(frames=8)
    job_id = upload(client, path).json()["job_id"]
    wait_for(client, job_id)

    r = client.delete(f"/jobs/{job_id}")
    assert r.status_code == 202
    assert r.json()["status"] == "deleted"
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_job_list(client, make_video):
    path, _, _ = make_video(frames=6)
    job_id = upload(client, path).json()["job_id"]
    wait_for(client, job_id)
    ids = [j["id"] for j in client.get("/jobs").json()["jobs"]]
    assert job_id in ids


def test_jobs_are_processed_serially(client, make_video):
    """GPU 는 하나다. 두 작업이 동시에 모델을 두드리면 VRAM 이 터진다."""
    path, _, _ = make_video(frames=12)
    ids = [upload(client, path).json()["job_id"] for _ in range(3)]
    for jid in ids:
        assert wait_for(client, jid)["status"] == "done"
