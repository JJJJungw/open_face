"""무DB 잡 러너 테스트 — 붙일 곳의 계약을 코드로 못 박는다.

네트워크는 쓰지 않는다. presigned URL 을 흉내 내는 것은 로컬 HTTP 서버 하나이고,
검출기는 가짜를 주입한다. 계약이 깨지는지를 보는 것이지 모델을 보는 게 아니다.
"""

import http.server
import json
import os
import threading

import pytest

from conftest import FakeDetector

pytest.importorskip("httpx", reason="pip install -r requirements-worker.txt")

from face_anonymizer import job_runner                      # noqa: E402
from face_anonymizer.core.pipeline import VideoAnonymizer   # noqa: E402
from face_anonymizer.storage import transfer                # noqa: E402


class _Handler(http.server.BaseHTTPRequestHandler):
    """GET 은 준비된 파일을, PUT 은 받아서 보관한다. 상태코드는 경로로 조종한다."""

    def log_message(self, *a):
        pass

    def _status_from_path(self):
        # /fail/503/... 처럼 앞에 붙이면 그 상태코드를 돌려준다
        parts = self.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "fail":
            return int(parts[1])
        return None

    def do_GET(self):
        code = self._status_from_path()
        if code:
            self.send_response(code); self.end_headers(); return
        body = self.server.blob
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):
        code = self._status_from_path()
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n)
        if code:
            self.send_response(code); self.end_headers(); return
        self.server.puts.append((self.path, self.headers.get("Content-Type"), data))
        self.send_response(200); self.end_headers()


@pytest.fixture
def bucket(make_video):
    """presigned URL 을 흉내 내는 로컬 서버. 원본 한 편을 들고 있다."""
    path, frames, size = make_video(frames=8)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.blob = open(path, "rb").read()
    srv.puts = []
    srv.frames, srv.size = frames, size
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield srv
    srv.shutdown()


def make_job(bucket, **over):
    job = {
        "video_id": "11111111-2222-3333-4444-555555555555",
        "token": "fencing-token",
        "input_url": f"{bucket.base}/in.mp4",
        "targets": [{"label": "deid-720p", "height": 720, "bitrate": "3500k",
                     "method": "mosaic", "conf": 0.25,
                     "put_url": f"{bucket.base}/out.mp4",
                     "content_type": "video/mp4"}],
        "heartbeat_every_s": 60,
    }
    job.update(over)
    return job


def anon(bucket):
    return VideoAnonymizer(detector=FakeDetector(bucket.size))


def test_runs_a_job_end_to_end_without_credentials(bucket):
    """페이로드 하나로 내려받기 → 익명화 → 올리기까지 완주한다.

    자격 증명도 버킷 이름도 DB 도 없다. 이 테스트가 통과한다는 것은 워커 이미지에
    비밀을 넣지 않아도 된다는 뜻이다.
    """
    beats = []
    out = job_runner.run_job(make_job(bucket), on_heartbeat=beats.append,
                             anonymizer=anon(bucket))

    assert out["elapsed_s"] >= 0
    assert out["targets"][0]["label"] == "deid-720p"
    assert len(bucket.puts) == 1
    path, ctype, data = bucket.puts[0]
    assert path == "/out.mp4" and ctype == "video/mp4" and len(data) > 0


def test_upload_only_happens_after_every_target_encodes(bucket):
    """타깃 하나가 실패하면 **아무것도 올리지 않는다.**

    반쪽 산출이 버킷에 남으면, 재시도가 성공할 때까지 잘못된 결과가 그 자리에 있다.
    """
    job = make_job(bucket)
    job["targets"].append({"label": "bad", "method": "없는방식",
                           "put_url": f"{bucket.base}/bad.mp4"})

    with pytest.raises(job_runner.JobError) as e:
        job_runner.run_job(job, anonymizer=anon(bucket))
    assert e.value.transient is False          # 다시 해도 같은 결과다
    assert e.value.stage == "process"
    assert bucket.puts == []


def test_expired_presign_is_transient(bucket):
    """403 은 presign 만료일 수 있어 일시로 분류한다 — 재큐잉되면 새 URL 을 받는다."""
    job = make_job(bucket, input_url=f"{bucket.base}/fail/403/in.mp4")
    with pytest.raises(job_runner.JobError) as e:
        job_runner.run_job(job, anonymizer=anon(bucket))
    assert e.value.transient is True and e.value.stage == "download"


def test_upload_5xx_is_transient_but_404_is_not(bucket):
    for code, transient in ((503, True), (404, False)):
        job = make_job(bucket)
        job["targets"][0]["put_url"] = f"{bucket.base}/fail/{code}/out.mp4"
        with pytest.raises(job_runner.JobError) as e:
            job_runner.run_job(job, anonymizer=anon(bucket))
        assert e.value.transient is transient, code
        assert e.value.stage == "upload"


def test_empty_targets_is_a_permanent_payload_error(bucket):
    with pytest.raises(job_runner.JobError) as e:
        job_runner.run_job(make_job(bucket, targets=[]), anonymizer=anon(bucket))
    assert e.value.transient is False and e.value.stage == "payload"


def test_heartbeat_is_throttled_by_time_not_by_frame(bucket):
    """콜백은 프레임마다 오지만 하트비트는 주기로 눌러 보낸다.

    누르지 않으면 8프레임짜리 영상에서도 큐에 수십 건이 쌓인다.
    """
    beats = []
    job_runner.run_job(make_job(bucket, heartbeat_every_s=3600),
                       on_heartbeat=beats.append, anonymizer=anon(bucket))
    assert beats == []          # 1시간 주기라 한 번도 안 나가는 게 맞다

    beats.clear()
    job_runner.run_job(make_job(bucket, heartbeat_every_s=1),
                       on_heartbeat=beats.append, anonymizer=anon(bucket))
    assert len(beats) <= 30     # 프레임 수만큼 나가지는 않는다


def test_stall_watchdog_fires_and_is_transient(monkeypatch):
    """진행이 멎으면 일시 실패로 끊는다. 환경 문제일 수 있어 영구가 아니다."""
    monkeypatch.setattr(job_runner, "STALL_S", 0)
    beat = job_runner._Beat(None, 60)
    beat(5)                                   # 첫 보고 — 기준점
    with pytest.raises(job_runner.JobError) as e:
        beat(5)                               # 같은 자리 — 정체
    assert e.value.transient is True and e.value.stage == "stall"


def test_height_null_means_keep_resolution(bucket):
    """저쪽 규약에서 height=null 은 '스케일 생략'이다. 키가 없는 것과 다르다."""
    assert job_runner.target_params({"height": None})["height"] == 0
    assert "height" not in job_runner.target_params({"label": "x"})


def test_crf_target_turns_off_our_bitrate_policy():
    """화질 정책은 둘 중 하나만 — 타깃 비트레이트와 CRF 를 같이 걸면 안 된다."""
    p = job_runner.target_params({"crf": 24})
    assert p["crf"] == 24 and p["bitrate"] == "" and p["max_bitrate"] == ""
    # 비트레이트로 말하면 우리 납품 기준 그대로다
    p = job_runner.target_params({"bitrate": "3500k", "max_bitrate": "4000k"})
    assert p["bitrate"] == "3500k" and "crf" not in p


def test_transfer_classifies_status_codes():
    """일시/영구 분류는 워커의 1차 판단이다 — 판정은 잡을 준 쪽이 한다."""
    assert 403 in transfer.TRANSIENT_STATUS      # presign 만료일 수 있다
    assert 429 in transfer.TRANSIENT_STATUS
    assert 503 in transfer.TRANSIENT_STATUS
    assert 404 not in transfer.TRANSIENT_STATUS
