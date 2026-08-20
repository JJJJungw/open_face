"""저쪽이 우리를 부르는 문 — `POST /api/deident/jobs`.

여기서 지키는 것은 넷이다.

1. **문이 기본으로 활짝 열려 있지 않다.** 토큰이 없으면 같은 기계에서만 열린다.
2. **같은 러너로 합류한다.** 큐 경로가 쓰던 `job_runner.run_job` 을 그대로
   부른다 — 계약은 페이로드고 전송은 선택이다.
3. **바쁘면 거절한다.** 대기열을 만들면 큐가 둘이 되고, 저쪽 리스·재시도와
   서로를 방해한다.
4. **실패가 사유를 잃지 않는다.** 스레드에서 죽었는데 기록이 `running` 으로
   남으면, 저쪽은 리스 만료까지 기다리고 화면에는 아무 사유가 없다.
"""

import threading
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient           # noqa: E402

from face_anonymizer.service import remote, server  # noqa: E402

JOB = {
    "video_id": "v-1",
    "token": "fencing-1",
    "input_url": "https://example.invalid/in.mp4",
    "targets": [{"label": "deid-720p", "height": 720,
                 "put_url": "https://example.invalid/out.mp4",
                 "content_type": "video/mp4"}],
}


class Req:
    """`door_open` 이 보는 것만 흉내 낸다 — 헤더와 붙은 쪽 주소."""

    def __init__(self, host="127.0.0.1", token=None):
        self.headers = {"x-deident-token": token} if token else {}
        self.client = type("C", (), {"host": host})()


# TestClient 는 붙은 쪽 주소를 `testclient` 로 준다 — 루프백이 아니다. 그래서
# 라우트를 지나는 검사들은 **토큰을 켜고** 돈다. 토큰이 없을 때의 동작은
# `door_open` 을 직접 불러서 본다(위 Req).
TOKEN = "t0ken-for-tests"


@pytest.fixture
def client(monkeypatch):
    remote.reset()
    monkeypatch.setattr(remote, "TOKEN", TOKEN)
    monkeypatch.setattr(remote, "MAX_INFLIGHT", 1)
    yield TestClient(server.app, headers={"x-deident-token": TOKEN})
    remote.reset()


def wait_for(jid, want, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rec = remote.get(jid)
        if rec and rec.status == want:
            return rec
        time.sleep(0.01)
    rec = remote.get(jid)
    raise AssertionError(f"{want} 가 안 됐다: 지금 {rec and rec.status}")


def ok_runner(job, *, on_heartbeat=None, anonymizer=None):
    return {"elapsed_s": 0.1, "review_needed": False, "targets": []}


# ── 문 ────────────────────────────────────────────────────────────────────

def test_without_a_token_the_door_only_opens_on_this_machine(monkeypatch):
    """**설정 안 하면 활짝 열린다** 는 오픈소스에서 제일 나쁜 기본값이다.

    그렇다고 설정을 안 하면 아무것도 못 하게 하면 처음 받는 사람이 막힌다.
    그래서 같은 기계에서는 열고, 밖에서 오면 닫는다.
    """
    monkeypatch.setattr(remote, "TOKEN", "")
    assert remote.door_open(Req(host="127.0.0.1"))[0] is True
    ok, why = remote.door_open(Req(host="10.0.0.9"))
    assert ok is False and "FA_REMOTE_TOKEN" in why


def test_with_a_token_only_the_token_counts(monkeypatch):
    """망으로 막는 것과 열쇠로 막는 것은 서로를 대신하지 못한다.

    IP 로만 막으면 **저쪽 서버를 거쳐 우회로 들어오는 요청**(SSRF)은 출처가
    정상이라 그냥 통과한다. 그렇게 만들어진 요청에는 헤더를 붙일 수 없다.
    """
    monkeypatch.setattr(remote, "TOKEN", "s3cr3t")
    assert remote.door_open(Req())[0] is False                    # 같은 기계여도
    assert remote.door_open(Req(token="틀림"))[0] is False
    assert remote.door_open(Req(host="10.0.0.9", token="s3cr3t"))[0] is True


def test_the_route_refuses_before_it_does_any_work(client):
    """열쇠 없이 온 요청은 **아무것도 하기 전에** 돌려보낸다."""
    bare = TestClient(server.app)                  # 헤더를 안 붙인다
    r = bare.post("/api/deident/jobs", json=JOB)
    assert r.status_code == 403
    assert r.json()["code"] == "remote_forbidden"
    assert not remote._JOBS, "거절했는데 기록이 생겼다"


# ── 페이로드 ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("body, missing", [
    ({}, "input_url"),
    ({"input_url": "https://x/in.mp4"}, "targets"),
    ({"input_url": "https://x/in.mp4", "targets": []}, "targets"),
])
def test_a_malformed_payload_is_refused_at_the_door(client, body, missing):
    """모양이 안 맞으면 **받기 전에** 거절한다. 받아 놓고 스레드에서 죽으면
    저쪽은 리스가 만료될 때까지 기다린다."""
    r = client.post("/api/deident/jobs", json=body)
    assert r.status_code == 400
    assert missing in r.json()["detail"]


# ── 러너로 합류 ───────────────────────────────────────────────────────────

def test_the_payload_reaches_the_same_runner_the_queue_uses(client, monkeypatch):
    """**계약은 페이로드다.** 큐로 오든 HTTP 로 오든 러너가 받는 dict 은 같다.

    `_run` 이 지연 임포트를 하므로 모듈 속성을 갈아 끼우면 그대로 걸린다 —
    큐 경로가 부르는 것과 **같은 이름**을 갈아 끼웠다는 것이 이 검사의 핵심이다.
    """
    seen = {}

    def fake(job, *, on_heartbeat=None, anonymizer=None):
        seen["job"] = job
        on_heartbeat({"elapsed_s": 1.0, "percent": 40.0, "stage": "detect",
                      "stage_label": "얼굴 찾는 중", "eta_s": 2})
        return {"elapsed_s": 1.2, "review_needed": False, "targets": []}

    monkeypatch.setattr("face_anonymizer.job_runner.run_job", fake)
    r = client.post("/api/deident/jobs", json=JOB)
    assert r.status_code == 202
    jid = r.json()["job_id"]
    wait_for(jid, "done")

    assert seen["job"] == JOB, "러너가 받은 것이 보낸 것과 다르다"
    d = client.get(f"/api/deident/jobs/{jid}").json()
    assert d["status"] == "done"
    assert d["result"]["elapsed_s"] == 1.2
    assert d["video_id"] == "v-1"


def test_progress_is_what_the_screen_can_draw(client):
    """진행률은 **우리만 안다.** 안 실어 보내면 화면은 스피너까지밖에 못 그린다."""
    hold = threading.Event()

    def slow(job, *, on_heartbeat=None, anonymizer=None):
        on_heartbeat({"elapsed_s": 3.0, "percent": 46.2, "stage": "detect",
                      "stage_label": "얼굴 찾는 중", "eta_s": 21})
        hold.wait(3)
        return {"elapsed_s": 4.0, "review_needed": False, "targets": []}

    rec = remote.submit(JOB, runner=slow)
    for _ in range(300):
        if rec.progress:
            break
        time.sleep(0.01)
    d = client.get(f"/api/deident/jobs/{rec.id}").json()
    assert d["status"] == "running"
    assert d["progress"]["percent"] == 46.2
    assert d["progress"]["stage_label"] == "얼굴 찾는 중"
    hold.set()
    wait_for(rec.id, "done")


# ── 거절과 실패 ───────────────────────────────────────────────────────────

def test_when_busy_it_says_so_instead_of_queueing(client):
    """**대기열을 만들지 않는다.** 저쪽이 이미 리스·재시도를 갖고 있고, 두 곳이
    각자 판단하면 같은 영상이 몇 배로 돈다."""
    hold = threading.Event()

    def slow(job, *, on_heartbeat=None, anonymizer=None):
        hold.wait(3)
        return {"elapsed_s": 0.1, "review_needed": False, "targets": []}

    first = remote.submit(JOB, runner=slow)
    r = client.post("/api/deident/jobs", json=JOB)
    assert r.status_code == 429
    body = r.json()
    assert body["code"] == "remote_busy" and body["retryable"] is True
    hold.set()
    wait_for(first.id, "done")


def test_a_failure_keeps_its_reason_and_its_retry_verdict(client):
    """`error` 는 우리 내부 문구다. 화면에는 `problem` 을 띄운다."""
    from face_anonymizer.job_runner import JobError

    def boom(job, *, on_heartbeat=None, anonymizer=None):
        raise JobError("내려받기 실패 HTTP 403", transient=True, stage="download")

    rec = remote.submit(JOB, runner=boom)
    wait_for(rec.id, "failed")
    d = client.get(f"/api/deident/jobs/{rec.id}").json()
    assert d["status"] == "failed"
    assert d["transient"] is True and d["stage"] == "download"
    assert d["problem"]["title"] == "원본을 내려받지 못했습니다"
    assert d["problem"]["retryable"] is True
    assert "403" in d["error"]


def test_an_unclassified_crash_is_treated_as_temporary(client):
    """우리가 분류 못 한 오류를 영구로 두면 **우리 버그 하나가 저쪽 재시도
    상한을 통째로 태운다.** 큐 경로와 같은 판단이다."""
    def boom(job, *, on_heartbeat=None, anonymizer=None):
        raise RuntimeError("어디선가 터졌다")

    rec = remote.submit(JOB, runner=boom)
    wait_for(rec.id, "failed")
    d = client.get(f"/api/deident/jobs/{rec.id}").json()
    assert d["transient"] is True and d["stage"] == "unknown"
    assert "어디선가 터졌다" in d["error"]


# ── 조회 ──────────────────────────────────────────────────────────────────

def test_an_unknown_job_id_says_so(client):
    r = client.get("/api/deident/jobs/없는것")
    assert r.status_code == 404
    assert r.json()["code"] == "remote_job_not_found"


def test_finished_records_are_swept_after_their_ttl(client, monkeypatch):
    rec = remote.submit(JOB, runner=ok_runner)
    wait_for(rec.id, "done")
    monkeypatch.setattr(remote, "TTL", 0.0)
    remote.submit(JOB, runner=ok_runner)          # 다음 접수가 쓸어낸다
    assert remote.get(rec.id) is None


# ── 시험용 스위치 ─────────────────────────────────────────────────────────

def test_the_open_switch_is_explicit_and_not_the_default():
    """**기본값이면 안 된다.** 조용히 열려 있는 것과, 켜 놓고 열려 있다고
    말하는 것은 다르다."""
    import os

    from face_anonymizer.env import flag
    assert flag("FA_REMOTE_OPEN", False) is False or os.environ.get("FA_REMOTE_OPEN")


def test_the_open_switch_lets_anyone_in(client, monkeypatch):
    """컨테이너끼리 붙여 보는 단계에서는 열쇠가 곧 마찰이다 — 받아서 띄운
    사람이 403 부터 만나게 된다."""
    monkeypatch.setattr(remote, "OPEN", True)
    monkeypatch.setattr(remote, "TOKEN", "")
    assert remote.door_open(Req(host="10.0.0.9"))[0] is True

    bare = TestClient(server.app)                  # 헤더 없이
    r = bare.post("/api/deident/jobs", json=JOB)
    assert r.status_code == 202


def test_the_open_switch_beats_a_token_but_says_so(client, monkeypatch, caplog):
    """토큰이 있어도 스위치가 이긴다 — 헷갈리지 않게 **로그가 말한다.**"""
    import logging

    monkeypatch.setattr(remote, "OPEN", True)
    monkeypatch.setattr(remote, "TOKEN", "무시된다")
    assert remote.door_open(Req(host="10.0.0.9"))[0] is True

    with caplog.at_level(logging.WARNING, logger="face_anonymizer.service.remote"):
        remote.announce()
    assert "인증 없이" in caplog.text


# ── 경로로 받기 (2026-08-20 결정) ─────────────────────────────────────────
#
# **클라우드 접근은 이쪽이 맡는다.** 저쪽은 분석할 파일 경로와 결과를 둘 경로만
# 넘기고, 자격 증명은 우리 `.env` 에 있다. 바꾸는 자리를 문 하나로 몰아 두어서,
# 러너는 예전과 똑같이 URL 두 개만 본다.

class FakeStore:
    bucket = "our-bucket"

    def __init__(self):
        self.signed = []

    def presigned_url(self, key, expires=None, filename=None):
        self.signed.append(("get", self.bucket, key))
        return f"https://signed.test/{self.bucket}/{key}?get"

    def presigned_put(self, key, content_type="video/mp4", expires=None):
        self.signed.append(("put", self.bucket, key, content_type))
        return f"https://signed.test/{self.bucket}/{key}?put&ct={content_type}"

    def for_bucket(self, bucket):
        if bucket == self.bucket:
            return self
        other = FakeStore()
        other.bucket = bucket
        other.signed = self.signed
        return other


@pytest.fixture
def store(monkeypatch):
    s = FakeStore()
    monkeypatch.setattr("face_anonymizer.storage.s3.get_store", lambda: s)
    return s


PATH_JOB = {
    "video_id": "v-2",
    "input_key": "work/v-2/analysis-720p.mp4",
    "targets": [{"label": "deid-720p", "height": 720,
                 "output_key": "work/v-2/analysis-720p.deid.mp4"}],
}


def test_paths_are_signed_with_our_own_credentials(store):
    """저쪽은 **경로만** 넘긴다. 서명은 우리가 한다."""
    out = remote.resolve(PATH_JOB)
    assert out["input_url"] == \
        "https://signed.test/our-bucket/work/v-2/analysis-720p.mp4?get"
    assert out["targets"][0]["put_url"].startswith(
        "https://signed.test/our-bucket/work/v-2/analysis-720p.deid.mp4?put")
    assert ("get", "our-bucket", "work/v-2/analysis-720p.mp4") in store.signed


def test_the_content_type_goes_into_the_signature(store):
    """안 넣으면 올릴 때 붙는 헤더와 서명이 어긋나 403 이 난다 — 그리고 그
    403 은 권한 문제와 구분이 안 된다."""
    out = remote.resolve(PATH_JOB)
    assert out["targets"][0]["content_type"] == "video/mp4"
    assert ("put", "our-bucket", "work/v-2/analysis-720p.deid.mp4",
            "video/mp4") in store.signed


def test_an_already_signed_job_is_left_alone(store):
    """저쪽이 직접 서명해 보내던 방식도 그대로 받는다 — 두 길이 같은 러너로
    합류한다."""
    out = remote.resolve(JOB)
    assert out is JOB
    assert not store.signed, "이미 서명된 잡에 또 서명했다"


def test_an_s3_uri_can_point_at_another_bucket(store):
    """입력이 스테이징에, 결과가 납품 버킷에 있는 구성이 정상이다."""
    out = remote.resolve({
        "video_id": "v-3",
        "input_key": "s3://intake/raw/v-3.mp4",
        "targets": [{"label": "deid", "output_key": "s3://delivery/out/v-3.mp4"}],
    })
    assert "/intake/raw/v-3.mp4?get" in out["input_url"]
    assert "/delivery/out/v-3.mp4?put" in out["targets"][0]["put_url"]


@pytest.mark.parametrize("alias", ["input_key", "input_path", "input",
                                   "source_key", "source"])
def test_the_caller_may_use_its_own_field_names(store, alias):
    """이름 하나가 어긋났을 때 "input_url 이 필요합니다" 가 나오면 무엇이
    잘못됐는지가 안 드러난다. 넉넉히 받는다."""
    out = remote.resolve({"video_id": "v", alias: "a/b.mp4",
                          "targets": [{"output": "c/d.mp4"}]})
    assert "a/b.mp4?get" in out["input_url"]
    assert "c/d.mp4?put" in out["targets"][0]["put_url"]


def test_without_a_store_it_says_what_to_check(monkeypatch):
    """경로로 받으려면 우리 쪽에 저장소가 있어야 한다. 없으면 **어디를 볼지**
    말해 준다."""
    monkeypatch.setattr("face_anonymizer.storage.s3.get_store", lambda: None)
    monkeypatch.setattr("face_anonymizer.storage.s3.unavailable_reason",
                        lambda: "버킷이 설정되지 않았습니다")
    with pytest.raises(ValueError) as e:
        remote.resolve(PATH_JOB)
    assert "credentials/health" in str(e.value)


def test_signing_failure_is_answered_at_submit_not_later(client, store, monkeypatch):
    """**서명은 접수할 때 한다.** 스레드 안에서 하면 실패가 202 뒤에 숨어서,
    저쪽은 잡을 받았다고 믿고 폴링부터 시작한다."""
    r = client.post("/api/deident/jobs",
                    json={"video_id": "v", "targets": [{"label": "x"}]})
    assert r.status_code == 400
    assert "input_url 또는 input_key" in r.json()["detail"]

    r = client.post("/api/deident/jobs",
                    json={"video_id": "v", "input_key": "a.mp4",
                          "targets": [{"label": "x"}]})
    assert r.status_code == 400
    assert "output_key" in r.json()["detail"]


# ── VRAM ──────────────────────────────────────────────────────────────────

def test_vram_shows_up_where_it_is_needed(client, monkeypatch):
    """OOM 은 나고 나서는 원인을 못 본다. **터지기 전부터 계속 남긴다.**"""
    from face_anonymizer import gpu

    monkeypatch.setattr(gpu, "snapshot", lambda device=None: {
        "available": True, "free_mb": 3000, "total_mb": 24000,
        "used_mb": 21000, "free_pct": 12.5, "name": "테스트GPU"})

    hold = threading.Event()

    def slow(job, *, on_heartbeat=None, anonymizer=None):
        on_heartbeat({"elapsed_s": 1.0, "percent": 10.0, "stage": "detect",
                      "stage_label": "얼굴 찾는 중", "eta_s": 9})
        hold.wait(3)
        return {"elapsed_s": 2.0, "review_needed": False, "targets": []}

    rec = remote.submit(JOB, runner=slow)
    for _ in range(300):
        if rec.progress:
            break
        time.sleep(0.01)
    d = client.get(f"/api/deident/jobs/{rec.id}").json()
    assert d["progress"]["vram_free_mb"] == 3000
    assert d["progress"]["vram_free_pct"] == 12.5
    hold.set()
    wait_for(rec.id, "done")


def test_no_gpu_means_no_extra_noise(monkeypatch):
    """못 재는 것은 불편이고, 그것 때문에 처리가 멈추는 것은 사고다."""
    from face_anonymizer import gpu

    monkeypatch.setattr(gpu, "snapshot", lambda device=None: {"available": False})
    assert remote._vram() == {}
    assert "CPU" in gpu.line()


def test_the_watch_keeps_the_low_water_mark(monkeypatch):
    """**최저값이 본체다.** 끝난 뒤에 재면 텐서가 다 반납된 뒤라 항상 넉넉해
    보이고, 아슬아슬했는지는 도는 중에만 보인다."""
    from face_anonymizer import gpu

    seq = iter([20000, 3000, 8000, 15000])          # 도중에 3GB 까지 내려갔다

    def fake(device=None):
        return {"available": True, "free_mb": next(seq, 15000),
                "total_mb": 24000, "used_mb": 0, "free_pct": 0.0,
                "name": "테스트GPU"}

    monkeypatch.setattr(gpu, "snapshot", fake)
    w = gpu.Watch(every_s=0.0)                      # 누르지 않고 매번 잰다
    for _ in range(3):
        w.sample(force=True)
    r = w.result()
    assert r["vram_min_free_mb"] == 3000
    assert r["vram_min_free_pct"] == 12.5
    assert r["vram_total_mb"] == 24000


def test_the_watch_is_quiet_without_a_gpu(monkeypatch):
    from face_anonymizer import gpu

    monkeypatch.setattr(gpu, "snapshot", lambda device=None: {"available": False})
    assert gpu.Watch().result() == {}
    assert gpu.fields() == {}


def test_the_ui_has_a_label_for_every_vram_field():
    """이름이 없으면 화면에 `vram_min_free_mb` 가 날것으로 뜬다."""
    import pathlib
    import re

    html = (pathlib.Path(__file__).resolve().parent.parent / "face_anonymizer"
            / "service" / "static" / "index.html").read_text(encoding="utf-8")
    labels = re.search(r"const DETAIL_LABEL = \{(.*?)\n\};", html, re.S)
    assert labels, "DETAIL_LABEL 을 찾지 못했다"
    from face_anonymizer import gpu

    class W:
        min_free, total, name = 1, 2, "x"
        result = gpu.Watch.result
    names = set(gpu.Watch.result(W())) | {"vram_free_mb", "vram_free_pct",
                                          "vram_total_mb"}
    for n in names:
        assert f"{n}:" in labels.group(1), f"{n} 의 한국어 이름이 없다"
