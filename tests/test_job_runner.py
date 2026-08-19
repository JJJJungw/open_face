"""무DB 잡 러너 테스트 — 붙일 곳의 계약을 코드로 못 박는다.

네트워크는 쓰지 않는다. presigned URL 을 흉내 내는 것은 로컬 HTTP 서버 하나이고,
검출기는 가짜를 주입한다. 계약이 깨지는지를 보는 것이지 모델을 보는 게 아니다.
"""

import http.server
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
    """저쪽 규약에서 height=null 은 '스케일 생략'이다. 말 안 한 것과 다르다."""
    assert job_runner.target_params({"height": None})["height"] == 0
    # 말을 안 하면 납품 기준(720p)이 적용된다 — 파이프라인 시그니처 기본값이 아니라.
    assert job_runner.target_params({"label": "x"})["height"] == 720


def test_unspecified_fields_get_the_service_defaults_not_the_signature_ones():
    """잡이 말하지 않은 것은 **튜닝된 기본값**으로 채운다.

    안 채우면 파이프라인 시그니처 기본값(batch_size=1, imgsz=960)으로 떨어진다.
    그건 '안전한 최소값' 이지 우리가 고른 값이 아니다 — L40S 에서 GPU 를 20% 만
    쓰고 한 편에 49.5초를 썼다(docs/issues/009).
    """
    from face_anonymizer import params

    p = job_runner.target_params({"label": "deid-720p"})
    assert p["batch_size"] == params.BATCH_SIZE != 1
    assert p["imgsz"] == params.IMGSZ != 960
    assert p["bitrate"] == "3500k"


def test_both_entry_points_share_one_set_of_defaults():
    """웹 화면과 큐 워커가 같은 영상을 다르게 처리하면 안 된다.

    기본값이 두 벌 있으면 언젠가 어긋난다. 실제로 어긋나 있었다.
    """
    pytest.importorskip("fastapi")
    from face_anonymizer import params
    from face_anonymizer.service import config

    worker = job_runner.target_params({})
    for k in params.JOB_OVERRIDABLE:
        assert worker[k] == config.JOB_DEFAULTS[k], k


def test_crf_target_turns_off_our_bitrate_policy():
    """화질 정책은 둘 중 하나만 — 타깃 비트레이트와 CRF 를 같이 걸면 안 된다."""
    p = job_runner.target_params({"crf": 24})
    assert p["crf"] == 24 and p["bitrate"] == "" and p["max_bitrate"] == ""
    # 비트레이트로 말하면 우리 납품 기준 그대로다. crf 기본값이 같이 실려 있어도
    # bitrate 가 살아 있으면 파이프라인은 비트레이트 쪽 정책을 쓴다.
    p = job_runner.target_params({"bitrate": "3500k", "max_bitrate": "4000k"})
    assert p["bitrate"] == "3500k" and p["max_bitrate"] == "4000k"
    # 아무 말도 안 하면 납품 기준(720p / 3500k)이 그대로 적용된다
    assert job_runner.target_params({})["bitrate"] == "3500k"


def test_transfer_classifies_status_codes():
    """일시/영구 분류는 워커의 1차 판단이다 — 판정은 잡을 준 쪽이 한다."""
    assert 403 in transfer.TRANSIENT_STATUS      # presign 만료일 수 있다
    assert 429 in transfer.TRANSIENT_STATUS
    assert 503 in transfer.TRANSIENT_STATUS
    assert 404 not in transfer.TRANSIENT_STATUS


# ── 검수 딱지 ────────────────────────────────────────────────────────────────

class _NoFace:
    """얼굴을 하나도 못 찾는 검출기. 풍경 영상과 설정 오류가 똑같이 이렇게 된다."""

    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        return [[] for _ in frames]


def test_zero_detections_completes_but_asks_for_review(bucket):
    """검출 0건은 **실패가 아니다.** 다만 사람이 봐야 한다고 딱지를 붙인다.

    얼굴이 없는 풍경 영상은 0 이 정당한 결과다. 그런데 가중치 손상·회전된 영상·
    잘못된 imgsz 도 결과가 똑같이 0 이고 그때는 원본이 그대로 나간다. 코드가 둘을
    구분할 수 없으므로 판단을 사람에게 넘기되 사실은 반드시 같이 보낸다.
    """
    out = job_runner.run_job(
        make_job(bucket),
        anonymizer=VideoAnonymizer(detector=_NoFace()))

    assert out["review_needed"] is True
    codes = [i["code"] for i in out["review"]]
    assert "no-detections" in codes
    assert "얼굴이 없는 영상이 맞는지" in out["review"][0]["message"]
    # 결과물은 그대로 올라간다 — 실패가 아니므로 재시도 대상도 아니다.
    assert len(bucket.puts) == 1
    assert out["targets"][0]["detected_frames"] == 0


def test_normal_result_asks_for_nothing(bucket):
    out = job_runner.run_job(make_job(bucket), anonymizer=anon(bucket))
    assert out["review_needed"] is False
    assert out["review"] == []


def test_only_human_worthy_warnings_become_review():
    """참고용 경고까지 사람을 부르면 딱지가 의미를 잃는다."""
    review = job_runner.review_of(["no-detections", "audio: ffmpeg-timeout",
                                   "decode-unverified", "low-detection-rate: 0.50%"])
    assert [i["code"] for i in review] == ["no-detections", "low-detection-rate"]
    # 수치는 원문 그대로 남는다 — 요약만 남기면 "얼마나 낮았는데?" 에 답 못 한다.
    assert review[1]["detail"] == "low-detection-rate: 0.50%"

    notices = job_runner.notices_of(["audio: ffmpeg-timeout", "decode-unverified"])
    assert {i["code"] for i in notices} == {"audio", "decode-unverified"}


def test_stage_timing_travels_with_the_result(bucket):
    """'느리다' 까지만 알면 아무 결정도 못 한다 — **어디가** 느린지가 필요하다.

    검출이 대부분이면 GPU 를 늘리는 수밖에 없고, 렌더·인제스트가 대부분이면 GPU 가
    놀고 있다는 뜻이라 대응이 정반대다. 워커를 몇 대 붙일지가 이 값으로 갈린다.
    """
    out = job_runner.run_job(make_job(bucket), anonymizer=anon(bucket))

    t = out["targets"][0]["timing"]
    assert {"ingest", "detect", "track", "render", "audio", "total"} <= set(t)
    assert t["total"] > 0
    # 짧은 클립에서 소수점 첫째 자리로 반올림하면 단계가 전부 0.0 이 되어
    # 어디가 느린지 안 보인다.
    assert all(isinstance(v, float) for v in t.values())
    assert sum(t[k] for k in ("ingest", "detect", "track", "render", "audio")) \
        <= t["total"] + 0.5


# ── 작은 인스턴스 대비 ────────────────────────────────────────────────────────

class _OOMOnce:
    """batch 가 threshold 보다 크면 CUDA OOM 을 흉내 낸다."""

    def __init__(self, threshold=4):
        self.threshold = threshold
        self.tried = []

    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        n = len(frames)
        self.tried.append(n)
        if n > self.threshold:
            raise RuntimeError(
                f"CUDA out of memory. Tried to allocate 2.00 GiB (batch {n})")
        return [[] for _ in frames]


def test_oom_shrinks_the_batch_instead_of_failing(bucket):
    """운영 인스턴스는 개발기보다 작다. **OOM 으로 큐가 통째로 죽으면 안 된다.**

    OOM 은 파이프라인 예외라 그냥 두면 '영구 실패' 로 분류되고, 그러면 인스턴스를
    줄이는 순간 들어오는 영상이 전부 재시도 없이 실패한다. 메모리 부족은 이 영상의
    문제가 아니라 환경의 문제다.
    """
    det = _OOMOnce(threshold=4)
    out = job_runner.run_job(
        make_job(bucket, targets=[{"label": "deid-720p", "batch_size": 32,
                                   "put_url": f"{bucket.base}/out.mp4"}]),
        anonymizer=VideoAnonymizer(detector=det))

    assert max(det.tried) <= 32 and min(det.tried) <= 4   # 줄여 가며 다시 했다
    assert len(bucket.puts) == 1                          # 결국 성공했다
    codes = [n["code"] for n in out["notices"]]
    assert "batch-reduced" in codes                       # 조용히 줄이지 않는다


def test_oom_at_batch_one_is_transient_not_permanent(bucket):
    """1까지 내려도 안 되면 더 큰 워커에서는 될 수 있다 — 일시 실패다."""
    det = _OOMOnce(threshold=0)                           # 무슨 배치든 터진다
    with pytest.raises(job_runner.JobError) as e:
        job_runner.run_job(make_job(bucket),
                           anonymizer=VideoAnonymizer(detector=det))
    assert e.value.transient is True and e.value.stage == "oom"


def test_only_oom_is_retried_other_errors_stay_permanent(bucket):
    """메모리 말고 다른 이유로 터진 것을 배치 줄여 다시 해 봐야 소용없다."""
    assert job_runner.is_oom(RuntimeError("CUDA out of memory")) is True
    assert job_runner.is_oom(ValueError("모르는 익명화 방식입니다")) is False


# ---------------------------------------------------------------------------
# 진행률 — 화면이 그릴 수 있는 값 중 **우리만 아는 것**


def _beat():
    got = []
    b = job_runner._Beat(got.append, 60)
    # 하트비트 간격은 1초 아래로 못 내려간다 — 프레임마다 큐로 보내면 초당
    # 수십 건이 된다. 테스트에서는 그 눌림을 풀어 매 호출을 본다.
    b.every = 0.0
    return b, got


def test_percent_never_goes_backwards_even_when_a_stage_is_skipped():
    """**되감기는 진행바는 고장으로 읽힌다.**

    h264 원본이면 전사 단계가 통째로 없고, 짧은 클립이면 검출 콜백이 몇 번 안
    불린다. 단계별 퍼센트를 그대로 내보내면 그때마다 100% → 0% 로 떨어진다.
    조금 부정확한 진행률은 읽히지만, 뒤로 가는 진행률은 못 읽는다.
    """
    b, got = _beat()
    seen = []
    for stage, done, total in [("download", 5, 10), ("detect", 1, 100),
                               ("detect", 90, 100), ("render", 1, 100),
                               ("detect", 5, 100),        # 늦게 도착한 콜백
                               ("upload", 1, 1)]:
        b(("x", done, stage), stage, done, total)
        seen.append(b.percent)
    assert seen == sorted(seen)
    assert seen[-1] == 100.0            # 다 끝나면 꽉 찬다
    assert got                           # every_s=0 이라 매번 보냈다


def test_eta_is_withheld_while_the_estimate_is_still_noise():
    """5% 아래에서는 추정이 요동친다 — "남은 시간 47분" 이 뜨고 곧 3분이 된다.

    모르는 구간에서는 **안 보내는 편이 낫다.** 화면은 None 을 받으면 '계산 중'
    을 띄우면 되지만, 틀린 숫자를 받으면 그대로 띄운다.
    """
    import time as _t
    b, _ = _beat()
    b(("x", 1), "download", 1, 100)              # 0.08%
    assert b.snapshot()["eta_s"] is None
    b(("x", 50), "detect", 50, 100)              # 45%
    _t.sleep(0.15)                               # 걸린 시간이 0 이면 되짚을 게 없다
    assert b.snapshot()["eta_s"] is not None


def test_progress_carries_a_sentence_for_the_screen():
    """화면에 `detect` 를 그대로 띄울 수는 없다. 문장까지 우리가 만들어 준다."""
    b, _ = _beat()
    b(("x", 1), "detect", 1, 10)
    assert b.snapshot()["stage_label"] == "얼굴 찾는 중"
    assert b.snapshot()["stage"] == "detect"      # 기계가 볼 코드도 같이


def test_stall_watchdog_still_fires_with_the_new_signature():
    """진행률을 얹느라 정체 감시가 죽으면, 멎은 잡이 리스 만료까지 매달린다."""
    b, _ = _beat()
    b(("x", 1), "detect", 1, 10)
    b.last_move -= (job_runner.STALL_S + 1)
    with pytest.raises(job_runner.JobError) as e:
        b(("x", 1), "detect", 1, 10)
    assert e.value.stage == "stall" and e.value.transient is True


def test_failure_carries_a_face_the_screen_can_show():
    """`str(e)` 만 보내면 화면에 우리 내부 문구가 뜬다.

    코드는 기계가, 제목·힌트는 사람이 읽는다. service/errors.py 와 같은 분담이되
    그쪽을 임포트하지 않는다 — fastapi 를 컨테이너에 끌고 오게 된다.
    """
    p = job_runner.JobError("presign 만료", transient=True,
                            stage="download").as_dict()
    assert p["code"] == "download" and p["retryable"] is True
    assert p["title"] and p["hint"]              # 사람이 읽을 두 줄
    assert p["detail"] == "presign 만료"          # 원문도 잃지 않는다

    q = job_runner.JobError("뭔가", transient=False, stage="process").as_dict()
    assert q["retryable"] is False


def test_download_percent_needs_the_total_but_survives_without_it():
    """Content-Length 를 안 주는 서버가 있다. 그때도 단계 이름은 떠야 한다."""
    b, _ = _beat()
    b(1, "download", 1, 0)                       # total 을 모른다
    s = b.snapshot()
    assert s["percent"] == 0.0 and s["stage_label"] == "원본 받는 중"
