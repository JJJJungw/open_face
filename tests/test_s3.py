"""S3 연동 테스트.

boto3 없이, 네트워크 없이 돈다 — 가짜 클라이언트를 주입한다. 실제 자격 증명이
필요한 검증(권한, 리전)은 여기서 볼 수 없고 별도 관심사다.
"""

import datetime as dt
import os

import pytest

from conftest import FakeDetector

pytest.importorskip("fastapi", reason="requirements-serve.txt 미설치")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient            # noqa: E402

from face_anonymizer import VideoAnonymizer, server  # noqa: E402
from face_anonymizer import s3 as s3mod              # noqa: E402


NOW = dt.datetime(2026, 8, 7, 0, 18)


class FakeS3Client:
    """list_objects_v2 / download_file / upload_file 만 흉내 낸다."""

    def __init__(self, objects=None):
        # {key: (bytes, modified)}
        self.objects = dict(objects or {})
        self.uploaded = {}
        self.downloads = []

    def list_objects_v2(self, Bucket, Prefix="", Delimiter=None, MaxKeys=1000,
                        ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        contents, prefixes = [], set()
        for k in keys:
            rest = k[len(Prefix):]
            if Delimiter and Delimiter in rest:
                prefixes.add(Prefix + rest.split(Delimiter, 1)[0] + Delimiter)
            else:
                data, mod = self.objects[k]
                contents.append({"Key": k, "Size": len(data), "LastModified": mod})
        out = {"Contents": contents}
        if prefixes:
            out["CommonPrefixes"] = [{"Prefix": p} for p in sorted(prefixes)]
        return out

    def download_file(self, bucket, key, dest):
        self.downloads.append(key)
        if key not in self.objects:
            raise KeyError(key)
        with open(dest, "wb") as f:
            f.write(self.objects[key][0])

    def generate_presigned_url(self, op, Params=None, ExpiresIn=None):
        return f"https://signed/{Params['Key']}?e={ExpiresIn}"

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {}

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        with open(path, "rb") as f:
            data = f.read()
        self.uploaded[key] = data
        self.objects[key] = (data, NOW)      # 올린 뒤에는 목록에도 보여야 한다


def wait(c, jid, timeout=30.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        s = c.get(f"/api/jobs/{jid}").json()
        if s["status"] in ("done", "failed", "cancelled"):
            return s
        time.sleep(0.02)
    raise AssertionError("작업이 끝나지 않았다")


def make_store(objects=None):
    return s3mod.S3Store(bucket="ax-mbc-label-data-storage",
                         client=FakeS3Client(objects),
                         output_prefix="v1/results/face/", root_prefix="")


def test_list_shows_one_level_like_console():
    store = make_store({
        "videos/2026-08/face4.mp4": (b"x" * 10, NOW),
        "videos/2026-08/face7.mp4": (b"x" * 20, NOW),
        "videos/2026-08/raw/inner.mp4": (b"x" * 5, NOW),
        "other/thing.txt": (b"x", NOW),
    })
    folders, objects = store.list("videos/2026-08/")

    assert folders == ["videos/2026-08/raw/"]
    assert [o["key"] for o in objects] == ["videos/2026-08/face4.mp4",
                                           "videos/2026-08/face7.mp4"]
    assert objects[0]["size"] == 10
    assert objects[0]["modified"].startswith("2026-08-07")


def test_output_key_follows_dataset_convention():
    """정체성 필드는 그대로 두고 STATE 만 raw -> deid."""
    store = make_store()
    assert store.output_key("videos/2026-08/f_00001_00_0000000_0042000_raw.mp4") \
        == "v1/results/face/2026-08_deid/f_00001_00_0000000_0042000_deid.mp4"
    # 규칙 밖 파일도 같은 규칙을 탄다 — 폴더를 따라간다
    assert store.output_key("a/b/c/clip.mov") == "v1/results/face/c_deid/clip_deid.mp4"


def test_output_lands_under_the_input_folder():
    """결과는 입력 폴더 이름을 따라 나뉜다.

    한곳에 몰면 결과 폴더가 몇만 건이 되고, 어느 묶음에서 나온 건지 목록만
    보고는 알 수 없다.
    """
    store = make_store()
    # 입력 폴더 이름을 그대로 쓰고 _deid 만 붙인다: kbs/ -> kbs_deid/
    assert store.output_key("kbs/a_00001_00_0000000_0001000_raw.mp4") \
        == "v1/results/face/kbs_deid/a_00001_00_0000000_0001000_deid.mp4"
    assert store.output_key("mbc/a_00001_00_0000000_0001000_raw.mp4") \
        .startswith("v1/results/face/mbc_deid/")
    assert store.output_key("sbs/a_00001_00_0000000_0001000_raw.mp4") \
        .startswith("v1/results/face/sbs_deid/")
    # 폴더가 없는 입력(직접 업로드)은 예전처럼 결과 프리픽스 바로 밑
    assert store.output_key("clip.mp4") == "v1/results/face/clip_deid.mp4"


def test_processed_keys_sees_subfolders():
    """폴더별로 나눠 쌓아도 '이미 처리됨' 판정이 살아 있어야 한다."""
    key = "videos/2026-08/f_00001_00_0000000_0042000_raw.mp4"
    store = make_store({
        "v1/results/face/2026-08_deid/f_00001_00_0000000_0042000_deid.mp4":
            (b"x", NOW)})
    assert store.output_key(key) in store.processed_keys()


def test_processed_keys_are_listed_once_not_per_object():
    """객체마다 HEAD 를 날리면 목록 한 번에 수백 번 왕복한다."""
    store = make_store({"v1/results/face/2026-08_deid/f_00001_00_0000000_0042000_deid.mp4":
                        (b"x", NOW)})
    calls = []
    orig = store.client.list_objects_v2
    store.client.list_objects_v2 = lambda **kw: (calls.append(kw), orig(**kw))[1]

    assert "v1/results/face/2026-08_deid/f_00001_00_0000000_0042000_deid.mp4" \
        in store.processed_keys()
    store.processed_keys()                      # 캐시가 먹어야 한다
    assert len(calls) == 1


def test_download_rejects_empty_result():
    store = make_store({"a.mp4": (b"", NOW)})
    with pytest.raises(s3mod.S3Error):
        store.download("a.mp4", "/tmp/_fa_empty.mp4")


def test_upload_invalidates_processed_cache(tmp_path):
    store = make_store()
    store.processed_keys()
    p = tmp_path / "out.mp4"
    p.write_bytes(b"video")
    store.upload(str(p), "v1/results/face/out_deid.mp4")
    assert "v1/results/face/out_deid.mp4" in store.processed_keys()


# ── 서버 경로 ────────────────────────────────────────────────────────────────

@pytest.fixture
def s3client(tmp_path, monkeypatch, make_video):
    """S3 가 설정된 서버. 버킷에 영상 하나가 들어 있다."""
    src, n, size = make_video(name="f_00001_00_0000000_0042000_raw.mp4",
                              frames=12)
    data = open(src, "rb").read()
    store = s3mod.S3Store(bucket="ax-mbc-label-data-storage",
                          client=FakeS3Client({
                              "videos/2026-08/f_00001_00_0000000_0042000_raw.mp4": (data, NOW),
                              "videos/2026-08/notes.txt": (b"hi", NOW)}),
                          output_prefix="v1/results/face/", root_prefix="")
    monkeypatch.setattr(s3mod, "get_store", lambda: store)
    monkeypatch.setattr(server, "JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(server, "_JOBS", {})
    monkeypatch.setattr(server, "_current", None)
    monkeypatch.setattr(server, "_model_error", None)
    anon = VideoAnonymizer(detector=FakeDetector(size))
    monkeypatch.setattr(server, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(server, "_anonymizer", anon)
    c = TestClient(server.app)
    c.store = store
    c.frames = n
    return c


def test_objects_endpoint(s3client):
    r = s3client.get("/api/s3/objects?prefix=videos/2026-08/")
    assert r.status_code == 200
    d = r.json()
    assert d["bucket"] == "ax-mbc-label-data-storage"
    assert d["output_prefix"] == "v1/results/face/"
    assert [o["key"] for o in d["objects"]] == [
        "videos/2026-08/f_00001_00_0000000_0042000_raw.mp4",
        "videos/2026-08/notes.txt"]
    assert d["objects"][0]["processed"] is False


def test_objects_endpoint_404_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(s3mod, "get_store", lambda: None)
    monkeypatch.setattr(server, "JOBS_DIR", str(tmp_path / "jobs"))
    c = TestClient(server.app)
    assert c.get("/api/s3/objects").status_code == 404


def test_s3_job_downloads_processes_and_uploads(s3client):
    r = s3client.post("/api/jobs", data={"s3_key": "videos/2026-08/f_00001_00_0000000_0042000_raw.mp4",
                                         "batch_size": "4", "keep_audio": "false"})
    assert r.status_code == 202, r.text
    jid = r.json()["accepted"][0]["id"]

    import time
    for _ in range(300):
        s = s3client.get(f"/api/jobs/{jid}").json()
        if s["status"] in ("done", "failed"):
            break
        time.sleep(0.02)

    assert s["status"] == "done", s.get("error")
    assert s["result"]["frames"] == s3client.frames
    assert s["result"]["s3_output"] \
        == "v1/results/face/2026-08_deid/f_00001_00_0000000_0042000_deid.mp4"
    assert s3client.store.client.downloads == [
        "videos/2026-08/f_00001_00_0000000_0042000_raw.mp4"]
    assert s3client.store.client.uploaded[
        "v1/results/face/2026-08_deid/f_00001_00_0000000_0042000_deid.mp4"]


def test_processed_flag_appears_after_run(s3client):
    s3client.store.client.objects[
        "v1/results/face/2026-08_deid/f_00001_00_0000000_0042000_deid.mp4"] = (b"x", NOW)
    s3client.store._out_cache = (0.0, set())
    d = s3client.get("/api/s3/objects?prefix=videos/2026-08/").json()
    assert d["objects"][0]["processed"] is True


def test_requires_exactly_one_input(s3client, make_video):
    """file 과 s3_key 를 둘 다 주거나 둘 다 안 주면 거절한다."""
    assert s3client.post("/api/jobs", data={}).status_code == 400

    src, n, size = make_video(name="both.mp4", frames=4)
    with open(src, "rb") as f:
        r = s3client.post("/api/jobs",
                          files={"file": ("both.mp4", f, "video/mp4")},
                          data={"s3_key": "videos/2026-08/f_00001_00_0000000_0042000_raw.mp4"})
    assert r.status_code == 400


def test_rejects_traversal_key(s3client):
    r = s3client.post("/api/jobs", data={"s3_key": "../../etc/passwd.mp4"})
    assert r.status_code == 400


def test_rejects_non_video_key(s3client):
    r = s3client.post("/api/jobs", data={"s3_key": "videos/2026-08/notes.txt"})
    assert r.status_code == 415
    assert r.json()["code"] == "unsupported_media"


# ── 배치 · 결과 URL ─────────────────────────────────────────────────────────

KEY = "videos/2026-08/f_00001_00_0000000_0042000_raw.mp4"


def test_batch_accepts_many_and_reports_each(s3client):
    """한 건이 거절돼도 나머지는 받는다 — 전체를 되돌리면 무엇이 들어갔는지
    호출하는 쪽이 알 수 없다."""
    r = s3client.post("/api/jobs", json={
        "s3_keys": [KEY, "videos/2026-08/notes.txt", "../escape.mp4", KEY]})

    assert r.status_code == 202
    d = r.json()
    assert len(d["accepted"]) == 2
    assert len(d["rejected"]) == 2
    codes = {x["error"]["code"] for x in d["rejected"]}
    assert codes == {"unsupported_media", "invalid_key"}
    for a in d["accepted"]:
        wait(s3client, a["id"], timeout=60)


def test_batch_applies_shared_params(s3client):
    r = s3client.post("/api/jobs",
                      json={"s3_keys": [KEY], "params": {"conf": 0.4}})
    jid = r.json()["accepted"][0]["id"]
    assert server._JOBS[jid].params["conf"] == 0.4
    assert server._JOBS[jid].params["method"] == server.JOB_DEFAULTS["method"]
    wait(s3client, jid, timeout=60)


def test_batch_rejects_empty_and_oversized(s3client, monkeypatch):
    assert s3client.post("/api/jobs", json={"s3_keys": []}).json()["code"] \
        == "missing_input"
    monkeypatch.setattr(server, "BATCH_MAX", 2)
    b = s3client.post("/api/jobs", json={"s3_keys": [KEY, KEY, KEY]}).json()
    assert b["code"] == "batch_too_large" and b["limit"] == 2


def test_result_gives_presigned_url_for_s3_job(s3client):
    jid = s3client.post("/api/jobs", data={"s3_key": KEY}).json()["accepted"][0]["id"]
    wait(s3client, jid, timeout=60)

    r = s3client.get(f"/api/jobs/{jid}/result")
    assert r.status_code == 200
    d = r.json()
    assert d["via"] == "s3"
    assert d["s3_key"] == "v1/results/face/2026-08_deid/f_00001_00_0000000_0042000_deid.mp4"
    assert d["download_url"].startswith("https://signed/")
    assert d["expires_in"] > 0


def test_download_redirects_to_s3_when_local_copy_is_gone(s3client):
    """보관 기간에 로컬 사본이 정리돼도 S3 원본은 남아 있다."""
    jid = s3client.post("/api/jobs", data={"s3_key": KEY}).json()["accepted"][0]["id"]
    wait(s3client, jid, timeout=60)

    os.remove(server._JOBS[jid].output)
    r = s3client.get(f"/api/jobs/{jid}/download", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"].startswith("https://signed/")


def test_result_before_done_is_409(s3client):
    jid = s3client.post("/api/jobs", data={"s3_key": KEY}).json()["accepted"][0]["id"]
    server._JOBS[jid].status = "running"
    b = s3client.get(f"/api/jobs/{jid}/result")
    assert b.status_code == 409 and b.json()["code"] == "job_not_finished"


def test_s3_access_denied_is_distinguished(s3client):
    """권한 문제와 키 오타는 사용자가 해야 할 일이 다르다."""
    class Denied(Exception):
        response = {"Error": {"Code": "AccessDenied"}}

    def boom(**kw):
        raise Denied()
    s3client.store.client.list_objects_v2 = boom

    r = s3client.get("/api/s3/objects?prefix=videos/")
    assert r.status_code == 502
    assert r.json()["code"] == "s3_access_denied"
    assert "권한" in r.json()["hint"]


# ── 단일 진입점 ─────────────────────────────────────────────────────────────
#
# 한 건이든 여러 건이든 폴더든 POST /api/jobs 하나로 들어간다. 진입점을 나누면
# 클라이언트가 경우마다 분기해야 하고 화면에도 버튼이 그만큼 늘어난다.

def test_folder_submission_expands_prefix(s3client):
    r = s3client.post("/api/jobs", json={"s3_prefix": "videos/2026-08/"})

    assert r.status_code == 202
    d = r.json()
    # 영상만 골라 넣는다 (notes.txt 는 제외)
    assert len(d["accepted"]) == 1
    assert d["accepted"][0]["s3_key"] == KEY
    wait(s3client, d["accepted"][0]["id"], timeout=60)


def test_folder_submission_can_skip_processed(s3client):
    """폴더를 다시 돌릴 때 이미 끝난 건 건너뛴다."""
    s3client.store.client.objects[
        "v1/results/face/2026-08_deid/f_00001_00_0000000_0042000_deid.mp4"] = (b"x", NOW)
    s3client.store._out_cache = (0.0, set())

    r = s3client.post("/api/jobs", json={"s3_prefix": "videos/2026-08/",
                                         "skip_processed": True})
    assert r.status_code == 400
    assert r.json()["code"] == "batch_empty"


def test_folder_recursive_includes_subfolders(s3client):
    s3client.store.client.objects["videos/2026-08/sub/f_00002_00_0000000_0010000_raw.mp4"] = \
        (s3client.store.client.objects[KEY][0], NOW)

    flat = s3client.post("/api/jobs", json={"s3_prefix": "videos/2026-08/"}).json()
    deep = s3client.post("/api/jobs", json={"s3_prefix": "videos/2026-08/",
                                            "recursive": True}).json()

    assert len(flat["accepted"]) == 1
    assert len(deep["accepted"]) == 2
    for d in (flat, deep):
        for a in d["accepted"]:
            wait(s3client, a["id"], timeout=60)


def test_folder_submission_excludes_deid_outputs(s3client):
    """결과물이 입력 폴더에 같이 있어도 다시 집어넣지 않는다.

    skip_processed 로는 못 막는다 — deid 파일의 output_key() 는 자기 자신이라
    "아직 결과물이 없다" 로 판정된다. 그대로 두면 모자이크 위에 모자이크가
    한 번 더 올라간다.
    """
    s3client.store.client.objects[
        "videos/2026-08/f_00009_00_0000000_0010000_deid.mp4"] = \
        (s3client.store.client.objects[KEY][0], NOW)

    r = s3client.post("/api/jobs", json={"s3_prefix": "videos/2026-08/",
                                         "skip_processed": True})

    assert r.status_code == 202, r.text
    assert [a["s3_key"] for a in r.json()["accepted"]] == [KEY]
    for a in r.json()["accepted"]:
        wait(s3client, a["id"], timeout=60)


def test_single_key_and_many_keys_use_the_same_endpoint(s3client):
    one = s3client.post("/api/jobs", json={"s3_keys": [KEY]})
    many = s3client.post("/api/jobs", json={"s3_keys": [KEY, KEY]})

    assert one.status_code == many.status_code == 202
    assert set(one.json()) == set(many.json())          # 응답 형태가 같다
    assert len(one.json()["accepted"]) == 1
    assert len(many.json()["accepted"]) == 2
    for r in (one, many):
        for a in r.json()["accepted"]:
            wait(s3client, a["id"], timeout=60)


def test_all_rejected_is_an_error_not_202(s3client):
    """하나도 못 받았으면 202 를 줄 수 없다. 단건이면 그 사유가 응답 코드다."""
    r = s3client.post("/api/jobs", json={"s3_keys": ["videos/2026-08/notes.txt"]})
    assert r.status_code == 415
    assert r.json()["code"] == "unsupported_media"

    mixed = s3client.post("/api/jobs", json={
        "s3_keys": ["a.txt", "../b.mp4"]})
    assert mixed.status_code == 400
    assert len(mixed.json()["rejected"]) == 2


def test_batch_size_is_unbounded_by_default(s3client, monkeypatch):
    """폴더 하나에 수천 건이 들어 있는 게 정상이다. 상한에 걸려서 사용자가
    폴더를 손으로 쪼개게 만들면 안 된다. 필요하면 FA_BATCH_MAX 로 다시 건다."""
    monkeypatch.setattr(server, "BATCH_MAX", 2)
    capped = s3client.post("/api/jobs", json={"s3_keys": [KEY, KEY, KEY]})
    assert capped.status_code == 400
    assert capped.json()["code"] == "batch_too_large"

    monkeypatch.setattr(server, "BATCH_MAX", 0)          # 기본값
    r = s3client.post("/api/jobs", json={"s3_keys": [KEY, KEY, KEY]})
    assert r.status_code == 202, r.text
    assert len(r.json()["accepted"]) == 3
    for a in r.json()["accepted"]:
        wait(s3client, a["id"], timeout=60)


def test_cannot_mix_input_kinds(s3client):
    r = s3client.post("/api/jobs", json={"s3_keys": [KEY],
                                         "s3_prefix": "videos/"})
    assert r.status_code == 400
    assert r.json()["code"] == "conflicting_input"
