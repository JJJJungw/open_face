"""S3 연동 테스트.

boto3 없이, 네트워크 없이 돈다 — 가짜 클라이언트를 주입한다. 실제 자격 증명이
필요한 검증(권한, 리전)은 여기서 볼 수 없고 별도 관심사다.
"""

import datetime as dt
import os
import time

import pytest

from conftest import FakeDetector

pytest.importorskip("fastapi", reason="pip install -r requirements-serve.txt")
pytest.importorskip("httpx", reason="pip install -r requirements-dev.txt")

from fastapi.testclient import TestClient            # noqa: E402

from face_anonymizer import VideoAnonymizer                # noqa: E402
from face_anonymizer.service import jobs as jobsmod        # noqa: E402
from face_anonymizer.service import config, server, worker  # noqa: E402
from face_anonymizer.storage import s3 as s3mod              # noqa: E402


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

    def download_file(self, bucket, key, dest, Callback=None):
        self.downloads.append(key)
        if key not in self.objects:
            raise KeyError(key)
        data = self.objects[key][0]
        with open(dest, "wb") as f:
            # 실제 boto3 처럼 청크마다 콜백을 부른다. 여기서 취소가 걸린다.
            for i in range(0, len(data), 4096) or [0]:
                chunk = data[i:i + 4096]
                f.write(chunk)
                if Callback:
                    Callback(len(chunk))
            if not data and Callback:
                Callback(0)

    def generate_presigned_url(self, op, Params=None, ExpiresIn=None):
        # 진짜 S3 는 ResponseContentDisposition 을 쿼리로 실어 서명한다.
        # 그게 있어야 브라우저가 재생 대신 내려받기를 한다 — 흉내에서도 실어 준다.
        extra = ""
        cd = (Params or {}).get("ResponseContentDisposition")
        if cd:
            from urllib.parse import quote as _q
            extra = "&response-content-disposition=" + _q(cd)
        return f"https://signed/{Params['Key']}?e={ExpiresIn}{extra}"

    def put_object(self, Bucket, Key, Body=b""):
        self.objects[Key] = (Body, NOW)
        return {}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)
        return {}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key][0])}

    def upload_file(self, path, bucket, key, ExtraArgs=None, Callback=None):
        with open(path, "rb") as f:
            data = f.read()
        if Callback:
            for i in range(0, len(data), 4096):
                Callback(len(data[i:i + 4096]))
        self.uploaded[key] = data
        self.objects[key] = (data, NOW)      # 올린 뒤에는 목록에도 보여야 한다


def wait(c, jid, timeout=30.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        s = c.get(f"/api/jobs/{jid}").json()
        # review 도 **워커가 손을 뗀** 상태다. 여기 빼면 검출 0건인 합성 클립이
        # 영원히 안 끝난 것으로 보인다 — 남은 것은 사람의 확인이지 처리가 아니다.
        if s["status"] in ("done", "review", "failed", "cancelled"):
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
                          output_prefix="v1/results/face/", root_prefix="",
                          config=s3mod.providers.StorageConfig(
                              provider="s3", bucket="ax-mbc-label-data-storage",
                              output_prefix="v1/results/face/"))
    monkeypatch.setattr(s3mod, "get_store", lambda: store)
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(jobsmod, "JOBS", {})
    monkeypatch.setattr(worker, "current", None)
    monkeypatch.setattr(worker, "model_error", None)
    anon = VideoAnonymizer(detector=FakeDetector(size))
    monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(worker, "_anonymizer", anon)
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
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path / "jobs"))
    c = TestClient(server.app)
    assert c.get("/api/s3/objects").status_code == 404


def test_s3_job_downloads_processes_and_uploads(s3client):
    r = s3client.post("/api/jobs",
                      json={"s3_keys": ["videos/2026-08/f_00001_00_0000000_0042000_raw.mp4"],
                            "params": {"batch_size": 4, "keep_audio": False}})
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
    assert s3client.post("/api/jobs", json={}).status_code == 400

    src, n, size = make_video(name="both.mp4", frames=4)
    with open(src, "rb") as f:
        r = s3client.post("/api/jobs",
                          files={"file": ("both.mp4", f, "video/mp4")},
                          json={"s3_keys": ["videos/2026-08/f_00001_00_0000000_0042000_raw.mp4"]})
    assert r.status_code == 400


def test_rejects_traversal_key(s3client):
    r = s3client.post("/api/jobs", json={"s3_keys": ["../../etc/passwd.mp4"]})
    assert r.status_code == 400


def test_rejects_non_video_key(s3client):
    r = s3client.post("/api/jobs", json={"s3_keys": ["videos/2026-08/notes.txt"]})
    assert r.status_code == 415
    assert r.json()["code"] == "unsupported_media"


# ── 배치 · 결과 URL ─────────────────────────────────────────────────────────

KEY = "videos/2026-08/f_00001_00_0000000_0042000_raw.mp4"


def seed(client, n):
    """버킷에 서로 다른 영상 n 개를 더 넣고 그 키들을 준다.

    같은 키를 여러 번 보내면 이제 한 번만 들어간다(중복 제거). 개수를 보는
    테스트는 서로 다른 키를 써야 한다.
    """
    data = client.store.client.objects[KEY][0]
    keys = []
    for i in range(2, 2 + n):
        k = f"videos/2026-08/f_{i:05d}_00_0000000_0010000_raw.mp4"
        client.store.client.objects[k] = (data, NOW)
        keys.append(k)
    return keys


def test_batch_accepts_many_and_reports_each(s3client):
    """한 건이 거절돼도 나머지는 받는다 — 전체를 되돌리면 무엇이 들어갔는지
    호출하는 쪽이 알 수 없다."""
    (other,) = seed(s3client, 1)
    r = s3client.post("/api/jobs", json={
        "s3_keys": [KEY, "videos/2026-08/notes.txt", "../escape.mp4", other]})

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
    assert jobsmod.JOBS[jid].params["conf"] == 0.4
    assert jobsmod.JOBS[jid].params["method"] == server.JOB_DEFAULTS["method"]
    wait(s3client, jid, timeout=60)


def test_batch_rejects_empty_and_oversized(s3client, monkeypatch):
    assert s3client.post("/api/jobs", json={"s3_keys": []}).json()["code"] \
        == "missing_input"
    monkeypatch.setattr(config, "BATCH_MAX", 2)
    b = s3client.post("/api/jobs",
                      json={"s3_keys": [KEY] + seed(s3client, 2)}).json()
    assert b["code"] == "batch_too_large" and b["limit"] == 2


def test_result_gives_presigned_url_for_s3_job(s3client):
    jid = s3client.post("/api/jobs", json={"s3_keys": [KEY]}).json()["accepted"][0]["id"]
    wait(s3client, jid, timeout=60)

    r = s3client.get(f"/api/jobs/{jid}/result")
    assert r.status_code == 200
    d = r.json()
    assert d["via"] == "s3"
    assert d["s3_key"] == "v1/results/face/2026-08_deid/f_00001_00_0000000_0042000_deid.mp4"
    assert d["download_url"].startswith("https://signed/")
    assert d["expires_in"] > 0


# ── 로컬 디스크 정리 (docs/issues/001) ──────────────────────────────────────
#
# 결과물을 버킷에 올린 뒤에도 로컬 사본을 들고 있으면 대량 처리에서 디스크가
# 먼저 찬다. 5분 클립 한 건이 입력 40~75MB + 결과물 131MB 다.

def workdir_files(client, jid):
    d = jobsmod.JOBS[jid].workdir
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def test_local_copy_is_removed_after_upload(s3client):
    """S3 에 올렸으면 로컬에 남길 이유가 없다. 기록만 남긴다."""
    jid = s3client.post("/api/jobs", json={"s3_keys": [KEY]}).json()["accepted"][0]["id"]
    s = wait(s3client, jid, timeout=60)

    assert s["status"] == "done"
    assert s3client.store.client.objects.get(s["s3_key"]), "결과물이 버킷에 있어야 한다"
    assert workdir_files(s3client, jid) == ["job.json"]


def test_keeping_the_local_copy_can_be_turned_back_on(s3client, monkeypatch):
    """로컬에서 결과를 바로 열어 보고 싶을 때가 있다."""
    monkeypatch.setattr(config, "KEEP_LOCAL", True)
    jid = s3client.post("/api/jobs", json={"s3_keys": [KEY]}).json()["accepted"][0]["id"]
    wait(s3client, jid, timeout=60)

    assert len(workdir_files(s3client, jid)) > 1


def test_failed_s3_job_keeps_only_the_record(s3client, monkeypatch):
    """실패 원인을 보는 데 필요한 건 job.json 뿐이다. 원본은 버킷에 있다."""
    monkeypatch.setattr(config, "MAX_ATTEMPTS", 1)

    class Broken:
        def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
            raise RuntimeError("일시적인 척하는 영구 오류")
    anon = VideoAnonymizer(detector=Broken())
    monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)

    jid = s3client.post("/api/jobs", json={"s3_keys": [KEY]}).json()["accepted"][0]["id"]
    s = wait(s3client, jid, timeout=60)

    assert s["status"] == "failed"
    assert workdir_files(s3client, jid) == ["job.json"]
    assert s["error"]["detail"]                      # 사유는 남아 있다


def test_result_before_done_is_409(s3client):
    jid = s3client.post("/api/jobs", json={"s3_keys": [KEY]}).json()["accepted"][0]["id"]
    jobsmod.JOBS[jid].status = "running"
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


def mark_done(client, key=None):
    """결과물이 이미 있는 상태로 만든다."""
    client.store.client.objects[client.store.output_key(key or KEY)] = (b"x", NOW)
    client.store._out_cache = (0.0, set())


def test_folder_submission_can_skip_processed(s3client):
    """폴더를 다시 돌릴 때 이미 끝난 건 건너뛴다."""
    mark_done(s3client)

    r = s3client.post("/api/jobs", json={"s3_prefix": "videos/2026-08/",
                                         "skip_processed": True})
    assert r.status_code == 409
    assert r.json()["code"] == "already_processed"


def test_picking_one_file_obeys_skip_processed_too(s3client):
    """체크박스 하나가 폴더에서는 먹고 파일을 골랐을 때는 안 먹으면 함정이다.

    화면에서 이미 처리된 파일을 골라 다시 누르면 그냥 다시 돌아 버렸다.
    """
    mark_done(s3client)

    r = s3client.post("/api/jobs", json={"s3_keys": [KEY],
                                         "skip_processed": True})

    assert r.status_code == 409
    assert r.json()["code"] == "already_processed"
    assert "건너뛰기" in r.json()["hint"]


def test_skip_can_be_turned_off_to_reprocess(s3client):
    """건너뛰기를 끄면 이미 끝난 것도 다시 돈다."""
    mark_done(s3client)

    r = s3client.post("/api/jobs", json={"s3_keys": [KEY],
                                         "skip_processed": False})

    assert r.status_code == 202, r.text
    wait(s3client, r.json()["accepted"][0]["id"], timeout=60)


def test_partial_skip_still_accepts_the_rest(s3client):
    """한 건이 이미 끝났다고 나머지까지 막으면 안 된다."""
    other, = seed(s3client, 1)
    mark_done(s3client, KEY)

    r = s3client.post("/api/jobs", json={"s3_keys": [KEY, other],
                                         "skip_processed": True})

    assert r.status_code == 202, r.text
    assert [a["s3_key"] for a in r.json()["accepted"]] == [other]
    assert r.json()["rejected"][0]["error"]["code"] == "already_processed"
    wait(s3client, r.json()["accepted"][0]["id"], timeout=60)


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
    many = s3client.post("/api/jobs", json={"s3_keys": seed(s3client, 2)})

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
    three = [KEY] + seed(s3client, 2)
    monkeypatch.setattr(config, "BATCH_MAX", 2)
    capped = s3client.post("/api/jobs", json={"s3_keys": three})
    assert capped.status_code == 400
    assert capped.json()["code"] == "batch_too_large"

    monkeypatch.setattr(config, "BATCH_MAX", 0)          # 기본값
    r = s3client.post("/api/jobs", json={"s3_keys": three})
    assert r.status_code == 202, r.text
    assert len(r.json()["accepted"]) == 3
    for a in r.json()["accepted"]:
        wait(s3client, a["id"], timeout=60)


def test_files_and_folders_can_be_submitted_together(s3client):
    """화면에서 파일 두 개와 폴더 하나를 같이 체크하는 게 자연스럽다.

    펼친 결과가 겹치면 한 번만 들어가야 한다 — 폴더 안에 있는 파일을 따로
    체크했다고 두 번 돌 이유가 없다.
    """
    other = "videos/2026-09/f_00002_00_0000000_0031000_raw.mp4"
    s3client.store.client.objects[other] = \
        (s3client.store.client.objects[KEY][0], NOW)

    r = s3client.post("/api/jobs", json={
        "s3_keys": [KEY],                       # 폴더를 펼치면 또 나오는 키
        "s3_prefix": ["videos/2026-08/", "videos/2026-09/"]})

    assert r.status_code == 202, r.text
    got = [a["s3_key"] for a in r.json()["accepted"]]
    assert got == [KEY, other]                  # 중복 없이, 고른 순서대로
    for a in r.json()["accepted"]:
        wait(s3client, a["id"], timeout=60)


def test_prefix_takes_a_single_string_too(s3client):
    """폴더 하나면 배열로 감싸지 않아도 된다."""
    r = s3client.post("/api/jobs", json={"s3_prefix": "videos/2026-08/"})
    assert r.status_code == 202, r.text
    for a in r.json()["accepted"]:
        wait(s3client, a["id"], timeout=60)



# ── S3 전송 중 취소 · 진행률 (docs/issues/004) ──────────────────────────────
#
# 취소는 협조적이다. 플래그를 세우면 작업 쪽이 스스로 확인해서 빠져나온다.
# 그런데 확인하는 자리가 파이프라인의 진행률 콜백 하나뿐이라, 그 앞뒤인 S3
# 전송 구간에서는 취소를 눌러도 전송이 끝날 때까지 안 멈췄다.

def test_download_reports_progress(s3client):
    """받는 동안 진행률이 멈춰 있으면 멈춘 것처럼 보인다."""
    seen = []
    s3client.store.download(KEY, "/tmp/_fa_dl_test.mp4",
                            callback=lambda n: seen.append(n))
    os.remove("/tmp/_fa_dl_test.mp4")

    assert seen, "콜백이 한 번도 안 불렸다"
    assert sum(seen) == len(s3client.store.client.objects[KEY][0])


def test_upload_reports_progress(s3client, tmp_path):
    p = tmp_path / "out.mp4"
    p.write_bytes(b"z" * 12000)
    seen = []

    s3client.store.upload(str(p), "v1/results/face/x_deid.mp4",
                          callback=lambda n: seen.append(n))

    assert sum(seen) == 12000


def test_cancel_during_download_stops_the_transfer(s3client):
    """전송 콜백에서 취소를 확인하지 않으면 다 받을 때까지 안 멈춘다."""
    def cancel_now(_chunk):
        raise s3mod.TransferAborted()

    with pytest.raises(s3mod.TransferAborted):
        s3client.store.download(KEY, "/tmp/_fa_cancel_test.mp4",
                                callback=cancel_now)
    if os.path.exists("/tmp/_fa_cancel_test.mp4"):
        os.remove("/tmp/_fa_cancel_test.mp4")


def test_transfer_abort_is_not_reported_as_an_s3_error(s3client, tmp_path):
    """사용자가 취소한 것을 'S3 호출 실패' 로 보고하면 원인이 뒤바뀐다."""
    p = tmp_path / "x.mp4"
    p.write_bytes(b"z" * 9000)

    def stop(_chunk):
        raise s3mod.TransferAborted()

    with pytest.raises(s3mod.TransferAborted):
        s3client.store.upload(str(p), "v1/results/face/y_deid.mp4", callback=stop)


def test_aborted_transfer_ends_as_cancelled_not_failed(s3client, monkeypatch):
    """전송 중 취소가 '실패' 로 기록되면 원인이 뒤바뀐다."""
    def aborted(key, dest, callback=None):
        raise s3mod.TransferAborted()
    monkeypatch.setattr(s3client.store, "download", aborted)

    jid = s3client.post("/api/jobs", json={"s3_keys": [KEY]}).json()["accepted"][0]["id"]
    s = wait(s3client, jid, timeout=30)

    assert s["status"] == "cancelled", s.get("error")
    assert s["error"]["code"] == "cancelled"


def test_worker_callback_aborts_when_cancel_is_requested(s3client, monkeypatch):
    """취소 플래그를 세우면 전송 콜백이 그 자리에서 중단을 요청해야 한다."""
    seen = {}

    def capture(key, dest, callback=None):
        seen["cb"] = callback
        raise s3mod.TransferAborted()          # 여기서 멈춰 콜백만 꺼내 온다
    monkeypatch.setattr(s3client.store, "download", capture)

    jid = s3client.post("/api/jobs", json={"s3_keys": [KEY]}).json()["accepted"][0]["id"]
    wait(s3client, jid, timeout=30)

    j = jobsmod.JOBS[jid]
    j.cancel = True
    with pytest.raises(s3mod.TransferAborted):
        seen["cb"](1024)


def test_review_does_not_hold_the_local_copy_hostage(s3client, monkeypatch):
    """검수 대기라고 로컬을 붙들지 않는다.

    붙들면 검수가 밀린 만큼 디스크가 차고, 결국 새 작업 제출이 거부된다
    (docs/issues/001 이 그 형태로 되살아난다). 다운로드가 "로컬에 없으면 S3 로
    302" 라서 검수하는 사람은 그대로 볼 수 있다 — 붙들 이유가 없다.
    """
    class Blind:
        def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
            return [[] for _ in frames]
    from face_anonymizer import VideoAnonymizer
    from face_anonymizer.service import worker
    anon = VideoAnonymizer(detector=Blind())
    monkeypatch.setattr(worker, "get_anonymizer", lambda: anon)
    monkeypatch.setattr(worker, "_anonymizer", anon)

    jid = s3client.post("/api/jobs", json={"s3_keys": [KEY]}).json()["accepted"][0]["id"]
    s = wait(s3client, jid, timeout=60)

    assert s["status"] == "review"
    # 정리는 상태가 바뀐 **뒤에** 돈다. 바로 보면 아직 파일이 남아 있을 수 있어
    # 테스트가 들쭉날쭉해진다 — 다운로드가 500 나던 것과 같은 틈이다.
    for _ in range(100):
        if workdir_files(s3client, jid) == ["job.json"]:
            break
        time.sleep(0.05)
    assert workdir_files(s3client, jid) == ["job.json"]
    # 그래도 볼 수 있어야 판정할 수 있다 — 버킷의 서명된 주소로 안내한다
    d = s3client.get(f"/api/jobs/{jid}/result").json()
    assert d["via"] == "s3" and d["download_url"].startswith("https://signed/")


# ---------------------------------------------------------------------------
# 어디에 붙을지 — 제공자


def test_s3_compatible_providers_need_no_adapter(monkeypatch):
    """NCP·R2·MinIO 는 **엔드포인트 주소만 다르다.**

    S3 API 를 그대로 쓰므로 어댑터가 아니라 설정으로 푼다. 이걸 어댑터로 만들면
    똑같은 코드를 제공자 수만큼 복사하게 된다.
    """
    from face_anonymizer.storage import providers

    ncp = providers.StorageConfig(provider="ncp", bucket="b")
    assert ncp.supported and ncp.ready
    assert ncp.endpoint == "https://kr.object.ncloudstorage.com"
    assert ncp.region == "kr-standard"          # 제공자 기본값이 채워진다

    aws = providers.StorageConfig(provider="s3", bucket="b")
    assert aws.endpoint is None                 # boto3 가 리전으로 정한다


def test_a_custom_endpoint_wins_over_the_provider_default(monkeypatch):
    """같은 NCP 라도 리전이 다르면 주소가 다르다. 직접 넣은 값이 이긴다."""
    from face_anonymizer.storage import providers
    c = providers.StorageConfig(provider="ncp", bucket="b",
                                endpoint="https://sg.object.ncloudstorage.com")
    assert c.endpoint == "https://sg.object.ncloudstorage.com"


def test_an_s3_compatible_provider_must_be_told_where(monkeypatch):
    """R2·MinIO 는 기본 주소가 없다. 주소 없이 준비됐다고 하면 안 된다."""
    from face_anonymizer.storage import providers
    c = providers.StorageConfig(provider="s3compat", bucket="b")
    assert c.supported and not c.ready
    assert providers.StorageConfig(provider="s3compat", bucket="b",
                                   endpoint="https://x").ready


def test_unsupported_providers_are_refused_out_loud(monkeypatch):
    """**조용히 안 되는 것이 제일 나쁘다.**

    GCS 를 골라 뒀는데 "S3 미설정" 으로 보이면 사람이 엉뚱한 데를 고친다.
    """
    from face_anonymizer.storage import providers, s3 as s3mod
    c = providers.StorageConfig(provider="gcs", bucket="b")
    assert not c.supported and not c.ready

    monkeypatch.setattr(s3mod, "CONFIG", c)
    monkeypatch.setattr(s3mod, "_store", None)
    assert s3mod.get_store() is None
    assert "지원하지 않습니다" in s3mod.unavailable_reason()


def test_the_endpoint_reaches_boto3(monkeypatch):
    """설정에 넣어 놓고 클라이언트에 안 넘기면 그 제공자가 통째로 못 쓰게 된다."""
    from face_anonymizer.storage import providers, s3 as s3mod
    seen = {}

    class FakeBoto:
        @staticmethod
        def client(name, **kw):
            seen.update(name=name, **kw)
            return object()

    monkeypatch.setitem(__import__("sys").modules, "boto3", FakeBoto)
    s3mod.make_client(providers.StorageConfig(provider="ncp", bucket="b"))
    assert seen["endpoint_url"] == "https://kr.object.ncloudstorage.com"
    assert seen["region_name"] == "kr-standard"

    seen.clear()
    s3mod.make_client(providers.StorageConfig(provider="s3", bucket="b"))
    assert "endpoint_url" not in seen           # AWS 는 리전으로 알아서 간다


def test_checksums_are_off_by_default():
    """NCP 는 botocore 가 붙이는 CRC32 를 AccessDenied 로 거절한다.

    권한 문제가 아니라 체크섬 문제인데 **돌아오는 말이 똑같다.** 그대로 두면
    이관 당일에 키·IAM·버킷 정책을 며칠 뒤지게 된다. 저쪽 레포도 같은 이유로
    같은 설정을 쓴다(rebornstudio 의 media/storage.py 주석).
    """
    pytest.importorskip("botocore")
    from face_anonymizer.storage import s3 as s3mod
    cfg = s3mod.client_config()
    assert cfg is not None and cfg.signature_version == "s3v4"
    # 낡은 botocore 에는 속성 자체가 없다 — 있을 때만 본다.
    if hasattr(cfg, "request_checksum_calculation"):
        assert cfg.request_checksum_calculation == "when_required"


def test_connection_check_separates_read_from_write(s3client):
    """**읽기만 되는 자격 증명이 흔하다.** 그걸 '연결됨' 이라고 하면 안 된다.

    잘못된 버킷에 900건을 넣고 나서 아는 것보다 넣기 전에 아는 편이 낫다.
    """
    assert s3client.post("/api/storage/test").json()["ok"] is True

    def no_write(**kw):
        raise RuntimeError("AccessDenied")
    s3client.store.client.put_object = no_write
    r = s3client.post("/api/storage/test")
    assert r.status_code >= 400
    assert "쓰지 못" in r.json()["detail"]


def test_storage_info_never_leaks_credentials(s3client):
    """우리는 키를 애초에 안 들고 있다. 응답에도 없어야 한다."""
    d = s3client.get("/api/storage").json()
    body = str(d).lower()
    assert "secret" not in body and "access_key" not in body
    assert d["current"]["bucket"] and d["first_run"] is False
    # 어디서 왔는지는 말하되 값은 절대 안 말한다.
    assert "source" in d["credentials"]
    ids = {p["id"] for p in d["providers"]}
    assert {"s3", "ncp", "gcs"} <= ids
    assert next(p for p in d["providers"] if p["id"] == "gcs")["supported"] is False


def test_a_key_whose_result_would_not_fit_is_refused_up_front(s3client):
    """**처리하고 나서 못 올리는 것보다 넣기 전에 아는 편이 낫다.**

    로컬 이름은 우리가 짧게 짓게 됐으니 더 이상 제약이 아니다. 남은 한계는
    결과물을 올릴 버킷 키뿐이고 그건 못 피한다 — 그때는 제출 시점에 건별로
    사유와 함께 돌려준다.
    """
    import unicodedata
    huge = unicodedata.normalize("NFD", "가" * 400) + ".mp4"
    r = s3client.post("/api/jobs", json={"s3_keys": ["videos/2026-08/" + huge]})
    assert r.status_code == 400
    d = r.json()
    assert d["code"] == "name_too_long"
    # 글자 수만 말하면 "60자밖에 안 되는데 왜?" 가 된다. 왜 긴지 같이 적는다.
    assert "자모가 분리" in d["detail"] and "바이트" in d["detail"]


def test_a_long_but_ordinary_name_still_goes_through(s3client):
    """한계는 버킷 키 1024바이트다. 그 아래는 길어도 받는다."""
    from face_anonymizer.service import server
    ok = "가" * 100 + ".mp4"                       # NFC 로 300바이트 남짓
    assert len(server.check_s3_key("videos/2026-08/" + ok)) > 0


def test_decomposed_names_get_a_notice_not_a_failure(s3client):
    """자모 분리는 처리에 문제가 없다 — 다만 화면 검색에 안 잡힌다.

    그걸 모르면 "파일이 분명히 있는데 검색하면 안 나온다" 로만 겪는다.
    """
    from face_anonymizer.service import server
    import unicodedata
    nfd = unicodedata.normalize("NFD", "나의아저씨.mp4")
    notes = server.name_notes(["v1/input/kbs/" + nfd, "v1/input/kbs/plain.mp4"])
    assert len(notes) == 1
    assert "1건" in notes[0] and "검색" in notes[0]
    assert server.name_notes(["v1/input/kbs/plain.mp4"]) == []
