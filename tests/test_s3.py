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

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        with open(path, "rb") as f:
            data = f.read()
        self.uploaded[key] = data
        self.objects[key] = (data, NOW)      # 올린 뒤에는 목록에도 보여야 한다


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


def test_output_key_collects_results_in_one_place():
    store = make_store()
    assert store.output_key("videos/2026-08/face4.mp4") \
        == "v1/results/face/face4_anon.mp4"
    assert store.output_key("a/b/c/clip.mov") == "v1/results/face/clip_anon.mp4"


def test_processed_keys_are_listed_once_not_per_object():
    """객체마다 HEAD 를 날리면 목록 한 번에 수백 번 왕복한다."""
    store = make_store({"v1/results/face/face4_anon.mp4": (b"x", NOW)})
    calls = []
    orig = store.client.list_objects_v2
    store.client.list_objects_v2 = lambda **kw: (calls.append(kw), orig(**kw))[1]

    assert "v1/results/face/face4_anon.mp4" in store.processed_keys()
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
    store.upload(str(p), "v1/results/face/out_anon.mp4")
    assert "v1/results/face/out_anon.mp4" in store.processed_keys()


# ── 서버 경로 ────────────────────────────────────────────────────────────────

@pytest.fixture
def s3client(tmp_path, monkeypatch, make_video):
    """S3 가 설정된 서버. 버킷에 영상 하나가 들어 있다."""
    src, n, size = make_video(name="clip.mp4", frames=12)
    data = open(src, "rb").read()
    store = s3mod.S3Store(bucket="ax-mbc-label-data-storage",
                          client=FakeS3Client({
                              "videos/2026-08/clip.mp4": (data, NOW),
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
    assert [o["key"] for o in d["objects"]] == ["videos/2026-08/clip.mp4",
                                               "videos/2026-08/notes.txt"]
    assert d["objects"][0]["processed"] is False


def test_objects_endpoint_404_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(s3mod, "get_store", lambda: None)
    monkeypatch.setattr(server, "JOBS_DIR", str(tmp_path / "jobs"))
    c = TestClient(server.app)
    assert c.get("/api/s3/objects").status_code == 404


def test_s3_job_downloads_processes_and_uploads(s3client):
    r = s3client.post("/api/jobs", data={"s3_key": "videos/2026-08/clip.mp4",
                                         "batch_size": "4", "keep_audio": "false"})
    assert r.status_code == 202, r.text
    jid = r.json()["id"]

    import time
    for _ in range(300):
        s = s3client.get(f"/api/jobs/{jid}").json()
        if s["status"] in ("done", "failed"):
            break
        time.sleep(0.02)

    assert s["status"] == "done", s.get("error")
    assert s["result"]["frames"] == s3client.frames
    assert s["result"]["s3_output"] == "v1/results/face/clip_anon.mp4"
    assert s3client.store.client.downloads == ["videos/2026-08/clip.mp4"]
    assert s3client.store.client.uploaded["v1/results/face/clip_anon.mp4"]


def test_processed_flag_appears_after_run(s3client):
    s3client.store.client.objects["v1/results/face/clip_anon.mp4"] = (b"x", NOW)
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
                          data={"s3_key": "videos/2026-08/clip.mp4"})
    assert r.status_code == 400


def test_rejects_traversal_key(s3client):
    r = s3client.post("/api/jobs", data={"s3_key": "../../etc/passwd.mp4"})
    assert r.status_code == 400


def test_rejects_non_video_key(s3client):
    r = s3client.post("/api/jobs", data={"s3_key": "videos/2026-08/notes.txt"})
    assert r.status_code == 400
