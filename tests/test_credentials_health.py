"""자격 증명 확인 — `GET /api/credentials/health`.

**붙이기 전에 이것부터 친다.** "설정했는데 왜 안 되지" 를 첫 영상에서 만나지
않게 하는 자리다. 읽기와 쓰기를 따로 보는 이유는 **읽기만 되는 자격 증명이
흔하고**, 그 둘은 사람이 할 일이 다르기 때문이다.
"""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient           # noqa: E402

from face_anonymizer.service import server          # noqa: E402
from face_anonymizer.storage import s3 as s3mod     # noqa: E402


class Store:
    bucket = "b"
    config = None

    def __init__(self, read=True, write=True):
        self._read, self._write = read, write
        self.client = self
        self.deleted = []

    def list(self, prefix=""):
        if not self._read:
            raise s3mod.S3Error("버킷을 읽지 못했습니다 (b)")
        return [], []

    def put_object(self, **kw):
        if not self._write:
            raise RuntimeError("AccessDenied")
        return {}

    def delete_object(self, **kw):
        self.deleted.append(kw.get("Key"))
        return {}


@pytest.fixture
def client():
    return TestClient(server.app)


def test_it_says_where_the_credentials_came_from(client, monkeypatch):
    """되는 날에는 아무래도 좋지만, **안 되는 날에는 이 한 줄이 없어서 엉뚱한
    데를 뒤지게 된다.**"""
    monkeypatch.setattr(s3mod, "get_store", lambda: Store())
    monkeypatch.setattr(s3mod, "credential_source",
                        lambda: ("~/.aws/credentials", True))
    d = client.get("/api/credentials/health").json()
    assert d["ok"] is True
    assert d["credentials"] == {"source": "~/.aws/credentials", "present": True}
    assert d["read"] is True and d["write"] is True
    assert "checked_ms" in d


def test_read_and_write_are_reported_separately(client, monkeypatch):
    """읽기만 되는 자격 증명이 흔하다. 그걸 '연결 실패' 로 뭉치면, 사람은
    정책에 PutObject 를 더해야 한다는 것을 못 읽는다."""
    monkeypatch.setattr(s3mod, "get_store", lambda: Store(write=False))
    monkeypatch.setattr(s3mod, "credential_source", lambda: ("환경 변수", True))
    r = client.get("/api/credentials/health")
    assert r.status_code == 503
    d = r.json()
    assert d["ok"] is False
    assert d["read"] is True and d["write"] is False
    assert d["problem"]["code"] in ("s3_access_denied", "s3_upstream")


def test_a_read_failure_stops_before_it_writes(client, monkeypatch):
    """못 읽는데 쓰기를 시도하면 버킷에 쓰레기만 남는다."""
    st = Store(read=False)
    monkeypatch.setattr(s3mod, "get_store", lambda: st)
    monkeypatch.setattr(s3mod, "credential_source", lambda: ("없습니다", False))
    r = client.get("/api/credentials/health")
    assert r.status_code == 503
    d = r.json()
    assert d["read"] is False and d["write"] is None
    assert not st.deleted, "읽기가 안 되는데 쓰기를 시도했다"


def test_the_probe_object_is_removed(client, monkeypatch):
    """확인하려고 남긴 파일이 버킷에 쌓이면 안 된다."""
    st = Store()
    monkeypatch.setattr(s3mod, "get_store", lambda: st)
    monkeypatch.setattr(s3mod, "credential_source", lambda: ("환경 변수", True))
    client.get("/api/credentials/health")
    assert st.deleted and st.deleted[0].endswith(".fa-credential-check")


def test_without_a_store_it_is_not_ok(client, monkeypatch):
    monkeypatch.setattr(s3mod, "get_store", lambda: None)
    monkeypatch.setattr(s3mod, "credential_source", lambda: ("없습니다", False))
    monkeypatch.setattr(s3mod, "unavailable_reason", lambda: "버킷이 없습니다")
    r = client.get("/api/credentials/health")
    assert r.status_code == 503
    assert r.json()["problem"]["code"] == "s3_not_configured"


def test_it_never_returns_the_key_itself(client, monkeypatch):
    """출처는 말하고 값은 안 말한다."""
    import json

    monkeypatch.setattr(s3mod, "get_store", lambda: Store())
    monkeypatch.setattr(s3mod, "credential_source",
                        lambda: ("환경 변수 (AWS_ACCESS_KEY_ID)", True))
    monkeypatch.setattr(s3mod, "credentials",
                        lambda: {"aws_access_key_id": "AKIAEXAMPLE",
                                 "aws_secret_access_key": "s3cr3t"})
    blob = json.dumps(client.get("/api/credentials/health").json(),
                      ensure_ascii=False)
    assert "AKIAEXAMPLE" not in blob and "s3cr3t" not in blob
