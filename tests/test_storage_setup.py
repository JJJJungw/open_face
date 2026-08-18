"""첫 실행 설정 — **가져다 쓰는 사람이 처음 보는 화면.**

이 프로젝트는 남이 클론해 가서 자기 버킷에 붙이는 것을 전제로 한다. 그러면
"서버는 떴는데 뭘 해야 하는지 아무 데도 안 적혀 있는" 상태가 첫인상이 된다.
그래서 붙을 곳이 아직 없으면 화면에서 정할 수 있게 열고, 한 번 정해지면
닫는다. 여기서 보는 것은 그 문이 제때 열리고 제때 닫히는가다.

닫는 쪽이 더 중요하다. **우리 API 에는 인증이 없다.** 남이 이걸 공인 IP 에
그냥 띄우는 일은 반드시 생기고, 그때 설정 화면이 계속 열려 있으면 서버에 닿는
누구나 저장소를 자기 것으로 갈아치울 수 있다.
"""

import json

import pytest

pytest.importorskip("fastapi", reason="pip install -r requirements-serve.txt")
pytest.importorskip("httpx", reason="pip install -r requirements-dev.txt")

from fastapi.testclient import TestClient          # noqa: E402

from face_anonymizer.service import config, server  # noqa: E402
from face_anonymizer.storage import providers      # noqa: E402
from face_anonymizer.storage import s3 as s3mod    # noqa: E402

GOOD = "good-bucket"


class FakeClient:
    """`good-bucket` 만 읽힌다. 쓰기는 되는 곳이면 다 된다."""

    def __init__(self, writable=True):
        self.writable = writable
        self.written = {}

    def list_objects_v2(self, **kw):
        if kw.get("Bucket") != GOOD:
            raise RuntimeError(f"NoSuchBucket: {kw.get('Bucket')}")
        return {"CommonPrefixes": [], "Contents": []}

    def put_object(self, **kw):
        if not self.writable:
            raise RuntimeError("AccessDenied")
        self.written[kw["Key"]] = kw.get("Body")

    def delete_object(self, **kw):
        self.written.pop(kw["Key"], None)


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """아직 아무것도 안 정해진 서버. 진짜 첫 실행처럼."""
    monkeypatch.setenv("FA_JOBS_DIR", str(tmp_path))
    monkeypatch.delenv("FA_ALLOW_STORAGE_EDIT", raising=False)
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(s3mod, "CONFIG",
                        providers.StorageConfig(provider="s3", bucket=""))
    monkeypatch.setattr(s3mod, "_store", None)
    monkeypatch.setattr(s3mod, "_creds", None)
    client = FakeClient()
    monkeypatch.setattr(s3mod, "make_client", lambda config=None: client)
    c = TestClient(server.app)
    c.fake = client
    c.jobs_dir = tmp_path
    return c


def test_first_run_says_so_and_opens_the_door(fresh):
    """아무것도 없으면 화면이 '고르세요' 라고 해야 한다."""
    d = fresh.get("/api/storage").json()
    assert d["first_run"] is True and d["editable"] is True
    assert not d["lock_reason"]
    assert "골라" in d["note"]


def test_a_typo_does_not_get_saved(fresh):
    """**연결이 되는 것을 보고 나서 저장한다.**

    첫 실행에만 열리는 문이라, 오타 하나로 잠겨 버리면 고치러 들어갈 길이 같이
    막힌다. 되는 것을 확인한 뒤에 저장하면 그 사고가 성립하지 않는다.
    """
    r = fresh.post("/api/storage", json={"provider": "s3", "bucket": "typo"})
    assert r.status_code >= 400
    assert not (fresh.jobs_dir / providers.SAVED_NAME).exists()
    # 문은 그대로 열려 있다 — 다시 시도할 수 있어야 한다.
    assert fresh.get("/api/storage").json()["editable"] is True


def test_write_only_failure_also_blocks_the_save(fresh, monkeypatch):
    """읽기만 되는 자격 증명이 흔하다. 그걸 '연결됨' 이라고 하면
    900건을 다 처리하고 나서 결과를 못 올린다는 걸 알게 된다."""
    fresh.fake.writable = False
    r = fresh.post("/api/storage", json={"provider": "s3", "bucket": GOOD})
    assert r.status_code >= 400
    assert not (fresh.jobs_dir / providers.SAVED_NAME).exists()


def test_saving_locks_the_door(fresh):
    """한 번 정해지면 닫힌다. 우리 API 에는 인증이 없다."""
    r = fresh.post("/api/storage", json={"provider": "ncp", "bucket": GOOD,
                                         "root_prefix": "v1/input/"})
    assert r.status_code == 200, r.text
    # NCP 를 골랐을 뿐인데 주소·리전이 알아서 채워진다.
    cur = r.json()["current"]
    assert cur["endpoint"] == "https://kr.object.ncloudstorage.com"
    assert cur["region"] == "kr-standard"

    d = fresh.get("/api/storage").json()
    assert d["editable"] is False and d["first_run"] is False
    assert "FA_ALLOW_STORAGE_EDIT" in d["lock_reason"]

    r2 = fresh.post("/api/storage", json={"provider": "s3",
                                          "bucket": "someone-elses"})
    assert r2.status_code == 409
    assert r2.json()["code"] == "storage_locked"
    # 잠긴 뒤에도 원래 값이 그대로여야 한다 — 거절만 하고 바꿔 놓으면 최악이다.
    assert s3mod.CONFIG.bucket == GOOD


def test_the_switch_opens_it_again(fresh, monkeypatch):
    """잠긴 뒤에도 바꿔야 하면 **띄우는 사람이 명시적으로** 연다."""
    fresh.post("/api/storage", json={"provider": "s3", "bucket": GOOD})
    assert fresh.get("/api/storage").json()["editable"] is False
    monkeypatch.setenv("FA_ALLOW_STORAGE_EDIT", "1")
    assert fresh.get("/api/storage").json()["editable"] is True


def test_what_gets_saved_has_no_secrets_in_it(fresh):
    """파일에 남는 것은 제공자·버킷·주소·프리픽스뿐이다."""
    fresh.post("/api/storage", json={"provider": "ncp", "bucket": GOOD,
                                     "access_key": "AKIAEXAMPLE",
                                     "secret_key": "s3cr3t"})
    saved = json.loads((fresh.jobs_dir / providers.SAVED_NAME).read_text())
    assert saved["bucket"] == GOOD and saved["provider"] == "ncp"
    blob = json.dumps(saved)
    assert "AKIAEXAMPLE" not in blob and "s3cr3t" not in blob


def test_keys_given_on_the_screen_never_come_back_out(fresh):
    """열쇠는 메모리에만 둔다. 어떤 라우트로도 안 돌려준다."""
    fresh.post("/api/storage", json={"provider": "s3", "bucket": GOOD,
                                     "access_key": "AKIAEXAMPLE",
                                     "secret_key": "s3cr3t"})
    body = json.dumps(fresh.get("/api/storage").json())
    assert "AKIAEXAMPLE" not in body and "s3cr3t" not in body
    # 대신 **어디서 왔는지**는 말해 준다 — 그게 없으면 왜 되는지를 모른다.
    assert "메모리" in fresh.get("/api/storage").json()["credentials"]["source"]


def test_it_tells_you_how_to_make_the_keys_permanent(fresh):
    """메모리에 든 것은 재시작하면 사라진다. 그 사실을 나중에 알게 하지 않는다."""
    d = fresh.post("/api/storage", json={"provider": "s3", "bucket": GOOD,
                                         "access_key": "AKIAEXAMPLE",
                                         "secret_key": "s3cr3t"}).json()
    hint = " ".join(d["persist_hint"])
    assert "AWS_ACCESS_KEY_ID" in hint and "AWS_SECRET_ACCESS_KEY" in hint
    assert "s3cr3t" not in hint                     # 값은 여기에도 안 적는다


def test_the_setting_survives_a_restart_but_the_keys_do_not(fresh, monkeypatch):
    """설정은 파일에 남고 열쇠는 안 남는다 — 그게 이 방식의 값이자 대가다."""
    fresh.post("/api/storage", json={"provider": "ncp", "bucket": GOOD,
                                     "root_prefix": "v1/input/",
                                     "access_key": "AKIAEXAMPLE",
                                     "secret_key": "s3cr3t"})
    # 다시 띄운 셈 치고 처음부터 읽는다.
    monkeypatch.setattr(s3mod, "_creds", None)
    again = providers.StorageConfig.from_env()
    assert again.bucket == GOOD and again.provider == "ncp"
    assert again.root_prefix == "v1/input/"
    assert s3mod.credentials() is None


def test_the_environment_still_wins(fresh, monkeypatch):
    """`.env` 를 고쳤는데 화면에서 눌러 둔 옛 값이 이기면, 사람은 있지도 않은
    문제를 찾게 된다."""
    fresh.post("/api/storage", json={"provider": "ncp", "bucket": GOOD})
    monkeypatch.setenv("FA_S3_BUCKET", "from-env")
    assert providers.StorageConfig.from_env().bucket == "from-env"


def test_unsupported_providers_are_refused_at_the_door(fresh):
    """고르면 분명하게 거절한다. 조용히 안 되는 것이 제일 나쁘다."""
    r = fresh.post("/api/storage", json={"provider": "gcs", "bucket": GOOD})
    assert r.status_code >= 400
    assert "지원하지 않습니다" in r.json()["detail"]


def test_a_broken_saved_file_does_not_stop_the_server(fresh):
    """설정 파일 하나 때문에 서버가 안 뜨면, 고치러 들어갈 화면도 같이 없어진다."""
    (fresh.jobs_dir / providers.SAVED_NAME).write_text("{ 이건 JSON 이 아니다")
    assert providers.load_saved() == {}
    assert providers.StorageConfig.from_env().provider == "s3"


def test_the_screen_does_not_open_before_it_is_configured(fresh):
    """관문이 화면에 실제로 있는가.

    붙을 곳이 없는데 본 화면을 그리면 진척률·큐가 0 으로 채워져 나오고
    사이드바 탭은 눌러도 빈 화면과 404 뿐이다. 그 화면들은 '아직 아니다' 라고
    말해 주지 않는다 — 그냥 아무것도 없는 도구처럼 보인다.
    """
    html = fresh.get("/").text
    assert 'id="setup"' in html
    assert '<div class="app" hidden>' in html      # 통과해야 열린다
    assert "어디에 붙을지" in html
