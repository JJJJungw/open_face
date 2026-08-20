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

pytest.importorskip("fastapi", reason="pip install -r requirements/serve.txt")
pytest.importorskip("httpx", reason="pip install -r requirements/dev.txt")

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
    """아직 아무것도 안 정해진 서버. 진짜 첫 실행처럼.

    **환경부터 비운다.** `from_env()` 는 환경 변수를 저장 파일보다 먼저 본다 —
    그게 맞는 순서다(`.env` 를 고쳤는데 안 바뀌면 사람은 있지도 않은 문제를
    찾는다). 그런데 이 뒷정리를 안 하면 "진짜 첫 실행" 이 그 기계의 `.env` 를
    물려받는다. 실제로 EC2 에서 이 검사가 실제 버킷 이름을 읽고 실패했다 —
    코드가 아니라 그 기계의 설정을 보고 있었던 것이다.
    """
    monkeypatch.setenv("FA_JOBS_DIR", str(tmp_path))
    monkeypatch.delenv("FA_ALLOW_STORAGE_EDIT", raising=False)
    for k in ("FA_S3_BUCKET", "FA_S3_REGION", "FA_S3_ENDPOINT",
              "FA_S3_ROOT_PREFIX", "FA_S3_OUTPUT_PREFIX",
              "FA_STORAGE_PROVIDER", "FA_STORAGE_STORE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(s3mod, "CONFIG",
                        providers.StorageConfig(provider="s3", bucket=""))
    monkeypatch.setattr(s3mod, "_store", None)
    monkeypatch.setattr(s3mod, "_creds", None)
    client = FakeClient()
    monkeypatch.setattr(s3mod, "make_client", lambda config=None: client)
    # **https 로 붙은 셈 친다.** 열쇠를 받는 라우트는 평문 연결을 거절하므로
    # (아래 test_keys_are_refused_over_plaintext), 평범한 흐름을 보려면
    # 안전한 연결이어야 한다.
    c = TestClient(server.app, base_url="https://testserver")
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


def test_saving_connects(fresh):
    r = fresh.post("/api/storage", json={"provider": "ncp", "bucket": GOOD,
                                         "root_prefix": "v1/input/"})
    assert r.status_code == 200, r.text
    # NCP 를 골랐을 뿐인데 주소·리전이 알아서 채워진다.
    cur = r.json()["current"]
    assert cur["endpoint"] == "https://kr.object.ncloudstorage.com"
    assert cur["region"] == "kr-standard"

    d = fresh.get("/api/storage").json()
    assert d["first_run"] is False and d["current"]["ready"] is True


def test_a_locked_down_server_refuses_to_change(fresh, monkeypatch):
    """공개된 자리에 띄우는 배포는 화면에서 못 바꾸게 잠글 수 있어야 한다."""
    fresh.post("/api/storage", json={"provider": "s3", "bucket": GOOD})
    monkeypatch.setenv("FA_ALLOW_STORAGE_EDIT", "0")

    d = fresh.get("/api/storage").json()
    assert d["editable"] is False and d["lock_reason"]

    r = fresh.post("/api/storage", json={"provider": "s3",
                                         "bucket": "someone-elses"})
    assert r.status_code == 409 and r.json()["code"] == "storage_locked"
    assert fresh.delete("/api/storage").status_code == 409
    # 거절만 하고 바꿔 놓으면 최악이다.
    assert s3mod.CONFIG.bucket == GOOD


def test_disconnect_goes_back_to_the_first_screen(fresh):
    """다른 버킷으로 옮기려고 서버를 다시 띄워야 한다면 '고를 수 있다' 가 아니다."""
    fresh.post("/api/storage", json={"provider": "ncp", "bucket": GOOD,
                                     "access_key": "AKIAEXAMPLE",
                                     "secret_key": "s3cr3t"})
    assert (fresh.jobs_dir / providers.SAVED_NAME).exists()

    r = fresh.delete("/api/storage")
    assert r.status_code == 200, r.text

    d = fresh.get("/api/storage").json()
    assert d["first_run"] is True and d["editable"] is True
    # 남겨 둔 설정도 메모리의 열쇠도 같이 지운다.
    assert not (fresh.jobs_dir / providers.SAVED_NAME).exists()
    assert s3mod.credentials() is None
    # 그리고 곧바로 다른 곳에 붙을 수 있어야 한다.
    assert fresh.post("/api/storage",
                      json={"provider": "s3", "bucket": GOOD}).status_code == 200


def test_disconnect_waits_for_work_in_flight(fresh, monkeypatch):
    """돌고 있는 작업 밑에서 저장소를 빼면 그 작업은 결과를 올릴 곳을 잃는다."""
    from face_anonymizer.service import jobs as jobsmod
    fresh.post("/api/storage", json={"provider": "s3", "bucket": GOOD})
    monkeypatch.setattr(jobsmod, "counts",
                        lambda: {"running": 1, "queued": 0, "review": 0})

    r = fresh.delete("/api/storage")
    assert r.status_code == 409 and r.json()["code"] == "storage_busy"
    assert s3mod.CONFIG.bucket == GOOD           # 그대로 붙어 있어야 한다


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


# ── 보안 ────────────────────────────────────────────────────────────────────

def test_keys_are_refused_over_plaintext(fresh):
    """평문으로 온 열쇠는 이미 경로 위의 누구나 봤다고 봐야 한다.

    그걸 받아서 '메모리에만 둡니다' 라고 하면, 안전하게 다뤘다는 인상만 주고
    실제로는 아니다. 받지 않는 편이 정직하다.
    """
    plain = TestClient(server.app, base_url="http://not-localhost")
    r = plain.post("/api/storage", json={"provider": "s3", "bucket": GOOD,
                                         "access_key": "AKIAEXAMPLE",
                                         "secret_key": "s3cr3t"})
    assert r.status_code == 400 and r.json()["code"] == "insecure_transport"
    assert s3mod.credentials() is None

    # 열쇠 없이 붙는 것은 평문에서도 된다 — 비밀이 오가지 않기 때문이다.
    assert plain.post("/api/storage",
                      json={"provider": "s3", "bucket": GOOD}).status_code == 200


def test_a_proxy_that_terminates_tls_counts_as_secure(fresh):
    """리버스 프록시 뒤가 정상적인 배포다. 그걸 무시하면 제대로 감싼 서버에서도
    키를 못 넣게 된다."""
    plain = TestClient(server.app, base_url="http://behind-proxy")
    r = plain.post("/api/storage",
                   headers={"X-Forwarded-Proto": "https"},
                   json={"provider": "s3", "bucket": GOOD,
                         "access_key": "AKIAEXAMPLE", "secret_key": "s3cr3t"})
    assert r.status_code == 200, r.text


def test_the_module_path_is_not_something_a_request_can_choose(fresh):
    """**`store` 는 파이썬 모듈 경로다.**

    받는 순간 `import_module()` 에 그대로 들어가고, 임포트만으로 코드가 도는
    모듈은 세상에 얼마든지 있다. 인증 없는 라우트에서 그걸 받는 것은 남의
    서버에서 무엇을 실행할지 고르게 해 주는 일이다. 구현을 갈아 끼우는 것은
    서버를 띄우는 사람의 일이라 환경 변수로만 한다.
    """
    r = fresh.post("/api/storage", json={"provider": "s3", "bucket": GOOD,
                                         "store": "webbrowser:Anything"})
    assert r.status_code == 200, r.text          # 그냥 무시하고 정상 연결된다
    saved = json.loads((fresh.jobs_dir / providers.SAVED_NAME).read_text())
    assert saved["store"] is None
    assert s3mod.CONFIG.store is None


def test_endpoints_that_are_never_storage_are_refused(fresh):
    """서버가 대신 요청을 보내 줄 주소다 — 임의의 주소를 받으면 SSRF 다.

    메타데이터 주소(169.254.169.254)는 인스턴스 역할의 임시 키를 그대로 내주는
    자리라, 이런 기능이 생기는 순간 첫 번째 표적이 된다.
    """
    for bad in ("http://169.254.169.254/", "http://metadata.google.internal",
                "file:///etc/passwd", "https://user:pw@example.com"):
        r = fresh.post("/api/storage", json={"provider": "s3compat",
                                             "bucket": GOOD, "endpoint": bad})
        assert r.status_code == 400, (bad, r.status_code)
        assert r.json()["code"] == "invalid_input"
    assert not (fresh.jobs_dir / providers.SAVED_NAME).exists()


def test_a_private_endpoint_is_still_allowed(fresh):
    """MinIO 를 사내망에 두는 것은 정상적인 용법이다 — 우리가 목록에 적어 뒀다.
    막으면 지원한다고 해 놓고 못 쓰게 하는 셈이다."""
    ok, why = providers.validate_endpoint("http://127.0.0.1:9000")
    assert ok, why
