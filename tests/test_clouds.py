"""붙을 수 있는 클라우드 — 카드에 불이 들어오는 근거.

여기서 지키는 것은 셋이다.

1. **불과 활성은 다르다.** 불은 여럿 들어올 수 있고 활성은 항상 하나다.
2. **둘 이상 켜져 있는데 활성이 안 정해졌으면 아무것도 고르지 않는다.**
   임의로 하나 고르면 결과가 엉뚱한 버킷에 조용히 쌓인다.
3. **정본은 `.env` 하나다.** 화면은 고르기만 하고 파일에 쓰지 않는다.
"""

import pytest

from face_anonymizer.storage import registry

ENV = ("FA_STORAGE_ACTIVE", "FA_S3_BUCKET", "FA_S3_ACCESS_KEY", "FA_S3_SECRET_KEY",
       "FA_NCP_BUCKET", "FA_NCP_ACCESS_KEY", "FA_NCP_SECRET_KEY",
       "FA_S3COMPAT_BUCKET", "FA_S3COMPAT_ENDPOINT")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(registry, "ACTIVE", None)
    registry.invalidate()
    yield
    registry.invalidate()


# ── 설정 읽기 ─────────────────────────────────────────────────────────────

def test_the_provider_id_is_the_prefix():
    assert registry.prefix_of("s3") == "FA_S3_"
    assert registry.prefix_of("s3compat") == "FA_S3COMPAT_"


def test_a_bucketless_provider_counts_as_unconfigured(monkeypatch):
    """버킷 없이 리전만 있는 설정은 아무것도 못 한다. 그걸 '설정됨' 으로 세면
    카드가 켜졌다 꺼졌다 하고 사람은 왜 안 되는지 모른다."""
    monkeypatch.setenv("FA_NCP_REGION", "kr-standard")
    assert registry.config_of("ncp") is None
    assert registry.configured() == []


def test_the_old_aws_only_setup_still_works(monkeypatch):
    """`FA_S3_*` 는 예전부터 쓰던 이름 그대로다 — 한 글자도 안 고치고 돈다."""
    monkeypatch.setenv("FA_S3_BUCKET", "old-bucket")
    cfg = registry.config_of("s3")
    assert cfg.bucket == "old-bucket" and cfg.provider == "s3"
    assert registry.configured() == ["s3"]


def test_keys_never_reach_the_config_object(monkeypatch):
    """열쇠가 설정에 섞이면 `as_dict()` 로 화면에 나간다."""
    monkeypatch.setenv("FA_S3_BUCKET", "b")
    monkeypatch.setenv("FA_S3_ACCESS_KEY", "AKIA…")
    monkeypatch.setenv("FA_S3_SECRET_KEY", "s3cr3t")
    assert registry.creds_of("s3")["aws_access_key_id"] == "AKIA…"
    blob = str(registry.config_of("s3").as_dict())
    assert "AKIA" not in blob and "s3cr3t" not in blob


# ── 활성 고르기 ───────────────────────────────────────────────────────────

def test_one_configured_cloud_needs_no_choosing(monkeypatch):
    monkeypatch.setenv("FA_S3_BUCKET", "only")
    assert registry.wanted() == ("s3", "")


def test_two_configured_clouds_with_no_choice_pick_nothing(monkeypatch):
    """**여기서 임의로 고르면 900건이 엉뚱한 버킷에 쌓인다.**"""
    monkeypatch.setenv("FA_S3_BUCKET", "a")
    monkeypatch.setenv("FA_NCP_BUCKET", "b")
    pid, why = registry.wanted()
    assert pid is None
    assert "여럿" in why and "FA_STORAGE_ACTIVE" in why


def test_an_explicit_choice_wins(monkeypatch):
    monkeypatch.setenv("FA_S3_BUCKET", "a")
    monkeypatch.setenv("FA_NCP_BUCKET", "b")
    monkeypatch.setenv("FA_STORAGE_ACTIVE", "ncp")
    assert registry.wanted() == ("ncp", "")


@pytest.mark.parametrize("value, hint", [
    ("없는것", "모르는 제공자"),
    ("ncp", "BUCKET 이 없습니다"),
])
def test_a_bad_choice_says_which_way_it_is_bad(monkeypatch, value, hint):
    monkeypatch.setenv("FA_S3_BUCKET", "a")
    monkeypatch.setenv("FA_STORAGE_ACTIVE", value)
    pid, why = registry.wanted()
    assert pid is None and hint in why


def test_activating_swaps_the_one_seam(monkeypatch):
    """**갈아 끼우는 자리는 하나다** — get_store() 를 지나는 열다섯 군데는
    자기가 어느 클라우드를 보는지 모른 채 그대로 돈다."""
    from face_anonymizer.storage import s3 as s3mod

    monkeypatch.setenv("FA_NCP_BUCKET", "ncp-bucket")
    monkeypatch.setenv("FA_NCP_ACCESS_KEY", "k")
    monkeypatch.setenv("FA_NCP_SECRET_KEY", "s")
    monkeypatch.setattr(s3mod, "_store", None)
    registry.activate("ncp")
    assert s3mod.CONFIG.bucket == "ncp-bucket"
    assert s3mod.CONFIG.provider == "ncp"
    assert s3mod.credentials()["aws_access_key_id"] == "k"
    assert registry.ACTIVE == "ncp"


def test_activating_an_unconfigured_cloud_is_refused():
    with pytest.raises(ValueError):
        registry.activate("ncp")


# ── 카드가 그릴 것 ────────────────────────────────────────────────────────

def test_the_listing_covers_every_provider(monkeypatch):
    """설정이 없는 것도 카드로 보여 준다 — "이건 되나?" 를 묻지 않게."""
    from face_anonymizer.storage import providers

    monkeypatch.setenv("FA_S3_BUCKET", "a")
    d = registry.listing()
    assert [c["id"] for c in d["clouds"]] == list(providers.PROVIDERS)
    s3row = next(c for c in d["clouds"] if c["id"] == "s3")
    assert s3row["configured"] is True and s3row["bucket"] == "a"
    ncp = next(c for c in d["clouds"] if c["id"] == "ncp")
    assert ncp["configured"] is False and "bucket" not in ncp


def test_the_listing_carries_no_secrets(monkeypatch):
    monkeypatch.setenv("FA_S3_BUCKET", "a")
    monkeypatch.setenv("FA_S3_ACCESS_KEY", "AKIAEXAMPLE")
    monkeypatch.setenv("FA_S3_SECRET_KEY", "s3cr3t")
    blob = str(registry.listing())
    assert "AKIAEXAMPLE" not in blob and "s3cr3t" not in blob
    # 어디서 왔는지는 말한다 — 그게 없으면 왜 되는지 아무도 모른다.
    assert "FA_S3_ACCESS_KEY" in blob


def test_probe_reports_read_and_write_separately(monkeypatch):
    """읽기만 되는 열쇠가 흔하다. 하나로 뭉치면 어디가 막혔는지 안 보인다."""
    from face_anonymizer.storage import s3 as s3mod

    monkeypatch.setenv("FA_S3_BUCKET", "b")

    class OnlyRead:
        def check(self):
            raise s3mod.S3Error("읽기는 되지만 결과물을 쓰지 못합니다 (v1/)")

    monkeypatch.setattr(registry, "store_for", lambda pid: OnlyRead())
    r = registry.probe("s3", force=True)
    assert r["ok"] is False and r["read"] is True and r["write"] is False


def test_probe_says_nothing_when_there_is_nothing_to_say():
    r = registry.probe("ncp", force=True)
    assert r["ok"] is None and "BUCKET" in r["detail"]


def test_probe_is_cached_so_the_card_screen_can_poll(monkeypatch):
    monkeypatch.setenv("FA_S3_BUCKET", "b")
    calls = []

    class Ok:
        def check(self):
            calls.append(1)
            return True

    monkeypatch.setattr(registry, "store_for", lambda pid: Ok())
    registry.probe("s3", force=True)
    registry.probe("s3")
    registry.probe("s3")
    assert len(calls) == 1, "카드가 폴링할 때마다 클라우드를 치고 있다"


def test_an_unresolved_active_does_not_quietly_fall_back(monkeypatch):
    """**여기가 조용한 사고를 막는 자리다.**

    전역 설정은 임포트할 때 `FA_S3_BUCKET` 을 읽어 둔다. 활성을 못 정했는데
    그냥 두면 잡이 **소리 없이 AWS 로 나간다** — 켜져 있는 다른 클라우드로
    보낼 생각이었는데도. 못 정했으면 붙을 곳을 비워서 거절하게 만든다.
    """
    from face_anonymizer.storage import s3 as s3mod

    monkeypatch.setenv("FA_S3_BUCKET", "aws-bucket")
    monkeypatch.setenv("FA_NCP_BUCKET", "ncp-bucket")
    monkeypatch.setattr(s3mod, "CONFIG",
                        s3mod.providers.StorageConfig(provider="s3",
                                                      bucket="aws-bucket"))
    monkeypatch.setattr(s3mod, "_store", None)
    monkeypatch.setattr(registry, "REASON", "")

    pid, why = registry.resolve()
    assert pid is None and "여럿" in why
    assert s3mod.get_store() is None, "옛 값으로 조용히 붙고 있다"
    assert "여럿" in s3mod.unavailable_reason()


def test_resolving_one_cloud_clears_the_reason(monkeypatch):
    from face_anonymizer.storage import s3 as s3mod

    monkeypatch.setenv("FA_S3_BUCKET", "only")
    monkeypatch.setattr(s3mod, "_store", None)
    monkeypatch.setattr(registry, "REASON", "옛 사유")
    pid, why = registry.resolve()
    assert pid == "s3" and why == "" and registry.REASON == ""


def test_disconnecting_lets_you_connect_again(monkeypatch, tmp_path):
    """**끊고 나서 다시 붙을 수 있어야 한다.**

    활성을 안 내려놓으면 카드는 계속 '사용 중' 인데 붙을 곳은 비어 있다 —
    그러면 화면에서 다시 연결할 길이 사라진다.
    """
    pytest = __import__("pytest")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from face_anonymizer.service import config, jobs, server
    from face_anonymizer.storage import s3 as s3mod

    monkeypatch.setenv("FA_S3_BUCKET", "b")
    monkeypatch.setenv("FA_JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(s3mod, "_store", None)
    monkeypatch.delenv("FA_ALLOW_STORAGE_EDIT", raising=False)

    registry.activate("s3")
    assert registry.ACTIVE == "s3"

    c = TestClient(server.app)
    assert c.delete("/api/storage").status_code == 200
    assert registry.ACTIVE is None, "끊었는데 활성이 남아 있다"

    # 다시 붙는다.
    registry.activate("s3")
    assert s3mod.CONFIG.bucket == "b"


def test_the_active_card_can_be_pressed_again():
    """붙을 곳이 비어 버렸을 때 다시 연결하는 길이 화면에 있어야 한다."""
    import pathlib
    import re

    html = (pathlib.Path(__file__).resolve().parent.parent / "face_anonymizer"
            / "service" / "static" / "index.html").read_text(encoding="utf-8")
    card = re.search(r"function cloudCard\(c\) \{.*?\n\}", html, re.S).group(0)
    assert "!c.active ? `data-pick" not in card, "활성 카드가 안 눌린다"
    assert "다시 연결합니다" in card


def test_the_first_screen_is_always_the_cards():
    """붙어 있어도 **한 번은 보여 준다.** 곧장 넘어가면 지금 어디에 붙어 있는지,
    애초에 붙기는 하는지를 볼 기회가 없다."""
    import pathlib

    html = (pathlib.Path(__file__).resolve().parent.parent / "face_anonymizer"
            / "service" / "static" / "index.html").read_text(encoding="utf-8")
    assert "openSetup(null);" in html
    assert "if (d && d.first_run) return openSetup(d);" not in html
