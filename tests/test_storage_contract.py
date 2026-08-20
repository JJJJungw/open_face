"""저장소 계약 — **새 클라우드를 꽂을 수 있다는 것을 여기서 증명한다.**

"클라우드를 고를 수 있어야 한다" 는 요구를 문서로만 두면, 실제로 꽂아 보는 날
`s3.py` 를 읽고 어느 게 계약이고 어느 게 S3 사정인지 추측하게 된다. 그래서
계약을 코드로 적고(`storage/base.py`) 그걸 지키는지 여기서 본다.

셋을 본다.
  1. 지금 쓰는 구현(S3Store)이 계약을 지키나
  2. 아직 없는 자리(NotImplementedStore)도 계약의 이름을 다 갖고 있나
     — 갖고 있어야 '꽂는 자리가 실제로 있다' 가 성립한다
  3. **한 줄만 바꾸면 통째로 갈아 끼워지나** — 이게 요구의 본체다
"""

import pathlib
import subprocess
import sys
import textwrap

import pytest

from face_anonymizer.storage import base, providers
from face_anonymizer.storage import s3 as s3mod


def _cfg(pid):
    return providers.StorageConfig(provider=pid, bucket="b",
                                   endpoint="https://example.invalid")


def test_the_real_one_keeps_the_contract():
    """지금 쓰는 구현이 계약을 어기면 계약 쪽이 틀린 것이다."""
    from face_anonymizer.storage.s3 import S3Store
    assert base.missing(S3Store(config=_cfg("s3"))) == ()


def test_the_empty_seat_keeps_it_too():
    """자리는 있어야 한다 — 없으면 '꽂으면 된다' 가 빈말이 된다."""
    assert base.missing(base.NotImplementedStore(config=_cfg("gcs"))) == ()


def test_every_registered_provider_resolves():
    """등록표에 적힌 클래스를 실제로 불러와서 만들 수 있나.

    오타는 **그 제공자를 고른 날** 드러난다 — 그날은 대개 이관 당일이다.
    만들기까지 해 본다. 이름만 맞고 생성자 모양이 다르면 똑같이 그날 터진다.
    """
    for pid in providers.PROVIDERS:
        store = providers.store_class(pid)(config=_cfg(pid))
        assert base.missing(store) == (), pid


def test_unbuilt_providers_refuse_out_loud():
    """조용한 목업이 제일 나쁘다.

    빈 목록을 돌려주는 가짜를 두면 화면에 '폴더 0개' 가 뜨고, 사람은 버킷이
    비었다고 믿는다. 900 건을 넣고 나서 아는 것보다 고른 순간 아는 편이 낫다.
    """
    from face_anonymizer.storage.s3 import S3Error
    cfg = providers.StorageConfig(provider="gcs", bucket="b")
    store = providers.store_class("gcs")(config=cfg)

    with pytest.raises(S3Error) as e:
        store.list()
    assert "지원하지 않습니다" in str(e.value)
    with pytest.raises(S3Error):
        store.upload("/tmp/x", "k")


def test_support_follows_the_registry_not_a_second_list(monkeypatch):
    """지원 여부를 따로 관리하면 언젠가 둘이 어긋난다.

    GCS 구현을 넣는 날 고칠 자리가 둘이면(등록 + 지원 목록) 하나를 빠뜨리고,
    '지원 안 함' 인데 동작하거나 그 반대가 된다. 한 줄에서 파생시킨다.
    """
    assert providers.is_supported("gcs") is False
    assert providers.StorageConfig(provider="gcs", bucket="b").supported is False

    # 구현을 꽂았다고 치자 — 등록표 한 줄만 바꾼다.
    monkeypatch.setitem(providers.PROVIDERS["gcs"], "store",
                        "face_anonymizer.storage.s3:S3Store")
    assert providers.is_supported("gcs") is True
    cfg = providers.StorageConfig(provider="gcs", bucket="b")
    assert cfg.supported and cfg.ready               # 손댄 곳은 그 한 줄뿐이다


def test_one_line_swaps_the_whole_thing(monkeypatch):
    """**요구의 본체.** 등록표 한 줄로 저장소가 통째로 갈리나.

    부르는 쪽(서버 라우트 · 워커 · 가중치 조달)은 전부 `get_store()` 하나를
    지나간다. 그래서 여기만 바뀌면 나머지는 안 고쳐도 된다.
    """
    from face_anonymizer.storage import s3 as s3mod

    class Elsewhere:
        """다른 클라우드라고 치자. 계약만 지키면 S3 를 하나도 안 쓴다."""
        bucket = "somewhere-else"
        root_prefix = "in/"
        output_prefix = "out/"

        def __init__(self, config=None, **kw):
            self.config = config

        def check(self):
            return True

        def list(self, prefix=""):
            return ["in/a/"], [{"key": "in/x.mp4", "size": 1, "modified": ""}]

        def list_all(self, prefix):
            return ["in/x.mp4"]

        def output_key(self, key):
            return "out/x_deid.mp4"

        def processed_keys(self):
            return set()

        def exists(self, key):
            return False

        def size_of(self, key):
            return 1

        def download(self, key, dest, callback=None):
            return 1

        def upload(self, path, key, content_type="video/mp4", callback=None):
            return None

        def presigned_url(self, key, expires=None, filename=None):
            return "https://elsewhere/" + key

    assert base.missing(Elsewhere(config=None)) == ()
    monkeypatch.setitem(providers.PROVIDERS, "elsewhere",
                        {"name": "다른 클라우드", "s3_compatible": False,
                         "endpoint": None, "region": None,
                         "needs_endpoint": False, "note": "",
                         "store": f"{__name__}:_Elsewhere"})
    globals()["_Elsewhere"] = Elsewhere

    monkeypatch.setattr(s3mod, "CONFIG",
                        providers.StorageConfig(provider="elsewhere", bucket="b"))
    monkeypatch.setattr(s3mod, "_store", None)

    store = s3mod.get_store()
    assert isinstance(store, Elsewhere)              # 고친 곳은 등록표 한 줄뿐
    assert store.bucket == "somewhere-else"
    assert s3mod.unavailable_reason() == ""


def test_a_stranger_can_plug_in_without_forking_us(monkeypatch):
    """등록표조차 안 고치고 환경 변수로 꽂는 길.

    우리가 모르는 저장소를 쓰게 될 쪽이 우리 레포를 포크해야 한다면 그건
    '고를 수 있다' 가 아니다. 이름이 등록표에 없어도 꽂은 것이 이긴다.
    """
    from face_anonymizer.storage import s3 as s3mod

    cfg = providers.StorageConfig(
        provider="gcs",                              # 등록표에는 '미지원' 인데
        bucket="b",
        store="face_anonymizer.storage.s3:S3Store",  # 직접 꽂은 것이 이긴다
    )
    assert cfg.supported and cfg.ready
    assert cfg.store_class is s3mod.S3Store
    assert cfg.as_dict()["store"] == "face_anonymizer.storage.s3:S3Store"

    monkeypatch.setenv("FA_STORAGE_STORE", "face_anonymizer.storage.s3:S3Store")
    monkeypatch.setenv("FA_S3_BUCKET", "b")
    monkeypatch.setenv("FA_STORAGE_PROVIDER", "azure")
    assert providers.StorageConfig.from_env().supported is True


def test_the_contract_stays_small():
    """계약이 넓어지면 새 저장소를 붙이기가 그만큼 어려워진다.

    부르는 쪽에서 실제로 쓰는 것만 남긴다. 늘려야 할 이유가 생기면 이 숫자를
    같이 고치면서 '정말 필요한가' 를 한 번 더 묻게 된다.
    """
    assert len(base.CONTRACT) == 13


# ── 워커 컨테이너에는 fastapi 가 없다 (docs/issues/024) ──────────────────────

def test_an_s3_error_survives_without_fastapi():
    """**큐 워커에는 웹 프레임워크를 일부러 안 깐다.**

    HTTP 를 안 받으니 requirements/worker.txt 에 fastapi 가 없다. 그런데
    `wrap()` 이 오류에 붙일 딱지를 가지러 `service/errors` 를 임포트했고, 그
    모듈이 fastapi 를 끈다. 그래서 워커에서 S3 오류가 나면 **진짜 원인이
    ImportError 로 바뀌어** 나갔다 — 권한 문제인데 설치가 잘못된 줄 알게 된다.

    딱지는 HTTP 상태 코드를 고르는 데 쓰는 부가 정보이고 워커에는 응답 자체가
    없다. 못 가져오면 안 붙이고 메시지는 그대로 내보낸다.

    **새 프로세스에서 확인한다.** 같은 프로세스에서 sys.modules 를 지워도 부모
    패키지가 속성으로 들고 있어서 재임포트가 안 일어난다 — 처음에 그렇게 짰다가
    통과하는 것처럼 보였다. 워커 컨테이너는 애초에 fastapi 가 **없는** 프로세스다.

    **그리고 차단이 먹었는지를 먼저 확인한다.** 처음에는 `find_module` /
    `load_module` 로 막았는데, 그 훅은 **파이썬 3.12 에서 제거됐다.** 3.11 에서는
    막히고 3.12 에서는 조용히 통과한다 — 즉 새 파이썬에서는 이 검사가 아무것도
    안 보면서 통과할 수 있었다. 검사가 죽는 것보다 검사가 살아 있는 척하는 게
    훨씬 나쁘다.
    """
    code = textwrap.dedent("""
        import sys
        class Block:
            # find_spec 이다. find_module/load_module 은 3.12 에서 사라졌다.
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "fastapi":
                    raise ImportError("No module named 'fastapi'")
                return None
        sys.meta_path.insert(0, Block())

        # 차단이 실제로 먹는가. 안 먹으면 아래 단언들은 아무 의미가 없다.
        try:
            import fastapi
        except ImportError:
            pass
        else:
            raise AssertionError("차단이 안 먹었다 — 이 검사는 아무것도 안 본다")

        from face_anonymizer.storage import s3 as s3mod
        class Denied(Exception):
            response = {"Error": {"Code": "AccessDenied"}}
        err = s3mod.wrap(Denied("Access Denied"), "내려받지 못했습니다")
        assert "Access Denied" in str(err), "진짜 원인이 사라졌다"
        assert err.problem is None, "워커에는 딱지를 붙일 수 없다"

        # 지원하지 않는 제공자도 같은 길을 탄다.
        from face_anonymizer.storage import base
        n = base.NotImplementedStore(type("C", (), {"provider": "gcs"})())
        try:
            n.check()
        except s3mod.S3Error as e:
            assert "지원하지 않습니다" in str(e), e
        print("OK")
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(pathlib.Path(__file__).parent.parent))
    assert p.returncode == 0, f"워커 경로가 죽었다:\n{p.stderr[-800:]}"
    assert "OK" in p.stdout


def test_the_label_is_still_attached_where_it_is_used(monkeypatch):
    """서버에서는 딱지가 붙어야 한다 — 권한·오타·장애가 다른 상태 코드다."""
    class Denied(Exception):
        response = {"Error": {"Code": "AccessDenied"}}

    class Missing(Exception):
        response = {"Error": {"Code": "NoSuchKey"}}

    pytest.importorskip("fastapi")
    assert s3mod.wrap(Denied("x"), "y").problem.code == "s3_access_denied"
    assert s3mod.wrap(Missing("x"), "y").problem.code == "s3_object_not_found"


# ── 붙지 않는 이유 다섯 ────────────────────────────────────────────────────
#
# 예전에는 넷이 하나로 뭉쳐 있었다. 열쇠가 아예 없어도 "리전 설정과 네트워크
# 상태를 확인해 주세요" 가 나갔고, 그게 재시도 대상이라 절대 성공할 수 없는
# 호출을 95초 창 안에서 네 번 반복했다.

# botocore 예외를 흉내 낸다. **실제 클래스를 임포트하지 않는다** — boto3 가
# 없는 환경(워커)에서도 이 검사가 돌아야 하고, wrap() 도 같은 이유로 클래스가
# 아니라 이름을 본다.
def _named(name):
    """자격 증명이 아예 없을 때 나오는 모양 — HTTP 응답이 없어 response 가 없다."""
    return type(name, (Exception,), {})


def _coded(code):
    """저장소가 대답은 했는데 거절한 모양."""
    return type("E", (Exception,), {"response": {"Error": {"Code": code}}})


@pytest.mark.parametrize("exc, want", [
    (_named("NoCredentialsError")("어디에도 없다"), "s3_no_credentials"),
    (_named("PartialCredentialsError")("시크릿이 없다"), "s3_no_credentials"),
    (_coded("InvalidAccessKeyId")("x"), "s3_bad_credentials"),
    (_coded("SignatureDoesNotMatch")("x"), "s3_bad_credentials"),
    (_coded("ExpiredToken")("x"), "s3_bad_credentials"),
    (_coded("AccessDenied")("x"), "s3_access_denied"),
    (_coded("NoSuchBucket")("x"), "s3_bucket_not_found"),
    (_coded("NoSuchKey")("x"), "s3_object_not_found"),
    (Exception("연결 실패"), "s3_upstream"),
])
def test_each_reason_gets_its_own_label(exc, want):
    """사람이 다음에 할 일이 다르면 딱지도 달라야 한다.

    키가 없다 / 키가 틀렸다 / 권한이 없다 / 버킷이 없다 / 못 닿는다 —
    다섯 중 넷을 '네트워크' 로 보고하면 나머지 넷은 멀쩡한 망을 뒤진다.
    """
    pytest.importorskip("fastapi")
    assert s3mod.wrap(exc, "무엇을").problem.code == want


def test_only_the_network_case_is_worth_retrying():
    """**재시도는 세상이 바뀔 수 있을 때만 쓸모가 있다.**

    열쇠가 없는데 다섯 번 더 물어봐도 없다. 그동안 뒤에 쌓인 정상 작업이 밀린다.
    """
    pytest.importorskip("fastapi")
    from face_anonymizer.service import errors

    assert errors.S3_UPSTREAM.retryable
    for p in (errors.S3_NO_CREDENTIALS, errors.S3_BAD_CREDENTIALS,
              errors.S3_ACCESS_DENIED, errors.S3_BUCKET_NOT_FOUND):
        assert not p.retryable, p.code


def test_the_old_wrong_hint_is_gone():
    """자격 증명이 없을 때 리전을 확인하라고 하지 않는다."""
    pytest.importorskip("fastapi")
    from face_anonymizer.service import errors

    assert "리전" not in errors.S3_NO_CREDENTIALS.hint
    assert "리전" not in errors.S3_UPSTREAM.hint
    # 대신 실제로 볼 자리를 말한다.
    assert "AWS_ACCESS_KEY_ID" in errors.S3_NO_CREDENTIALS.hint
    assert "엔드포인트" in errors.S3_UPSTREAM.hint
