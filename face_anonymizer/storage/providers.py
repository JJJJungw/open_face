"""저장소 제공자 — **어디에 붙을지는 설정으로 정한다.**

지금은 AWS S3 를 쓰지만 나중에 NCP 로 옮길 수 있고, 그 뒤는 아무도 모른다.
그때마다 코드를 고치는 대신 여기서 고른다.

두 종류로 갈린다
----------------
**프로토콜이 같은 것**(``s3_compatible=True``) — AWS S3 · NCP Object Storage ·
Cloudflare R2 · MinIO · Wasabi. 전부 S3 API 를 그대로 쓴다. 코드가 달라질 게
없고 **엔드포인트 주소만 다르다.** 실제로 붙을 대상의 대부분이 여기 있다.

**프로토콜이 다른 것** — GCS · Azure Blob. API 모양 자체가 달라서 구현을 하나
더 써서 꽂아야 한다. 목록에는 두되 **고르면 분명하게 거절한다.** 조용히 안 되는
것보다 "아직 지원하지 않습니다" 가 낫다 — 붙일 대상이 정해지면 그때 만든다.

꽂는 자리
---------
제공자마다 ``store`` 에 **어느 구현을 쓸지**가 적혀 있다(``"모듈:클래스"``).
그래서 새 저장소를 붙이는 일은 이렇게 끝난다.

1. `storage/base.py` 의 계약(``CONTRACT``)을 지키는 클래스를 하나 쓴다
2. 여기 ``store`` 한 줄을 그 클래스로 바꾼다

**그게 전부다.** 부르는 쪽은 전부 `s3.get_store()` 하나만 지나가므로
(`service/` · `job_runner` · `weights`) 고칠 데가 없고, 지원 여부(``supported``)
도 이 한 줄에서 저절로 따라온다 — 따로 관리하는 목록이 없으니 둘이 어긋날 수가
없다. 계약을 지켰는지는 `tests/test_storage_contract.py` 가 본다.

이 파일조차 안 고치고 싶으면 ``FA_STORAGE_STORE=내모듈:내클래스`` 로 직접
꽂는다. 등록표보다 이게 이긴다 — 우리가 모르는 저장소를 쓰게 될 쪽이 우리
레포를 포크하지 않아도 되게 하려는 것이다.

바꿀 때 주의
------------
저장소를 바꾸는 것은 접속처만 바꾸는 게 아니다. "이미 처리했나" 판정은 결과
버킷 대조로 하고(``processed_keys``), 진척률 폴더 목록과 저널의 폴더 이름도
그 저장소 기준이다. 바꾸면 **이미 처리한 것이 전부 미처리로 보인다.** 사실상
새 작업 공간을 여는 일이라, 화면에서 바꿀 수 있게 만들 때는 그걸 먼저 알려야
한다(docs/issues 참고).
"""

import json
import os

# 제공자 정의. endpoint 가 None 이면 그 제공자의 기본 엔드포인트를 쓴다는 뜻이고
# (AWS 는 리전으로 알아서 정해진다), 빈 문자열이면 사람이 넣어야 한다는 뜻이다.
PROVIDERS = {
    "s3": {
        "name": "AWS S3",
        "s3_compatible": True,
        "endpoint": None,               # boto3 가 리전으로 정한다
        "region": None,                 # 기본 체인
        "needs_endpoint": False,
        "store": "face_anonymizer.storage.s3:S3Store",
        "note": "자격 증명은 EC2 인스턴스 역할이면 자동으로 잡힌다.",
    },
    "ncp": {
        "name": "네이버 클라우드 Object Storage",
        "s3_compatible": True,
        "endpoint": "https://kr.object.ncloudstorage.com",
        "region": "kr-standard",
        "needs_endpoint": False,
        "store": "face_anonymizer.storage.s3:S3Store",
        "note": "S3 API 를 그대로 쓴다. 액세스 키가 필요하다.",
    },
    "s3compat": {
        "name": "기타 S3 호환 (R2 · MinIO · Wasabi …)",
        "s3_compatible": True,
        "endpoint": "",                 # 사람이 넣는다
        "region": None,
        "needs_endpoint": True,
        "store": "face_anonymizer.storage.s3:S3Store",
        "note": "엔드포인트 주소를 직접 넣는다.",
    },
    # ── 아직 없는 것들 ──────────────────────────────────────────────────────
    # 목록에 두는 이유는 "이건 되나?" 를 묻지 않게 하기 위해서다. 고르면
    # 분명하게 거절한다 — 조용히 안 되는 것이 제일 나쁘다.
    "gcs": {
        "name": "Google Cloud Storage",
        "s3_compatible": False,
        "endpoint": None, "region": None, "needs_endpoint": False,
        "store": "face_anonymizer.storage.base:NotImplementedStore",
        "note": "아직 지원하지 않는다. 구현을 하나 써서 store 를 바꾸면 된다.",
    },
    "azure": {
        "name": "Azure Blob Storage",
        "s3_compatible": False,
        "endpoint": None, "region": None, "needs_endpoint": False,
        "store": "face_anonymizer.storage.base:NotImplementedStore",
        "note": "아직 지원하지 않는다. 구현을 하나 써서 store 를 바꾸면 된다.",
    },
}

DEFAULT = "s3"


def get(name=None):
    """제공자 정의. 모르는 이름이면 기본값."""
    return PROVIDERS.get((name or "").strip().lower() or DEFAULT,
                         PROVIDERS[DEFAULT])


STUB = "face_anonymizer.storage.base:NotImplementedStore"


def store_class(name=None):
    """이 제공자를 실제로 다룰 클래스. **꽂는 자리는 여기 하나뿐이다.**

    문자열로 적어 두고 부를 때 불러온다 — 임포트 시점에 모든 구현을 끌어오면
    S3 만 쓰는 사람이 GCS 라이브러리를 깔아야 한다. 새 저장소가 무거운
    의존성을 들고 와도 고른 사람만 그 값을 치른다.
    """
    from importlib import import_module              # noqa: PLC0415
    mod, _, cls = get(name).get("store", STUB).partition(":")
    return getattr(import_module(mod), cls)


def is_supported(name=None):
    """진짜 구현이 붙어 있나. **목록을 따로 관리하지 않는다.**

    예전에는 ``s3_compatible`` 로 판단했다. 그러면 GCS 구현을 넣는 날 고칠
    자리가 둘이 되고(구현 등록 + 지원 목록), 하나를 빠뜨리면 "지원 안 함" 인데
    동작하거나 그 반대가 된다. ``store`` 한 줄에서 파생시키면 어긋날 수가 없다.
    """
    return get(name).get("store", STUB) != STUB


def listing():
    """화면이 그릴 목록. 지원 여부까지 같이 준다."""
    return [{"id": k, **{kk: vv for kk, vv in v.items()
                         if kk not in ("endpoint", "store")},
             "endpoint": v["endpoint"], "supported": is_supported(k)}
            for k, v in PROVIDERS.items()]


# ── 화면에서 정한 것을 어디에 두나 ──────────────────────────────────────────
# 파일 하나. **여기 들어가는 것 중에 비밀은 없다** — 제공자·버킷·주소·프리픽스
# 뿐이고 열쇠는 애초에 안 받는다(s3.set_credentials 주석). 그래서 파일로 남겨도
# 되고, 남겨야 서버를 다시 띄워도 설정이 살아 있다.
SAVED_NAME = "_storage.json"
SAVED_FIELDS = ("provider", "bucket", "region", "endpoint",
                "root_prefix", "output_prefix", "store")


def saved_path():
    """작업 디렉터리 밑. `service.config` 를 임포트하지 않으려고 환경을 직접 본다
    — storage 가 service 를 알면 의존이 거꾸로 돈다."""
    return os.path.join(os.environ.get("FA_JOBS_DIR", "jobs"), SAVED_NAME)


def load_saved():
    """저장해 둔 설정. 없거나 깨졌으면 빈 dict.

    **깨졌다고 기동을 막지 않는다.** 설정 파일 하나 때문에 서버가 안 뜨면
    고치러 들어갈 화면도 같이 없어진다.
    """
    try:
        with open(saved_path(), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save(cfg):
    """설정을 남긴다. 원자적으로 — 쓰다 만 파일이 남으면 다음 기동이 막힌다."""
    p = saved_path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({k: getattr(cfg, k) for k in SAVED_FIELDS},
                  f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p


class StorageConfig:
    """지금 어디에 붙어 있나. **모듈 상수가 아니라 객체다.**

    예전에는 버킷·리전이 모듈 상수라 임포트할 때 한 번 읽고 끝이었다. 그래서
    설정을 바꾸려면 서버를 다시 띄워야 했고, 화면에서 고르게 하는 길이 아예
    막혀 있었다. 객체로 두면 나중에 갈아 끼울 수 있다.
    """

    __slots__ = ("provider", "bucket", "region", "endpoint",
                 "root_prefix", "output_prefix", "store")

    def __init__(self, provider=None, bucket=None, region=None, endpoint=None,
                 root_prefix=None, output_prefix=None, store=None):
        p = get(provider)
        self.provider = (provider or DEFAULT).strip().lower()
        # 등록표를 안 거치고 직접 꽂는 길. **우리 저장소를 고치지 않아도 된다.**
        # 아무도 안 쓰면 None 이고 그때는 제공자 이름이 클래스를 정한다.
        self.store = (store or "").strip() or None
        self.bucket = bucket or ""
        # 제공자 기본값 위에 준 값만 덮는다 — NCP 를 고르면 엔드포인트·리전이
        # 알아서 채워지고, 그래도 다르면 직접 넣을 수 있다.
        self.region = region or p["region"]
        self.endpoint = endpoint or p["endpoint"] or None
        self.root_prefix = root_prefix or ""
        self.output_prefix = (output_prefix if output_prefix is not None
                              else "v1/results/face/")

    @classmethod
    def from_env(cls):
        """환경 변수 → 저장해 둔 것 → 제공자 기본값 순으로 채운다.

        **환경 변수가 이긴다.** 띄우는 사람이 명시적으로 준 값이 화면에서
        예전에 눌러 둔 것보다 뒤에 오면, `.env` 를 고쳐도 안 바뀌는 것처럼
        보인다 — 그때 사람은 있지도 않은 문제를 찾게 된다.
        """
        s = load_saved()

        def pick(env, key, default=None):
            v = os.environ.get(env)
            return v if v not in (None, "") else s.get(key, default)

        return cls(
            provider=pick("FA_STORAGE_PROVIDER", "provider"),
            bucket=pick("FA_S3_BUCKET", "bucket"),
            region=pick("FA_S3_REGION", "region"),
            endpoint=pick("FA_S3_ENDPOINT", "endpoint"),
            root_prefix=pick("FA_S3_ROOT_PREFIX", "root_prefix", ""),
            output_prefix=pick("FA_S3_OUTPUT_PREFIX", "output_prefix",
                               "v1/results/face/"),
            store=pick("FA_STORAGE_STORE", "store"),
        )

    @property
    def info(self):
        return get(self.provider)

    @property
    def supported(self):
        """구현이 꽂혀 있나. `store` 한 줄에서 따라온다(is_supported 주석)."""
        return bool(self.store) or is_supported(self.provider)

    @property
    def store_class(self):
        """이 설정을 다룰 클래스. 직접 꽂은 것이 있으면 그게 이긴다."""
        if self.store:
            from importlib import import_module      # noqa: PLC0415
            mod, _, cls = self.store.partition(":")
            return getattr(import_module(mod), cls)
        return store_class(self.provider)

    @property
    def ready(self):
        """지금 이 설정으로 붙을 수 있나."""
        if not self.bucket or not self.supported:
            return False
        return bool(self.endpoint) or not self.info["needs_endpoint"]

    def as_dict(self):
        """화면에 보여 줄 것. **자격 증명은 여기 없다** — 애초에 안 들고 있다."""
        return {"provider": self.provider, "name": self.info["name"],
                "store": self.store,
                "bucket": self.bucket, "region": self.region,
                "endpoint": self.endpoint, "root_prefix": self.root_prefix,
                "output_prefix": self.output_prefix,
                "supported": self.supported, "ready": self.ready,
                # 직접 꽂았으면 제공자 설명이 거짓말이 된다 — '아직 지원하지
                # 않는다' 옆에 '연결됨' 이 같이 떠 있으면 둘 다 못 믿게 된다.
                "note": (f"직접 꽂은 구현으로 돕니다 ({self.store})"
                         if self.store else self.info["note"])}
