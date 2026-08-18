"""저장소 제공자 — **어디에 붙을지는 설정으로 정한다.**

지금은 AWS S3 를 쓰지만 나중에 NCP 로 옮길 수 있고, 그 뒤는 아무도 모른다.
그때마다 코드를 고치는 대신 여기서 고른다.

두 종류로 갈린다
----------------
**프로토콜이 같은 것**(``s3_compatible=True``) — AWS S3 · NCP Object Storage ·
Cloudflare R2 · MinIO · Wasabi. 전부 S3 API 를 그대로 쓴다. 코드가 달라질 게
없고 **엔드포인트 주소만 다르다.** 실제로 붙을 대상의 대부분이 여기 있다.

**프로토콜이 다른 것** — GCS · Azure Blob. API 모양 자체가 달라서 어댑터가
필요하다. 목록에는 두되 **고르면 분명하게 거절한다.** 조용히 안 되는 것보다
"아직 지원하지 않습니다" 가 낫다 — 붙일 대상이 정해지면 그때 만든다.

바꿀 때 주의
------------
저장소를 바꾸는 것은 접속처만 바꾸는 게 아니다. "이미 처리했나" 판정은 결과
버킷 대조로 하고(``processed_keys``), 진척률 폴더 목록과 저널의 폴더 이름도
그 저장소 기준이다. 바꾸면 **이미 처리한 것이 전부 미처리로 보인다.** 사실상
새 작업 공간을 여는 일이라, 화면에서 바꿀 수 있게 만들 때는 그걸 먼저 알려야
한다(docs/issues 참고).
"""

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
        "note": "자격 증명은 EC2 인스턴스 역할이면 자동으로 잡힌다.",
    },
    "ncp": {
        "name": "네이버 클라우드 Object Storage",
        "s3_compatible": True,
        "endpoint": "https://kr.object.ncloudstorage.com",
        "region": "kr-standard",
        "needs_endpoint": False,
        "note": "S3 API 를 그대로 쓴다. 액세스 키가 필요하다.",
    },
    "s3compat": {
        "name": "기타 S3 호환 (R2 · MinIO · Wasabi …)",
        "s3_compatible": True,
        "endpoint": "",                 # 사람이 넣는다
        "region": None,
        "needs_endpoint": True,
        "note": "엔드포인트 주소를 직접 넣는다.",
    },
    # ── 아직 없는 것들 ──────────────────────────────────────────────────────
    # 목록에 두는 이유는 "이건 되나?" 를 묻지 않게 하기 위해서다. 고르면
    # 분명하게 거절한다 — 조용히 안 되는 것이 제일 나쁘다.
    "gcs": {
        "name": "Google Cloud Storage",
        "s3_compatible": False,
        "endpoint": None, "region": None, "needs_endpoint": False,
        "note": "아직 지원하지 않는다. API 모양이 달라 어댑터가 필요하다.",
    },
    "azure": {
        "name": "Azure Blob Storage",
        "s3_compatible": False,
        "endpoint": None, "region": None, "needs_endpoint": False,
        "note": "아직 지원하지 않는다. API 모양이 달라 어댑터가 필요하다.",
    },
}

DEFAULT = "s3"


def get(name=None):
    """제공자 정의. 모르는 이름이면 기본값."""
    return PROVIDERS.get((name or "").strip().lower() or DEFAULT,
                         PROVIDERS[DEFAULT])


def listing():
    """화면이 그릴 목록. 지원 여부까지 같이 준다."""
    return [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "endpoint"},
             "endpoint": v["endpoint"], "supported": v["s3_compatible"]}
            for k, v in PROVIDERS.items()]


class StorageConfig:
    """지금 어디에 붙어 있나. **모듈 상수가 아니라 객체다.**

    예전에는 버킷·리전이 모듈 상수라 임포트할 때 한 번 읽고 끝이었다. 그래서
    설정을 바꾸려면 서버를 다시 띄워야 했고, 화면에서 고르게 하는 길이 아예
    막혀 있었다. 객체로 두면 나중에 갈아 끼울 수 있다.
    """

    __slots__ = ("provider", "bucket", "region", "endpoint",
                 "root_prefix", "output_prefix")

    def __init__(self, provider=None, bucket=None, region=None, endpoint=None,
                 root_prefix=None, output_prefix=None):
        p = get(provider)
        self.provider = (provider or DEFAULT).strip().lower()
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
        return cls(
            provider=os.environ.get("FA_STORAGE_PROVIDER"),
            bucket=os.environ.get("FA_S3_BUCKET"),
            region=os.environ.get("FA_S3_REGION"),
            endpoint=os.environ.get("FA_S3_ENDPOINT"),
            root_prefix=os.environ.get("FA_S3_ROOT_PREFIX", ""),
            output_prefix=os.environ.get("FA_S3_OUTPUT_PREFIX",
                                         "v1/results/face/"),
        )

    @property
    def info(self):
        return get(self.provider)

    @property
    def supported(self):
        return bool(self.info["s3_compatible"])

    @property
    def ready(self):
        """지금 이 설정으로 붙을 수 있나."""
        if not self.bucket or not self.supported:
            return False
        return bool(self.endpoint) or not self.info["needs_endpoint"]

    def as_dict(self):
        """화면에 보여 줄 것. **자격 증명은 여기 없다** — 애초에 안 들고 있다."""
        return {"provider": self.provider, "name": self.info["name"],
                "bucket": self.bucket, "region": self.region,
                "endpoint": self.endpoint, "root_prefix": self.root_prefix,
                "output_prefix": self.output_prefix,
                "supported": self.supported, "ready": self.ready,
                "note": self.info["note"]}
