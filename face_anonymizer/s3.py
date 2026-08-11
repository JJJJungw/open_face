"""S3 연동.

입력 영상을 S3 에서 읽고 결과물을 다시 S3 에 올린다. 파이프라인은 로컬 경로만
다루므로(그게 단순하고 테스트하기 쉽다) 여기서 받아 내리고 올리는 것으로 감싼다.

boto3 는 **함수 안에서 지연 임포트**한다. S3 를 안 쓰는 사용자에게 의존성을
강요하지 않고, 테스트는 가짜 클라이언트를 주입해 네트워크 없이 돈다.

환경 변수
    FA_S3_BUCKET          버킷 이름. 없으면 S3 기능 자체가 꺼진다
    FA_S3_REGION          리전 (기본: AWS 기본 체인)
    FA_S3_ROOT_PREFIX     브라우저 최상위 (기본: '')
    FA_S3_OUTPUT_PREFIX   결과물 위치 (기본: v1/results/face/)
    FA_S3_LIST_TTL        처리됨 표시용 목록 캐시 초 (기본: 30)

자격 증명은 boto3 기본 체인을 쓴다 — EC2 인스턴스 역할이 있으면 그대로 잡힌다.
"""

import logging
import os
import time

from . import naming

log = logging.getLogger(__name__)

BUCKET = os.environ.get("FA_S3_BUCKET") or ""
REGION = os.environ.get("FA_S3_REGION") or None
ROOT_PREFIX = os.environ.get("FA_S3_ROOT_PREFIX", "")
OUTPUT_PREFIX = os.environ.get("FA_S3_OUTPUT_PREFIX", "v1/results/face/")
LIST_TTL = int(os.environ.get("FA_S3_LIST_TTL", 30))
URL_TTL = int(os.environ.get("FA_S3_URL_TTL", 3600))

PAGE_MAX = 1000


class S3Error(RuntimeError):
    """S3 호출 실패. 작업은 실패로 남기되 서버는 계속 산다.

    ``problem`` 에 구체 원인이 붙는다 — 권한 문제와 키 오타와 네트워크 장애는
    사용자가 해야 할 일이 전혀 다르다.
    """

    problem = None


def wrap(e, what):
    """botocore 예외를 원인이 드러나는 S3Error 로."""
    from . import errors                             # 지연 임포트 (순환 방지)
    code = ""
    resp = getattr(e, "response", None)
    if isinstance(resp, dict):
        code = str(resp.get("Error", {}).get("Code", ""))
    if code in ("AccessDenied", "403", "SignatureDoesNotMatch",
                "InvalidAccessKeyId", "ExpiredToken", "AllAccessDisabled"):
        p = errors.S3_ACCESS_DENIED
    elif code in ("NoSuchKey", "NoSuchBucket", "404"):
        p = errors.S3_OBJECT_NOT_FOUND
    else:
        p = errors.S3_UPSTREAM
    err = S3Error(f"{what}: {e}")
    err.problem = p
    return err


def make_client():
    import boto3                                   # noqa: PLC0415 — 지연 임포트
    return boto3.client("s3", region_name=REGION)


class S3Store:
    """버킷 하나를 다루는 얇은 래퍼."""

    def __init__(self, bucket=None, client=None, output_prefix=None,
                 root_prefix=None):
        self.bucket = bucket or BUCKET
        self.output_prefix = (output_prefix if output_prefix is not None
                              else OUTPUT_PREFIX)
        self.root_prefix = (root_prefix if root_prefix is not None
                            else ROOT_PREFIX)
        self._client = client
        self._out_cache = (0.0, set())

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    # ── 조회 ──────────────────────────────────────────────────────────────

    def list(self, prefix=""):
        """한 단계만 나열한다 (S3 콘솔과 같은 방식).

        Returns (folders, objects) — folders 는 전체 프리픽스 문자열,
        objects 는 {key, size, modified}.
        """
        prefix = prefix or self.root_prefix
        folders, objects, token = [], [], None
        try:
            while True:
                kw = {"Bucket": self.bucket, "Prefix": prefix, "Delimiter": "/",
                      "MaxKeys": PAGE_MAX}
                if token:
                    kw["ContinuationToken"] = token
                r = self.client.list_objects_v2(**kw)
                folders += [p["Prefix"] for p in r.get("CommonPrefixes", [])]
                for o in r.get("Contents", []):
                    if o["Key"] == prefix:          # 프리픽스 자체를 나타내는 키
                        continue
                    m = o.get("LastModified")
                    objects.append({
                        "key": o["Key"],
                        "size": o.get("Size"),
                        "modified": m.isoformat() if hasattr(m, "isoformat") else m,
                    })
                token = r.get("NextContinuationToken")
                if not token:
                    break
        except Exception as e:                      # noqa: BLE001
            raise wrap(e, "목록 조회 실패") from e
        return folders, objects

    def list_all(self, prefix):
        """하위 폴더까지 전부. 폴더 단위 제출에 쓴다."""
        objects, token = [], None
        try:
            while True:
                kw = {"Bucket": self.bucket, "Prefix": prefix,
                      "MaxKeys": PAGE_MAX}
                if token:
                    kw["ContinuationToken"] = token
                r = self.client.list_objects_v2(**kw)
                for o in r.get("Contents", []):
                    if o["Key"].endswith("/"):
                        continue
                    m = o.get("LastModified")
                    objects.append({
                        "key": o["Key"], "size": o.get("Size"),
                        "modified": m.isoformat() if hasattr(m, "isoformat") else m,
                    })
                token = r.get("NextContinuationToken")
                if not token:
                    break
        except Exception as e:                      # noqa: BLE001
            raise wrap(e, "목록 조회 실패") from e
        return objects

    def output_key(self, key):
        """입력 키에 대응하는 결과물 키.

        데이터셋 규칙(naming.py)을 따른다 — 정체성 필드는 그대로 두고 STATE 만
        raw -> deid 로 바꾼다. 결과는 **입력 폴더별로 나눠 쌓는다.**

            videos/2026-08/f_00001_00_0000000_0042000_raw.mp4
            -> v1/results/face/2026-08-deid/f_00001_00_0000000_0042000_deid.mp4

        한곳에 몰아 두면 폴더 하나가 몇만 건이 되고, 어느 원본 묶음에서 나온
        결과인지 목록만 보고는 알 수 없다. 폴더 이름을 그대로 따라가면 입력과
        출력이 일대일로 붙는다.

        폴더가 없는 입력(직접 업로드 등)은 예전처럼 결과 프리픽스 바로 밑에
        떨어진다. 출력 키는 **입력 키만으로 결정된다** — 폴더로 넣든 한 건씩
        넣든 같은 자리에 떨어져야 중복 판정이 성립한다.
        """
        return self.output_prefix + self.output_folder(key) + naming.output_name(key)

    @staticmethod
    def output_folder(key):
        """입력 키가 들어갈 결과 하위 폴더 ('2026-08-deid/' 또는 '')."""
        parent = os.path.basename(os.path.dirname(key or ""))
        return f"{parent}-deid/" if parent else ""

    def processed_keys(self):
        """결과물 프리픽스에 이미 있는 키 집합. 짧게 캐시한다.

        객체마다 HEAD 를 날리면 목록 한 번에 수백 번 왕복한다. 결과물 폴더를
        한 번 나열해서 대조하는 편이 훨씬 싸다.
        """
        now = time.time()
        ts, cached = self._out_cache
        if now - ts < LIST_TTL:
            return cached
        keys, token = set(), None
        try:
            while True:
                kw = {"Bucket": self.bucket, "Prefix": self.output_prefix,
                      "MaxKeys": PAGE_MAX}
                if token:
                    kw["ContinuationToken"] = token
                r = self.client.list_objects_v2(**kw)
                keys.update(o["Key"] for o in r.get("Contents", []))
                token = r.get("NextContinuationToken")
                if not token:
                    break
        except Exception as e:                      # noqa: BLE001
            log.warning("결과물 목록을 읽지 못했다: %s", e)
            return cached
        self._out_cache = (now, keys)
        return keys

    # ── 전송 ──────────────────────────────────────────────────────────────

    def download(self, key, dest):
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        try:
            self.client.download_file(self.bucket, key, dest)
        except Exception as e:                      # noqa: BLE001
            raise wrap(e, f"내려받기 실패 ({key})") from e
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise S3Error(f"내려받은 파일이 비어 있다: {key}")
        return dest

    def presigned_url(self, key, expires=None):
        """결과물을 바로 받을 수 있는 임시 URL.

        GPU 서버가 파일 전송까지 떠안을 이유가 없고, 로컬 사본이 보관 기간에
        정리돼도 S3 원본은 남아 있다.
        """
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=int(expires or URL_TTL))
        except Exception as e:                      # noqa: BLE001
            raise wrap(e, f"URL 생성 실패 ({key})") from e

    def exists(self, key):
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:                           # noqa: BLE001
            return False

    def upload(self, path, key, content_type="video/mp4"):
        try:
            self.client.upload_file(path, self.bucket, key,
                                    ExtraArgs={"ContentType": content_type})
        except Exception as e:                      # noqa: BLE001
            raise wrap(e, f"올리기 실패 ({key})") from e
        self._out_cache = (0.0, set())              # 방금 올린 것이 반영되게
        return key


_store = None


def get_store():
    """설정돼 있으면 S3Store, 아니면 None."""
    global _store
    if not BUCKET:
        return None
    if _store is None:
        _store = S3Store()
    return _store
