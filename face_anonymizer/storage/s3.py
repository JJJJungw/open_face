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
from urllib.parse import quote
import time

from . import naming, providers

log = logging.getLogger(__name__)

# 지금 어디에 붙어 있나. **객체 하나로 모아 둔다** — 예전에는 모듈 상수라
# 임포트할 때 한 번 읽고 끝이어서, 설정을 바꾸려면 서버를 다시 띄워야 했고
# 화면에서 고르게 하는 길이 아예 막혀 있었다(providers.StorageConfig 주석).
CONFIG = providers.StorageConfig.from_env()

LIST_TTL = int(os.environ.get("FA_S3_LIST_TTL", 30))
URL_TTL = int(os.environ.get("FA_S3_URL_TTL", 3600))

PAGE_MAX = 1000


class TransferAborted(Exception):
    """전송 콜백이 중단을 요청했다.

    **S3 오류가 아니므로 감싸지 않고 그대로 올려보낸다.** 사용자가 취소를
    누른 것을 "S3 호출 실패" 로 보고하면 원인이 완전히 뒤바뀐다.
    """


class S3Error(RuntimeError):
    """S3 호출 실패. 작업은 실패로 남기되 서버는 계속 산다.

    ``problem`` 에 구체 원인이 붙는다 — 권한 문제와 키 오타와 네트워크 장애는
    사용자가 해야 할 일이 전혀 다르다.
    """

    problem = None


def wrap(e, what):
    """botocore 예외를 원인이 드러나는 S3Error 로."""
    from ..service import errors                     # 지연 임포트 (순환 방지)
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


# 화면에서 받은 열쇠. **메모리에만 있다.**
#
# 파일에 안 쓰고, 어떤 라우트로도 돌려주지 않는다. 서버를 다시 띄우면 사라진다
# — 그게 이 방식의 값이자 대가다. 값은 웹 화면이 있는 도구인데 시작하려고
# 터미널을 먼저 치지 않아도 된다는 것이고, 대가는 영구히 두려면 결국 환경
# (인스턴스 역할 · aws configure · 환경 변수)으로 옮겨야 한다는 것이다.
#
# 대부분은 여기까지 안 온다. EC2 인스턴스 역할이 붙어 있거나 `aws configure`
# 가 되어 있으면 boto3 기본 체인이 알아서 잡는다.
_creds = None


def set_credentials(access_key=None, secret_key=None, session_token=None):
    """화면에서 받은 열쇠를 메모리에 둔다. 비우려면 인자 없이 부른다."""
    global _creds, _store
    _creds = ({"aws_access_key_id": access_key,
               "aws_secret_access_key": secret_key,
               **({"aws_session_token": session_token} if session_token else {})}
              if access_key and secret_key else None)
    _store = None                                  # 다음 get_store 부터 새 열쇠
    return _creds is not None


def credentials():
    """지금 메모리에 든 열쇠. **되돌리기용이다** — 화면에 보내지 않는다."""
    return _creds


def restore(config, creds):
    """시험에 실패했을 때 있던 자리로. 설정과 열쇠를 같이 돌려놓는다."""
    global _creds, _store
    _creds = creds
    reconfigure(config)
    _store = None
    return CONFIG


def credential_source():
    """열쇠가 **어디서** 오고 있나. (설명, 있나)

    이게 없으면 왜 되는지 왜 안 되는지를 아무도 모른다. 되는 날에는 아무래도
    좋지만 안 되는 날에는 이 한 줄이 없어서 엉뚱한 데를 뒤지게 된다.
    """
    if _creds:
        return "화면에서 받음 (메모리 — 서버를 다시 띄우면 사라집니다)", True
    try:
        import boto3                               # noqa: PLC0415
        c = boto3.Session().get_credentials()
    except Exception:                              # noqa: BLE001
        return "확인할 수 없습니다", False
    if c is None:
        return "없습니다", False
    m = getattr(c, "method", "") or ""
    return {
        "iam-role": "EC2 인스턴스 역할",
        "env": "환경 변수 (AWS_ACCESS_KEY_ID)",
        "shared-credentials-file": "~/.aws/credentials",
        "config-file": "~/.aws/config",
        "container-role": "컨테이너 역할",
        "assume-role": "역할 위임 (assume-role)",
        "sso": "AWS SSO",
    }.get(m, m or "알 수 없는 경로"), True


def editable(config=None):
    """화면에서 저장소를 바꿀 수 있나. (가능?, 사유)

    **연결하고 끊는 것은 화면이 한다.** 처음에는 첫 실행에만 열고 그 뒤로는
    잠갔는데, 그러면 다른 버킷으로 옮기려고 서버를 다시 띄워야 했다. 도구를
    쓰는 사람 입장에서 그건 "고를 수 있다" 가 아니다.

    잠금을 푼 대가는 분명히 적어 둔다. **이 API 에는 인증이 없다.** 공인 IP 에
    띄우면 서버에 닿는 누구나 저장소를 자기 것으로 바꿀 수 있다 — 다만 그 사람은
    이미 작업을 넣고 지우고 결과 주소를 받아 갈 수도 있다. 즉 저장소 설정만
    잠가 두는 것은 문 하나만 잠그고 나머지를 다 열어 두는 일이었다. 진짜 답은
    인증이고, 그때까지 이 도구는 **믿을 수 있는 망 안에서** 띄우는 물건이다.

    잠가야 하는 배포는 ``FA_ALLOW_STORAGE_EDIT=0`` 으로 띄운다.
    """
    v = os.environ.get("FA_ALLOW_STORAGE_EDIT", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        if not (config or CONFIG).bucket:
            return True, ""                        # 첫 실행은 그래도 열어 준다
        return False, ("이 서버는 화면에서 저장소를 바꿀 수 없게 띄워져 "
                       "있습니다. .env 를 고치고 다시 띄워 주세요.")
    return True, ""


def disconnect():
    """붙어 있던 곳에서 떨어진다. 저장해 둔 설정과 메모리의 열쇠를 같이 지운다.

    **버킷은 건드리지 않는다.** 파일은 그대로 있고 우리가 안 볼 뿐이다.
    """
    global _creds
    _creds = None
    path = providers.saved_path()
    try:
        os.remove(path)
    except OSError:
        pass
    reconfigure(providers.StorageConfig(provider=CONFIG.provider, bucket=""))
    return path


def client_config():
    """botocore 설정. **체크섬 한 줄이 NCP 이관을 좌우한다.**

    botocore 1.36 부터 PutObject 에 CRC32 체크섬을 기본으로 붙이는데 **NCP 가
    이걸 AccessDenied 로 거절한다.** 같은 키로 체크섬만 빼면 200 이다 — 권한
    문제가 아니라 체크섬 문제인데 돌아오는 말은 똑같이 "AccessDenied" 라서,
    이관 당일에 키·IAM·버킷 정책을 며칠 뒤지게 된다. 미리 막아 둔다.

    ``when_required`` 는 1.36 이전의 동작이다 — AWS S3 에서도 그대로 안전하다.
    옵션을 모르는 낡은 botocore 는 TypeError 를 내므로 그때는 빼고 만든다.
    """
    try:
        from botocore.client import Config         # noqa: PLC0415 — 지연 임포트
    except ImportError:                            # boto3 가 없는 환경(테스트)
        return None
    try:
        return Config(signature_version="s3v4",
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required")
    except TypeError:                              # botocore < 1.36
        return Config(signature_version="s3v4")


def make_client(config=None):
    """boto3 클라이언트. **엔드포인트를 넘길 수 있다.**

    NCP Object Storage · Cloudflare R2 · MinIO · Wasabi 는 전부 S3 API 를 그대로
    쓴다. 코드가 달라질 게 없고 이 주소 하나만 다르다 — 그래서 어댑터가 아니라
    설정으로 푼다(storage/providers.py).

    자격 증명은 boto3 기본 체인이다. EC2 인스턴스 역할이 있으면 그대로 잡히고,
    다른 제공자는 환경 변수(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)를 쓴다 —
    **우리가 키를 파일에 들고 있지 않는다.** 첫 실행 화면에서 받은 것이 있으면
    그게 이기는데, 그것도 메모리에만 있다(`set_credentials`).
    """
    import boto3                                   # noqa: PLC0415 — 지연 임포트
    c = config or CONFIG
    kw = {"region_name": c.region, **(_creds or {})}
    cfg = client_config()
    if cfg is not None:
        kw["config"] = cfg
    if c.endpoint:
        kw["endpoint_url"] = c.endpoint
    return boto3.client("s3", **kw)


class S3Store:
    """버킷 하나를 다루는 얇은 래퍼."""

    def __init__(self, bucket=None, client=None, output_prefix=None,
                 root_prefix=None, config=None):
        self.config = config or CONFIG
        self.bucket = bucket or self.config.bucket
        self.output_prefix = (output_prefix if output_prefix is not None
                              else self.config.output_prefix)
        self.root_prefix = (root_prefix if root_prefix is not None
                            else self.config.root_prefix)
        self._client = client
        self._out_cache = (0.0, set())

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(self.config)
        return self._client

    def check(self):
        """지금 설정으로 **실제로 붙는지** 본다. 되면 (True, 설명).

        잘못된 버킷에 900건을 넣고 나서 아는 것보다, 넣기 전에 아는 편이 낫다.
        읽기와 쓰기를 따로 본다 — 읽기만 되는 자격 증명이 흔하다.
        """
        try:
            self.client.list_objects_v2(Bucket=self.bucket,
                                        Prefix=self.root_prefix, MaxKeys=1)
        except Exception as e:                      # noqa: BLE001
            raise wrap(e, f"버킷을 읽지 못했습니다 ({self.bucket})") from e
        probe = self.output_prefix + ".fa-write-check"
        try:
            self.client.put_object(Bucket=self.bucket, Key=probe, Body=b"ok")
            self.client.delete_object(Bucket=self.bucket, Key=probe)
        except Exception as e:                      # noqa: BLE001
            raise wrap(e, "읽기는 되지만 결과물을 쓰지 못합니다 "
                          f"({self.output_prefix})") from e
        return True

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
            raise wrap(e, "목록을 불러오지 못했습니다") from e
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
            raise wrap(e, "목록을 불러오지 못했습니다") from e
        return objects

    def output_key(self, key):
        """입력 키에 대응하는 결과물 키.

        데이터셋 규칙(naming.py)을 따른다 — 정체성 필드는 그대로 두고 STATE 만
        raw -> deid 로 바꾼다. 결과는 **입력 폴더별로 나눠 쌓는다.**

            videos/2026-08/f_00001_00_0000000_0042000_raw.mp4
            -> v1/results/face/2026-08_deid/f_00001_00_0000000_0042000_deid.mp4

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
        """입력 키가 들어갈 결과 하위 폴더 ('2026-08_deid/' 또는 '')."""
        parent = os.path.basename(os.path.dirname(key or ""))
        return f"{parent}_deid/" if parent else ""

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

    def size_of(self, key):
        """객체 크기(bytes). 알 수 없으면 0.

        전송 진행률의 분모다. 작업 하나에 한 번이라 왕복이 아깝지 않다 —
        목록에서 크기를 들고 다니게 만드는 쪽이 더 번거롭다.
        """
        try:
            return int(self.client.head_object(Bucket=self.bucket,
                                               Key=key).get("ContentLength", 0))
        except Exception:                           # noqa: BLE001
            return 0

    def download(self, key, dest, callback=None):
        """``callback(전송된 바이트)`` 는 boto3 가 청크마다 부른다.

        여기서 취소를 확인할 수 있다. 그러지 않으면 큰 파일을 받는 동안에는
        취소를 눌러도 다 받을 때까지 안 멈춘다(docs/issues/004).
        """
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        try:
            self.client.download_file(self.bucket, key, dest, Callback=callback)
        except TransferAborted:
            raise
        except Exception as e:                      # noqa: BLE001
            raise wrap(e, f"내려받지 못했습니다 ({key})") from e
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise S3Error(f"내려받은 파일이 비어 있습니다: {key}")
        return dest

    def presigned_url(self, key, expires=None, filename=None):
        """결과물을 바로 받을 수 있는 임시 URL.

        GPU 서버가 파일 전송까지 떠안을 이유가 없고, 로컬 사본이 보관 기간에
        정리돼도 S3 원본은 남아 있다.

        ``filename`` 을 주면 **내려받기**가 되고, 없으면 브라우저에서 바로 튼다.
        이게 없으면 S3 가 헤더 없이 mp4 를 주고, 브라우저는 내려받는 대신
        페이지를 떠나 영상을 재생한다 — '내려받기' 를 눌렀는데 화면이 사라진다.
        검수처럼 **보는 것이 목적**일 때는 일부러 비운다.
        """
        params = {"Bucket": self.bucket, "Key": key}
        if filename:
            # 한글 파일명은 그대로 헤더에 못 넣는다(RFC 5987).
            quoted = quote(filename)
            params["ResponseContentDisposition"] = (
                f'attachment; filename="{quoted}"; '
                f"filename*=UTF-8''{quoted}")
        try:
            return self.client.generate_presigned_url(
                "get_object", Params=params, ExpiresIn=int(expires or URL_TTL))
        except Exception as e:                      # noqa: BLE001
            raise wrap(e, f"다운로드 주소를 만들지 못했습니다 ({key})") from e

    def exists(self, key):
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:                           # noqa: BLE001
            return False

    def upload(self, path, key, content_type="video/mp4", callback=None):
        try:
            self.client.upload_file(path, self.bucket, key,
                                    ExtraArgs={"ContentType": content_type},
                                    Callback=callback)
        except TransferAborted:
            raise
        except Exception as e:                      # noqa: BLE001
            raise wrap(e, f"올리지 못했습니다 ({key})") from e
        self._out_cache = (0.0, set())              # 방금 올린 것이 반영되게
        return key


_store = None


def get_store():
    """붙을 수 있으면 스토어, 아니면 None.

    **어느 클래스를 쓸지는 여기서 정하지 않는다.** 제공자 등록표가 정한다
    (`providers.store_class`). 그래서 GCS 구현을 넣는 날 이 함수는 안 고친다 —
    부르는 쪽 열다섯 군데도 마찬가지다. 전부 이 한 줄을 지나가기 때문이다.

    지원하지 않는 제공자(GCS·Azure)를 골라 두면 **조용히 None 이 되지 않고**
    라우트가 그 사유를 돌려준다 — 설정이 잘못됐는데 "S3 미설정" 으로 보이면
    사람이 엉뚱한 데를 고친다.
    """
    global _store
    if not CONFIG.ready:
        return None
    if _store is None:
        _store = CONFIG.store_class(config=CONFIG)
    return _store


def unavailable_reason():
    """왜 못 붙는지 한 줄. 붙을 수 있으면 빈 문자열."""
    if CONFIG.ready:
        return ""
    if not CONFIG.supported:
        return f"{CONFIG.info['name']} 는 아직 지원하지 않습니다"
    if not CONFIG.bucket:
        return "버킷이 설정되어 있지 않습니다"
    return "엔드포인트 주소가 필요합니다"


def reconfigure(config):
    """설정을 갈아 끼운다. 다음 get_store() 부터 새 곳을 본다.

    **이미 처리한 기록과의 연결이 끊긴다.** '이미 처리했나' 판정은 결과 버킷
    대조이고, 진척률 폴더와 저널의 폴더 이름도 옛 저장소 기준이다. 사실상 새
    작업 공간을 여는 일이라, 부르는 쪽이 그걸 사람에게 먼저 알려야 한다.
    """
    global CONFIG, _store
    CONFIG, _store = config, None
    return CONFIG
