"""저장소 계약 — **갈아 끼우려면 무엇을 지켜야 하는지 여기 적는다.**

왜 따로 쓰나
------------
"클라우드를 고를 수 있어야 한다" 는 요구는 두 층으로 나뉜다.

**1층 — 주소만 다른 경우.** AWS S3 · NCP · R2 · MinIO · Wasabi 는 전부 S3 API
를 그대로 쓴다. 이건 `providers.py` 의 설정으로 끝난다. 코드가 달라질 게 없다.
실제로 붙을 대상의 대부분이 여기 있다.

**2층 — 말이 아예 다른 경우.** GCS · Azure Blob 은 API 모양 자체가 다르다.
이건 설정으로 안 되고 **구현을 하나 더 써서 꽂아야** 한다. 그때 "무엇을 쓰면
되나" 에 답하는 것이 이 파일이다. 이게 없으면 새 저장소를 붙이는 사람이
`s3.py` 900 줄을 읽고 어느 게 계약이고 어느 게 S3 사정인지 추측해야 한다.

계약이 이미 돌아간다는 증거
---------------------------
테스트가 `S3Store` 대신 완전히 다른 객체(``tests/conftest.py`` 의 가짜 스토어)
를 꽂아서 서버 전체를 돌린다. boto3 도 네트워크도 없이 돈다. 즉 **틈은 원래
있었고 적어 두지 않았을 뿐**이다. 여기서 하는 일은 그 틈에 이름을 붙이는 것,
그리고 꽂는 자리를 한 곳(``providers.STORES``)으로 모으는 것이다.

계약에 없는 것
--------------
자격 증명은 계약이 아니다. 우리는 키를 들고 있지 않는다 — 각 구현이 자기
방식대로(EC2 인스턴스 역할 · 환경 변수 · 서비스 계정) 알아서 얻는다.

'이미 처리했나' 판정도 계약이 아니라 **결과**다. `processed_keys()` 가 결과
버킷을 대조해서 답하므로, 저장소를 바꾸면 그 답이 통째로 바뀐다. 갈아 끼우는
쪽은 그걸 사람에게 먼저 알려야 한다(`s3.reconfigure` 주석).
"""

# 저장소 하나가 반드시 가져야 하는 이름들. 부르는 쪽에서 실제로 쓰는 것만 넣는다
# — 계약이 넓어지면 새 구현을 쓰기가 그만큼 어려워진다.
#
# 값(속성):
#   bucket          어디에 붙어 있나. 화면과 로그에 그대로 나간다
#   root_prefix     입력을 찾기 시작할 자리
#   output_prefix   결과물을 쌓을 자리
#
# 동작(메서드):
#   check()                        지금 설정으로 읽고 쓸 수 있나. 못 하면 예외
#   list(prefix)                   한 단계만 -> (폴더 목록, 객체 목록)
#   list_all(prefix)               그 아래 전부 -> 키 목록
#   output_key(key)                입력 키 -> 결과물 키 (naming.py 규칙)
#   processed_keys()               결과물 프리픽스에 이미 있는 키 집합
#   exists(key)                    있나
#   size_of(key)                   바이트. 모르면 None
#   download(key, dest, callback)  받아 내린다. 콜백은 (받은 바이트, 전체)
#   upload(path, key, ...)         올린다. 콜백 규약은 위와 같다
#   presigned_url(key, ...)        브라우저가 바로 열 수 있는 주소
ATTRS = ("bucket", "root_prefix", "output_prefix")

METHODS = ("check", "list", "list_all", "output_key", "processed_keys",
           "exists", "size_of", "download", "upload", "presigned_url")

CONTRACT = ATTRS + METHODS


def missing(store):
    """계약에서 빠진 이름들. 다 지켰으면 빈 튜플.

    **런타임에 부르지 않는다.** 새 구현을 붙이는 사람이 테스트에서 쓰는
    물건이다(`tests/test_storage_contract.py`). 요청을 받고 나서
    AttributeError 로 아는 것보다 붙이는 날 아는 편이 낫다.
    """
    return tuple(n for n in CONTRACT if not hasattr(store, n))


class NotImplementedStore:
    """아직 안 만든 저장소 — **자리는 있고 동작은 없다.**

    GCS · Azure 처럼 프로토콜이 다른 곳을 고르면 이게 나온다. 계약의 이름은
    전부 갖고 있어서 꽂는 자리가 실제로 존재한다는 것은 증명하되, 부르면
    분명하게 거절한다.

    **조용한 목업이 제일 나쁘다.** 빈 목록을 돌려주는 가짜를 두면 화면에는
    "폴더 0개" 가 뜨고, 사람은 버킷이 비었다고 믿는다. 900 건을 넣었는데
    아무것도 안 나오는 것보다, 고른 순간 "아직 지원하지 않습니다" 를 보는 편이
    낫다. 실제로 붙일 날이 오면 이 자리에 진짜 구현을 놓으면 된다.
    """

    def __init__(self, config=None, **kw):
        self.config = config
        info = getattr(config, "info", None) or {}
        self.name = info.get("name") or "이 저장소"
        self.bucket = getattr(config, "bucket", "") or ""
        self.root_prefix = getattr(config, "root_prefix", "") or ""
        self.output_prefix = getattr(config, "output_prefix", "") or ""

    def _no(self, what):
        from .s3 import S3Error, _problems          # noqa: PLC0415 (순환 방지)
        e = S3Error(f"{self.name} 는 아직 지원하지 않습니다 ({what})")
        # 딱지는 HTTP 계층에서만 쓴다. 워커에는 fastapi 가 없고 응답도 없다.
        errors = _problems()
        if errors is not None:
            e.problem = errors.S3_NOT_CONFIGURED
        return e

    def check(self):
        raise self._no("연결 확인")

    def list(self, prefix=""):
        raise self._no("목록")

    def list_all(self, prefix):
        raise self._no("목록")

    def output_key(self, key):
        raise self._no("결과물 경로")

    def processed_keys(self):
        raise self._no("처리 여부 대조")

    def exists(self, key):
        raise self._no("존재 확인")

    def size_of(self, key):
        raise self._no("크기 조회")

    def download(self, key, dest, callback=None):
        raise self._no("내려받기")

    def upload(self, path, key, content_type="video/mp4", callback=None):
        raise self._no("올리기")

    def presigned_url(self, key, expires=None, filename=None):
        raise self._no("서명된 주소")
