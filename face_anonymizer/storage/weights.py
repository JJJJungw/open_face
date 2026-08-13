"""가중치 확보 — 조달처를 갈아 끼울 수 있게 둔다.

**세 가지가 있고 지금 쓰는 것은 ``s3`` 다.** 어느 것을 쓰든 바깥에서 보는 계약은
같다 — ``ensure(path)`` 를 부르고 나면 그 경로에 온전한 가중치가 있다.

    s3     우리 버킷에서 받는다 (기본). 자격 증명이 필요하다
    baked  이미지에 이미 구워져 있다. 네트워크 호출 0, 있는지만 확인한다
    url    잡 페이로드가 준 서명된 URL 로 받는다. 자격 증명 0

``url`` 은 RebornStudio 무DB 워커 규약("워커는 자격 증명을 갖지 않는다")에 맞춘
길이고, ``baked`` 는 그 규약을 이미지 빌드 시점으로 미루는 길이다. **셋 중
무엇으로 갈지는 아직 정하지 않았다** — 상대와 합의할 사항이라 우리끼리 못 정한다
(docs/integration/rebornstudio.md 의 미결 D1). 그래서 결정을 코드에 굳히지 않고
스위치로 남긴다. ``FA_WEIGHTS_SOURCE`` 하나가 바뀌고 부르는 쪽은 안 바뀐다.

아래는 지금 기본값인 ``s3`` 의 근거다.
배포할 때마다 GitHub 릴리스에 의존하면 곤란하다. 레이트 리밋에 걸리고, 네트워크
정책에 막히고, 업스트림 태그가 바뀌면 어제와 다른 파일을 받는다. 가중치는 이미
자격 증명이 있는 우리 버킷에 두고 거기서 받는다.

**이미 있으면 아무것도 하지 않는다.** 네트워크 호출조차 없다. 그래서 이 함수를
검출기 만들기 직전에 불러도 평소에는 비용이 0 이고, 새 EC2 나 컨테이너에서만
실제로 내려받는다.

받을 때는 **임시 파일에 받고 원자적으로 옮긴다.** 중간에 끊긴 파일이 제자리에
남으면 다음 기동에서 "있다" 로 판정되고, 그때 나는 오류는 원인이 전혀 드러나지
않는다(체크포인트 언피클 실패).

리포(third_party/YOLO-FaceV2)는 그대로 GitHub 에서 클론한다. 바꾸는 범위를
가중치 하나로 좁힌다.
"""

import logging
import os
import shutil
import tempfile

log = logging.getLogger(__name__)

# 버킷 안의 위치. 결과물과 같은 v1/ 아래 model/ 을 쓴다.
WEIGHTS_KEY = os.environ.get("FA_S3_WEIGHTS_KEY", "v1/model/yolo-facev2.pt")

# s3 | baked | url — 위 독스트링 참고. 미결 D1 이 정해지면 기본값만 바꾸면 된다.
SOURCE = (os.environ.get("FA_WEIGHTS_SOURCE") or "s3").strip().lower()

# 이보다 작으면 받다 만 파일로 본다. 실제 가중치는 40MB 남짓이다.
MIN_BYTES = int(os.environ.get("FA_WEIGHTS_MIN_BYTES", 1_000_000))


class WeightsUnavailable(RuntimeError):
    """가중치를 갖추지 못했다. 사유가 메시지에 담긴다."""


def looks_complete(path):
    return os.path.exists(path) and os.path.getsize(path) >= MIN_BYTES


def _atomic(path):
    """받는 중인 파일을 옆에 두는 임시 경로. 같은 파일시스템이라야 os.replace 가 된다."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".weights-",
                               dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    return tmp


def ensure(path, key=None, source=None, url=None):
    """``path`` 에 가중치를 갖춰 놓고 그 경로를 돌려준다.

    이미 온전한 파일이 있으면 **아무것도 하지 않는다** — 네트워크 호출조차 없다.
    그래서 검출기 만들기 직전에 불러도 평소 비용이 0 이다.

    Args:
        source: ``s3`` | ``baked`` | ``url``. 생략하면 ``FA_WEIGHTS_SOURCE``.
        url:    ``source='url'`` 일 때 쓸 서명된 GET. 잡 페이로드가 준다.
    """
    if looks_complete(path):
        return path

    src = (source or SOURCE)
    if src == "baked":
        # 이미지에 굽는 방식에서는 여기 도달한 것 자체가 빌드 사고다. 조용히
        # 네트워크로 흘러가면 "왜 첫 요청이 40초 걸리지" 로 나타난다.
        raise WeightsUnavailable(
            f"가중치가 이미지에 없습니다 ({path}). FA_WEIGHTS_SOURCE=baked 는 "
            f"빌드 때 넣어 두는 방식이라 실행 중에 받아 오지 않습니다.")

    if src == "url":
        return _from_url(path, url)

    return _from_s3(path, key)


def _from_url(path, url):
    """서명된 GET 으로 받는다. **자격 증명이 없다** — 워커 이미지에 비밀이 안 들어간다."""
    from . import transfer

    if not url:
        raise WeightsUnavailable(
            "FA_WEIGHTS_SOURCE=url 인데 가중치 URL 이 없습니다. "
            "잡 페이로드의 weights_url 을 확인해 주세요.")
    tmp = _atomic(path)
    try:
        log.info("가중치를 서명된 URL 로 받는다")
        transfer.fetch(url, tmp)
        if not looks_complete(tmp):
            raise WeightsUnavailable(
                f"내려받은 가중치가 너무 작습니다 ({os.path.getsize(tmp)} bytes).")
        os.replace(tmp, path)
        log.info("가중치 준비 완료: %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
        return path
    except transfer.TransferError as e:
        raise WeightsUnavailable(f"가중치를 내려받지 못했습니다: {e}") from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _from_s3(path, key=None):
    from . import s3 as s3mod                     # 지연 임포트 (boto3 를 늦게 연다)
    key = key or WEIGHTS_KEY
    store = s3mod.get_store()
    if store is None:
        raise WeightsUnavailable(
            f"가중치가 없고({path}) S3 도 설정되어 있지 않습니다. "
            f"FA_S3_BUCKET 을 설정하시거나 python setup_weights.py 를 실행해 주세요.")

    tmp = _atomic(path)
    try:
        log.info("가중치를 S3 에서 받는다: s3://%s/%s", store.bucket, key)
        store.download(key, tmp)
        if not looks_complete(tmp):
            raise WeightsUnavailable(
                f"내려받은 가중치가 너무 작습니다 ({os.path.getsize(tmp)} bytes). "
                f"s3://{store.bucket}/{key} 를 확인해 주세요.")
        os.replace(tmp, path)                     # 원자적. 부분 파일이 남지 않는다
        log.info("가중치 준비 완료: %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
        return path
    except s3mod.S3Error as e:
        raise WeightsUnavailable(
            f"가중치를 내려받지 못했습니다 (s3://{store.bucket}/{key}): {e}") from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
