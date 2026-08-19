"""가중치 확보 — 조달처를 갈아 끼울 수 있게 둔다.

**어느 것을 쓰든 바깥에서 보는 계약은 같다** — ``ensure(path)`` 를 부르고 나면
그 경로에 온전한 가중치가 있다.

    auto   있으면 그대로, 없으면 S3 → 공개 URL 순으로 시도한다 (기본)
    s3     버킷에서만 받는다. 자격 증명이 필요하다
    baked  이미지에 이미 구워져 있다. 네트워크 호출 0, 있는지만 확인한다
    url    잡 페이로드가 준 서명된 URL 로 받는다. 자격 증명 0

기본이 ``auto`` 인 이유
-----------------------
예전 기본은 ``s3`` 였고 그건 **우리 버킷을 전제**한 값이었다. 남이 이걸 클론해
자기 버킷을 붙이면 그 버킷에 가중치가 없다 — 첫 실행 화면에서 저장소를 골라
통과시켜 놓고, 정작 첫 영상에서 "가중치가 없습니다" 로 멎는다. 저장소를 고를 수
있게 만들어 놓고 모델은 우리 것에 묶어 두면 앞의 노력이 통째로 무의미하다.

``auto`` 는 셋을 순서대로 본다. 이미 있으면 아무것도 안 하고(네트워크 호출조차
없다), 없으면 버킷을 보고, 거기도 없으면 공개 릴리스에서 받는다. 우리 EC2 는
1번이나 2번에서 끝나므로 **동작이 예전과 같고**, 남의 환경은 3번으로 산다.

버킷을 먼저 보는 이유는 그대로다. 배포할 때마다 GitHub 릴리스에 의존하면
레이트 리밋에 걸리고, 네트워크 정책에 막히고, 업스트림 태그가 바뀌면 어제와
다른 파일을 받는다. 그래서 **우리는** 버킷을 쓰고, 그게 없는 사람에게는 길을
막지 않는다.

가중치를 저장소에 넣지 않는 이유
--------------------------------
40MB 짜리 이진 파일이라 git 이 감당할 물건이 아니고, 애초에 **우리가 만든 것이
아니다.** 업스트림(clibdev/YOLO-FaceV2, GPL-3.0)이 배포하는 릴리스 자산을
그대로 가리킨다 — 우리가 사본을 떠서 다시 배포하는 것보다 출처가 분명하다.

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
import tempfile

log = logging.getLogger(__name__)

# 버킷 안의 위치. 결과물과 같은 v1/ 아래 model/ 을 쓴다.
WEIGHTS_KEY = os.environ.get("FA_S3_WEIGHTS_KEY", "v1/model/yolo-facev2.pt")

# auto | s3 | baked | url — 위 독스트링 참고. MSA 미결 D1 이 정해지면 그쪽만
# 기본값을 바꾸면 된다(컨테이너는 url 또는 baked 로 간다).
SOURCE = (os.environ.get("FA_WEIGHTS_SOURCE") or "auto").strip().lower()

# 없을 때 마지막으로 가는 곳. 업스트림이 배포하는 릴리스 자산을 그대로 가리킨다
# — 우리가 사본을 떠서 다시 배포하는 것보다 출처가 분명하다. 사내망처럼 여기에
# 못 닿는 곳이면 FA_WEIGHTS_URL 로 자기 미러를 넣으면 된다.
PUBLIC_URL = os.environ.get("FA_WEIGHTS_URL") or (
    "https://github.com/clibdev/YOLO-FaceV2/releases/latest/download/"
    "yolo-facev2.pt")

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
        source: ``auto`` | ``s3`` | ``baked`` | ``url``. 생략하면
                ``FA_WEIGHTS_SOURCE``.
        url:    ``source='url'`` 일 때 쓸 서명된 GET. 잡 페이로드가 준다.
    """
    if looks_complete(path):
        return path

    src = (source or SOURCE)
    if src == "auto":
        return _auto(path, key)
    if src == "baked":
        # 이미지에 굽는 방식에서는 여기 도달한 것 자체가 빌드 사고다. 조용히
        # 네트워크로 흘러가면 "왜 첫 요청이 40초 걸리지" 로 나타난다.
        raise WeightsUnavailable(
            f"가중치가 이미지에 없습니다 ({path}). FA_WEIGHTS_SOURCE=baked 는 "
            f"빌드 때 넣어 두는 방식이라 실행 중에 받아 오지 않습니다.")

    if src == "url":
        return _from_url(path, url)

    return _from_s3(path, key)


def _auto(path, key=None):
    """버킷 → 공개 URL. **먼저 것이 실패해도 멈추지 않는다.**

    남의 버킷에는 가중치가 없는 것이 정상이다. 그걸 실패로 끝내면 저장소를
    고를 수 있게 만들어 둔 의미가 없다. 다만 **왜 넘어갔는지는 남긴다** —
    조용히 넘어가면 우리 버킷 설정이 잘못된 날에도 그냥 GitHub 에서 받아 와서,
    한참 뒤에 "왜 매번 40MB 를 받지" 로 나타난다.
    """
    from . import s3 as s3mod                     # 지연 임포트

    tried = []
    if s3mod.get_store() is not None:
        try:
            return _from_s3(path, key)
        except WeightsUnavailable as e:
            tried.append(f"버킷: {e}")
            log.info("버킷에서 가중치를 못 받았다 — 공개 릴리스로 넘어간다: %s", e)
    else:
        tried.append("버킷: 설정되어 있지 않습니다")

    try:
        return _from_url(path, PUBLIC_URL)
    except WeightsUnavailable as e:
        tried.append(f"공개 릴리스: {e}")

    # 여기까지 왔으면 사람이 손을 대야 한다. **무엇을 하면 되는지 다 적는다** —
    # 세 갈래를 다 시도하고 실패한 상황이라, 하나만 알려 주면 그것만 붙들게 된다.
    raise WeightsUnavailable(
        "모델 가중치를 갖추지 못했습니다.\n  " + "\n  ".join(tried) +
        f"\n다음 중 하나로 해결됩니다 — {path} 에 파일을 직접 두거나, "
        "FA_WEIGHTS_URL 에 받을 수 있는 주소를 넣거나, "
        "python scripts/setup_weights.py 를 실행하시면 됩니다.")


def status(path=None):
    """지금 가중치가 있나, 없으면 어디서 받게 되나. **화면이 쓴다.**

    첫 실행에서 저장소만 정해 놓고 모델은 말해 주지 않으면, 통과한 사람이
    첫 영상에서야 문제를 만난다. 그때는 이미 900건을 넣은 뒤일 수도 있다.
    """
    from ..core.paths import DEFAULT_WEIGHTS      # 지연 임포트 (torch 안 끌어옴)

    path = path or DEFAULT_WEIGHTS
    if looks_complete(path):
        return {"present": True, "path": path,
                "size_mb": round(os.path.getsize(path) / 1e6, 1),
                "source": SOURCE,
                "detail": "준비되어 있습니다"}
    return {"present": False, "path": path, "size_mb": None, "source": SOURCE,
            "detail": ("처음 처리할 때 자동으로 내려받습니다 (약 40MB)"
                       if SOURCE == "auto" else
                       f"FA_WEIGHTS_SOURCE={SOURCE} 로 조달합니다"),
            "url": PUBLIC_URL if SOURCE == "auto" else None}


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
            f"FA_S3_BUCKET 을 설정하시거나 python scripts/setup_weights.py 를 실행해 주세요.")

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
