"""가중치 확보 — 없으면 S3 에서 받아 온다.

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

# 이보다 작으면 받다 만 파일로 본다. 실제 가중치는 40MB 남짓이다.
MIN_BYTES = int(os.environ.get("FA_WEIGHTS_MIN_BYTES", 1_000_000))


class WeightsUnavailable(RuntimeError):
    """가중치를 갖추지 못했다. 사유가 메시지에 담긴다."""


def looks_complete(path):
    return os.path.exists(path) and os.path.getsize(path) >= MIN_BYTES


def ensure(path, key=None):
    """``path`` 에 가중치를 갖춰 놓고 그 경로를 돌려준다.

    이미 온전한 파일이 있으면 그대로 둔다. 없으면 S3 에서 받는다.
    """
    if looks_complete(path):
        return path

    from . import s3 as s3mod                     # 지연 임포트 (boto3 를 늦게 연다)
    key = key or WEIGHTS_KEY
    store = s3mod.get_store()
    if store is None:
        raise WeightsUnavailable(
            f"가중치가 없고({path}) S3 도 설정되어 있지 않습니다. "
            f"FA_S3_BUCKET 을 설정하시거나 python setup_weights.py 를 실행해 주세요.")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".weights-", dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
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
