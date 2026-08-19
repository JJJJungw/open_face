"""처리 파라미터 기본값 — **두 진입점의 단일 출처.**

웹 화면(`service/`)과 큐 워커(`msa/`)는 같은 영상을 같은 설정으로 처리해야 한다.
그런데 이 값들은 원래 `service/config.py` 에만 있었고, 큐 워커는 잡 페이로드가
말하지 않은 항목을 **파이프라인 시그니처 기본값**으로 떨어뜨렸다.

    파이프라인 시그니처:  batch_size=1,  imgsz=960
    서비스 기본값:        batch_size=32, imgsz=1280

그래서 큐 경로는 한 장씩 추론하고 있었다. L40S 에서 GPU 사용률 20%, 메모리
721 MiB — 45GB 짜리 카드를 거의 안 쓰는 상태로 한 편에 49.5초를 썼다.

**기본값이 두 벌 있으면 언젠가 어긋난다.** 여기 한 벌만 둔다. `service` 는 이걸
`JOB_DEFAULTS` 로 쓰고, `job_runner` 는 잡 페이로드가 덮어쓰지 않은 자리에 채운다.
어긋나지 않는다는 것은 테스트가 지킨다.

값 자체를 코드에 박지 않고 환경 변수로 여는 이유는 납품 기준이 바뀔 때 코드를
고치게 하면 안 되기 때문이다.
"""

import os

from .env import flag as _bool

from .core.pipeline import (
    DEFAULT_BITRATE_RATIO,
    DEFAULT_CRF,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_BITRATE,
    DEFAULT_TARGET_BITRATE,
)





# 검출기 해상도. 잡이 말하지 않으면 이 값으로 검출한다.
IMGSZ = int(os.environ.get("FA_IMGSZ", 1280))

# 한 번에 GPU 에 올리는 프레임 수. **이 값이 1이면 GPU 가 논다.**
#
# 32 는 개발기(L40S 45GB) 기준이다. **운영 인스턴스는 더 작을 수 있으므로 이 값은
# 배포마다 정해야 한다** — T4(16GB)나 L4(24GB)에서 32/1280 은 OOM 이 날 수 있다.
# 그래서 코드에 박지 않고 FA_BATCH_SIZE 로 열어 두고, 그래도 모자라면 잡 러너가
# 절반씩 줄여 가며 다시 시도한다(job_runner.run_target). 조용히 실패하지 않게
# 하는 것이 목적이지, 튜닝을 대신하는 것은 아니다 — 줄여서 돌면 그만큼 느리다.
BATCH_SIZE = int(os.environ.get("FA_BATCH_SIZE", "32"))

# 여기까지 줄여도 안 되면 그 워커에는 이 영상이 버거운 것이다.
BATCH_MIN = int(os.environ.get("FA_BATCH_MIN", "1"))

DEFAULTS = {
    "method": os.environ.get("FA_METHOD", "mosaic"),
    "conf": float(os.environ.get("FA_CONF", "0.25")),
    "imgsz": IMGSZ,
    "batch_size": BATCH_SIZE,
    "pad": float(os.environ.get("FA_PAD", "0.15")),
    "mosaic_scale": float(os.environ.get("FA_MOSAIC_SCALE", "0.06")),
    "linger": int(os.environ.get("FA_LINGER", "5")),
    "interp": _bool("FA_INTERP", True),
    "keep_audio": _bool("FA_KEEP_AUDIO", True),
    "crf": DEFAULT_CRF,
    "bitrate_ratio": DEFAULT_BITRATE_RATIO,
    # 납품 스펙. 값만 바꾸면 되도록 열어 둔다 — 기준이 바뀔 때 코드를 고치게
    # 하면 안 된다. height=0 이면 원본 유지, bitrate="" 면 예전 CRF 방식.
    "height": DEFAULT_HEIGHT,
    "bitrate": DEFAULT_TARGET_BITRATE,
    "max_bitrate": DEFAULT_MAX_BITRATE,
}

# 잡 페이로드가 정할 수 있는 것들. 여기 없는 키는 잡이 못 바꾼다 — 임의의
# 파이프라인 인자를 페이로드로 넘기게 두면 계약이 없는 것과 같다.
JOB_OVERRIDABLE = (
    "method", "conf", "imgsz", "batch_size", "pad", "mosaic_scale", "linger",
    "interp", "keep_audio", "height", "bitrate", "max_bitrate", "crf",
)
