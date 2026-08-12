"""서비스 설정 — 환경 변수와 처리 기본값.

값이 여기 한곳에 모여 있어야 "무엇을 조절할 수 있는가" 를 한 화면에서 본다.
이 모듈은 아무것도 임포트하지 않는다(코어의 기본값 제외) — 설정이 로직을
끌고 오면 임포트 순서를 타기 시작한다.

환경 변수
    FA_DEVICE          'cuda:0' | 'cpu'    (기본: 자동)
    FA_IMGSZ           검출기 기본 해상도  (기본: 1280)
    FA_JOBS_DIR        작업 디렉터리       (기본: ./jobs)
    FA_MAX_UPLOAD_MB   업로드 상한         (기본: 2048)
    FA_JOB_TTL_MIN     완료 후 자동 삭제   (기본: 120, 0이면 안 지움)
    FA_FAILED_TTL_MIN  실패 보관           (기본: 0 = 안 지움)
    FA_SWEEP_SEC       정리 주기           (기본: 300)
    FA_PRELOAD         기동 시 모델 로드   (기본: 1)
    FA_QUEUE_MAX       대기열 개수 상한    (기본: 0 = 무제한)
    FA_BATCH_MAX       한 번에 넣을 개수   (기본: 0 = 무제한)
    FA_MIN_FREE_MB     최소 여유 디스크    (기본: 2048, 미달이면 507)
    FA_LIST_LIMIT      목록 기본 개수      (기본: 100)
    FA_MAX_ATTEMPTS    일시적 오류 재시도  (기본: 3)

처리 파라미터 기본값 (JOB_DEFAULTS)
    FA_METHOD mosaic · FA_CONF 0.25 · FA_BATCH_SIZE 32 · FA_PAD 0.15
    FA_MOSAIC_SCALE 0.06 · FA_LINGER 5 · FA_INTERP 1 · FA_KEEP_AUDIO 1
    FA_CRF 23 · FA_BITRATE_RATIO 1.0
    FA_OUTPUT_HEIGHT 720 (0=원본 유지) · FA_TARGET_BITRATE 3500k
    FA_MAX_BITRATE 4000k  (납품 대역 720p / 3000~4000 kbps)

S3 설정은 storage/s3.py 를 참고 (FA_S3_BUCKET 등).
"""

import os

from ..core.pipeline import (
    DEFAULT_BITRATE_RATIO,
    DEFAULT_CRF,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_BITRATE,
    DEFAULT_TARGET_BITRATE,
    VideoOpenError,
    VideoWriteError,
)

DEVICE = os.environ.get("FA_DEVICE") or None
IMGSZ = int(os.environ.get("FA_IMGSZ", 1280))
JOBS_DIR = os.path.abspath(os.environ.get("FA_JOBS_DIR", "jobs"))
MAX_BYTES = int(os.environ.get("FA_MAX_UPLOAD_MB", 2048)) * 1024 * 1024
JOB_TTL = int(os.environ.get("FA_JOB_TTL_MIN", 120)) * 60
SWEEP_SEC = int(os.environ.get("FA_SWEEP_SEC", 300))
PRELOAD = os.environ.get("FA_PRELOAD", "1") not in ("0", "false", "False")
RETRY_AFTER = int(os.environ.get("FA_RETRY_AFTER", 30))
# 대기열은 기본적으로 개수로 제한하지 않는다. 전체 수행처럼 한꺼번에 수백 건을
# 넣는 사용이 정상이고, 개수는 애초에 잘못된 기준이다 — 10건이 50MB 짜리면
# 아무것도 아니고 2GB 짜리면 이미 위험하다. 진짜 제약은 디스크다(MIN_FREE_MB).
QUEUE_MAX = int(os.environ.get("FA_QUEUE_MAX", 0))          # 0 = 무제한
# 한 번에 넣을 개수도 막지 않는다. 폴더 하나에 수천 건이 들어 있는 게 정상이고,
# 상한에 걸리면 사용자가 폴더를 손으로 쪼개야 한다 — 그게 훨씬 나쁘다.
# S3 입력은 대기 중에 디스크를 쓰지 않는다(내려받기는 _run 에서 한다). 그래서
# 대기열이 길어도 드는 건 작업 디렉터리와 job.json 뿐이다.
BATCH_MAX = int(os.environ.get("FA_BATCH_MAX", 0))          # 0 = 무제한
# 실패/취소 작업은 기본적으로 지우지 않는다. 배치로 수백 건 돌린 뒤 몇 건이
# 실패했을 때, 입력과 사유가 남아 있어야 원인을 볼 수 있다.
FAILED_TTL = int(os.environ.get("FA_FAILED_TTL_MIN", 0)) * 60
MIN_FREE_MB = int(os.environ.get("FA_MIN_FREE_MB", 2048))   # 0 = 검사 안 함
LIST_LIMIT = int(os.environ.get("FA_LIST_LIMIT", 100))
MAX_ATTEMPTS = int(os.environ.get("FA_MAX_ATTEMPTS", 3))

# 다시 시도해도 결과가 같은 오류들. 깨진 파일이나 잘못된 인자를 세 번 돌리는 건
# 그냥 낭비이고, 그동안 뒤에 쌓인 정상 작업이 밀린다.
PERMANENT_ERRORS = (VideoOpenError, VideoWriteError, ValueError, FileNotFoundError)
STATE_FILE = "job.json"
GPU_LOCK_FILE = ".gpu.lock"
PROGRESS_FLUSH_SEC = 0.5      # 진행률을 디스크에 쓰는 최소 간격

CHUNK = 1 << 20
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _bool_env(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() not in ("0", "false", "no")


# 처리 파라미터 기본값.
#
# **호출하는 쪽은 입력만 주면 된다.** 튜닝된 값은 서비스가 들고 있어야지,
# 호출자마다 들고 다니면 어느 설정으로 처리됐는지가 호출 지점마다 달라진다.
# 운영 중 조정은 환경 변수로 하고, 필요할 때만 요청에서 개별 항목을 덮는다.
#
# imgsz 는 검출기와 같은 값을 쓴다(FA_IMGSZ). 둘이 어긋나면 워밍업한 커널과
# 실제 추론이 달라진다.
JOB_DEFAULTS = {
    "method": os.environ.get("FA_METHOD", "mosaic"),
    "conf": float(os.environ.get("FA_CONF", "0.25")),
    "imgsz": IMGSZ,
    "batch_size": int(os.environ.get("FA_BATCH_SIZE", "32")),
    "pad": float(os.environ.get("FA_PAD", "0.15")),
    "mosaic_scale": float(os.environ.get("FA_MOSAIC_SCALE", "0.06")),
    "linger": int(os.environ.get("FA_LINGER", "5")),
    "interp": _bool_env("FA_INTERP", True),
    "keep_audio": _bool_env("FA_KEEP_AUDIO", True),
    "crf": DEFAULT_CRF,
    "bitrate_ratio": DEFAULT_BITRATE_RATIO,
    # 납품 스펙. 값만 바꾸면 되도록 열어 둔다 — 기준이 바뀔 때 코드를 고치게
    # 하면 안 된다. height=0 이면 원본 유지, bitrate="" 면 예전 CRF 방식.
    "height": DEFAULT_HEIGHT,
    "bitrate": DEFAULT_TARGET_BITRATE,
    "max_bitrate": DEFAULT_MAX_BITRATE,
}


# 다시 시도해도 결과가 같은 오류들. 깨진 파일이나 잘못된 인자를 세 번 돌리는 건
# 그냥 낭비이고, 그동안 뒤에 쌓인 정상 작업이 밀린다.
PERMANENT_ERRORS = (VideoOpenError, VideoWriteError, ValueError, FileNotFoundError)
STATE_FILE = "job.json"
GPU_LOCK_FILE = ".gpu.lock"
PROGRESS_FLUSH_SEC = 0.5      # 진행률을 디스크에 쓰는 최소 간격
CHUNK = 1 << 20
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
