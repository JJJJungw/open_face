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
    FA_KEEP_LOCAL_RESULT  S3 업로드 뒤에도 로컬 사본 유지 (기본: 0)
    FA_RECOVER         기동 시 중단 작업 복구 (기본: 1)
    FA_RETRY_DELAYS    재시도 간격 목록    (기본: 5,30,60 — 95초 창)
    FA_DEFER_SEC       보류 재확인 간격    (기본: 60)
    FA_DEFER_MAX_SEC   보류 상한           (기본: 1800, 넘으면 실패)
    FA_LIST_LIMIT      API 기본 상한       (기본: 100)
    FA_PAGE_SIZE       화면 한 쪽 카드 수  (기본: 5)
    FA_MAX_ATTEMPTS    일시적 오류 재시도  (기본: 4 = 처음 1회 + 재시도 3회)

처리 파라미터 기본값 (JOB_DEFAULTS)
    FA_METHOD mosaic · FA_CONF 0.25 · FA_BATCH_SIZE 32 · FA_PAD 0.15
    FA_MOSAIC_SCALE 0.06 · FA_LINGER 5 · FA_INTERP 1 · FA_KEEP_AUDIO 1
    FA_CRF 23 · FA_BITRATE_RATIO 1.0
    FA_OUTPUT_HEIGHT 720 (0=원본 유지) · FA_TARGET_BITRATE 3500k
    FA_MAX_BITRATE 4000k  (납품 대역 720p / 3000~4000 kbps)

S3 설정은 storage/s3.py 를 참고 (FA_S3_BUCKET 등).
"""

import os

from .. import params
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
IMGSZ = params.IMGSZ          # 단일 출처는 face_anonymizer/params.py
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
# 한 페이지에 몇 건을 그릴까. 목록은 페이지로 넘긴다(docs/issues/006).
# 실측으로 100건을 그리는 데 9.5ms — 폴링 예산(0.7초)의 1% 다. 300건이 4%,
# 1000건이 26%, 3000건이면 75% 라 화면이 눈에 띄게 버벅인다. 상한을 없애는 대신
# 페이지를 나누면 폴더가 몇천 건이어도 한 페이지 값만 낸다.
LIST_LIMIT = int(os.environ.get("FA_LIST_LIMIT", 100))
# 화면 한 쪽에 그리는 카드 수. API 기본 상한(LIST_LIMIT)과 다른 값인 이유는
# 둘이 다른 질문에 답하기 때문이다. LIST_LIMIT 은 "한 번에 얼마나 내줄 수
# 있나"(스크립트·오케스트레이터용), PAGE_SIZE 는 "사람이 한 화면에서 읽을
# 만한 양이 얼마인가" 다. 카드는 크고, 큐 전체 모양은 카드가 아니라 큐 UI 가
# 보여 준다.
PAGE_SIZE = int(os.environ.get("FA_PAGE_SIZE", 5))
# 처음 1회 + 재시도 3회. RETRY_DELAYS 와 짝이다.
MAX_ATTEMPTS = int(os.environ.get("FA_MAX_ATTEMPTS", 4))

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
# S3 작업은 결과를 버킷에 올린 뒤 로컬 사본을 바로 지운다. 다운로드 라우트가
# 이미 "로컬에 없으면 S3 로 302" 이므로 들고 있을 이유가 없고, 안 지우면
# 대량 처리에서 디스크가 먼저 찬다(docs/issues/001). 1 이면 예전처럼 남긴다.
KEEP_LOCAL = _bool_env("FA_KEEP_LOCAL_RESULT", False)
# 기동 시 중단된 작업 정리. --workers N 으로 띄울 때는 한 프로세스만 켜 둔다 —
# 여럿이 켜면 각자 같은 작업을 재큐해 중복 처리한다.
RECOVER = _bool_env("FA_RECOVER", True)

# 재시도 간격. 세 번을 같은 순간에 시도하면 세 번 다 같은 세상을 본다 —
# 시도는 했는데 기다리지는 않은 것이다(docs/issues/003).
#
#   5초 -> 30초 -> 60초.  처음 1회 + 재시도 3회로 95초짜리 창을 덮는다.
#
# 공식(base x factor^n) 대신 **목록**으로 둔다. 재시도가 서너 번뿐이라 지수의
# 이점(적은 시도로 자릿수를 덮는 것)이 거의 없고, 목록은 보는 순간 동작이
# 보인다. 시도가 목록보다 많으면 마지막 값을 계속 쓴다.
#
# 첫 간격이 5초인 것은 S3 순단이 대부분 여기서 끝나기 때문이다 — 흔한 경우를
# 빨리 통과시킨다. 창이 95초로 긴 것은 거의 순수한 이득이다. 대기가 큐를
# 막지 않으므로(예약 방식) 처리량 손해가 없고, 그 시간을 쓰는 것은 정말
# 일시적일 수 있는 오류뿐이다 — 권한 없음이나 키 오타는 애초에 재시도 대상이
# 아니라 즉시 실패한다.
RETRY_DELAYS = tuple(
    float(x) for x in os.environ.get("FA_RETRY_DELAYS", "5,30,60").split(",") if x.strip()
) or (5.0,)
RETRY_JITTER = float(os.environ.get("FA_RETRY_JITTER", 0.2))   # ±20%

# 보류는 재시도와 다르다. 실패가 아니라 "아직 시작할 조건이 안 됐다" 이므로
# 시도 횟수를 쓰지 않는다 — 디스크를 세 번 확인했다고 포기할 일이 아니다.
# 대신 상한을 둔다. 영구히 찬 디스크를 영원히 숨기지 않기 위해서다.
DEFER_SEC = float(os.environ.get("FA_DEFER_SEC", 60))
DEFER_MAX_SEC = float(os.environ.get("FA_DEFER_MAX_SEC", 1800))


# 처리 파라미터는 **이 파일이 소유하지 않는다.** 큐 워커(msa/)도 같은 값을
# 써야 하는데, 두 벌로 두면 언젠가 어긋난다 — 실제로 어긋나서 큐 경로가
# batch_size=1 로 돌고 있었다(docs/issues/009).
JOB_DEFAULTS = dict(params.DEFAULTS)


# 다시 시도해도 결과가 같은 오류들. 깨진 파일이나 잘못된 인자를 세 번 돌리는 건
# 그냥 낭비이고, 그동안 뒤에 쌓인 정상 작업이 밀린다.
PERMANENT_ERRORS = (VideoOpenError, VideoWriteError, ValueError, FileNotFoundError)
STATE_FILE = "job.json"
GPU_LOCK_FILE = ".gpu.lock"
PROGRESS_FLUSH_SEC = 0.5      # 진행률을 디스크에 쓰는 최소 간격
CHUNK = 1 << 20
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
