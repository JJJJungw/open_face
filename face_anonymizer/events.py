"""이벤트 저널 — **기계가 읽는 기록.** 사람이 읽는 로그와 별개다.

왜 둘로 나누나
--------------
로그는 문장이다. 문구는 읽기 좋게 계속 바뀌고, 바뀌어야 한다. 거기에 파서를
붙이면 **문장을 고칠 때마다 집계가 깨진다.** 그래서 기계가 볼 것은 처음부터
따로 남긴다.

`job.json` 과도 겹치지 않는다. 저쪽은 작업의 **최종 상태**를 들고 있어서
"지금 어떤가" 에 답하지만, "01:04 에 시작해서 01:05 에 한 번 실패하고 01:06 에
다시 시작했다" 같은 **시간 축**은 남지 않는다. 저널은 그 축이다.

무엇에 쓰나
-----------
납품 근거다. "이 파일 언제, 어떤 설정으로, 얼마나 걸려 처리했고, 검출률은
얼마였나" 에 한 줄로 답한다. 나중에 CloudWatch·S3 로 그대로 실어 보낼 수 있다 —
JSONL 은 어디서나 읽힌다.

형식
----
한 줄에 한 사건. 날짜별 파일(``2026-08-13.jsonl``)로 쌓는다.

    {"at":"2026-08-13T01:04:37+09:00","ts":1786...,"event":"job.finished",
     "job":"a1b2c3","name":"K_00297_....mp4","batch":"kbs",
     "seconds":40.7,"frames":1027,"detected_frames":768,"review":[]}

규칙 둘
-------
**절대 예외를 밖으로 내보내지 않는다.** 기록하려다 작업이 죽으면 본말전도다.
**절대 개인정보를 넣지 않는다.** 파일명·키·수치까지다. 서명된 URL 은 서명이
들어 있으므로 넣지 않는다.
"""

import json
import os
import threading
import time

from . import timefmt

DIR = os.environ.get("FA_EVENTS_DIR") or os.path.join(
    os.environ.get("FA_JOBS_DIR", "jobs"), "_events")
ENABLED = (os.environ.get("FA_EVENTS", "1").strip().lower()
           not in ("0", "false", "no", "off"))

# 한 번에 읽어 줄 수 있는 최대. 파일이 아무리 커도 응답이 터지지 않게 한다.
READ_MAX = int(os.environ.get("FA_EVENTS_READ_MAX", 2000))

# 어느 얼굴로 돈 기록인가. 같은 저널에 둘이 섞이므로 줄마다 밝힌다.
#   api  — 우리가 서버. 웹 화면·HTTP 로 들어온 작업
#   msa  — 우리가 소비자. 남의 큐에서 꺼내 온 작업
# 진입점이 자기 값으로 바꾼다(msa/celery_app.py). 안 바꾸면 api 다.
MODE = os.environ.get("FA_MODE") or "api"

# 파일 말고 **stdout 으로도** 같은 줄을 내보낼까.
#
# 두 얼굴의 사는 곳이 달라서 기본값이 다르다. API 는 EC2 에 계속 떠 있으니
# 파일로 쌓아 두면 되고(`/api/events` 로 다시 읽는다), MSA 는 컨테이너다 —
# KEDA 가 큐가 비면 0대로 줄이므로 **파일과 함께 기록이 사라진다.** 컨테이너
# 쪽 기록은 stdout 으로 나가야 CloudWatch·Loki 가 걷어 간다. 그래서 msa
# 진입점이 configure(stdout=True) 로 켠다.
STDOUT = (os.environ.get("FA_EVENTS_STDOUT", "").strip().lower()
          in ("1", "true", "yes", "on"))

_lock = threading.Lock()


def configure(mode=None, stdout=None):
    """진입점이 자기 얼굴을 밝힌다. 환경 변수가 있으면 그쪽이 이긴다."""
    global MODE, STDOUT
    if mode and not os.environ.get("FA_MODE"):
        MODE = mode
    if stdout is not None and not os.environ.get("FA_EVENTS_STDOUT"):
        STDOUT = bool(stdout)

# 서명된 URL 은 서명이 들어 있어 기록에 남기면 안 된다. 실수로 넘겨도 걸러 낸다.
_SECRET = ("url", "input_url", "put_url", "weights_url", "token")


def path_for(epoch=None):
    """그날의 파일. 날짜별로 나눠야 오래된 것을 파일 단위로 버릴 수 있다."""
    t = timefmt.iso(epoch or time.time()) or ""
    return os.path.join(DIR, f"{t[:10] or 'unknown'}.jsonl")


def emit(event, **fields):
    """사건 한 줄. 실패해도 조용히 넘어간다."""
    if not ENABLED:
        return None
    now = time.time()
    row = {"at": timefmt.iso(now), "ts": round(now, 3), "mode": MODE,
           "event": event}
    row.update({k: v for k, v in fields.items()
                if v is not None and k not in _SECRET})
    try:
        line = json.dumps(row, ensure_ascii=False)
        if STDOUT:
            # 파일보다 먼저. 디스크가 차서 아래가 실패해도 이건 나가야 한다 —
            # 컨테이너에서는 이쪽이 유일한 사본이다.
            print(line, flush=True)
        os.makedirs(DIR, exist_ok=True)
        with _lock:
            with open(path_for(now), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except (OSError, TypeError, ValueError):
        # 기록 실패로 작업을 망치지 않는다. 로그에도 안 남긴다 — 디스크가 찬
        # 상황이면 그 로그가 다시 디스크를 쓴다.
        return None
    return row


def files(limit_days=7):
    """최근 날짜 파일들, 최신 순."""
    try:
        names = sorted((n for n in os.listdir(DIR) if n.endswith(".jsonl")),
                       reverse=True)
    except OSError:
        return []
    return [os.path.join(DIR, n) for n in names[:limit_days]]


def read(job=None, batch=None, event=None, since=None, limit=200):
    """저널을 뒤에서부터 읽어 조건에 맞는 것만. 최신 순.

    파일을 통째로 파싱하지 않는다 — 뒤에서부터 필요한 만큼만 읽는다. 하루치가
    수만 줄이 되어도 응답이 일정하다.
    """
    limit = max(1, min(int(limit or 200), READ_MAX))
    out = []
    for path in files():
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue                     # 반쪽 줄(쓰다 만 것)은 건너뛴다
            if job and row.get("job") != job:
                continue
            if batch and row.get("batch") != batch:
                continue
            if event and not str(row.get("event", "")).startswith(event):
                continue
            if since and (row.get("ts") or 0) < float(since):
                continue
            out.append(row)
            if len(out) >= limit:
                return out
    return out
