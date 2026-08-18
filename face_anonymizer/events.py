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


def files(limit_days=7, from_day=None, to_day=None):
    """날짜 파일들, 최신 순.

    파일 이름이 곧 날짜(``2026-08-18.jsonl``)라 **범위를 파일 단위로 자른다** —
    범위 밖 파일은 열지도 않는다.

    기본은 최근 7일이다. 화면은 그 정도만 보면 되고, 더 오래된 것을 뒤지려고
    매번 전부 여는 것은 낭비다. 다만 **날짜를 지정하면 그 상한을 풀어야 한다** —
    "지난달 것을 받겠다" 는데 7일 창에 걸려 빈 파일이 나오면 안 된다.
    """
    try:
        names = sorted((n for n in os.listdir(DIR) if n.endswith(".jsonl")),
                       reverse=True)
    except OSError:
        return []
    if from_day or to_day:
        lo, hi = str(from_day or "0000-00-00"), str(to_day or "9999-99-99")
        names = [n for n in names if lo <= n[:10] <= hi]
    else:
        names = names[:limit_days]
    return [os.path.join(DIR, n) for n in names]


# 뒤에서부터 읽을 때 한 번에 가져오는 크기. 저널 한 줄이 대략 300~500바이트라
# 64KB 면 150줄쯤 된다 — 화면 한 쪽(60줄)을 대개 한 번에 채운다.
_TAIL_CHUNK = 64 * 1024


def tail_lines(path):
    """파일을 **끝에서부터** 한 줄씩 내놓는다.

    예전에는 ``readlines()`` 로 통째로 올렸다. 최신 60줄을 보려고 하루치를 다
    메모리에 얹는 셈인데, 900건짜리를 돌리면 하루에 수천 줄이 쌓이고 그게 폴링
    때마다 반복된다. 뒤에서 필요한 만큼만 읽으면 파일이 아무리 커도 비용이 같다.

    UTF-8 은 여러 바이트짜리 글자가 있어서 아무 데나 자르면 안 된다. 줄바꿈
    경계에서만 자르고, 맨 앞의 반쪽 줄은 다음 덩이와 이어 붙인다.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            rest = b""
            while pos > 0:
                size = min(_TAIL_CHUNK, pos)
                pos -= size
                f.seek(pos)
                chunk = f.read(size) + rest
                parts = chunk.split(b"\n")
                rest = parts.pop(0)          # 앞쪽 반쪽 줄은 다음 덩이가 채운다
                for raw in reversed(parts):
                    if raw.strip():
                        yield raw
            if rest.strip():
                yield rest
    except OSError:
        return


# 목록이 실제로 그리는 값들. **상세는 펼칠 때 따로 가져온다.**
#
# 예전에는 저널 줄을 통째로 내려보냈다. 단계별 소요(timing)나 경고 원문처럼
# 펼쳐야 보이는 것까지 60줄에 다 붙어 오는 셈이라, 목록 한 번에 몇 배가 실렸다.
LIST_FIELDS = ("at", "ts", "mode", "event", "job", "name", "batch",
               "seconds", "elapsed_s", "frames", "detection_rate",
               "review_needed", "transcoded", "source_codec", "stage",
               "transient", "detail", "attempts", "percent", "eta_s",
               "action", "note", "codes", "done", "failed", "avg_elapsed_s",
               "cold_s", "queue")


def read(job=None, batch=None, event=None, since=None, limit=200,
         mode=None, before=None, q=None, fields=None,
         from_day=None, to_day=None):
    """저널을 뒤에서부터 읽어 조건에 맞는 것만. 최신 순.

    파일을 통째로 파싱하지 않는다 — 뒤에서부터 필요한 만큼만 읽는다(tail_lines).
    하루치가 수만 줄이 되어도 응답이 일정하다.

    ``fields`` 를 주면 그 키만 남긴다. 목록은 상세까지 필요 없다.

    ``before`` 는 '더 보기' 용이다. 받은 마지막 줄의 ``ts`` 를 그대로 넣으면 그
    아래부터 이어 읽는다. ``offset`` 을 쓰지 않는 이유는, 읽는 사이에도 줄이
    계속 쌓여서 offset 기준이 밀리기 때문이다 — 같은 줄을 두 번 보거나 건너뛴다.
    """
    limit = max(1, min(int(limit or 200), READ_MAX))
    needle = (q or "").strip().lower()
    keep = set(fields) if fields else None
    # 날짜 범위는 **파일 단위로 먼저** 자른다. 범위 밖 파일은 열지도 않는다.
    day_since, day_before = timefmt.day_range(from_day, to_day)
    if day_since and (since is None or day_since > float(since)):
        since = day_since
    if day_before and (before is None or day_before < float(before)):
        before = day_before
    out = []
    for path in files(from_day=from_day, to_day=to_day):
        for line in tail_lines(path):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue                     # 반쪽 줄(쓰다 만 것)은 건너뛴다
            if job and row.get("job") != job:
                continue
            if batch and row.get("batch") != batch:
                continue
            if event and not str(row.get("event", "")).startswith(event):
                continue
            if mode and row.get("mode") != mode:
                continue
            if since and (row.get("ts") or 0) < float(since):
                continue
            if before and (row.get("ts") or 0) >= float(before):
                continue
            if needle and needle not in " ".join(
                    str(row.get(k) or "") for k in ("name", "batch", "job")).lower():
                continue
            out.append({k: v for k, v in row.items() if k in keep}
                       if keep else row)
            if len(out) >= limit:
                return out
    return out


def detail_of(job=None, ts=None, event=None):
    """줄 하나를 **원본 그대로** 찾는다. 펼쳤을 때 쓴다.

    저널 줄에는 id 가 없다. 대신 (시각, 사건, 작업) 셋이면 사실상 유일하다 —
    같은 작업의 같은 사건이 같은 밀리초에 두 번 일어나지 않는다. id 를 새로
    넣지 않는 이유는 **이미 쌓인 파일에는 그게 없기** 때문이다.
    """
    if ts is None:
        return None
    target = round(float(ts), 3)
    for path in files():
        for line in tail_lines(path):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if round(float(row.get("ts") or 0), 3) != target:
                continue
            if job and row.get("job") != job:
                continue
            if event and row.get("event") != event:
                continue
            return row
    return None


def days():
    """저널이 있는 날짜들, 최신 순. 날짜 고르기 화면이 쓴다."""
    try:
        return sorted((n[:10] for n in os.listdir(DIR) if n.endswith(".jsonl")),
                      reverse=True)
    except OSError:
        return []


def batches(limit=None):
    """저널에 나타난 폴더 이름들, 최근에 보인 것이 앞.

    **작업 목록이 아니라 저널에서 뽑는다.** 작업은 TTL 로 정리되므로 거기서
    뽑으면 어제 돌린 폴더가 필터 목록에서 사라진다 — 정작 저널에는 그 줄들이
    그대로 남아 있는데 걸러 볼 방법이 없어진다.
    """
    seen = []
    for row in read(limit=limit or READ_MAX):
        b = row.get("batch")
        if b and b not in seen:
            seen.append(b)
    return seen


# ---------------------------------------------------------------------------
# 읽는 쪽 — 줄 하나를 사람이 읽을 한 문장으로.
#
# **문장을 서버가 만든다.** 화면이 만들면 로그 파일과 화면이 다른 말을 하게 되고,
# 나중에 다른 화면이 하나 더 붙으면 같은 계산을 또 짜야 한다. 저널에 넣는 것은
# 수치뿐이고(위쪽 규칙), 그 수치를 문장으로 바꾸는 일은 여기 한 곳에서만 한다.

EVENT_LABEL = {
    "job.queued": "대기 등록", "job.started": "시작", "job.finished": "완료",
    "job.failed": "실패", "job.retry": "재시도", "job.cancelled": "취소",
    "job.review": "검수 대기", "job.reviewed": "검수 판정",
    "job.progress": "진행", "worker.ready": "워커 준비",
    "worker.stopped": "워커 종료", "server.started": "서버 기동",
}

# 화면이 색을 고르는 근거. 이름을 색으로 두지 않은 것은, 색은 화면 사정이고
# 여기서 정하는 것은 '어떤 종류의 소식인가' 이기 때문이다.
EVENT_TONE = {
    "job.finished": "ok", "job.failed": "bad", "job.cancelled": "muted",
    "job.review": "warn", "job.reviewed": "ok",
    "job.retry": "warn", "job.started": "run", "job.progress": "run",
    "job.queued": "muted", "worker.ready": "muted", "worker.stopped": "muted",
}


def _pct(v):
    return f"{float(v) * 100:.1f}%" if isinstance(v, (int, float)) else None


def describe(row):
    """줄 하나 → 짧은 한 문장. 모르는 사건이면 빈 문자열."""
    e = row.get("event") or ""
    bits = []
    if e == "job.finished":
        if row.get("seconds") is not None:
            bits.append(f"{row['seconds']}초")
        elif row.get("elapsed_s") is not None:
            bits.append(f"{row['elapsed_s']}초")
        if row.get("frames"):
            bits.append(f"{row['frames']}프레임")
        rate = _pct(row.get("detection_rate"))
        if rate:
            bits.append(f"검출 {rate}")
        if row.get("review_needed"):
            bits.append("⚠ 검수 필요")
        if row.get("transcoded"):
            bits.append(f"{row.get('source_codec') or '원본'} 전사")
    elif e == "job.failed":
        bits.append(f"[{row.get('stage') or '?'}]")
        bits.append("일시적" if row.get("transient") else "영구")
        if row.get("detail"):
            bits.append(str(row["detail"])[:120])
    elif e == "job.retry":
        bits.append(f"{row.get('attempts') or '?'}회째")
        if row.get("detail"):
            bits.append(str(row["detail"])[:120])
    elif e == "job.review":
        bits.append(", ".join(row.get("codes") or []) or "확인 필요")
    elif e == "job.reviewed":
        bits.append("승인 → 완료" if row.get("action") == "approve"
                    else "반려 → 실패")
        if row.get("note"):
            bits.append(str(row["note"])[:120])
    elif e == "job.progress":
        if row.get("percent") is not None:
            bits.append(f"{row['percent']}%")
        if row.get("stage"):
            bits.append(str(row["stage"]))
        if row.get("eta_s") is not None:
            bits.append(f"남은 {row['eta_s']}초")
    elif e == "job.started":
        if row.get("attempts"):
            bits.append(f"{row['attempts']}회째 시도")
    elif e == "worker.ready":
        if row.get("cold_s") is not None:
            bits.append(f"기동 {row['cold_s']}초")
        if row.get("queue"):
            bits.append(str(row["queue"]))
    elif e == "worker.stopped":
        bits.append(f"성공 {row.get('done', 0)} · 실패 {row.get('failed', 0)}")
        if row.get("avg_elapsed_s"):
            bits.append(f"평균 {row['avg_elapsed_s']}초")
    return " · ".join(b for b in bits if b)


def decorate(row):
    """화면이 바로 그릴 수 있게 라벨·문장·색조를 붙인 사본."""
    out = dict(row)
    out["label"] = EVENT_LABEL.get(row.get("event") or "", row.get("event") or "")
    out["text"] = describe(row)
    out["tone"] = EVENT_TONE.get(row.get("event") or "", "muted")
    out["time"] = (row.get("at") or "")[11:19]
    return out
