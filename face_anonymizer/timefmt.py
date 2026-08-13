"""시각 표기 — 로그와 화면이 같은 문장을 쓰게 한다.

**서버는 UTC 로 돌고 사람은 한국 시각으로 읽는다.** 컨테이너의 타임존은 배포마다
다르고(대개 UTC), 거기 맞춰 로그를 남기면 "8월 13일 01시" 가 무슨 시각인지
읽는 사람이 매번 9시간을 더해야 한다. 그래서 표기는 여기서 한 번만 정한다.

``zoneinfo`` 를 쓰지 않는다. tzdata 가 없는 슬림 이미지에서 `ZoneInfo("Asia/Seoul")`
는 예외를 던지는데, **로그를 남기려다 작업이 죽는 것**만큼 나쁜 일이 없다. 한국은
서머타임이 없어 고정 +9 로 충분하다.
"""

import datetime as _dt
import os

# 표기용 오프셋. 다른 지역에서 운영하면 이것만 바꾼다.
OFFSET_HOURS = float(os.environ.get("FA_TZ_OFFSET", 9))
LABEL = os.environ.get("FA_TZ_LABEL", "KST")

TZ = _dt.timezone(_dt.timedelta(hours=OFFSET_HOURS))


def stamp(epoch=None):
    """``8월 13일 01:04:37`` — 사람이 읽는 짧은 표기."""
    if not epoch:
        return "—"
    t = _dt.datetime.fromtimestamp(epoch, TZ)
    return f"{t.month}월 {t.day}일 {t:%H:%M:%S}"


def iso(epoch=None):
    """``2026-08-13T01:04:37+09:00`` — 기계가 읽는 표기. 로그 집계용."""
    if not epoch:
        return None
    return _dt.datetime.fromtimestamp(epoch, TZ).isoformat(timespec="seconds")


def span(start, end):
    """``8월 13일 01:04:37 ~ 01:05:26 (49초)``.

    같은 날이면 끝 시각의 날짜를 생략한다 — 한 편이 몇 분이라 날짜를 두 번 적으면
    읽는 눈이 오히려 흐려진다. 날이 바뀌면 그때는 적는다.
    """
    if not start:
        return "—"
    if not end:
        return f"{stamp(start)} ~ (진행 중)"
    a = _dt.datetime.fromtimestamp(start, TZ)
    b = _dt.datetime.fromtimestamp(end, TZ)
    tail = f"{b:%H:%M:%S}" if a.date() == b.date() else stamp(end)
    return f"{stamp(start)} ~ {tail} ({dur(end - start)})"


def dur(seconds):
    """``49초`` · ``4분 5초`` · ``4시간 5분``."""
    s = int(max(0, seconds or 0))
    if s < 60:
        return f"{s}초"
    if s < 3600:
        return f"{s // 60}분 {s % 60}초"
    return f"{s // 3600}시간 {(s % 3600) // 60}분"
