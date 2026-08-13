"""로깅 설정 — **한 번만, 진입점에서.**

이게 없으면 우리 `log.info` 는 **전부 버려진다.** uvicorn 은 자기 로거만 구성하고
루트는 건드리지 않아서, 우리 모듈의 유효 수준이 WARNING 이 된다(실측: root
handlers=[], level=30). 그러면 `▶ 시작` / `■ 완료` 같은 것이 서버 모드에서 한 줄도
안 찍힌다. celery 는 루트를 스스로 설정해서 보였고, 그래서 더 오래 몰랐다.

라이브러리 코드는 절대 로깅을 설정하지 않는다(각 모듈은 `getLogger(__name__)` 만
쓴다). 설정은 **프로세스를 띄우는 쪽**의 일이라 여기 모아 둔다.

시각은 한국 기준으로 찍는다. 컨테이너 타임존은 대개 UTC 인데, 로그를 읽는 사람이
매번 9시간을 더하게 만들 이유가 없다(timefmt 참고).
"""

import logging
import logging.handlers
import os

from . import timefmt

LEVEL = os.environ.get("FA_LOG_LEVEL", "INFO").upper()
FILE = os.environ.get("FA_LOG_FILE") or ""
KEEP_DAYS = int(os.environ.get("FA_LOG_KEEP_DAYS", 14))

_done = False


class KstFormatter(logging.Formatter):
    """``2026-08-13 01:04:37 KST  INFO   worker    ▶ 시작 …``"""

    def format(self, record):
        when = timefmt.stamp(record.created)
        name = record.name.rsplit(".", 1)[-1]
        head = f"{when} {timefmt.LABEL}  {record.levelname:<7} {name:<10}"
        text = record.getMessage()
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return f"{head} {text}"


def setup(level=None, force=False):
    """루트 로거를 구성한다. 두 번 불러도 한 번만 먹는다.

    파일 출력은 선택이다(`FA_LOG_FILE`). 컨테이너에서는 stdout 만 있으면 되고,
    EC2 에 직접 띄울 때는 파일이 편하다 — 날짜별로 돌리고 오래된 것은 지운다.
    """
    global _done
    if _done and not force:
        return
    root = logging.getLogger()
    root.setLevel(getattr(logging, level or LEVEL, logging.INFO))

    fmt = KstFormatter()
    have = {type(h).__name__ for h in root.handlers}
    if "StreamHandler" not in have:
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        root.addHandler(h)
    else:
        for h in root.handlers:
            h.setFormatter(fmt)

    if FILE and "TimedRotatingFileHandler" not in have:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(FILE)) or ".", exist_ok=True)
            fh = logging.handlers.TimedRotatingFileHandler(
                FILE, when="midnight", backupCount=KEEP_DAYS, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError as e:                        # 파일에 못 쓴다고 안 뜨면 안 된다
            root.warning("로그 파일을 열지 못했다 (%s) — 화면에만 남긴다", e)

    # 우리 것만 INFO 로 보고 싶을 때가 있다. 남의 라이브러리 INFO 는 시끄럽다.
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _done = True
