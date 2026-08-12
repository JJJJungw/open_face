"""`.env` 파일 읽기.

파이썬은 `.env` 를 알아서 읽지 않는다. 그래서 지금까지는 서버를 띄울 때마다
`export FA_S3_BUCKET=... FA_S3_REGION=...` 를 손으로 쳐야 했고, 창을 새로 열면
다시 쳐야 했고, 무엇을 조절할 수 있는지는 코드를 뒤져야 알 수 있었다.

의존성을 하나 더 들이지 않으려고 직접 읽는다. 필요한 문법이 `KEY=VALUE` 와
주석뿐이라 스무 줄이면 끝난다.

**규칙 두 가지만 기억하면 된다.**

1. **실제 환경 변수가 이긴다.** 파일은 비어 있는 자리만 채운다. 반대로 하면
   `export FA_CRF=19` 로 한 번 돌려 보려는데 파일이 조용히 덮어써서, 왜 안
   바뀌는지 한참 찾게 된다.
2. **파일이 없어도 된다.** 없으면 그냥 넘어간다. 배포에서는 파일 대신
   컨테이너 환경 변수나 systemd EnvironmentFile 을 쓰는 게 정상이다.

찾는 순서는 ``FA_ENV_FILE`` → 현재 디렉터리 → 리포 루트다.
"""

import os

ENV_NAME = ".env"


def parse(text):
    """`.env` 본문 -> dict. 파싱할 수 없는 줄은 조용히 건너뛴다."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):          # 셸에서 source 해도 되게
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]                 # 따옴표는 벗긴다
        out[key] = value
    return out


def find():
    """읽을 `.env` 경로. 없으면 None."""
    explicit = os.environ.get("FA_ENV_FILE")
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (os.path.join(os.getcwd(), ENV_NAME), os.path.join(here, ENV_NAME)):
        if os.path.isfile(path):
            return path
    return None


def load(path=None, override=False):
    """`.env` 를 os.environ 에 채운다. 채운 키 목록을 돌려준다.

    ``override=False`` 가 기본이다 — 이미 있는 환경 변수는 건드리지 않는다.
    """
    path = path or find()
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        pairs = parse(fh.read())
    applied = []
    for key, value in pairs.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    return applied
