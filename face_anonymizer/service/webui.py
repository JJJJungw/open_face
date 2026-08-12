"""웹 UI 로더.

화면은 ``static/index.html`` 파일 하나다. 빌드 도구도 CDN 의존도 없다 —
HTML·CSS·JS 가 한 파일에 들어 있고 서버는 그걸 그대로 내보낸다.

전에는 이 내용을 파이썬 문자열 상수로 들고 있었다. 파일을 못 찾는 사고를
막으려던 것인데, 대가가 컸다. 900줄짜리 .py 안에 HTML 이 갇혀 있어서 편집기
문법 강조가 안 되고, 한 글자만 고쳐도 파이썬 파일 diff 로 잡혔다. 파일로
빼고, 못 찾는 경우는 기동 시점에 바로 드러나게 한다 — 임포트에서 터지면
서버가 아예 안 뜨므로 조용히 잘못될 여지가 없다.
"""

import os

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")

with open(INDEX_PATH, encoding="utf-8") as _fh:
    INDEX_HTML = _fh.read()
