"""큐 껍데기 설정 — 전부 환경 변수. 이 파일은 아무것도 임포트하지 않는다.

**이름이 왜 죄다 바꿀 수 있게 되어 있나.** 태스크 이름과 큐 이름의 주인은
우리가 아니라 **잡을 넣는 쪽**이다. 저쪽이 `send_task("...deidentify_one")` 로
보내면 우리는 정확히 그 이름으로 등록되어 있어야 받는다. 한 글자만 달라도
메시지는 큐에 남아 아무도 안 먹는다 — 오류도 안 난다.

아직 저쪽에 이 태스크들이 없으므로 아래 값은 **제안**이다. 저쪽이 이름을 정하면
코드를 고치지 않고 환경 변수만 맞춘다.
"""

import os

# 브로커 = 저쪽 Redis. 컨테이너가 필요로 하는 설정은 사실상 이것 하나다.
# REDIS_URL 도 보는 이유: 저쪽 compose·k8s 가 그 이름을 쓴다(x-app-env).
BROKER_URL = (os.environ.get("FA_BROKER_URL")
              or os.environ.get("REDIS_URL")
              or "redis://localhost:6379/0")

# 우리가 구독하는 큐. 큐 = 스케일 단위이므로 GPU 워크로드는 자기 큐를 가져야 한다.
QUEUE = os.environ.get("FA_MSA_QUEUE", "q.deidentify")

# 우리가 등록할 태스크 이름 = 저쪽이 보낼 이름.
TASK_NAME = os.environ.get("FA_MSA_TASK", "worker_io.tasks.deidentify_one")

# 되돌려 보낼 곳. 저쪽에서 DB 를 만지는 것은 이 둘이고, 가벼워서 default 큐다.
CALLBACK_QUEUE = os.environ.get("FA_MSA_CALLBACK_QUEUE", "default")
HEARTBEAT_TASK = os.environ.get("FA_MSA_HEARTBEAT_TASK",
                                "worker_io.tasks.deidentify_heartbeat")
COMPLETE_TASK = os.environ.get("FA_MSA_COMPLETE_TASK",
                               "worker_io.tasks.deidentify_complete")

# GPU 한 장에 검출기 하나. 이 값을 올릴 이유가 생기면 그건 GPU 를 늘렸다는 뜻이고,
# 그때는 컨테이너를 늘리는 게 맞다(KEDA 가 큐 깊이로 그렇게 한다).
CONCURRENCY = int(os.environ.get("FA_MSA_CONCURRENCY", 1))

# 기동할 때 모델을 미리 올릴까. 첫 잡이 로딩 수십 초를 뒤집어쓰지 않게 한다.
PRELOAD = (os.environ.get("FA_MSA_PRELOAD", "1").strip().lower()
           not in ("0", "false", "no", "off"))
