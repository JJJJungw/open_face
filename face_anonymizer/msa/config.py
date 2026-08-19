"""큐 껍데기 설정 — 전부 환경 변수. 이 파일은 아무것도 임포트하지 않는다.

**이름이 왜 죄다 바꿀 수 있게 되어 있나.** 태스크 이름과 큐 이름의 주인은
우리가 아니라 **잡을 넣는 쪽**이다. 저쪽이 `send_task("...deidentify_one")` 로
보내면 우리는 정확히 그 이름으로 등록되어 있어야 받는다. 한 글자만 달라도
메시지는 큐에 남아 아무도 안 먹는다 — 오류도 안 난다.

아직 저쪽에 이 태스크들이 없으므로 아래 값은 **제안**이다. 저쪽이 이름을 정하면
코드를 고치지 않고 환경 변수만 맞춘다.
"""

import os

from ..env import flag

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

# 브로커가 "이 메시지는 아직 처리 중" 으로 봐 주는 시간. **넘기면 다른 워커에게
# 다시 배달된다.**
#
# Redis 브로커에는 ack 라는 개념이 없다. acks_late 를 켜도 kombu 가 이 시간으로
# 흉내 낼 뿐이라, 컨테이너가 통째로 죽었을 때 메시지가 되살아나는 시점이 곧
# 이 값이다(kombu 기본 3600초). 반대로 이 값이 **한 건 처리 시간보다 짧으면**
# 멀쩡히 돌고 있는 작업이 중복 배달된다 — GPU 를 두 배로 쓴다.
#
# 그래서 길게 둔다. 진짜 회수는 저쪽 리스 만료 스윕(5분)이 하고, 늦게 도착한
# 우리 보고는 펜싱 토큰이 거른다. 기본값에 기대지 않고 여기 적어 두는 이유는,
# 이 숫자가 "죽었을 때 얼마나 빨리 되살아나나" 를 정하는 값이기 때문이다.
VISIBILITY_TIMEOUT = int(os.environ.get("FA_MSA_VISIBILITY_TIMEOUT", 3600))

# 기동할 때 모델을 미리 올릴까. 첫 잡이 로딩 수십 초를 뒤집어쓰지 않게 한다.
PRELOAD = flag("FA_MSA_PRELOAD", True)
