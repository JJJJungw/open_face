# msa — 큐를 지켜보는 껍데기

**우리를 호출하는 사람이 없다.** 이 폴더는 인바운드 포트가 없는 워커다. Redis
큐를 지켜보다가 잡을 스스로 꺼내 간다.

`service/` 와 목적이 정반대다. 저쪽은 우리가 서버이고 사람이 웹 화면으로 일을
시킨다. 여기는 우리가 소비자이고 저쪽 시스템(RebornStudio)이 큐에 일을 꽂아
둔다. 둘 다 같은 `core/` 를 쓴다 — 얼굴이 둘이고 몸은 하나다.

## 흐름

```
저쪽 API ──① 잡 넣기──▶ Redis q.deidentify ──② 꺼내기──▶ deidentify_one
   ▲                                                          │
   │                                                     ③ run_job
   └──── default 큐 ◀── ④ 하트비트 · 완료 (토큰 첨부) ─────────┘
```

① 저쪽이 먼저 자기 DB 에서 상태를 `queued` 로 바꾸고, **펜싱 토큰**을 발급하고,
**리스**(시한부 소유권)를 걸고, S3 입출력용 **서명된 URL** 을 만든다. 그걸 잡
페이로드 한 장으로 말아 큐에 넣는다.

② 우리 컨테이너가 꺼낸다. `-c 1 --prefetch-multiplier 1` 이라 한 번에 한 장만.

③ `job_runner.run_job()` 이 내려받기 → 비식별화 → 올리기를 한다. 이 폴더는 영상
처리를 모른다.

④ 60초마다 하트비트로 리스를 연장하고, 끝나면 **받았던 토큰을 그대로 붙여**
완료를 보낸다. 저쪽이 토큰을 대조해서 자기가 들고 있는 것과 같을 때만 인정한다.

## 왜 HTTP 가 아니라 큐인가

**0대까지 줄일 수 있다.** 저쪽 KEDA 는 큐 깊이를 보고 0↔8대를 오간다. HTTP 라면
받을 놈이 항상 있어야 해서 0대가 불가능하고, 로드밸런서·서비스 디스커버리·
헬스체크가 전부 따라붙는다.

**작업이 분 단위다.** HTTP 요청 하나를 몇 분씩 붙들면 게이트웨이 타임아웃에
걸리고, 그 재시도가 곧 중복 처리가 된다.

**인바운드가 없으면 인증도 없다.** 우리 API 에 인증이 없다는 문제가 여기서는
성립하지 않는다.

## 우리가 하지 않는 일

순번 매기기, 재시도 횟수 세기, 백오프 기다리기, 재시작 후 복구, 중복 제출 막기,
S3 자격 증명 들고 있기 — **전부 저쪽 몫이다.** `service/worker.py` 에 있는 그
기능들(issues/002·003)은 우리가 큐를 소유하는 단독 운영에서만 쓴다. 같은 일을 두
곳에서 하면 서로를 방해한다.

우리는 실패를 **일시(transient) / 영구로 1차 분류만** 해서 돌려준다. 몇 번까지
다시 해볼지는 우리가 알 수 있는 정보가 아니다.

그래서 태스크가 **예외를 밖으로 던지지 않는다.** 던지면 celery 가 메시지를 다시
돌리는데, 저쪽도 리스로 같은 판단을 하고 있어서 같은 영상이 몇 배로 돈다.

## 파일

| 파일 | 하는 일 |
|---|---|
| `config.py` | 환경 변수. 아무것도 임포트하지 않는다 |
| `celery_app.py` | 브로커 연결 · 태스크 등록 · 되돌려 보내기 · 모델 예열 |

## 이름은 우리 것이 아니다

태스크 이름과 큐 이름의 주인은 **잡을 넣는 쪽**이다. 저쪽이
`send_task("worker_io.tasks.deidentify_one")` 로 보내면 우리는 정확히 그 이름으로
등록돼 있어야 받는다. 한 글자만 달라도 메시지는 큐에 남아 아무도 안 먹는다 —
**오류도 안 난다.** 그래서 전부 환경 변수로 뺐고, 지금 기본값은 제안일 뿐이다.

## 실행

```bash
export REDIS_URL=redis://<저쪽-redis>:6379/0
celery -A face_anonymizer.msa.celery_app worker \
       -Q q.deidentify -c 1 --prefetch-multiplier 1 -l info
```

컨테이너가 필요로 하는 설정은 사실상 `REDIS_URL` 하나다. 버킷 이름도 자격 증명도
DB 도 없다. 가중치 조달만 아직 미결이다(docs/integration/rebornstudio.md D1).

## 환경 변수

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `FA_BROKER_URL` / `REDIS_URL` | `redis://localhost:6379/0` | 브로커 |
| `FA_MSA_QUEUE` | `q.deidentify` | 구독할 큐 |
| `FA_MSA_TASK` | `worker_io.tasks.deidentify_one` | 등록할 태스크 이름 |
| `FA_MSA_CALLBACK_QUEUE` | `default` | 하트비트·완료를 보낼 큐 |
| `FA_MSA_HEARTBEAT_TASK` | `worker_io.tasks.deidentify_heartbeat` | |
| `FA_MSA_COMPLETE_TASK` | `worker_io.tasks.deidentify_complete` | |
| `FA_MSA_CONCURRENCY` | `1` | GPU 한 장에 검출기 하나 |
| `FA_MSA_PRELOAD` | `1` | 기동 때 모델 예열 |
