# 돌리는 법

루트 README 는 **API 서버로 쓰는 법**만 적는다(그게 이 물건의 목적이라서).
도커 없이 돌리거나, 명령줄로 한 편만 해 보거나, 웹 화면으로 쓰는 법은 여기 있다.

## 깔기

**파이썬 3.11 이다**(`.python-version`). 3.12 이상은 막아 두었다 — 붙을 곳이
3.11 로 묶여 있고, 개발과 배포가 다른 파이썬에서 도는 것이 실제로 우리를 물었다
([issues/026](issues/026-two-tests-were-passing-for-the-wrong-reason.md)).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements/base.txt

# ffmpeg 는 시스템 패키지 (오디오 합성·코덱 정규화)
#   Ubuntu/Debian: sudo apt install ffmpeg   |   macOS: brew install ffmpeg

python scripts/setup_weights.py     # 검출기 리포 + 가중치 (한 번만)
```

가중치는 **S3 → 공개 릴리스** 순으로 받는다. 사내망처럼 바깥에 못 닿으면
`FA_WEIGHTS_URL` 에 자기 미러 주소를 넣는다 —
[`storage/README`](../face_anonymizer/storage/README.md).

## 명령줄로 한 편

서버도 클라우드도 없이 파일 하나를 넣어 파일 하나를 받는다.

```bash
face-anonymize input.mp4                              # 기본 (모자이크)
face-anonymize input.mp4 --method box                 # 완전히 가리기
face-anonymize input.mp4 --imgsz 1600 --conf 0.20     # 작은 얼굴 대응
```

> **블러는 복원될 수 있다.** 약한 블러는 디블러링 모델로 부분 복원될 여지가
> 있다. 강한 비식별화가 필요하면 `mosaic`(강하게) 또는 `box` 를 쓴다.

## 웹 화면으로 여러 편

사람이 버킷을 훑어 폴더째 제출하고 진행률·검수·로그를 화면에서 본다. **이쪽은
우리가 저장소에 직접 붙으므로 자격 증명이 필요하다** — API 로 부르는 경로와
다른 점이 그것 하나다.

```bash
pip install -r requirements/base.txt -r requirements/serve.txt
uvicorn face_anonymizer.service.server:app --host 127.0.0.1 --port 8000
```

처음 열면 **어디에 붙을지부터 묻는다**(AWS S3 · NCP · S3 호환). 제공자를 고르면
그 제공자의 자격 증명 안내가 나온다 —
[`service/README`](../face_anonymizer/service/README.md).

## 도커 없이 API 서버만

같은 서버다. 컨테이너는 포장일 뿐이라, 위 명령으로 띄워도 `/api/deident/jobs`
는 그대로 열린다. 저장소 설정을 안 해도 된다(서명된 URL 만 쓰므로).

## 설정

조절할 수 있는 값은 전부 환경 변수다. 띄울 때마다 `export` 를 치는 대신 `.env`
를 둔다.

```bash
cp .env.example .env      # 필요한 줄만 고친다
```

규칙은 둘이다. **실제 환경 변수가 파일보다 우선**하고(한 번만 다르게 돌려
보려면 `FA_CRF=19 uvicorn ...` 처럼 앞에 붙인다), **파일이 없어도 전부 기본값으로
돈다.** `.env` 는 커밋하지 않는다 — 버킷 이름 같은 실제 값은 거기만 둔다.

목록과 기본값은 [`.env.example`](../.env.example) 에 주석으로 다 적혀 있다.
코드가 읽는 값과 그 파일이 어긋나면 테스트가 실패한다.

## 테스트

```bash
pip install -r requirements/dev.txt
pytest
```

가중치도 torch 도 GPU 도 없이 2분 안에 돈다 — 가짜 검출기를 주입하기 때문이다.
[`tests/README`](../tests/README.md).

## 폴더

```
face_anonymizer/   패키지 — 코어 · 서비스 · 저장소 · 큐 워커
scripts/           준비 스크립트 (가중치 내려받기, EC2 세팅)
requirements/      의존성 — base · serve · worker · dev
tests/             회귀 테스트
tools/             손으로 돌리는 도구 (큐 왕복 검증)
docs/              문제와 해결 기록 · 연동 규약 · 보안 재고
weights/           가중치 (커밋하지 않는다)
third_party/       YOLO-FaceV2 리포 (체크포인트 unpickle 에 필요)
```

## 세 가지 문, 하나의 러너

이 서비스는 일을 세 방식으로 받는다. **뒤의 둘은 같은 러너**
(`job_runner.run_job`)로 합류한다 — 계약은 페이로드고 전송은 선택이다.

| 문 | 어떻게 | 자격 증명 |
|---|---|---|
| 웹 화면 | 사람이 버킷을 훑어 제출 | **필요하다** (우리가 붙는다) |
| HTTP | 남의 시스템이 잡을 POST | 없다 (서명된 URL) |
| 큐 (MSA) | 우리가 큐에서 꺼내 온다 | 없다 (서명된 URL) |

처리 기본값도 한 벌을 나눠 쓴다. 두 벌로 뒀더니 큐 경로가 조용히 다른 설정으로
돌고 있었다([issues/009](issues/009-queue-path-ran-untuned.md)).
