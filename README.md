# face-anonymizer

영상 속 얼굴을 검출해 **모자이크·블러·박스**로 비식별화한다. 검출은
**YOLO-FaceV2**, 추적은 **ByteTrack**, 끊긴 프레임은 보간으로 메꿔 순간 누출을
막고, 원본 오디오는 유지한다.

```
검출(YOLO-FaceV2) → 추적(ByteTrack) → 트랙 보간 → 익명화 렌더 → 오디오 합성
```

프레임 단위 검출은 어떤 프레임에서 얼굴을 순간적으로 놓칠 수 있고, 비식별화
파이프라인에서 그건 곧 프라이버시 누출이다. 그래서 트랙을 잇고, 관측 사이의 빈
구간을 보간으로 채우고, 마지막 관측 이후에도 몇 프레임 더 가린다.

> **블러는 복원될 수 있다.** 약한 블러는 디블러링 모델로 부분 복원될 여지가
> 있다. 강한 비식별화가 필요하면 `mosaic`(강하게) 또는 `box` 를 쓴다.

---

## 시작하기

**파이썬 3.11 이다**(`.python-version`). 3.12 이상은 막아 두었다 — 붙을 곳이 3.11 로
묶여 있고, 개발과 배포가 다른 파이썬에서 도는 것이 실제로 우리를 물었다
(docs/issues/026).

```bash
git clone <저장소> face-anonymizer && cd face-anonymizer
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements/base.txt

# ffmpeg 는 시스템 패키지 (오디오 합성용)
#   Ubuntu/Debian: sudo apt install ffmpeg   |   macOS: brew install ffmpeg

python scripts/setup_weights.py     # 검출기 리포 + 가중치 (한 번만)
```

**명령줄로 한 편**

```bash
face-anonymize input.mp4                              # 기본 (모자이크)
face-anonymize input.mp4 --method box                 # 완전히 가리기
face-anonymize input.mp4 --imgsz 1600 --conf 0.20     # 작은 얼굴 대응
```

**서버로 여러 편** — 웹 화면에서 버킷을 훑어 폴더째 제출하고 진행률을 본다.

```bash
pip install -r requirements/base.txt -r requirements/serve.txt
uvicorn face_anonymizer.service.server:app --host 127.0.0.1 --port 8000
```

처음 열면 **어디에 붙을지부터 묻는다**(AWS S3 · NCP · S3 호환). 정하고 나면
파일 브라우저로 들어간다.

> ⚠ **이 API 에는 아직 인증이 없다.** 믿을 수 있는 망 안에서 띄우는 것을
> 전제한다. 무엇이 열려 있고 무엇을 막았는지는
> [`docs/security.md`](docs/security.md) 에 재고 목록으로 적어 뒀다.

---

## 구조

```
face_anonymizer/   패키지 — 두 얼굴(서버 · 큐 워커)이 한 몸을 쓴다
scripts/           준비·배포 스크립트 (가중치 내려받기, EC2 세팅)
requirements/      의존성 — base · serve · worker · dev
tests/             회귀 테스트. 가중치·torch·GPU 없이 1분 안에 돈다
tools/             손으로 돌리는 도구 (MSA 큐 왕복 검증)
docs/              문제와 해결 기록 · 연동 규약 · 보안 재고
weights/           가중치 (커밋하지 않는다)
third_party/       YOLO-FaceV2 리포 (체크포인트 unpickle 에 필요)
```

**자세한 것은 폴더마다 README 를 둔다.** 이 파일은 지도이고, 왜 그렇게
만들었는지는 각 폴더에 있다.

| 폴더 | 무엇이 있나 |
|---|---|
| [`face_anonymizer/`](face_anonymizer/README.md) | 패키지 지도 · 처리 파라미터 · 두 얼굴의 관계 |
| [`core/`](face_anonymizer/core/README.md) | 한 편이 처리되는 순서 · 입력 코덱 · 알려진 제약 |
| [`service/`](face_anonymizer/service/README.md) | HTTP API · 웹 UI · 큐 · 검수 · 오류 체계 |
| [`msa/`](face_anonymizer/msa/README.md) | 남의 큐에서 일을 꺼내 오는 워커 |
| [`storage/`](face_anonymizer/storage/README.md) | 저장소 고르기 · 이름 규칙 · 가중치 조달 |
| [`requirements/`](requirements/README.md) | 쓰임새별로 무엇을 깔아야 하는가 |
| [`scripts/`](scripts/README.md) | 가중치 준비 · EC2 부트스트랩 |
| [`tests/`](tests/README.md) | 무엇을 가짜로 두는지 · 골라서 돌리는 법 |
| [`tools/`](tools/README.md) | 큐 왕복 검증 |
| [`docs/`](docs/README.md) | 겪은 문제와 판단의 기록 |

---

## 설정

조절할 수 있는 값은 전부 환경 변수다. 띄울 때마다 `export` 를 치는 대신 `.env`
를 둔다.

```bash
cp .env.example .env      # 필요한 줄만 고친다
```

규칙은 둘이다. **실제 환경 변수가 파일보다 우선**하고(한 번만 다르게 돌려
보려면 `FA_CRF=19 uvicorn ...` 처럼 앞에 붙인다), **파일이 없어도 전부
기본값으로 돈다.** `.env` 는 커밋하지 않는다 — 버킷 이름 같은 실제 값은 거기만
둔다.

목록과 기본값은 [`.env.example`](.env.example) 에 주석으로 다 적혀 있다. 코드가
읽는 값과 그 파일이 어긋나면 테스트가 실패한다.

---

## 세 가지로 쓴다

**단독 운영** — 우리가 서버를 갖고, 사람이 웹 화면으로 일을 시킨다. 버킷을 훑어
폴더째 제출하고 진행률·검수·로그를 화면에서 본다.
→ [`service/README`](face_anonymizer/service/README.md)

**HTTP 로 부르기** — 남의 시스템이 **잡 하나를 우리 API 에 보낸다.** 서명된 URL
두 개(입력·출력)만 받고, 저장소 자격 증명은 우리 쪽에 없다. 아래 참고.

**큐 워커(MSA)** — 남의 시스템이 큐에 넣은 일을 우리가 꺼내 온다. 인바운드
포트가 없다.
→ [`msa/README`](face_anonymizer/msa/README.md) ·
[연동 규약](docs/integration/rebornstudio.md)

셋 다 같은 `core/` 를 쓰고, 뒤의 둘은 **같은 러너**(`job_runner.run_job`)로
합류한다 — 계약은 페이로드고 전송은 선택이다. 처리 기본값도 한 벌을 나눠 쓴다.
두 벌로 뒀더니 큐 경로가 조용히 다른 설정으로 돌고 있었다([issues/009](docs/issues/009-queue-path-ran-untuned.md)).

---

## 컨테이너로 붙이기

```bash
docker compose up --build          # 8000 포트로 뜬다
curl localhost:8000/api/health
```

잡 하나를 보내고 결과를 가져가는 것이 전부다.

```bash
curl -X POST localhost:8000/api/deident/jobs \
  -H 'Content-Type: application/json' \
  -d '{"video_id":"v-1","token":"fencing-1",
       "input_url":"<presigned GET>",
       "targets":[{"label":"deid-720p","height":720,
                   "put_url":"<presigned PUT>","content_type":"video/mp4"}]}'
# → 202 {"job_id":"…","status":"running"}

curl localhost:8000/api/deident/jobs/<job_id>
# → {"status":"running","progress":{"percent":46.2,"stage_label":"얼굴 찾는 중",…}}
# → {"status":"done","result":{…}}  또는  {"status":"failed","problem":{…}}
```

**문은 기본으로 활짝 열려 있지 않다.** 아무 설정이 없으면 **같은 기계에서만**
열린다. 밖에서 부르려면 둘 중 하나다.

| 설정 | 뜻 |
|---|---|
| `FA_REMOTE_TOKEN=아무값` | `X-Deident-Token` 헤더에 같은 값을 보내야 열린다 |
| `FA_REMOTE_OPEN=1` | **인증 없이** 연다. 붙여 보는 단계용 — 기동 로그가 경고한다 |

바쁘면 `429` 로 거절한다(`FA_REMOTE_MAX_INFLIGHT`, 기본 1). **대기열을 만들지
않는다** — 부르는 쪽이 이미 재시도를 갖고 있고, 두 곳이 각자 판단하면 같은
영상이 몇 배로 돈다.

GPU 는 있으면 쓰고 없으면 CPU 로 돈다(느리다). `docker-compose.yml` 의 `deploy`
블록을 지우면 CPU 로 뜬다. 자세한 계약은
[연동 규약 §4-1](docs/integration/rebornstudio.md).

---

## 테스트

```bash
pip install -r requirements/dev.txt
pytest -q
```

가중치도 GPU 도 없이 1분 안에 돈다 — 가짜 검출기를 주입하기 때문이다. 자세한
것은 [`tests/README`](tests/README.md).

---

## 기록

만들면서 겪은 문제와 그것을 어떻게 판단하고 고쳤는지는
[`docs/`](docs/README.md) 에 남긴다. 고친 코드만 보면 **고르지 않은 선택지**가
사라지기 때문이다 — 더 쉬운 방법이 있는데 왜 안 썼는지, 어떤 대가를 받아들였는지.

---

## 라이선스

**GPL-3.0-or-later.** 검출기 YOLO-FaceV2(및 베이스 YOLOv5)가 GPL-3.0 이므로
파생물인 이 프로젝트도 같은 조건이다. supervision(ByteTrack)은 MIT 로 호환된다.
가중치는 저장소에 넣지 않고 업스트림 릴리스를 그대로
가리킨다([issues/012](docs/issues/012-the-model-was-tied-to-our-bucket.md)).

> `LICENSE` 의 `<YOUR NAME OR ORG>` 는 배포 전에 실제 저작자명으로 바꿔야 한다.

## 크레딧

- **YOLO-FaceV2** — Krasjet-Yu
  ([논문](https://www.sciencedirect.com/science/article/abs/pii/S0031320324004655)),
  clibdev 포크
- **ByteTrack** — supervision / Roboflow
