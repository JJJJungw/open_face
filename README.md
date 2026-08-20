> **이 저장소는 여기서 멈춥니다.** 작업은 sniperfactory-official/rebornstudio-deid 에서 이어집니다.
> 이 트리는 웹 화면·큐·검수까지 있던 마지막 상태이고, 그것들을 만들며 겪은 것은 `docs/` 에 남아 있습니다.

# face-anonymizer

**사용자가 지정한 클라우드의 영상에서 얼굴을 지운다.** 목적은 그것 하나다.

영상 한 편을 받아 얼굴을 검출하고(YOLO-FaceV2) 추적으로 이어 붙여(ByteTrack)
모자이크·블러·박스로 가린 뒤 돌려준다. 원본 오디오는 유지한다. 프레임 하나를
놓치면 그게 곧 누출이라, 끊긴 구간은 보간으로 채우고 마지막 관측 뒤에도 몇
프레임 더 가린다.

---

## 띄우기

```bash
docker compose up --build          # 8000 포트
curl localhost:8000/api/health
```

## 먼저 자격 증명부터

**클라우드 접근은 이 서버가 맡는다.** 부르는 쪽은 경로만 넘긴다. 그래서 붙이기
전에 이것부터 친다 — 되는지 안 되는지, 안 되면 왜 안 되는지가 한 번에 나온다.

```bash
curl localhost:8000/api/credentials/health
# → {"ok":true,"credentials":{"source":"환경 변수 (AWS_ACCESS_KEY_ID)"},
#    "bucket":"…","read":true,"write":true}
# → 503 {"ok":false,"read":true,"write":false,"problem":{"title":"이 버킷에 대한 권한이 없습니다",…}}
```

자격 증명은 `.env` 에 둔다(`FA_S3_BUCKET` · `FA_S3_REGION` · `FA_S3_ENDPOINT` 와
`AWS_ACCESS_KEY_ID` · `AWS_SECRET_ACCESS_KEY`). AWS·NCP·R2·MinIO·Wasabi 가 전부
같은 모양이고, 엔드포인트 한 줄만 다르다. EC2 라면 IAM 역할을 붙여도 된다 —
**어디서 온 자격 증명이든 위 응답의 `source` 가 말해 준다.**

## 부르기

일 하나를 넣고, 결과를 가져간다. 그게 전부다.

```bash
curl -X POST localhost:8000/api/deident/jobs \
  -H 'Content-Type: application/json' \
  -d '{"video_id":"v-1","token":"fencing-1",
       "input_key":"work/v-1/analysis-720p.mp4",
       "targets":[{"label":"deid-720p","height":720,
                   "output_key":"work/v-1/analysis-720p.deid.mp4"}]}'
# → 202 {"job_id":"…","status":"running"}

curl localhost:8000/api/deident/jobs/<job_id>
# running → {"progress":{"percent":46.2,"stage_label":"얼굴 찾는 중","eta_s":21,
#                        "vram_free_mb":3000,"vram_free_pct":12.5}}
# done    → {"result":{…}}
# failed  → {"problem":{"title":…,"hint":…},"transient":true}
```

`s3://다른버킷/키` 도 받는다 — 입력이 스테이징에, 결과가 납품 버킷에 있어도 된다.
**이미 서명해 둔 URL 이 있으면** `input_url` · `put_url` 로 보내도 된다. 그때는
우리 자격 증명을 안 쓴다. 두 방식이 **같은 러너로 합류하므로** 섞어 써도 된다.

### 알아 둘 것 셋

| | |
|---|---|
| **문** | 기본은 **같은 기계에서만** 열린다. `FA_REMOTE_OPEN=1` 이면 아무나, `FA_REMOTE_TOKEN=값` 이면 `X-Deident-Token` 헤더로 |
| **동시에 한 편** | 넘치면 `429`. **대기열을 만들지 않는다** — 부르는 쪽이 이미 재시도를 갖고 있다 (`FA_REMOTE_MAX_INFLIGHT`) |
| **GPU 여유** | `/api/health` 의 `vram` 과 진행률·로그에 계속 남는다. OOM 은 **나고 나서는 원인을 못 본다** |

GPU 는 있으면 쓰고 없으면 CPU 로 돈다(느리다). `docker-compose.yml` 의 `deploy`
블록을 지우면 CPU 로 뜬다.

---

## 그 밖에

이 파일은 지도다. **자세한 것은 폴더마다 README 를 둔다.**

| | |
|---|---|
| [돌리는 법](docs/running.md) | 도커 없이 · 명령줄 한 편 · 웹 화면 · 설정 · 테스트 |
| [`face_anonymizer/`](face_anonymizer/README.md) | 패키지 지도 · 처리 파라미터 |
| [`core/`](face_anonymizer/core/README.md) | 한 편이 처리되는 순서 · 입력 코덱 · 알려진 제약 |
| [`service/`](face_anonymizer/service/README.md) | HTTP API · 웹 UI · 큐 · 검수 · 오류 체계 |
| [`storage/`](face_anonymizer/storage/README.md) | 저장소 고르기 · 이름 규칙 · 가중치 조달 |
| [`msa/`](face_anonymizer/msa/README.md) | 큐에서 일을 꺼내 오는 워커 (전송만 다른 같은 러너) |
| [연동 규약](docs/integration/rebornstudio.md) | 잡 스키마 · 응답 · 진행률 · 실패 분류 |
| [`docs/`](docs/README.md) | 겪은 문제와 판단의 기록 |
| [`tests/`](tests/README.md) · [`scripts/`](scripts/README.md) · [`tools/`](tools/README.md) · [`requirements/`](requirements/README.md) | 검사 · 준비 · 도구 · 의존성 |

> ⚠ **이 서버에는 아직 인증이 없다**(위 "문" 한 줄이 전부다). 믿을 수 있는 망
> 안에서 띄우는 것을 전제한다 — [`docs/security.md`](docs/security.md).

---

## 라이선스

**GPL-3.0-or-later.** 검출기 YOLO-FaceV2(및 베이스 YOLOv5)가 GPL-3.0 이므로
파생물인 이 프로젝트도 같은 조건이다. supervision(ByteTrack)은 MIT 로 호환된다.
가중치는 저장소에 넣지 않고 릴리스를 가리킨다([issues/012](docs/issues/012-the-model-was-tied-to-our-bucket.md)).

- **YOLO-FaceV2** — Krasjet-Yu
  ([논문](https://www.sciencedirect.com/science/article/abs/pii/S0031320324004655)), clibdev 포크
- **ByteTrack** — supervision / Roboflow
