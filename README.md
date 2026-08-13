# face-anonymizer

영상 속 얼굴을 검출해 **모자이크/블러/박스**로 비식별화하는 파이프라인.
검출은 **YOLO-FaceV2**, 추적은 **ByteTrack**, 끊긴 프레임은 보간으로 메꿔 순간 누출을 막고, 원본 오디오는 유지한다.

```
검출(YOLO-FaceV2) → 추적(ByteTrack) → 트랙 보간 → 익명화 렌더 → 오디오 합성
```

프레임 단위 검출은 특정 프레임에서 얼굴을 순간적으로 놓칠 수 있고, 블러 파이프라인에서 그건 곧 프라이버시 누출이다. 그래서 ByteTrack 으로 프레임 간 트랙을 잇고, 관측된 프레임 사이의 빈 구간을 선형 보간으로 채우며, 마지막 관측 이후 몇 프레임 더 박스를 유지한다.

> **블러는 복원될 수 있다**: 약한 블러는 디블러링 모델로 부분 복원될 여지가 있다. 강한 익명화가 필요하면 `--method mosaic`(강하게) 또는 `--method box` 가 안전하다.

## 설치

```bash
git clone https://github.com/JJJJungw/open_face.git face-anonymizer
cd face-anonymizer

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ffmpeg 는 시스템 패키지 (오디오 합성용)
#   Ubuntu/Debian: sudo apt install ffmpeg   |   macOS: brew install ffmpeg

python setup_weights.py     # 검출기 리포 클론 + 가중치 (한 번만)
```

`setup_weights.py` 는 `third_party/YOLO-FaceV2` 에 검출기 코드를, `weights/yolo-facev2.pt` 에 가중치를 준비한다. 둘 다 gitignore 되어 저장소에는 안 들어간다.

## 사용법

```bash
# CLI
face-anonymize input.mp4                              # 기본 (모자이크)
face-anonymize input.mp4 --method box                 # 완전히 가리기
face-anonymize input.mp4 --imgsz 1280 --conf 0.20     # 작은 얼굴 대응
face-anonymize input.mp4 --batch-size 16              # GPU 처리량 우선
```

설치 전이라면 `python -m face_anonymizer.cli` 로도 실행된다.

```python
from face_anonymizer import VideoAnonymizer

anonymizer = VideoAnonymizer()          # 가중치/디바이스 자동, 재사용 가능
res = anonymizer.process("input.mp4", "output_anon.mp4",
                         method="mosaic", conf=0.25, batch_size=8)
print(res.frames, res.raw_boxes, res.filled_boxes, res.audio)
```

## 구조

```
face_anonymizer/
├── core/         영상 처리. 서버도 S3 도 없이 돈다 (fastapi·boto3 임포트 안 함)
├── service/      HTTP API · 웹 UI · 운영 지표          ← 단독 운영의 얼굴
├── msa/          큐를 지켜보는 워커 (인바운드 포트 없음) ← MSA 의 얼굴
├── storage/      S3 입출력 · 서명된 URL · 이름 규칙
├── params.py     처리 파라미터 기본값 — **두 진입점의 단일 출처**
├── job_runner.py 잡 페이로드 한 장 = 일 한 건
└── cli.py        명령줄 진입점
tools/          손으로 돌리는 도구 (MSA 큐 왕복 검증)
tests/          회귀 테스트 271개. 가중치·torch·GPU 없이 1분
weights/        가중치 (setup_weights.py 가 받는다)
third_party/    YOLO-FaceV2 리포 (체크포인트 unpickle 에 필요)
```

**얼굴이 둘이고 몸은 하나다.** `service/` 는 우리가 서버여서 사람이 웹으로 일을
시키고, `msa/` 는 우리가 소비자여서 남의 큐에서 일을 꺼내 온다. 둘 다 같은
`core/` 를 쓰고, 처리 기본값도 `params.py` 한 벌을 나눠 쓴다 — 두 벌로 두었더니
큐 경로가 조용히 다른 설정으로 돌고 있었다(docs/issues/009).
자세한 연동 방식은 `docs/integration/rebornstudio.md`.

의존은 **한 방향**이다. `service` 와 `storage` 는 `core` 를 쓰지만 `core` 는
둘을 모른다. 덕분에 코어만 떼어 배치 워커로 쓸 수 있고, 테스트가 가짜 검출기로
torch 없이 돈다. 폴더마다 README 가 있으니 자세한 건 거기를 보면 된다.

## 문제와 해결 기록

만들면서 겪은 문제와 그것을 어떻게 판단하고 고쳤는지는 [`docs/`](docs/) 에
따로 남긴다. 고친 코드만 보면 **고르지 않은 선택지**가 사라지기 때문이다 —
더 쉬운 방법이 있는데 왜 안 썼는지, 어떤 대가를 받아들였는지.

## 설정

조절할 수 있는 값은 전부 환경 변수다. 서버를 띄울 때마다 `export` 를 치는 대신
`.env` 를 두면 된다.

```bash
cp .env.example .env      # 필요한 줄만 고친다
```

`.env.example` 에 31개 항목이 기본값과 함께 주석으로 적혀 있다. 규칙은 둘이다 —
**실제 환경 변수가 파일보다 우선**하고(한 번만 다르게 돌려 보려면
`FA_CRF=19 uvicorn ...` 처럼 앞에 붙인다), **파일이 없어도 전부 기본값으로 돈다.**
`.env` 는 커밋하지 않는다.

## HTTP API + 웹 UI

```bash
pip install -r requirements.txt -r requirements-serve.txt
uvicorn face_anonymizer.service.server:app --host 0.0.0.0 --port 8000
```

브라우저로 열면 S3 를 콘솔처럼 훑으면서 파일이나 폴더를 골라 제출하고, 진행률·fps·남은 시간을 보면서 결과를 내려받을 수 있다.

테스트를 돌리려면 `requirements-dev.txt` 가 더 필요하다. 없으면 서버 테스트가 통째로 skip 된다(`TestClient` 가 httpx 를 쓴다).

```bash
pip install -r requirements-dev.txt
pytest -q
```

| 엔드포인트 | 설명 |
|---|---|
| `GET /` | 웹 UI |
| `GET /api/status` | `{ready, busy, queued, free_mb}` — 오케스트레이터용 최소 응답 |
| `GET /api/health` | 준비 전 **503**. 디바이스·모델 상태·작업 수 |
| `POST /api/jobs` | **제출은 여기 하나.** 한 건 · 여러 건 · 폴더 전부 |
| `POST /api/jobs/{id}/cancel` | 취소 (대기 중이면 즉시, 수행 중이면 다음 진행 보고에서) |
| `GET /api/jobs/{id}/result` | 결과 받는 방법 (S3 면 presigned URL) |
| `GET /api/problems` | 이 서비스가 낼 수 있는 오류 목록 |
| `GET /api/jobs/{id}` | 진행률 · 단계 · fps · ETA · 완료 시 통계와 경고 |
| `GET /api/jobs/{id}/download` | 결과 영상 (완료 전 409, 파일 만료 410) |
| `DELETE /api/jobs/{id}` | 작업과 파일 삭제 (진행 중이면 409) |
| `GET /api/jobs?limit=&status=` | 작업 목록 (최신순, 기본 100건, 상태 필터) |
| `GET /api/s3/objects?prefix=` | 버킷 한 단계 나열 (미설정 시 404) |
| `GET /api/defaults` | 서비스가 쓰는 처리 파라미터 기본값 |

## S3

`FA_S3_BUCKET` 을 주면 입력을 S3 에서 내려받고 결과물을 다시 올린다. 자격 증명은 boto3 기본 체인이라 EC2 인스턴스 역할이 있으면 그대로 잡힌다.

```bash
export FA_S3_BUCKET=ax-mbc-label-data-storage
export FA_S3_REGION=us-east-1
export FA_S3_OUTPUT_PREFIX=v1/results/face/     # 기본값
```

`POST /api/jobs` 에 `file` 대신 `s3_key` 를 주면 된다. 둘 다 주거나 둘 다 안 주면 400 이다. 내려받기는 **워커가** 한다 — 접수 요청을 붙들고 수백 MB 를 받으면 클라이언트가 그동안 응답을 기다리게 된다.

결과물은 입력 위치와 무관하게 한곳에 모인다.

## 파일 이름 규칙

```
C_NNNNN_SS_STARTMS_ENDMS[_STATE].ext
```

| 필드 | 의미 | 예 |
|---|---|---|
| `f` | 카테고리 (face) | `f` |
| `NNNNN` | 원본 영상 번호 5자리 | `00001` |
| `SS` | 세그먼트 2자리 (한 영상에서 여러 클립일 때) | `00` |
| `STARTMS` | 클립 시작 ms 7자리 | `0000000` |
| `ENDMS` | 클립 끝 ms 7자리 | `0042000` |
| `STATE` | `raw`(입력) / `deid`(비식별 출력) | `deid` |

비식별화는 **정체성 필드를 건드리지 않고 STATE 만 바꾼다.** 클립을 자르거나 합치지 않으므로 번호·세그먼트·구간은 입력 그대로여야 하고, 여기가 틀리면 결과물이 어느 원본의 어느 구간인지 추적할 수 없게 된다.

```
videos/2026-08/f_00001_00_0000000_0042000_raw.mp4
  -> v1/results/face/f_00001_00_0000000_0042000_deid.mp4
```

확장자는 입력이 `.mov` 여도 항상 `.mp4` 다 — H.264/mp4 로 다시 뜨기 때문이다. 자릿수는 고정으로 검사한다(느슨하게 받으면 정렬이 깨지고 잘못 붙은 이름이 결과 폴더에 남는다). 규칙 밖 이름도 처리는 되며 `<이름>_deid.mp4` 로 떨어진다 — 직접 업로드한 임의 파일을 막지 않기 위해서다. 규칙은 `face_anonymizer/naming.py` 한곳에 있고 S3·로컬 출력·다운로드 파일명·CLI 기본 경로가 모두 이걸 쓴다.

목록의 `processed` 표시는 결과물 프리픽스를 **한 번 나열해서** 대조한다(`FA_S3_LIST_TTL` 초 캐시). 객체마다 HEAD 를 날리면 목록 한 번에 수백 번 왕복한다.

boto3 는 지연 임포트라 S3 를 안 쓰면 설치할 필요가 없다.

## 처리 파라미터

**호출하는 쪽은 입력만 주면 된다.** 튜닝된 값은 서비스가 들고 있어야지, 호출자마다 들고 다니면 어느 설정으로 처리됐는지가 호출 지점마다 달라진다.

| 항목 | 기본 | 환경 변수 |
|---|---|---|
| `method` | `mosaic` | `FA_METHOD` |
| `conf` | `0.25` | `FA_CONF` |
| `imgsz` | `1280` | `FA_IMGSZ` (검출기와 공유) |
| `batch_size` | `32` | `FA_BATCH_SIZE` |
| `pad` | `0.15` | `FA_PAD` |
| `mosaic_scale` | `0.06` | `FA_MOSAIC_SCALE` |
| `linger` | `5` | `FA_LINGER` |
| `interp` / `keep_audio` | on | `FA_INTERP` / `FA_KEEP_AUDIO` |
| `crf` / `bitrate_ratio` | `23` / `1.0` | `FA_CRF` / `FA_BITRATE_RATIO` |

운영 중 조정은 환경 변수로 하고, 필요할 때만 요청에서 개별 항목을 덮는다(보낸 것만 덮이고 나머지는 기본값). 현재 값은 `GET /api/defaults` 로 확인한다 — 웹 UI 도 컨트롤 초깃값을 여기서 받아 가므로 서버 설정과 화면이 어긋나지 않는다.

**한 번에 한 편.** 추론은 워커 스레드 하나가 순차로 돌린다(GPU 한 장에 검출기 하나).

**대기열은 개수로 막지 않는다.** 전체 수행처럼 한꺼번에 수백 건을 넣는 사용이 정상이고, 개수는 애초에 잘못된 기준이다 — 10건이 50MB 짜리면 아무것도 아니고 2GB 짜리면 이미 위험하다. 대기 중인 작업은 입력 파일을 디스크에 들고 있으므로 진짜 제약은 거기다. 여유 공간이 `FA_MIN_FREE_MB`(기본 2048) 밑이면 `507` 로 거절한다. 개수 상한이 필요하면 `FA_QUEUE_MAX` 로 켤 수 있다(기본 0 = 무제한).

수백 건이 쌓여도 폴링이 느려지지 않게, `GET /api/jobs/{id}` 는 디스크를 훑지 않고 메모리만 본다. 목록은 대기 순번을 한 번에 계산하고(작업마다 전체를 다시 훑으면 O(N²)) 기본 100건까지만 준다. 작업 500건 기준 폴링 0.1ms, 목록 13ms.

작업 상태는 다섯이다.

| 상태 | 뜻 |
|---|---|
| `queued` | 대기 |
| `running` | 수행중 (`stage` 가 `detect`/`render`, `overall` 이 진행률) |
| `done` | 완료 (`result` 에 통계) |
| `failed` | 실패 (`error.code` 에 사유, `attempts` 에 시도 횟수) |
| `cancelled` | 취소됨 |

## 오류

응답은 **RFC 9457 Problem Details**(`application/problem+json`)를 따른다.

```json
{
  "type": "/problems/queue-full",
  "title": "대기열이 가득 찼다",
  "status": 429,
  "detail": "대기 10건",
  "code": "queue_full",
  "hint": "Retry-After 뒤에 다시 보내거나 다른 인스턴스로 보내라.",
  "retryable": true,
  "instance": "/api/jobs"
}
```

호출하는 쪽은 재시도할지, 다른 인스턴스로 보낼지, 사람을 불러야 할지를 정해야 한다. 한국어 문장을 파싱해서 정할 수는 없으므로 **`code`(안정된 식별자)와 `retryable`(서버가 내린 판단)** 로 분기한다. `detail` 과 `hint` 는 사람이 읽는 용도라 문구가 바뀔 수 있다.

오류 정의는 `face_anonymizer/errors.py` 의 `CATALOG` 한곳에 있고 `GET /api/problems` 로 그대로 나온다 — 호출하는 쪽이 code 별 대응을 미리 짜 둘 수 있다. 종류는 30가지이고, 크게 이렇게 나뉜다.

| 갈래 | 예 |
|---|---|
| 요청이 잘못됨 | `missing_input` `conflicting_input` `invalid_key` `unsupported_media` `payload_too_large` |
| 서비스 상태 | `not_ready` `model_load_failed` `queue_full` `insufficient_storage` |
| 작업 | `job_not_found` `job_not_finished` `result_expired` `job_not_cancellable` |
| S3 | `s3_not_configured` `s3_object_not_found` `s3_access_denied` `s3_upstream` |
| 처리 실패 | `video_unreadable` `decode_incomplete` `encode_failed` `no_detections` `gpu_out_of_memory` `ffmpeg_missing` |

작업이 실패하면 같은 코드 체계가 `error` 에 담긴다. 권한 문제와 키 오타와 GPU 메모리 부족은 사용자가 해야 할 일이 전혀 다르므로 구분해서 남긴다.

```json
"error": { "code": "gpu_out_of_memory", "title": "GPU 메모리가 부족하다",
           "detail": "CUDA out of memory", "hint": "batch_size 나 imgsz 를 낮춰라.",
           "retryable": true }
```

## 제출

**진입점은 하나다.** 진입점을 나누면 클라이언트가 경우마다 분기해야 하고, 화면에도 버튼이 그만큼 늘어난다. 입력이 무엇이냐만 다르고 나머지는 전부 같다.

```bash
# 한 건 업로드
curl -F file=@clip.mp4 localhost:8000/api/jobs

# 고른 것들
curl -X POST localhost:8000/api/jobs -H 'Content-Type: application/json' \
  -d '{"s3_keys": ["videos/2026-08/f_00001_00_0000000_0042000_raw.mp4"]}'

# 폴더 통째로
curl -X POST localhost:8000/api/jobs -H 'Content-Type: application/json' \
  -d '{"s3_prefix": "videos/2026-08/", "recursive": true, "skip_processed": true}'
```

`file` · `s3_keys` · `s3_prefix` 중 **하나만** 보낸다. 옵션은 JSON 이면 `params`, multipart 면 폼 필드로 주고, 안 주면 서비스 기본값이다.

응답은 세 경우 모두 같다.

```json
{"accepted": [{"id": "ab12...", "name": "...", "s3_key": "..."}],
 "rejected": [{"s3_key": "...", "error": {"code": "unsupported_media", ...}}],
 "queued": 3}
```

**한 건이 거절돼도 나머지는 받는다.** 수백 건에서 키 하나가 오타라고 전체를 되돌리면 호출하는 쪽이 무엇이 들어갔는지 알 수 없다. 다만 **하나도 못 받았으면 `202` 를 주지 않는다** — 단건 제출이면 그 사유가 곧 응답 코드가 되고(예: `415 unsupported_media`), 여러 건이면 `400` 에 항목별 사유가 담긴다.

폴더는 서버가 펼친다. 클라이언트가 목록을 먼저 받아 오게 하면 그 사이에 파일이 늘거나 줄 수 있고 왕복도 한 번 더 든다. `recursive` 는 하위 폴더까지, `skip_processed` 는 이미 결과물이 있는 건 건너뛴다(폴더를 다시 돌릴 때). 영상 확장자만 골라 넣는다. 상한은 `FA_BATCH_MAX`(기본 500).

## 결과 받기

```bash
curl -s localhost:8000/api/jobs/<id>/result
# {"via":"s3","s3_key":"v1/results/face/..._deid.mp4",
#  "download_url":"https://...","expires_in":3600}
```

S3 작업이면 **presigned URL** 을 준다 — GPU 서버가 파일 전송까지 떠안을 이유가 없고, 로컬 사본이 보관 기간에 정리돼도 S3 원본은 남아 있다. `/download` 도 로컬 사본이 없으면 S3 로 302 리다이렉트한다.

**실패·취소된 작업은 기본적으로 지우지 않는다**(`FA_FAILED_TTL_MIN=0`). 배치로 수백 건 돌린 뒤 몇 건이 실패했을 때 입력과 사유가 남아 있어야 원인을 볼 수 있다.

**실패하면 재시도한다.** 일시적 오류(CUDA OOM, ffmpeg 실패, 디스크 문제)는 `FA_MAX_ATTEMPTS`(기본 3)회까지 다시 큐에 넣고, 소진되면 `failed` 로 남긴다. 다만 **같은 입력으로 같은 결과가 나올 오류는 재시도하지 않는다** — 깨진 파일이나 잘못된 인자(`VideoOpenError`, `VideoWriteError`, `ValueError`, `FileNotFoundError`)를 세 번 돌려도 결과가 같고, 그동안 뒤에 쌓인 정상 작업만 밀린다. 이 경우 `attempts` 가 1 로 남는다.

호출 흐름은 이렇다.

```bash
# 1. 여유 있는지 확인 (선택)
curl -s localhost:8000/api/status          # {"ready":true,"busy":false,"queued":0,"free_mb":41230}

# 2. 제출 — 바쁘면 429
curl -sf -F file=@in.mp4 localhost:8000/api/jobs   # {"id":"ab12...","status":"queued"}
# S3 입력이면:  -F s3_key=videos/2026-08/f_00001_00_0000000_0042000_raw.mp4

# 3. 진행률 폴링
curl -s localhost:8000/api/jobs/ab12...     # {"status":"running","stage":"detect","overall":31,"fps":98.4,"eta":42}

# 4. 결과 받기
curl -sf -OJ localhost:8000/api/jobs/ab12.../download

# 5. 정리
curl -sX DELETE localhost:8000/api/jobs/ab12...
```

완료 응답의 `result.warnings` 는 결과를 그대로 믿으면 안 되는 사유다(`no-detections`, `decode-short`, `frame-loss` 등). 비어 있지 않으면 사람이 확인해야 한다.

**모델은 기동 시 미리 올린다**(`FA_PRELOAD`, 기본 1). 첫 요청 때 로드하면 헬스체크는 이미 통과한 상태라 오케스트레이터가 보낸 첫 요청이 모델 로딩 수십 초를 기다린다. 로드에 실패하면 죽지 않고 `/api/health` 가 `503` 과 사유(`model_error`)를 돌려준다 — 크래시 루프로 재시작하면 로그가 흘러가 원인을 찾기 어렵다.

**추론은 한 번에 하나만 돈다.** GPU 한 장에 검출기 하나를 올려 두고 워커 스레드 하나가 큐를 소비한다. 요청마다 스레드를 띄우면 VRAM 이 터지거나 서로 느려지기만 하고, 총 처리량은 오히려 직렬화하는 쪽이 높다. 프로세스를 여러 개 띄워도(`--workers N`) 작업 디렉터리의 잠금 파일로 직렬화한다.

**작업 상태는 디스크에 둔다.** 작업별 디렉터리의 `job.json` 이 정본이고 메모리는 캐시다. 전역 dict 에만 두면 재시작 시 전부 사라져 폴링 중인 클라이언트가 404 를 받고, `--workers 2` 로 띄우는 순간 업로드는 A 프로세스 · 폴링은 B 프로세스로 가서 계속 404 가 난다. 진행률은 0.5초 간격으로 흘려 쓰고, 쓰기는 임시 파일 + rename 이라 읽는 쪽이 반쪽짜리 JSON 을 보지 않는다.

기동 시 `queued`/`running` 상태로 남은 작업은 실패로 표시한다(프로세스가 죽으면 상태 파일만 남아 클라이언트가 영원히 '처리 중' 을 폴링한다). TTL 정리는 백그라운드 스레드가 `FA_SWEEP_SEC` 주기로 돈다 — 예전에는 새 업로드가 있을 때만 돌아서 업로드가 끊기면 디스크가 안 비워졌다.

환경 변수로 `FA_DEVICE`, `FA_IMGSZ`, `FA_JOBS_DIR`, `FA_MAX_UPLOAD_MB`(기본 2048), `FA_JOB_TTL_MIN`(기본 120, 완료 후 자동 삭제), `FA_FFMPEG_TIMEOUT`(기본 600)을 조정한다.

인증이 없으므로 공개 주소에 그대로 띄우지 말 것. 원격 테스트는 SSH 터널이 안전하다.

```bash
ssh -i key.pem -L 8000:localhost:8000 ubuntu@<host>
```

## 주요 파라미터

| 옵션 | 기본 | 설명 |
| --- | --- | --- |
| `--method` | `mosaic` | `mosaic` / `blur` / `box` |
| `--imgsz` | `960` | 추론 해상도. 720p 작은 얼굴이면 `1280` 권장 |
| `--conf` | `0.25` | 낮출수록 재현율↑(누출↓)·오탐↑. 프라이버시 우선이면 `0.15~0.20` |
| `--mosaic-scale` | `0.06` | 낮출수록 블록 커짐(더 강하게 가림) |
| `--pad` | `0.15` | 박스 확장 비율(턱/헤어라인 커버) |
| `--linger` | `5` | 트랙 소실 후 박스 유지 프레임 수 |
| `--batch-size` | `1` | 한 번에 모델에 넣을 프레임 수. GPU 에서는 8~32 권장 |
| `--half` / `--no-half` | 자동 | FP16 추론. CUDA 에서 기본 활성화 |
| `--no-interp` | off | 트랙 보간 끄기 |

## 테스트

**가중치도 torch 도 없이 파이프라인 전 구간이 돈다.** 검출기를 주입할 수 있게 만들어 두었기 때문에, 얼굴 위치를 아는 합성 영상 + 가짜 검출기로 배선을 확인한다.

```bash
pip install -e ".[dev]"     # torch 없음
pytest
```

핵심은 두 개다. 검출기가 프레임 12~16 을 통째로 놓쳐도 보간이 덮는다는 것, 그리고 보간을 끄면 실제로 노출된다는 음성 대조 — 후자가 없으면 앞 테스트가 보간 덕분에 통과한 건지 알 수 없다. 익명화 여부는 평균 색이 아니라 라플라시안 분산으로 판정한다(모자이크는 평균 색을 유지하므로 표준편차로는 알 수 없다). 나머지는 좌표 왕복, 박스 정수화 방향, 조용한 실패 몇 가지다.

모델 정확도(WIDERFace 성능)는 가중치가 필요한 별도 관심사라 분리했다.

## 실패했을 때

가장 위험한 실패는 "빈 결과물이 성공으로 보고되는 것"이다. 영상을 못 열거나 인코더를 못 잡으면 예외를 던지고(`VideoOpenError` / `VideoWriteError`), 디코딩 크기가 컨테이너 메타와 다르거나 1·2차 패스의 프레임 수가 어긋나도 실패로 처리한다.

**디코딩이 중간에 끊긴 것도 실패다**(`DecodeIncompleteError`). `cap.read()` 가 False 를 주는 건 "스트림 끝"과 "디코드 실패" 둘 다인데 구분 없이 break 하면 영상 뒷부분이 결과물에 통째로 없는 채 정상 종료한다(실측: 600프레임 선언 파일에서 241프레임만 렌더 후 성공). 1·2차 패스 비교는 둘 다 같은 지점에서 끊기면 통과하므로 이걸 못 잡는다.

기대 프레임 수는 **비디오 스트림 길이 x fps** 로 잡는다. 다른 후보들은 전부 오탐을 낸다.

| 기준 | 문제 |
|---|---|
| `CAP_PROP_FRAME_COUNT` | 추정값이라 틀린다 |
| `nb_frames` / 패킷 수 | 앞뒤를 잘라낸 영상(edit list)은 패킷이 그대로 남는다. 실측: 0.5초 자른 파일이 패킷 150 / 실제 129 |
| `format.duration` | 모든 스트림 중 가장 긴 값. 오디오가 길면 과대추정. 실측: 영상 1.33초 + 오디오 3초에서 20프레임을 45프레임으로 계산 |

판정도 2단이다. 누락이 2% 이하면 정상, 20% 이상이면 실패, 그 사이는 경고(`decode-short`)만 남기고 진행한다. **서비스에서는 정상 영상을 거부하는 쪽이 더 큰 사고**라, 명백한 절단일 때만 실패시킨다. 의도한 부분 처리면 `--allow-partial`.

## 입력 코덱

**들어오는 코덱은 가리지 않고, 나가는 것은 H.264 로 고정한다.**

OpenCV 의 FFmpeg 빌드는 ffmpeg 본체보다 코덱 지원이 좁다. AV1 이 대표적인데, 파일을 **열기는 열면서 한 프레임도 못 뽑는다**(실측: OpenCV 4.13 에서 `isOpened()` 는 True, 디코딩 0프레임, `Your platform doesn't support hardware accelerated AV1 decoding`). ffmpeg 는 libdav1d 로 잘 읽는다.

그래서 `ingest.py` 가 **코덱 이름이 아니라 실제로 한 프레임을 뽑아 본다.** 목록으로 관리하면 빌드마다 다르고 새 코덱이 나올 때마다 어긋난다. 못 뽑으면 ffmpeg 로 H.264 로 옮겨 담고 그 파일을 파이프라인에 준다.

```
av1 입력  →  [전사 0.4s]  →  검출/렌더  →  h264 출력
h264 입력 →  (그대로)     →  검출/렌더  →  h264 출력
```

H.264 입력은 이 경로에 들어오지 않으므로 비용이 없다. 전사본은 검출 **전에** 만들어지므로 화질이 떨어지면 검출률이 같이 떨어진다 — `FA_INGEST_CRF`(기본 16)로 고화질을 쓴다. 중간 산출물이라 용량이 커도 되고 작업이 끝나면 지워진다. 최종 결과물의 화질·용량은 `--crf` / `--bitrate-ratio` 로 따로 관리한다.

오디오는 전사본에 담지 않는다. 최종 합성은 **원본**에서 가져온다. 결과에는 `source_codec` 과 `transcoded` 가 남고 소요 시간은 `timing.ingest` 로 분리된다.

읽을 수 없는 입력은 `video_unreadable` 로 실패하며 **재시도하지 않는다** — 같은 파일로 다시 시도해도 결과가 같다.

**출력은 H.264 로 다시 뜬다.** `cv2.VideoWriter` 가 쓸 수 있는 mp4v(MPEG-4 Part 2)는 같은 화질에 H.264 대비 약 10배 크다(1280x720 실측 10.8배). 어차피 ffmpeg 를 거치므로 그 단계에서 인코딩한다. GPU 가 있으면 NVENC 를 쓰고(자동 판정) 없으면 libx264 로 떨어진다. 품질은 `--crf`(기본 23), 인코더 강제는 `FA_ENCODER`.

**출력 비트레이트에는 원본 기준 상한을 건다**(`--bitrate-ratio`, 기본 1.0). CRF 만 쓰면 "목표 화질"로 인코딩하므로 이미 압축된 원본을 받으면 결과물이 더 커진다(실측: 1.89 → 2.96 Mbps, 46MB → 70MB). 비식별화 결과물에 원본 이상의 화질이 필요할 이유가 없고 서비스에서는 다운로드 용량이 곧 비용이라, CRF 는 그대로 두고 상한만 걸어(capped CRF) 단순한 장면은 더 작게, 복잡한 장면도 원본을 넘지 않게 한다. `--bitrate-ratio 0` 이면 무제한.

**세로 촬영 영상은 회전을 명시적으로 처리한다.** 폰 세로 영상은 픽셀이 가로로 저장되고 회전 메타데이터가 붙어, 비율로는 가로 영상과 구분되지 않는다. 누운 프레임에 검출을 돌리면 얼굴을 거의 못 잡는데 크기 검사는 통과해 조용히 새어 나간다. OpenCV 4.5.2+ 는 자동 적용하지만 빌드에 따라 꺼져 있을 수 있어, `open_capture()` 가 명시적으로 켜고 안 켜지면 직접 돌린다. 필요하면 `--rotate` 로 지정한다.

**검출 0건은 실패가 아니라 경고다.** 얼굴이 없는 영상은 정당하게 0 이기 때문이다. 다만 가중치 손상·회전된 영상·잘못된 `imgsz`·HDR 톤매핑 실패도 결과가 똑같이 0 이고 그때 원본이 그대로 나간다. 원인이 무엇이든 결과가 조용한 게 문제이므로 `Result.warnings` 에 `no-detections` 를 담고 CLI 는 stderr 로, 웹 UI 는 빨간 배너로 드러낸다. 반드시 얼굴이 있어야 하는 파이프라인이면 `--min-detection-rate 0.5` 로 실패시킬 수 있다. 익명화를 무력화하는 파라미터(음수 `pad`, `mosaic_scale >= 1`)도 입구에서 거부한다.

반대로 오디오 합성 실패는 결과물을 버릴 이유가 아니다. ffmpeg 가 없거나 실패해도 익명화된 영상은 출력 경로에 남기고 `Result.audio` 로 이유를 알린다.

단 **합성이 프레임을 삼키는 것은 실패로 친다.** 예전에는 `-shortest` 를 썼는데 이건 짧은 쪽에 맞춰 자르므로, 오디오가 영상보다 짧으면 잘리는 게 영상이었다(20초/600프레임 결과물이 10초/300프레임으로 잘린 채 `audio='ok'` 로 보고됨). 지금은 `-shortest` 를 쓰지 않고, 합성 결과의 프레임 수를 실제로 세어 렌더한 수와 다르면 합성본을 버리고 무음본을 내보낸다(`audio='frame-loss: 300/600'`). 검증을 통과하기 전에는 무음본을 지우지 않는다.

`ffmpeg`/`ffprobe` 호출에는 타임아웃이 걸려 있다(`FA_FFMPEG_TIMEOUT`, 기본 600초). 서버는 워커가 하나라 한 건이 매달리면 큐 전체가 정지하기 때문이다.

서버(`server.py`)도 같은 원칙이다. 작업이 실패해도 상태와 사유를 `status=error` 로 남기고 프로세스는 계속 산다.

## 알려진 제약

프레임 스킵(N 프레임마다 검출)은 쓰지 않는다. 건너뛴 구간의 커버리지를 보장할 수 없기 때문이다. 처리량은 배치 추론과 FP16 으로 올린다.

`supervision` 0.30 에서 `sv.ByteTrack` 이 제거될 예정이라 `<0.30` 상한을 걸어 뒀다. 올릴 때는 `tracking.py` 의 트래커 교체가 필요하다.

영상을 두 번 읽는다(1차 검출/추적, 2차 렌더). 프레임을 메모리에 쌓지 않는 대신 디스크 I/O 를 두 배 쓴다.

`Result.filled_boxes` 가 `raw_boxes` 에 비해 유난히 크면 검출이 자주 끊긴다는 신호다. `--conf` 를 낮추거나 `--imgsz` 를 올리는 게 낫다. 반대로 이 값이 **작다고 안전한 건 아니다** — 보간 자체가 돌지 않아도 0 이 나온다.

추적기 문턱은 `--conf` 에 연동된다. ByteTrack 은 활성화 임계값에 0.1 을 더한 값을 실질 문턱으로 쓰고, 매칭 비용에도 검출 점수를 곱한다(`1 - IoU x 점수 < 0.8`). 기본값으로 두면 `--conf 0.25` 로 통과시킨 검출의 상당수가 트랙조차 만들지 못하고, 만들어져도 다음 프레임에 죽는다. 즉 **간헐적으로 놓치는 저신뢰 얼굴 — 보간이 존재하는 이유인 바로 그 대상 — 에서만 안전망이 꺼진다.** `tracking.py` 가 문턱을 `conf` 에 맞추고 추적용 점수를 정규화해 이를 막는다.

## 프로젝트 구조

```
face_anonymizer/
├── geometry.py     # letterbox / 좌표 역변환 (torch 불필요)
├── detector.py     # YOLO-FaceV2 로더 + 배치 추론
├── tracking.py     # ByteTrack + 트랙 보간
├── anonymize.py    # 모자이크/블러/박스
├── pipeline.py     # 전체 파이프라인
├── cli.py          # 커맨드라인
├── server.py       # HTTP API (FastAPI, 작업 큐)
└── webui.py        # 웹 UI (단일 HTML 문자열)
```

## 라이선스

**GPL-3.0.** 검출기 YOLO-FaceV2(및 베이스 YOLOv5)가 GPL-3.0 이므로 파생물인 이 프로젝트도 GPL-3.0 이다. supervision(ByteTrack)은 MIT 로 호환된다. `LICENSE` 의 `<YOUR NAME OR ORG>` 는 배포 전 실제 저작자명으로 교체할 것.

## 크레딧

- YOLO-FaceV2 — Krasjet-Yu ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320324004655)), clibdev 포크
- ByteTrack — supervision / Roboflow
