# rebornstudio 에 붙기 — 점검 결과

> 읽기만 했다. `../rebornstudio` 는 한 글자도 안 건드렸다.
> 기준 커밋: 방금 `git pull` 한 `main`.

## 한 줄

**자리는 이미 다 만들어져 있다.** 우리가 새로 만들 것은 서비스가 아니라 **어댑터 함수
하나**다. 그리고 그 사실 때문에 지금까지 붙잡고 있던 자격 증명 문제가 이 경로에서는
**통째로 사라진다.**

---

## 1. 저쪽이 이미 만들어 둔 것 (Step 324)

`worker-deident` 는 골격이 아니라 **다 지어진 자리**다. 없는 것은 모델뿐이다.

| 층 | 파일 | 상태 |
|---|---|---|
| 큐 | `q.deident` | 있다 |
| 컨테이너 | `worker-deident` (local · prod compose 둘 다) | 있다 (`--profile deident`) |
| Celery 태스크 | `dispatch_deidents` · `deident_one` · `deident_heartbeat` · `deident_complete` · `reclaim_expired_deidents` | 있다 |
| DB 쪽 | `api_gateway.services.media.deident` — `build_jobs` · `extend_lease` · 펜싱 완료 | 있다 |
| 러너 | `reborn_transcode.deident.run_deident_job` | 있다 |
| 어댑터 | `_ADAPTERS = {"passthrough": _passthrough}` | **여기가 빈칸이다** |
| 스키마 | `videos.deident_status` · `deident_token` · `deident_lease_until` · `deident_attempts` | 있다 |
| 설정 | `projects.deident_enabled` (기본 false) · `DEIDENT_MODEL` 환경 변수 | 있다 |

리스·펜싱·회수·재시도·진행률 축·상태 기계가 전부 트랜스코딩과 **같은 프로토콜(M5)** 로
이미 돌고 있다. 설계 문서가 그걸 명시한다 — *"새 프로토콜을 만들지 않는다"*
(`docs/design/local-ingest-pipeline.md` L13).

## 2. 우리가 채워야 하는 계약 — 이게 전부다

`packages/reborn-transcode/src/reborn_transcode/deident.py` 의 어댑터 규약:

```python
def adapter(src: Path, dst: Path) -> dict:
    """로컬 파일 하나 → 로컬 파일 하나. 보고서 dict 를 돌려준다."""
```

- **입출력은 로컬 경로다.** S3 도, HTTP 도, DB 도 안 만진다 — 러너가 presigned GET 으로
  받아 두고, 끝나면 presigned PUT 으로 올린다.
- 실패는 `JobError(메시지, transient=bool)` 로 던진다. `transient` 가 재시도 여부를 정한다.
- 워커 **스레드**에서 돈다. 본 스레드가 하트비트를 보낸다 — 우리는 신경 안 써도 된다.
- `DEIDENT_MODEL` 환경 변수 이름으로 고른다. 이름이 없거나 모르는 이름이면 **잡이 실패**하고
  파이프라인이 멈춘다(L14 — 조용한 통과를 막는 장치다).

### 돌려줘야 하는 보고서 (L15)

> *"비식별화했다" 는 나중에 증명해야 할 수 있는 주장이다. 그리고 **아무것도 안 한 모델도
> 멀쩡한 파일을 낸다.**

```
model / model_version / 설정(블러 강도 등)
검출 수 (얼굴 N · 번호판 M) · 처리 프레임 수
```

**검출 0건을 실패로 보지 않는다** — 우리가 issue 008 에서 내린 결론과 정확히 같다.
다만 화면에 그대로 실어서 사람이 판단하게 한다.

## 3. 자격 증명 — 이 경로에서는 **우리가 들 게 없다**

지금까지 게이트·`providers.py`·키 입력을 붙잡고 있었는데, rebornstudio 안에서는 그게
**한 줄도 안 쓰인다.**

```
build_jobs (DB 쪽)                        워커 (우리)
  presign_get_for(bucket, key)   ──────▶   input_url  (그냥 GET)
  presign_put_for(bucket, key)   ──────▶   put_url    (그냥 PUT)
```

자격 증명은 `api-gateway` 의 `S3_ACCESS_KEY` / `S3_SECRET_KEY` 하나뿐이고, 그건 저쪽
`.env` 에 있다. **워커는 키를 본 적이 없다.** 이게 M5 가 "워커는 DB를 모른다" 로 얻어 낸
것의 연장이다.

그러니 그동안 한 작업이 헛일이냐 — 아니다. **두 제품이 갈린다.**

| | 오픈소스 단독 실행 | rebornstudio 모듈 |
|---|---|---|
| 입출력 | 우리가 버킷에 직접 붙는다 | presigned URL 두 개 |
| 자격 증명 | 우리 게이트가 받는다 | **없다** |
| 웹 UI | 우리 것 | 저쪽 화면 |
| 작업 상태 | 우리 `jobs.STATUSES` | 저쪽 `deident_status` |
| 쓰는 코드 | `core` + `service` + `storage` | **`core` 만** |

`core/` 를 `service`·`storage` 와 갈라 둔 게 여기서 값을 한다 — 그 경계 덕분에
**떼어 낼 게 없다.** `core/README.md` 에 적어 둔 세 가지 이유 중 세 번째가 이것이다.

## 4. 맞춰야 할 것 — 우선순위 순

### ① 파이썬 3.11 (막힘)

저쪽은 전부 `requires-python = ">=3.11,<3.12"` 이고 워커 이미지가 `python:3.11-slim` 이다.
우리는 `>=3.9` 로 열어 뒀고 **EC2 는 3.12** 로 돌고 있다. 어제 3.12 에서만 죽는 검사를
하나 밟은 게 우연이 아니다(issue 026).

→ 우리 CI 에 3.11 을 넣고, 3.11 에서 전체를 한 번 돌린다.

### ② 라이선스 (막힘 — 코드가 아니라 결정)

| | 라이선스 |
|---|---|
| rebornstudio (`pyproject.toml`) | Apache-2.0 |
| rebornstudio (`package.json`) | CC-BY-NC-SA-4.0 |
| **우리** | **GPL-3.0-or-later** (YOLO-FaceV2 에서 옴) |

어댑터는 저쪽 프로세스 **안에서** 임포트된다(`_ADAPTERS` 에 등록되는 함수다). GPL 은
전염성 조항이 있어서, 그 상태로 두면 저쪽 결합 저작물의 라이선스 해석이 걸린다.

선택지는 셋으로 보인다. **법률 판단은 제 몫이 아니니 사실만 적는다.**

1. **별도 프로세스로 뗀다** — 어댑터는 얇은 호출자만 두고 우리 모델은 자기 프로세스에서
   돈다. 결합이 아니라 실행이 된다. 대신 계약이 하나 늘어난다.
2. **검출기를 GPL 아닌 것으로 바꾼다** — YOLO-FaceV2 가 GPL 의 출처다. 그러면 검출
   성능을 다시 재야 한다.
3. **deident 워커 이미지만 GPL 로 간다** — 다만 그 이미지가 `reborn_transcode` 를
   임포트하므로 그 경계가 어디까지인지가 애매하다.

이건 코드보다 먼저 정해야 한다.

### ③ 어댑터 함수 (본체)

`core/pipeline.py` 를 감싸는 얇은 함수 하나. 우리가 이미 갖고 있는 값들이 보고서에
그대로 들어맞는다.

```
model          "yolo-facev2+bytetrack"
model_version  가중치 해시 또는 릴리스 태그
masked         True
설정           method · conf · imgsz · pad · mosaic_scale · linger
검출 수        raw_boxes (얼굴)  ← 이미 있다
처리 프레임 수  frames            ← 이미 있다
```

**한 가지 정직하게 적어야 할 것**: 저쪽 보고서 예시는 `얼굴 N · 번호판 M` 이다.
**우리는 번호판을 안 한다.** 0 으로 조용히 채우면 "번호판도 봤는데 없었다" 로 읽힌다.
`plates: null` 이나 `"번호판: 미지원"` 으로 명시해야 한다.

### ④ 이미지 (CUDA 베이스)

`worker-io.Dockerfile` 은 `python:3.11-slim` + ffmpeg 이고 의존성은 `httpx` · `xxhash`
둘뿐이다. 우리는 torch · opencv · supervision 에 YOLO-FaceV2 리포까지 끌고 온다.

저쪽 문서도 이걸 예상해 뒀다 — *"그때 이 모듈은 자기 이미지(CUDA 베이스)로 옮겨가겠지만
계약은 안 바뀐다."* 그러니 새 Dockerfile 은 **우리가 내는 것**이 맞다.

### ⑤ 진행률 — 지금 계약으로는 못 준다 (저쪽에 요청할 것)

러너가 하트비트에 **`None`** 을 보낸다.

> *진행 초는 `None` 으로 보낸다 — 모델이 프레임 진행을 보고하기 전까지는 **모르는 것이
> 사실**이다. 0 을 보내면 화면이 "0%에서 멎었다"고 거짓말한다.*

맞는 판단이다. 그런데 **우리는 프레임 진행을 안다.** 그걸 넘기려면 어댑터 시그니처가
`adapter(src, dst, on_progress=None)` 로 늘어야 하는데, 그건 **저쪽 파일**이다.
우리가 못 고친다 — 요청 항목으로 남긴다.

### ⑥ 코딩 규약

우리 코드가 저쪽 저장소 안에 들어간다면 루트 `[tool.ruff]` 가 걸린다: `line-length = 100`,
`select = ["E","F","I","N","UP","B","SIM"]`, `target-version = "py311"`. `import-linter`
계약은 `root_packages = ["api_gateway"]` 라서 우리에겐 안 걸린다. mypy 는 `strict = true` 다.

별도 패키지(`packages/reborn-deident/`)로 들어가는 형태면 ruff 만 맞추면 된다.

### ⑦ 품질 정책이 두 벌이 된다 — 지금 로직이 이미 맞다

저쪽 파이프라인에서 우리 입력은 **이미 720p CRF 24 로 인코딩된 분석본**이고, 우리 출력에서
**480p 재생본이 파생된다**(L4). 즉 우리 출력 품질이 두 번 쓰인다.

여기서 어제 정한 규칙이 그대로 맞는다 — **`목표 = min(원본 비트레이트, 납품 목표)`**.
언론재단 납품(3000~3500 kbps)에서는 목표가 걸리고, rebornstudio 에서는 원본(CRF 24 산출)이
낮으므로 **올려 담지 않고 그대로 나간다.** 정책 하나로 둘 다 덮인다.

## 5. 안 건드려도 되는 것

- `service/` 전체 (FastAPI · 웹 UI · 게이트 · 검수 화면) — 저쪽이 자기 것을 갖고 있다
- `storage/` 전체 (S3Store · providers · 자격 증명) — presigned URL 이 대신한다
- `msa/` 워커 — 저쪽 Celery 가 그 역할을 한다
- 우리 작업 상태 기계 — 저쪽 `deident_status` 가 정본이다

**전부 오픈소스 단독 실행용으로는 그대로 산다.** 지우는 게 아니라 안 쓰이는 것뿐이다.

## 6. 다음에 할 일

| # | 할 일 | 막는 것 |
|---|---|---|
| 1 | 라이선스 방향 결정 (①~③ 중) | 코드 전부 |
| 2 | 파이썬 3.11 로 전체 검사 통과 | 통합 |
| 3 | 어댑터 함수 + 보고서 (번호판 미지원 명시) | — |
| 4 | CUDA 베이스 Dockerfile | 배포 |
| 5 | 어댑터에 `on_progress` 를 달아 달라고 요청 | 진행률 표시 |
| 6 | 흐린 얼굴이 MI 요약 품질을 얼마나 깎는지 실측 | 저쪽 Phase 4 첫 검증 항목 |

6번은 저쪽 설계 문서가 *"지금은 모델이 없어 측정할 것이 없다"* 라고 남겨 둔 자리다.
**모델을 가진 쪽이 우리니까 그 측정도 우리가 하게 된다.**
