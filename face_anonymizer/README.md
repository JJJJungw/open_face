# face_anonymizer — 패키지 지도

**얼굴이 둘이고 몸은 하나다.**

```
service/  우리가 서버다 — 사람이 웹 화면으로 일을 시킨다
msa/      우리가 소비자다 — 남의 큐에서 일을 꺼내 온다
              ↘         ↙
               core/  영상 처리 (fastapi·boto3 를 모른다)
```

둘은 입구만 다르고 처리하는 몸은 같다. 처리 기본값도 `params.py` 한 벌을 나눠
쓴다 — 두 벌로 두었더니 큐 경로가 조용히 다른 설정으로 돌고 있었다
([issues/009](../docs/issues/009-queue-path-ran-untuned.md)).

---

## 의존은 한 방향이다

`service` 와 `storage` 는 `core` 를 쓰지만 **`core` 는 둘을 모른다.** 이걸 지키면
셋이 가능해진다 — 코어만 쓰는 사람에게 웹 프레임워크와 AWS SDK 를 강요하지 않고,
테스트가 가짜 검출기를 꽂아 torch 없이 전 구간을 돌고, 배치 실행기를 붙일 때 코어만
떼어 낼 수 있다.

`boto3` 는 `storage` 안에서도 지연 임포트다. S3 를 안 쓰면 설치할 필요가 없다.

## 폴더

| 폴더 | 하는 일 |
|---|---|
| [`core/`](core/README.md) | 파일 하나 → 파일 하나. 검출 · 추적 · 보간 · 렌더 · 인코딩 |
| [`service/`](service/README.md) | HTTP API · 웹 UI · 작업 큐 · 검수 · 지표 |
| [`msa/`](msa/README.md) | 큐를 지켜보는 워커. 인바운드 포트도 자격 증명도 없다 |
| [`storage/`](storage/README.md) | 저장소 고르기 · 파일 이름 규칙 · 가중치 조달 |

## 폴더에 안 들어간 것들

두 얼굴이 **같이 쓰기 때문에** 위로 올라온 모듈들이다. 한쪽에 두면 다른 쪽이
그쪽을 임포트하게 되고, 그 순간 의존 방향이 무너진다.

| 파일 | 하는 일 |
|---|---|
| `params.py` | 처리 파라미터 기본값 — **두 진입점의 단일 출처** |
| `job_runner.py` | 잡 페이로드 한 장 = 일 한 건. 우리 서버를 모른다 |
| `progress.py` | 진행률 계산. 화면과 하트비트가 **같은 자로 잰다** |
| `events.py` | 이벤트 저널(JSONL) — 기계가 읽는 기록. 사람 로그와 별개다 |
| `env.py` | `.env` 읽기 + `flag()` (참/거짓 환경 변수 해석 한 벌) |
| `timefmt.py` | 시각 표기. 서버는 UTC, 사람은 KST |
| `logsetup.py` | 로깅 설정 — **진입점에서 한 번만.** 라이브러리 코드는 안 건드린다 |
| `cli.py` | 명령줄 진입점 (`face-anonymize`) |

---

## 처리 파라미터

**호출하는 쪽은 입력만 주면 된다.** 튜닝된 값은 서비스가 들고 있어야지, 호출자마다
들고 다니면 어느 설정으로 처리됐는지가 호출 지점마다 달라진다.

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
| `height` | `720` (0=원본) | `FA_OUTPUT_HEIGHT` |
| `bitrate` (CBR 목표) | `3200k` | `FA_TARGET_BITRATE` |
| `min_bitrate` / `max_bitrate` | `3000k` / `3500k` | `FA_MIN_BITRATE` / `FA_MAX_BITRATE` |
| `crf` / `bitrate_ratio` | `23` / `1.0` | `FA_CRF` / `FA_BITRATE_RATIO` |

**납품 대역(3000~3500 kbps)은 강제한다.** `-b:v` 는 목표 평균이지 하한이 아니라서,
정지에 가까운 컷은 인코더가 얼마든지 아낀다 — 예전 설정에서 실측 14 kbps 가
나왔다. 그래서 목표를 대역 한가운데 두고 CBR 스터핑을 켠다. 단순한 장면에 비트를
낭비하는 것은 의도한 대가다. 아껴서 미달하는 것이 곧 반려이기 때문이다.
그러고도 결과물을 다시 재서 대역을 벗어나면 검수로 넘긴다
([issues/018](../docs/issues/018-the-delivery-band-was-not-enforced.md)).

`min_bitrate` 는 잡이 못 바꾼다. 대역은 계약이지 요청마다 고를 수 있는 값이 아니다.

`batch_size` 는 **배포마다 정해야 한다.** 32 는 개발기(L40S 45GB) 기준이고
T4(16GB)나 L4(24GB)에서 32/1280 은 OOM 이 날 수 있다. 그래도 모자라면 잡 러너가
절반씩 줄여 가며 다시 시도한다 — 조용히 실패하지 않게 하려는 것이지 튜닝을
대신하는 것은 아니다. 줄여서 돌면 그만큼 느리다.

운영 중 조정은 환경 변수로 하고, 필요할 때만 요청에서 개별 항목을 덮는다(보낸
것만 덮이고 나머지는 기본값). 현재 값은 `GET /api/defaults` 로 확인한다 — 웹 UI 도
컨트롤 초깃값을 여기서 받아 가므로 서버 설정과 화면이 어긋나지 않는다.

**잡 페이로드가 아무 값이나 덮을 수는 없다.** `JOB_OVERRIDABLE` 에 있는 것만
받는다. 임의의 파이프라인 인자를 페이로드로 넘기게 두면 계약이 없는 것과 같다.

## 파이썬으로 직접 쓰기

```python
from face_anonymizer import VideoAnonymizer

anonymizer = VideoAnonymizer()          # 가중치·디바이스 자동, 재사용 가능
res = anonymizer.process("input.mp4", "output_anon.mp4",
                         method="mosaic", conf=0.25, batch_size=8)
print(res.frames, res.raw_boxes, res.filled_boxes, res.audio)
```

`VideoAnonymizer` 는 모델을 들고 있으므로 **한 번 만들어 재사용한다.** 매번 새로
만들면 편마다 수십 초를 로딩에 쓴다.

패키지를 임포트하면 `.env` 를 **먼저** 읽는다. 설정 모듈들이 임포트 시점에
`os.environ` 을 읽기 때문에 그 뒤에 채우면 아무 효과가 없다.
