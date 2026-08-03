# face-anonymizer

영상 속 얼굴을 검출해 **모자이크/블러/박스**로 비식별화하는 파이프라인.
검출은 **YOLO-FaceV2**, 추적은 **ByteTrack**, 끊긴 프레임은 보간으로 메꿔 순간 누출을 막고, 원본 오디오는 유지한다.

- 검출: [YOLO-FaceV2](https://github.com/Krasjet-Yu/YOLO-FaceV2) (WIDERFace Hard ≈ 91.9) — clibdev 유지보수 포크 사용
- 추적: [supervision](https://github.com/roboflow/supervision) ByteTrack (MIT)
- 익명화: 모자이크(기본) / 가우시안 블러 / 단색 박스
- 인터페이스: 파이썬 API · CLI · HTTP API(FastAPI)
- 입력 기준: 720p 30fps (다른 해상도도 동작)

## 동작 방식

```
검출(YOLO-FaceV2) → 추적(ByteTrack) → 트랙 보간 → 익명화 렌더 → 오디오 합성
```

프레임 단위 검출은 특정 프레임에서 얼굴을 순간적으로 놓칠 수 있고, 블러 파이프라인에서 그건 곧 프라이버시 누출이다. 그래서 ByteTrack 으로 프레임 간 트랙을 잇고, 관측된 프레임 사이의 빈 구간을 선형 보간으로 채우며, 마지막 관측 이후 몇 프레임 더 박스를 유지한다. 마지막 관측 이후에는 대상 위치를 알 수 없으므로 박스를 고정하지 않고 최근 이동 속도에 비례해 넓힌다 — 방향을 추측해 틀리는 것보다 양쪽으로 여유를 두는 편이 누출 관점에서 안전하다.

> **블러는 복원될 수 있다**: 약한 블러는 최근 디블러링 모델로 부분 복원될 여지가 있다. 강한 익명화가 필요하면 `--method mosaic`(강하게) 또는 `--method box` 를 쓰는 것이 안전하다.

## 설치

```bash
git clone https://github.com/JJJJungw/open_face.git face-anonymizer
cd face-anonymizer

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 서빙까지 하려면 requirements-serve.txt

# ffmpeg 는 시스템 패키지 (오디오 합성용)
#   Ubuntu/Debian: sudo apt install ffmpeg
#   macOS:         brew install ffmpeg

python setup_weights.py                # 검출기 리포 클론 + 가중치 (한 번만)
```

`setup_weights.py` 는 `third_party/YOLO-FaceV2` 에 검출기 코드를, `weights/yolo-facev2.pt` 에 가중치를 준비한다. 둘 다 `.gitignore` 처리되어 저장소에는 커밋되지 않는다.

## 테스트

**가중치도 torch 도 없이 파이프라인 전 구간이 검증된다.** 검출기를 주입할 수 있게 만들어 두었기 때문에, 얼굴 위치를 이미 아는 합성 영상 + 가짜 검출기로 "검출 → 추적 → 보간 → 렌더 → 오디오" 배선을 결정적으로 확인한다.

```bash
pip install -r requirements-dev.txt    # torch 없음, 설치 10초 남짓
pytest
```

여기서 검증하는 것은 모델 정확도가 아니라 파이프라인이다. 특히 다음을 회귀 테스트로 고정해 두었다.

검출기가 프레임 12~16 을 통째로 놓쳐도 보간이 그 구간을 덮는다는 것(이 프로젝트의 핵심 주장), 그리고 보간을 끄면 실제로 노출된다는 음성 대조. 익명화 여부는 평균 색이 아니라 라플라시안 분산(고주파 디테일)으로 판정한다 — 모자이크는 영역 평균 색을 유지하므로 표준편차로는 가려졌는지 알 수 없다. 그 밖에 프레임 스킵 구간이 보간으로 덮이는지, 배치 크기가 결과를 바꾸지 않는지, 깨진 입력·없는 코덱·경로에 `.mp4` 가 섞인 디렉터리·동시 실행 충돌에서 조용히 실패하지 않는지, 그리고 HTTP API 의 작업 수명주기와 업로드 파일명 경로 탈출까지 포함한다.

모델 정확도(WIDERFace 성능)는 별도 관심사로 분리했다. 가중치가 필요하고 CI 를 무겁게 만들며, 파이프라인 회귀와 섞으면 무엇이 깨졌는지 알기 어려워진다.

## 사용법

### CLI

```bash
# 기본 (모자이크)
face-anonymize input.mp4

# 옵션 지정
face-anonymize input.mp4 -o out.mp4 \
    --method mosaic --imgsz 1280 --conf 0.20 --mosaic-scale 0.05 --pad 0.15

# 완전히 가리기
face-anonymize input.mp4 --method box

# GPU 처리량 우선 (3프레임마다 검출, 16장씩 배치)
face-anonymize input.mp4 --detect-every 3 --batch-size 16
```

`pip install -e .` 전이라면 `python -m face_anonymizer.cli` 로도 실행된다.

### 파이썬 API

```python
from face_anonymizer import VideoAnonymizer

anonymizer = VideoAnonymizer()          # 가중치/디바이스 자동, 프로세스당 1개 재사용
res = anonymizer.process(
    "input.mp4", "output_anon.mp4",
    method="mosaic", imgsz=960, conf=0.25,
    detect_every=1, batch_size=8,
    progress=lambda stage, done, total: print(stage, done, total),
)
print(res.frames, res.raw_boxes, res.filled_boxes, res.audio)
```

`process()` 는 `Result` 를 돌려준다: 렌더한 프레임 수, 모델이 실제 검출한 박스 수, 보간으로 채운 박스 수, 오디오 합성 상태, 입력 영상 메타데이터.

### HTTP API

```bash
pip install -r requirements-serve.txt
uvicorn face_anonymizer.server:app --host 0.0.0.0 --port 8000
```

```bash
# 작업 등록 (202 + job_id)
curl -F file=@input.mp4 -F method=mosaic -F batch_size=16 localhost:8000/jobs

# 진행 상황 폴링
curl localhost:8000/jobs/<job_id>

# 결과 내려받기
curl -o out.mp4 localhost:8000/jobs/<job_id>/result
```

| 엔드포인트 | 설명 |
| --- | --- |
| `GET /healthz` | 모델 로드 상태·큐 길이. 준비 전이면 503 |
| `POST /jobs` | 영상 업로드 → 202 + `job_id` |
| `GET /jobs/{id}` | 상태·단계·진행률·결과 요약 |
| `GET /jobs/{id}/result` | 익명화된 영상 |
| `DELETE /jobs/{id}` | 취소 또는 정리 |
| `GET /jobs` | 최근 작업 목록 |

설계상 모델은 프로세스당 한 번 로드하고 기동 시 warmup 까지 돌린다. GPU 작업은 워커 스레드 하나가 직렬로 처리한다 — 요청마다 스레드를 띄우면 VRAM 이 터지고 배치 추론 이득도 사라지기 때문이다. 처리량은 워커 수가 아니라 `FA_BATCH_SIZE` / `FA_DETECT_EVERY` 로 올린다.

**인증은 없다.** 외부에 노출한다면 앞단에 리버스 프록시를 두고 인증·업로드 크기 제한·레이트 리밋을 거는 것을 전제로 한다.

환경변수: `FA_WORKDIR`, `FA_MAX_UPLOAD_MB`(512), `FA_JOB_TTL_SEC`(3600), `FA_QUEUE_MAX`(32), `FA_EAGER_LOAD`(1), `FA_DEVICE`, `FA_BATCH_SIZE`(8), `FA_DETECT_EVERY`(1), 그리고 `FA_METHOD`/`FA_IMGSZ`/`FA_CONF`/`FA_PAD`/`FA_MOSAIC_SCALE`/`FA_LINGER` 기본값.

### Docker (GPU)

```bash
docker build -f docker/Dockerfile -t face-anonymizer .
docker run --gpus all -p 8000:8000 \
    -v $PWD/weights:/app/weights -v $PWD/third_party:/app/third_party \
    face-anonymizer
```

가중치와 검출기 리포는 이미지에 굽지 않고 볼륨으로 넣는다.

## 주요 파라미터

| 옵션 | 기본 | 설명 |
| --- | --- | --- |
| `--method` | `mosaic` | `mosaic` / `blur` / `box` |
| `--imgsz` | `960` | 추론 해상도. 720p 작은 얼굴이면 `1280` 권장 |
| `--conf` | `0.25` | 낮출수록 재현율↑(누출↓)·오탐↑. 프라이버시 우선이면 `0.15~0.20` |
| `--mosaic-scale` | `0.06` | 낮출수록 블록 커짐(더 강하게 가림) |
| `--pad` | `0.15` | 박스 확장 비율(턱/헤어라인 커버) |
| `--linger` | `5` | 트랙 소실 후 박스 유지 프레임 수 |
| `--detect-every` | `1` | N 프레임마다 검출. 사이 구간은 보간이 덮는다 |
| `--batch-size` | `1` | 한 번에 모델에 넣을 프레임 수. GPU 에서는 8~32 권장 |
| `--half` / `--no-half` | 자동 | FP16 추론. CUDA 에서 기본 활성화 |
| `--no-interp` | off | 트랙 보간 끄기 |
| `--no-audio` | off | 오디오 합성 생략 |

`--detect-every` 는 GPU 시간을 거의 선형으로 줄여 주지만, 검출을 건너뛴 프레임은 보간만이 덮는다. 그래서 `--no-interp` 와 같이 쓰는 것은 파이프라인 단계에서 거부한다.

## 실패했을 때의 동작

비식별화 파이프라인에서 가장 위험한 실패는 "빈 결과물이 성공으로 보고되는 것"이다. 그래서 영상을 못 열거나 인코더를 못 잡으면 예외를 던지고(`VideoOpenError` / `VideoWriteError`), 0프레임이나 빈 파일이 나오면 실패로 처리한다.

반대로 오디오 합성 실패는 결과물을 버릴 이유가 되지 않는다. ffmpeg 가 없거나 실패해도 익명화된 영상 자체는 출력 경로에 남기고 `Result.audio` 로 이유를 알린다. 중간 산출물은 출력 파일 옆 임시 디렉터리에 만들어 같은 폴더를 노리는 동시 작업끼리 충돌하지 않는다.

## 프로젝트 구조

```
face-anonymizer/
├── face_anonymizer/
│   ├── geometry.py     # letterbox / 좌표 역변환 (torch 불필요)
│   ├── detector.py     # YOLO-FaceV2 로더 + 배치 추론 디코더
│   ├── tracking.py     # ByteTrack + 트랙 보간
│   ├── anonymize.py    # 모자이크/블러/박스
│   ├── pipeline.py     # 전체 파이프라인
│   ├── cli.py          # 커맨드라인
│   └── server.py       # FastAPI HTTP API
├── tests/              # 가중치·torch 없이 도는 테스트
├── docker/Dockerfile   # GPU 서빙 이미지
├── notebooks/
│   └── colab_demo.ipynb
├── setup_weights.py    # 리포 클론 + 가중치 다운로드
└── LICENSE             # GPL-3.0
```

## 알려진 제약

`supervision` 0.29 에서 `sv.ByteTrack` 이 deprecated 되었고 0.30 에서 제거 예정이라 `<0.30` 으로 상한을 걸어 두었다. 올릴 때는 `tracking.py` 의 트래커 교체가 함께 필요하다.

파이프라인은 영상을 두 번 읽는다(1차 검출/추적, 2차 렌더). 프레임을 메모리에 쌓지 않는 대신 디스크 I/O 를 두 배로 쓰는 트레이드오프다.

`Result.filled_boxes` 가 `raw_boxes` 에 비해 비정상적으로 크면 검출이 자주 끊기고 있다는 신호다. `--conf` 를 낮추거나 `--imgsz` 를 올리는 편이 낫다.

## 라이선스

**GPL-3.0.** 검출기 YOLO-FaceV2(및 베이스 YOLOv5)가 GPL-3.0 이므로 파생물인 이 프로젝트도 GPL-3.0 으로 배포된다. 추적에 쓰는 supervision(ByteTrack)은 MIT 로 GPL 과 호환된다. `LICENSE` 의 저작권자 표기(`<YOUR NAME OR ORG>`)는 배포 전 실제 저작자명으로 교체할 것.

## 크레딧

- YOLO-FaceV2 — Krasjet-Yu ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320324004655)), clibdev 포크
- ByteTrack — supervision / Roboflow
