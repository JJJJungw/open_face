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

## HTTP API + 웹 UI

```bash
pip install -r requirements.txt -r requirements-serve.txt
uvicorn face_anonymizer.server:app --host 0.0.0.0 --port 8000
```

브라우저로 열면 영상을 끌어다 놓고 진행률·fps·남은 시간을 보면서 처리하고 결과를 내려받을 수 있다.

| 엔드포인트 | 설명 |
|---|---|
| `GET /` | 웹 UI |
| `GET /api/health` | 모델 로드 여부, 디바이스, 큐 상태 |
| `POST /api/jobs` | 영상 업로드 → `202` + 작업 id (multipart) |
| `GET /api/jobs` | 작업 목록 |
| `GET /api/jobs/{id}` | 진행률 · 단계 · fps · ETA · 완료 시 통계 |
| `GET /api/jobs/{id}/download` | 결과 영상 |
| `DELETE /api/jobs/{id}` | 작업과 파일 삭제 (진행 중이면 409) |

```bash
curl -F file=@in.mp4 -F method=mosaic -F conf=0.4 -F batch_size=32 \
     http://localhost:8000/api/jobs
curl http://localhost:8000/api/jobs/<id>
curl -O -J http://localhost:8000/api/jobs/<id>/download
```

**추론은 한 번에 하나만 돈다.** GPU 한 장에 검출기 하나를 올려 두고 워커 스레드 하나가 큐를 소비한다. 요청마다 스레드를 띄우면 VRAM 이 터지거나 서로 느려지기만 하고, 총 처리량은 오히려 직렬화하는 쪽이 높다.

환경 변수로 `FA_DEVICE`, `FA_IMGSZ`, `FA_JOBS_DIR`, `FA_MAX_UPLOAD_MB`(기본 2048), `FA_JOB_TTL_MIN`(기본 120, 완료 후 자동 삭제)을 조정한다.

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

가장 위험한 실패는 "빈 결과물이 성공으로 보고되는 것"이다. 영상을 못 열거나 인코더를 못 잡으면 예외를 던지고(`VideoOpenError` / `VideoWriteError`), 디코딩 크기가 컨테이너 메타와 다르거나 1·2차 패스의 프레임 수가 어긋나도 실패로 처리한다. 익명화를 무력화하는 파라미터(음수 `pad`, `mosaic_scale >= 1`)도 입구에서 거부한다.

반대로 오디오 합성 실패는 결과물을 버릴 이유가 아니다. ffmpeg 가 없거나 실패해도 익명화된 영상은 출력 경로에 남기고 `Result.audio` 로 이유를 알린다.

서버(`server.py`)도 같은 원칙이다. 작업이 실패해도 상태와 사유를 `status=error` 로 남기고 프로세스는 계속 산다.

## 알려진 제약

프레임 스킵(N 프레임마다 검출)은 만들었다가 걷어냈다. 커버리지가 대상의 움직임에 좌우돼서, 무작위 보행 영상 100개 중 33개에서 얼굴이 노출되고도 파이프라인은 정상 종료했다. 조용히 새는 손잡이보다 느린 편이 낫다고 판단했다. 처리량은 배치 추론과 FP16 으로 올린다.

`supervision` 0.30 에서 `sv.ByteTrack` 이 제거될 예정이라 `<0.30` 상한을 걸어 뒀다. 올릴 때는 `tracking.py` 의 트래커 교체가 필요하다.

영상을 두 번 읽는다(1차 검출/추적, 2차 렌더). 프레임을 메모리에 쌓지 않는 대신 디스크 I/O 를 두 배 쓴다.

`Result.filled_boxes` 가 `raw_boxes` 에 비해 유난히 크면 검출이 자주 끊긴다는 신호다. `--conf` 를 낮추거나 `--imgsz` 를 올리는 게 낫다.

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
