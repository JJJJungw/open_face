# face-anonymizer

영상 속 얼굴을 검출해 **모자이크/블러/박스**로 비식별화하는 파이프라인.
검출은 **YOLO-FaceV2**, 추적은 **ByteTrack**, 끊긴 프레임은 보간으로 메꿔 순간 누출을 막고, 원본 오디오는 유지한다.

- 검출: [YOLO-FaceV2](https://github.com/Krasjet-Yu/YOLO-FaceV2) (WIDERFace Hard ≈ 91.9) — clibdev 유지보수 포크 사용
- 추적: [supervision](https://github.com/roboflow/supervision) ByteTrack (MIT)
- 익명화: 모자이크(기본) / 가우시안 블러 / 단색 박스
- 입력 기준: 720p 30fps (다른 해상도도 동작)

## 동작 방식

```
검출(YOLO-FaceV2) → 추적(ByteTrack) → 트랙 보간 → 익명화 렌더 → 오디오 합성
```

프레임 단위 검출은 특정 프레임에서 얼굴을 순간적으로 놓칠 수 있고, 블러 파이프라인에서 그건 곧 프라이버시 누출이다. 그래서 ByteTrack 으로 프레임 간 트랙을 잇고, 관측된 프레임 사이의 빈 구간을 선형 보간으로 채우며, 마지막 관측 이후 몇 프레임 더 박스를 유지한다.

> **블러는 복원될 수 있다**: 약한 블러는 최근 디블러링 모델로 부분 복원될 여지가 있다. 강한 익명화가 필요하면 `--method mosaic`(강하게) 또는 `--method box` 를 쓰는 것이 안전하다.

## 설치

```bash
git clone <YOUR_REPO_URL> face-anonymizer
cd face-anonymizer

# (권장) 가상환경
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
# ffmpeg 는 시스템 패키지 (오디오 합성용)
#   Ubuntu/Debian: sudo apt install ffmpeg
#   macOS:         brew install ffmpeg

# YOLO-FaceV2 리포 클론 + 가중치 다운로드 (한 번만)
python setup_weights.py
```

`setup_weights.py` 는 `third_party/YOLO-FaceV2` 에 검출기 코드를, `weights/yolo-facev2.pt` 에 가중치를 준비한다. (둘 다 `.gitignore` 처리되어 저장소에는 커밋되지 않음)

## 사용법

### CLI

```bash
# 기본 (모자이크)
python -m face_anonymizer.cli input.mp4

# 옵션 지정
python -m face_anonymizer.cli input.mp4 -o out.mp4 \
    --method mosaic --imgsz 1280 --conf 0.20 --mosaic-scale 0.05 --pad 0.15

# 완전히 가리기
python -m face_anonymizer.cli input.mp4 --method box
```

`pip install -e .` 로 설치하면 `face-anonymize input.mp4` 로도 실행 가능.

### 파이썬 API

```python
from face_anonymizer import VideoAnonymizer

anonymizer = VideoAnonymizer()          # 기본 가중치/디바이스 자동
anonymizer.process("input.mp4", "output_anon.mp4",
                   method="mosaic", imgsz=960, conf=0.25,
                   mosaic_scale=0.06, pad=0.15, linger=5)
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
| `--no-interp` | off | 트랙 보간 끄기 |

## 프로젝트 구조

```
face-anonymizer/
├── face_anonymizer/
│   ├── detector.py     # YOLO-FaceV2 로더 + 범용 디코더
│   ├── tracking.py     # ByteTrack + 트랙 보간
│   ├── anonymize.py    # 모자이크/블러/박스
│   ├── pipeline.py     # 전체 파이프라인
│   └── cli.py          # 커맨드라인
├── notebooks/
│   └── colab_demo.ipynb
├── setup_weights.py    # 리포 클론 + 가중치 다운로드
├── requirements.txt
├── pyproject.toml
└── LICENSE             # GPL-3.0
```

## 라이선스

**GPL-3.0.** 검출기 YOLO-FaceV2(및 베이스 YOLOv5)가 GPL-3.0 이므로 파생물인 이 프로젝트도 GPL-3.0 으로 배포된다. 추적에 쓰는 supervision(ByteTrack)은 MIT 로 GPL 과 호환된다. 배포 전 `LICENSE` 파일에 GPL-3.0 전체 텍스트와 실제 저작자명을 채워 넣을 것.

## 크레딧

- YOLO-FaceV2 — Krasjet-Yu ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320324004655)), clibdev 포크
- ByteTrack — supervision / Roboflow
