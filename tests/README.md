# tests — 회귀 테스트

**가중치도 torch 도 GPU 도 없이 40초 안에 도는 것**이 이 폴더의 설계 목표다.
가짜 검출기를 주입하고 얼굴 위치를 아는 합성 영상을 쓰므로, 모델 정확도가
아니라 **배선**을 검증한다. 정확도는 가중치가 필요한 별개 관심사다.

```bash
pip install -r requirements/base.txt -r requirements/serve.txt \
            -r requirements/worker.txt -r requirements/dev.txt
pytest                      # 402개, 약 1분
```

**`dev.txt` 만 깔면 402개 중 247개가 조용히 빠진다.** 서버·S3·저장소 설정
201개는 fastapi 와 httpx 를, 큐 워커 46개는 celery 를 `importorskip` 으로
확인한다. 없으면 실패가 아니라 skip 이라 초록색으로 끝나는데, 정작 그 기계에서
띄울 서버는 한 줄도 검증되지 않은 상태다. 무엇이 왜 빠졌는지 보려면 `-rs` 를
붙인다.

```bash
pytest -rs                  # 건너뛴 것과 그 사유를 끝에 모아 준다
```

`pytest -q` 는 쓰지 말 것. `addopts` 에 이미 `-q` 가 있어서 `-qq` 가 되고,
그러면 "402 passed" 요약 줄까지 사라져 점만 찍히고 끝난다.

## 여기서 잡은 것들

테스트가 왜 있는지는 목록이 제일 잘 설명한다. 전부 실제로 통과하던 코드에서
나온 사고다.

- `-shortest` 때문에 20초/600프레임 결과물이 10초/300프레임으로 잘렸다.
  ffmpeg 는 리턴코드 0 을 줬고 파일도 멀쩡해 보였다.
- ByteTrack 점수 정규화가 없어 추적이 한 프레임 만에 죽었다.
- 목록 API 가 O(N²) 라 500건에서 폴링이 느려졌다.
- 배치 크기 비교가 인코더 설정에 따라 깨졌다 — 그리고 그 원인이 테스트
  헬퍼가 OpenCV 내부 버퍼를 그대로 들고 있던 것이었다.
- 결과물이 입력 폴더에 섞였을 때 다시 큐에 들어갔다.

## 파일

| 파일 | 무엇을 지키는가 |
|---|---|
| `conftest.py` | 합성 영상 · 가짜 검출기 · 프레임 읽기 헬퍼 |
| `test_pipeline.py` | 전 구간 스모크. 모든 프레임의 얼굴이 실제로 가려졌는가 |
| `test_tracking.py` | 추적 연결과 보간 |
| `test_audio_mux.py` | 오디오 합성이 영상을 자르지 않는가 |
| `test_verification.py` | 디코딩 완결성 · 검출 신뢰도 · 납품 해상도/비트레이트 |
| `test_orientation.py` | 회전 메타가 붙은 영상 |
| `test_naming.py` | 이름 규칙과 결과 키 변환 |
| `test_s3.py` | 제출 진입점 · 중복 방지 · presigned URL (가짜 S3 클라이언트) |
| `test_server.py` | 큐 · 재시도 · 취소 · 오류 응답 |
| `test_ingest.py` | AV1 등 OpenCV 가 못 읽는 입력의 정규화 |
| `test_metrics.py` | 큐 지표와 폴더별 진척률 |
| `test_storage_setup.py` | 첫 실행 관문 · 저장소 연결/해제 · 입력 검증 |
| `test_storage_contract.py` | 제공자를 갈아 끼워도 계약이 지켜지는가 |
| `test_weights.py` | 가중치 조달 세 갈래(있음 → 버킷 → 공개 릴리스) |
| `test_job_runner.py` | 잡 페이로드 한 장으로 도는 러너 · OOM 재시도 |
| `test_msa_worker.py` | 큐에서 꺼내 오는 껍데기 · 펜싱 토큰 |
| `test_msa_journal.py` | 큐 경로의 이벤트 기록 |
| `test_env.py` | `.env` 해석 · **설정이 문서와 갈라지지 않는가** |

## 방식

고칠 때는 **되돌려서 깨지는지**까지 본다. 테스트를 먼저 쓰고, 고치고, 고친
것을 되돌려 그 테스트가 실제로 실패하는지 확인한 다음 커밋한다. 통과하는
테스트가 아무것도 지키지 않는 경우를 막는 유일한 방법이다.

## 골라서 돌리기

```bash
pytest tests/test_s3.py                  # 파일 하나
pytest -k "retr or permanent"            # 이름으로
pytest -x                                # 첫 실패에서 멈춤
pytest --lf                              # 지난번 실패한 것만
```

## 무엇을 가짜로 두는가

| 진짜로 쓰는 것 | 가짜로 두는 것 | 왜 |
|---|---|---|
| OpenCV · ffmpeg | 검출기 (`FakeDetector`) | 가중치 2GB 와 GPU 없이 배선을 본다 |
| 파이프라인 전 구간 | S3 (`FakeS3Client`) | 네트워크 없이 키 계산과 중복 판정을 본다 |
| FastAPI 라우팅 | — | 실제 앱에 요청을 넣는다 |

그래서 **모델 정확도는 여기서 보지 않는다.** 얼굴 위치를 아는 합성 영상에
가짜 검출기를 물려 "검출 → 추적 → 보간 → 렌더 → 인코딩" 이 어긋나지 않는지만
확인한다. 정확도는 가중치가 필요한 별개 관심사다.
