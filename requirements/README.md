# requirements — 무엇을 깔아야 하는가

**쓰임새마다 다르게 깐다.** 한 파일에 몰아넣으면 큐 워커 이미지에 웹 프레임워크가
들어가고, 코어만 쓰려는 사람이 AWS SDK 를 받는다.

| 파일 | 언제 | 무엇이 들어 있나 |
|---|---|---|
| `base.txt` | 항상 | torch · opencv · supervision + 검출기 리포가 임포트하는 것들 |
| `serve.txt` | 단독 운영(웹 화면) | fastapi · uvicorn · boto3 |
| `worker.txt` | MSA 큐 워커 | httpx · celery[redis]. **fastapi 도 boto3 도 없다** |
| `dev.txt` | 테스트 | pytest · httpx |

```bash
pip install -r requirements/base.txt                          # CLI 만
pip install -r requirements/base.txt -r requirements/serve.txt # 서버
pip install -r requirements/base.txt -r requirements/worker.txt # 큐 워커
pip install -r requirements/dev.txt                            # + 테스트
```

## 왜 워커에 boto3 가 없나

그 컨테이너는 **버킷에 자기 열쇠로 붙지 않는다.** 서명된 URL 하나로만 드나들기
때문에 SDK 도 자격 증명도 필요 없다. 이미지에 비밀이 안 들어가는 이유가 여기
있다([연동 규약](../docs/integration/rebornstudio.md)).

## 왜 httpx 가 dev 에도 있나

`fastapi` 의 `TestClient` 가 내부적으로 쓴다. 서버를 **띄우는** 데는 필요 없어서
`serve.txt` 에는 없다. 없으면 `test_server.py` 가 통째로 skip 되는데, 그게 조용히
지나가면 서버 테스트가 안 돈다는 걸 아무도 모른다.

## ffmpeg 는 여기 없다

시스템 패키지다(오디오 합성용). `sudo apt install ffmpeg` 또는
`brew install ffmpeg`. `scripts/ec2_setup.sh` 가 EC2 에서는 같이 깔아 준다.

## 상한이 걸린 것

`supervision>=0.18,<0.30` — 0.30 에서 `sv.ByteTrack` 이 제거될 예정이다. 올릴 때는
`core/tracking.py` 의 트래커 교체가 함께 필요하다.

`base.txt` 뒤쪽의 pandas·matplotlib·seaborn·thop 은 **우리 코드가 쓰지 않는다.**
체크포인트를 unpickle 하려면 YOLO-FaceV2 리포의 `models/*`, `utils/*` 가 임포트돼야
하고 그 과정에서 필요해진다. 빼면 검출기 생성이 `ModuleNotFoundError` 로 실패한다.
