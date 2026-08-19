# scripts — 준비·배포 스크립트

**손으로 한 번씩 돌리는 것들.** 서버가 기동 중에 부르지 않는다.

| 파일 | 하는 일 |
|---|---|
| `setup_weights.py` | YOLO-FaceV2 리포 클론 + 가중치 준비 |
| `ec2_setup.sh` | 새 EC2 부트스트랩 + 스모크 테스트 (Ubuntu) |

---

## setup_weights.py

```bash
python scripts/setup_weights.py            # 준비 (버킷 우선, 없으면 GitHub)
python scripts/setup_weights.py --upload   # 지금 있는 가중치를 버킷에 올린다
```

받아 두는 자리는 **저장소 루트**다(`third_party/YOLO-FaceV2`, `weights/yolo-facev2.pt`).
둘 다 gitignore 라 저장소에는 안 들어간다.

가중치는 버킷을 먼저 본다 — 배포마다 GitHub 릴리스에 의존하면 레이트 리밋에
걸리고, 네트워크 정책에 막히고, 업스트림 태그가 바뀌면 어제와 다른 파일을 받는다.
버킷에 없거나 저장소가 설정돼 있지 않으면 GitHub 로 물러선다.

**이 스크립트를 안 돌려도 서버는 산다.** 런타임도 같은 순서로 가중치를 조달할 줄
안다(`storage/weights.py`) — 한동안 준비 스크립트만 폴백을 알고 런타임은 몰라서,
남의 버킷으로 붙은 사람은 첫 영상에서 멎었다
([issues/012](../docs/issues/012-the-model-was-tied-to-our-bucket.md)).

리포 클론까지는 아직 런타임이 안 한다. 서버가 기동 중에 `git clone` 을 하는 것은
그것대로 놀라운 일이라, 여기서만 한다.

## ec2_setup.sh

```bash
bash scripts/ec2_setup.sh
```

`$HOME/face-anonymizer` 를 전제하고 여섯 단계를 돈다 — 환경 확인 → 시스템
패키지(ffmpeg·libgl) → venv + torch(GPU 유무로 인덱스를 바꾼다) → 나머지 의존성 →
단위 테스트 → **실모델 스모크**.

마지막 단계는 합성 영상 60프레임을 실제 가중치로 통과시킨다. 여기서 검출 0건은
정상이다 — 그린 것이 원이지 얼굴이 아니다. 이 단계가 보는 것은 **배선**이다.
mp4v 인코더가 있는지, CUDA 가 잡히는지, ffmpeg 가 결과를 다시 뜨는지.
