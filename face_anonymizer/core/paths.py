"""기본 경로 — 가중치와 YOLO-FaceV2 리포가 어디 있는가.

**torch 를 끌고 오지 않는다.** 경로 하나 알자고 2GB 짜리를 임포트할 수는 없다.
가중치를 갖춰 놓는 쪽(service/worker)이나 준비 스크립트가 이 값만 필요하다.

리포 루트는 **패키지의 부모**로 계산한다. `..` 를 세는 방식은 파일이 폴더를
옮길 때 조용히 어긋난다 — 실제로 detector.py 가 core/ 로 내려가면서 패키지
**안쪽**을 가리키게 됐고, 가중치가 face_anonymizer/weights/ 에 받아진 채
서버는 "리포가 없다" 로 기동에서 터졌다.
"""

import os

_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_PACKAGE)

# 컨테이너에서는 가중치를 볼륨에 두는 게 흔하다. 그때는 환경 변수로 돌린다.
DEFAULT_REPO = os.environ.get("FA_REPO_DIR") or os.path.join(
    ROOT, "third_party", "YOLO-FaceV2")
DEFAULT_WEIGHTS = os.environ.get("FA_WEIGHTS") or os.path.join(
    ROOT, "weights", "yolo-facev2.pt")
