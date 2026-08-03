"""YOLO-FaceV2 리포 클론 + 가중치 다운로드.

한 번만 실행하면 third_party/YOLO-FaceV2 (model/utils 코드) 와
weights/yolo-facev2.pt 가 준비된다.

    python setup_weights.py
"""

import os
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(ROOT, "third_party", "YOLO-FaceV2")
WEIGHTS = os.path.join(ROOT, "weights", "yolo-facev2.pt")

REPO_URL = "https://github.com/clibdev/YOLO-FaceV2.git"
WEIGHTS_URL = "https://github.com/clibdev/YOLO-FaceV2/releases/latest/download/yolo-facev2.pt"


def main():
    os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)

    if not os.path.isdir(REPO_DIR):
        print(f"cloning {REPO_URL} ...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
    else:
        print("repo already present:", REPO_DIR)

    if not (os.path.exists(WEIGHTS) and os.path.getsize(WEIGHTS) > 1_000_000):
        print(f"downloading weights → {WEIGHTS}")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    else:
        print("weights already present:", WEIGHTS)

    size = os.path.getsize(WEIGHTS) / 1e6 if os.path.exists(WEIGHTS) else 0
    print(f"ready. weights {size:.1f} MB")


if __name__ == "__main__":
    main()
