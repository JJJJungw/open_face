"""YOLO-FaceV2 리포 클론 + 가중치 준비.

    python scripts/setup_weights.py            준비 (S3 우선, 없으면 GitHub)
    python scripts/setup_weights.py --upload   지금 있는 가중치를 S3 에 올린다

가중치는 **S3 를 먼저 본다.** 배포할 때마다 GitHub 릴리스에 의존하면 레이트
리밋에 걸리고, 네트워크 정책에 막히고, 업스트림 태그가 바뀌면 어제와 다른
파일을 받는다. 버킷에 없거나 S3 가 설정되어 있지 않으면 GitHub 로 물러선다 —
처음 한 번은 어딘가에서 받아 와야 하고, 그게 곧 `--upload` 로 올릴 파일이다.

리포는 그대로 GitHub 에서 클론한다. 바꾸는 범위를 가중치 하나로 좁힌다.
"""

import argparse
import os
import subprocess
import sys
import urllib.request

# 이 파일은 scripts/ 안에 있고 받아 두는 자리는 저장소 루트다. 한 칸 올라간다 —
# 여기를 scripts/ 로 두면 third_party 와 weights 가 scripts/ 밑에 생겨서 런타임이
# 보는 자리(core/paths.py 의 DEFAULT_WEIGHTS)와 어긋난다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.join(ROOT, "third_party", "YOLO-FaceV2")
WEIGHTS = os.path.join(ROOT, "weights", "yolo-facev2.pt")   # paths.DEFAULT_WEIGHTS 와 같은 자리

REPO_URL = os.environ.get("FA_WEIGHTS_REPO") or \
    "https://github.com/clibdev/YOLO-FaceV2.git"

# 가중치 주소는 **여기 안 적는다.** `storage/weights.py` 의 PUBLIC_URL 하나가
# 정본이고, 그게 `FA_WEIGHTS_URL` 을 본다. 예전에는 같은 주소가 두 곳에 있었고
# 이쪽만 환경 변수를 안 봤다 — 사내 미러를 넣어도 이 폴백만 엉뚱한 데로 갔다.
# 지금은 안 터지지만, 주소를 바꾸는 날 한 곳만 바뀐다(docs/issues/014 의 패턴).

sys.path.insert(0, ROOT)


def ensure_repo():
    os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)
    if os.path.isdir(REPO_DIR):
        print("리포 있음 :", REPO_DIR)
        return
    print(f"리포 클론 : {REPO_URL}")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)


def ensure_weights():
    from face_anonymizer.storage import weights as store

    if store.looks_complete(WEIGHTS):
        print(f"가중치 있음: {WEIGHTS} ({os.path.getsize(WEIGHTS)/1e6:.1f} MB)")
        return

    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    try:
        store.ensure(WEIGHTS)
        print("가중치 준비: S3 에서 받음")
        return
    except store.WeightsUnavailable as e:
        print(f"S3 에서 받지 못함 — GitHub 릴리스로 받습니다.\n  ({e})")

    # **주소는 store 가 갖고 있다.** 여기서 다시 적으면 두 곳이 되고, 그때부터
    # `FA_WEIGHTS_URL` 로 미러를 넣어도 이 줄만 안 따라온다.
    print(f"내려받는 중: {store.PUBLIC_URL}")
    urllib.request.urlretrieve(store.PUBLIC_URL, WEIGHTS)
    print("가중치 준비: 공개 릴리스에서 받음")
    print("  → 다음부터 S3 에서 받으려면: python scripts/setup_weights.py --upload")


def upload():
    """지금 있는 가중치를 S3 에 올린다. 한 번만 하면 된다."""
    from face_anonymizer.storage import s3 as s3mod
    from face_anonymizer.storage import weights as store

    if not store.looks_complete(WEIGHTS):
        sys.exit(f"올릴 가중치가 없습니다: {WEIGHTS}")
    bucket = s3mod.get_store()
    if bucket is None:
        sys.exit("S3 가 설정되어 있지 않습니다. FA_S3_BUCKET 을 설정해 주세요.")
    key = store.WEIGHTS_KEY
    print(f"올리는 중: {WEIGHTS} → s3://{bucket.bucket}/{key}")
    bucket.upload(WEIGHTS, key)
    print("완료. 이제 새 서버는 여기서 받습니다.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upload", action="store_true",
                    help="지금 있는 가중치를 S3 에 올린다")
    # 컨테이너 빌드용. **리포는 건너뛸 수 없다** — 체크포인트를 unpickle 하려면
    # 그 리포의 모듈이 임포트돼야 해서, 없으면 모델 로드가 통째로 실패한다.
    # 가중치는 볼륨이나 S3 로 나중에 넣을 수 있으므로 이쪽만 뺀다.
    ap.add_argument("--repo-only", action="store_true",
                    help="검출기 리포만 준비하고 가중치는 건너뛴다")
    args = ap.parse_args()

    if args.upload:
        return upload()

    ensure_repo()
    if args.repo_only:
        print(f"\n리포만 준비했습니다. 가중치는 따로 넣어 주세요: {WEIGHTS}")
        return
    ensure_weights()
    size = os.path.getsize(WEIGHTS) / 1e6 if os.path.exists(WEIGHTS) else 0
    print(f"\n준비 완료. 가중치 {size:.1f} MB")


if __name__ == "__main__":
    main()
