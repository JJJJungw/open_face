"""가중치 확보 테스트.

여기가 조용히 틀리면 제일 나쁘다 — 받다 만 파일이 제자리에 남으면 다음 기동은
"있다" 로 판정하고, 그때 나는 오류는 체크포인트 언피클 실패라 원인이 전혀
드러나지 않는다.
"""

import os

import pytest

from face_anonymizer.storage import s3 as s3mod
from face_anonymizer.storage import weights as store

from test_s3 import NOW, FakeS3Client            # noqa: E402

KEY = "v1/model/yolo-facev2.pt"
BLOB = b"x" * 2_000_000                          # 그럴듯한 크기


def make_store(objects):
    return s3mod.S3Store(bucket="b", client=FakeS3Client(objects),
                         output_prefix="v1/results/face/", root_prefix="")


def test_existing_weights_are_not_downloaded_again(tmp_path, monkeypatch):
    """이미 있으면 네트워크 호출조차 없어야 한다 — 매 기동마다 40MB 를 받을 수 없다."""
    p = tmp_path / "w.pt"
    p.write_bytes(BLOB)

    def boom():
        raise AssertionError("S3 를 보면 안 된다")
    monkeypatch.setattr(s3mod, "get_store", boom)

    assert store.ensure(str(p)) == str(p)


def test_downloads_from_s3_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(s3mod, "get_store", lambda: make_store({KEY: (BLOB, NOW)}))
    p = tmp_path / "w.pt"

    store.ensure(str(p), key=KEY)

    assert p.read_bytes() == BLOB


def test_partial_download_is_not_left_behind(tmp_path, monkeypatch):
    """중간에 끊긴 파일이 제자리에 남으면 다음 기동이 '있다' 로 속는다."""
    monkeypatch.setattr(s3mod, "get_store",
                        lambda: make_store({KEY: (b"tiny", NOW)}))
    p = tmp_path / "w.pt"

    with pytest.raises(store.WeightsUnavailable):
        store.ensure(str(p), key=KEY)

    assert not p.exists()
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".weights-")]
    assert leftovers == [], leftovers


def test_missing_key_says_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setattr(s3mod, "get_store", lambda: make_store({}))
    p = tmp_path / "w.pt"

    with pytest.raises(store.WeightsUnavailable) as e:
        store.ensure(str(p), key=KEY)

    assert KEY in str(e.value)
    assert not p.exists()


def test_without_s3_the_message_points_at_the_setup_script(tmp_path, monkeypatch):
    monkeypatch.setattr(s3mod, "get_store", lambda: None)

    with pytest.raises(store.WeightsUnavailable) as e:
        store.ensure(str(tmp_path / "w.pt"))

    assert "setup_weights.py" in str(e.value)
    assert "FA_S3_BUCKET" in str(e.value)


def test_weights_failure_is_classified_as_model_load_failed():
    """작업이 실패했을 때 '왜' 가 코드로 남아야 한다."""
    from face_anonymizer.service import errors
    p = errors.classify(store.WeightsUnavailable("없다"))
    assert p.code == "model_load_failed"


# ── 기본 경로 ────────────────────────────────────────────────────────────────
#
# 리팩토링으로 detector.py 가 core/ 로 내려가면서 `..` 가 한 단계 모자라
# 패키지 안쪽(face_anonymizer/weights/)을 가리켰다. 서버는 기동에서 터졌고,
# 그전에 가중치는 엉뚱한 곳에 받아져 있었다. 경로를 테스트로 못 박는다.

def test_default_paths_point_at_the_repo_root():
    import os

    from face_anonymizer.core import paths

    root = os.path.dirname(os.path.dirname(os.path.abspath(paths.__file__)))
    root = os.path.dirname(root)                     # face_anonymizer/ 의 부모

    assert paths.ROOT == root
    assert paths.DEFAULT_WEIGHTS == os.path.join(root, "weights", "yolo-facev2.pt")
    assert paths.DEFAULT_REPO == os.path.join(root, "third_party", "YOLO-FaceV2")
    # 패키지 안쪽을 가리키면 안 된다
    assert os.path.join("face_anonymizer", "weights") not in paths.DEFAULT_WEIGHTS
    assert os.path.join("face_anonymizer", "third_party") not in paths.DEFAULT_REPO


def test_paths_can_be_overridden_by_env(monkeypatch):
    """컨테이너에서는 가중치를 볼륨에 두는 게 흔하다."""
    import importlib

    monkeypatch.setenv("FA_WEIGHTS", "/models/w.pt")
    monkeypatch.setenv("FA_REPO_DIR", "/opt/yolo")
    from face_anonymizer.core import paths
    importlib.reload(paths)
    try:
        assert paths.DEFAULT_WEIGHTS == "/models/w.pt"
        assert paths.DEFAULT_REPO == "/opt/yolo"
    finally:
        monkeypatch.undo()
        importlib.reload(paths)
