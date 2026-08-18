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


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """**테스트가 GitHub 을 치면 안 된다.**

    기본이 `auto` 라 버킷에서 못 받으면 공개 릴리스로 넘어간다. 그걸 막지
    않으면 테스트가 40MB 를 받고, 인터넷이 없는 곳에서는 그냥 실패한다.
    여기서 끊고, 넘어가는 동작 자체는 아래에서 따로 본다.
    """
    from face_anonymizer.storage import transfer

    def blocked(url, dest, **kw):
        raise transfer.TransferError(f"테스트에서는 바깥으로 안 나간다: {url}",
                                     transient=False)
    monkeypatch.setattr(transfer, "fetch", blocked)


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
        store.ensure(str(p), key=KEY, source="s3")

    assert not p.exists()
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".weights-")]
    assert leftovers == [], leftovers


def test_missing_key_says_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setattr(s3mod, "get_store", lambda: make_store({}))
    p = tmp_path / "w.pt"

    with pytest.raises(store.WeightsUnavailable) as e:
        store.ensure(str(p), key=KEY, source="s3")

    assert KEY in str(e.value)
    assert not p.exists()


def test_without_s3_the_message_points_at_the_setup_script(tmp_path, monkeypatch):
    monkeypatch.setattr(s3mod, "get_store", lambda: None)

    with pytest.raises(store.WeightsUnavailable) as e:
        store.ensure(str(tmp_path / "w.pt"), source="s3")

    assert "setup_weights.py" in str(e.value)
    assert "FA_S3_BUCKET" in str(e.value)


# ── 남의 버킷에는 가중치가 없다 ──────────────────────────────────────────────
#
# 저장소를 고를 수 있게 만들어 놓고 모델은 우리 버킷에 묶어 두면 앞의 노력이
# 통째로 무의미하다. 첫 실행 화면을 통과시켜 놓고 첫 영상에서 멎기 때문이다.

def test_auto_falls_through_to_the_public_release(tmp_path, monkeypatch):
    """버킷에 없으면 **멈추지 않고** 공개 릴리스로 간다."""
    from face_anonymizer.storage import transfer
    monkeypatch.setattr(s3mod, "get_store", lambda: make_store({}))   # 빈 버킷
    seen = {}

    def fake_fetch(url, dest, **kw):
        seen["url"] = url
        with open(dest, "wb") as f:
            f.write(BLOB)
    monkeypatch.setattr(transfer, "fetch", fake_fetch)

    p = tmp_path / "w.pt"
    store.ensure(str(p), key=KEY, source="auto")

    assert p.read_bytes() == BLOB
    assert "YOLO-FaceV2" in seen["url"]           # 업스트림 릴리스를 그대로 가리킨다


def test_auto_still_prefers_the_bucket(tmp_path, monkeypatch):
    """우리 EC2 는 예전과 똑같이 버킷에서 받아야 한다 — 배포마다 GitHub 을
    치면 레이트 리밋에 걸리고 태그가 바뀌면 다른 파일을 받는다."""
    from face_anonymizer.storage import transfer
    monkeypatch.setattr(s3mod, "get_store", lambda: make_store({KEY: (BLOB, NOW)}))

    def boom(url, dest, **kw):
        raise AssertionError("버킷에 있는데 바깥으로 나가면 안 된다")
    monkeypatch.setattr(transfer, "fetch", boom)

    p = tmp_path / "w.pt"
    store.ensure(str(p), key=KEY, source="auto")
    assert p.read_bytes() == BLOB


def test_when_everything_fails_it_lists_every_way_out(tmp_path, monkeypatch):
    """세 갈래를 다 시도하고 실패한 상황이다. 하나만 알려 주면 그것만 붙들게 된다."""
    monkeypatch.setattr(s3mod, "get_store", lambda: None)

    with pytest.raises(store.WeightsUnavailable) as e:
        store.ensure(str(tmp_path / "w.pt"), source="auto")

    msg = str(e.value)
    assert "FA_WEIGHTS_URL" in msg and "setup_weights.py" in msg
    assert "버킷" in msg


def test_status_tells_the_screen_what_will_happen(tmp_path):
    """첫 실행에서 저장소만 정해 주고 모델은 말 안 해 주면, 통과한 사람이
    첫 영상에서야 문제를 만난다."""
    p = tmp_path / "w.pt"
    assert store.status(str(p))["present"] is False
    assert "내려받습니다" in store.status(str(p))["detail"]

    p.write_bytes(BLOB)
    st = store.status(str(p))
    assert st["present"] is True and st["size_mb"] == 2.0


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
