"""명령줄 진입점 — **README 첫 예제가 실제로 도는가.**

이 파일이 생긴 이유가 있다. `core/` 로 폴더를 나눌 때 `cli.py` 의 임포트 하나가
옛 경로에 남았고, 그래서 `face-anonymize` 가 **통째로 죽어 있었다.**

    ModuleNotFoundError: No module named 'face_anonymizer.pipeline'

테스트 400개가 통과하는 동안 아무도 몰랐다. 그 임포트가 **함수 안에 있는 지연
임포트**여서다 — torch 를 `--help` 에서까지 끌고 오지 않으려고 그렇게 둔 것인데,
덕분에 모듈을 임포트해 보는 것으로는 안 걸린다. 그리고 테스트가 `main()` 을 한
번도 부르지 않았다.

여기서 지키는 것은 둘이다. 이 경로가 실제로 끝까지 도는 것, 그리고 같은 종류의
잔재가 패키지 어디에도 없는 것.
"""

import ast
import importlib.util
import pathlib
import sys

import pytest

from face_anonymizer import cli
from face_anonymizer.core.pipeline import Result, Timing, VideoInfo


class FakeAnonymizer:
    """모델을 안 든다. CLI 배선만 본다."""

    made = []
    seen = {}

    def __init__(self, **kw):
        FakeAnonymizer.made.append(kw)

    def process(self, src, out, **kw):
        FakeAnonymizer.seen.clear()
        FakeAnonymizer.seen.update(kw)
        return Result(output=out, frames=30, raw_boxes=30, filled_boxes=2,
                      method=kw.get("method", "mosaic"), audio="ok",
                      video=VideoInfo(fps=30.0, width=1280, height=720,
                                      frame_count=30),
                      timing=Timing(total=1.0, detect=0.5, track=0.1,
                                    render=0.3, audio=0.1),
                      detected_frames=30)


@pytest.fixture
def fake_pipeline(monkeypatch):
    """지연 임포트는 **부르는 순간** 풀리므로 원본 모듈을 갈아 끼우면 잡힌다."""
    import face_anonymizer.core.pipeline as P
    FakeAnonymizer.made = []
    FakeAnonymizer.seen = {}
    monkeypatch.setattr(P, "VideoAnonymizer", FakeAnonymizer)
    return FakeAnonymizer


def test_the_readme_first_example_actually_runs(tmp_path, monkeypatch,
                                                fake_pipeline):
    """`face-anonymize input.mp4` — 이게 안 돌면 다른 게 다 돌아도 소용없다."""
    src = tmp_path / "input.mp4"
    src.write_bytes(b"not really a video")      # 파이프라인은 가짜라 안 읽는다
    monkeypatch.setattr(sys, "argv",
                        ["face-anonymize", str(src), "-o", str(tmp_path / "o.mp4")])

    assert cli.main() == 0, "CLI 가 0 이 아닌 값으로 끝났다"
    assert fake_pipeline.made, "파이프라인이 만들어지지도 않았다"


def test_cli_options_reach_the_pipeline(tmp_path, monkeypatch, fake_pipeline):
    """인자를 받아 놓고 안 넘기면 조용히 기본값으로 처리된다."""
    src = tmp_path / "input.mp4"
    src.write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", [
        "face-anonymize", str(src), "-o", str(tmp_path / "o.mp4"),
        "--method", "box", "--conf", "0.15", "--imgsz", "1600",
        "--batch-size", "4", "--quiet"])

    assert cli.main() == 0
    seen = FakeAnonymizer.seen
    # 검출기 쪽으로 간 것
    assert fake_pipeline.made[0]["imgsz"] == 1600
    # 파이프라인 쪽으로 간 것
    assert seen["method"] == "box"
    assert seen["conf"] == 0.15
    assert seen["batch_size"] == 4
    assert seen["imgsz"] == 1600


def test_bad_arguments_do_not_crash(tmp_path, monkeypatch, fake_pipeline):
    """잘못된 인자는 예외가 아니라 종료 코드로 답해야 한다."""
    src = tmp_path / "input.mp4"
    src.write_bytes(b"x")

    def boom(self, *a, **kw):
        raise ValueError("mosaic_scale 은 0 과 1 사이여야 합니다: 2.0")

    monkeypatch.setattr(FakeAnonymizer, "process", boom)
    monkeypatch.setattr(sys, "argv",
                        ["face-anonymize", str(src), "--mosaic-scale", "2.0"])
    assert cli.main() == 2               # 2 = 잘못된 인자


def test_no_stale_relative_imports_anywhere(monkeypatch):
    """**지연 임포트는 임포트 검사에 안 걸린다.** 그래서 여기서 전수로 본다.

    패키지 안의 모든 상대 임포트를 훑어서 그 모듈이 실제로 존재하는지 확인한다.
    `cli.py` 를 고친 것은 한 건이지만, 함수 안에 숨은 상대 임포트가 스물아홉 개
    더 있다 — 폴더를 또 옮기면 그중 아무거나 같은 방식으로 죽는다.
    """
    root = pathlib.Path(__file__).resolve().parent.parent / "face_anonymizer"
    broken = []
    for f in sorted(root.rglob("*.py")):
        if "__pycache__" in str(f):
            continue
        rel = f.relative_to(root.parent).with_suffix("")
        mod = str(rel).replace("/", ".")
        pkg = mod if f.name == "__init__.py" else mod.rsplit(".", 1)[0]
        if f.name == "__init__.py":
            pkg = mod.rsplit(".", 1)[0] if mod.endswith(".__init__") else mod
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            up = pkg.split(".")
            if node.level > 1:
                up = up[:-(node.level - 1)]
            target = ".".join(up + ([node.module] if node.module else []))
            try:
                ok = importlib.util.find_spec(target) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                ok = False
            if not ok:
                broken.append(f"{f.name}:{node.lineno} → {target}")
    assert not broken, "없는 모듈을 가리키는 상대 임포트:\n  " + "\n  ".join(broken)
