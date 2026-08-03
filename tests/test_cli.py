"""CLI 테스트 — 인자 파싱과 종료 코드.

실제 검출기를 만들지 않도록 VideoAnonymizer 를 가짜로 갈아 끼운다.
CLI 가 넘기는 파라미터 이름이 파이프라인 시그니처와 어긋나는 사고를 잡는 게 목적이다.
"""

import io

import pytest

from conftest import FakeDetector
from face_anonymizer import cli
from face_anonymizer.pipeline import VideoOpenError


@pytest.fixture
def patched_cli(monkeypatch):
    """VideoAnonymizer 를 가짜 검출기 버전으로 대체하고 호출 인자를 기록한다."""
    from face_anonymizer import pipeline as pl

    seen = {}
    real = pl.VideoAnonymizer

    class Spy(real):
        def __init__(self, detector=None, **kw):
            seen["detector_kwargs"] = kw
            super().__init__(detector=FakeDetector((320, 240)))

        def process(self, *a, **kw):
            seen["process_kwargs"] = kw
            return super().process(*a, **kw)

    monkeypatch.setattr(pl, "VideoAnonymizer", Spy)
    return seen


def test_defaults_are_forwarded(patched_cli, make_video, tmp_path):
    path, _, _ = make_video(frames=6)
    out = tmp_path / "o.mp4"
    assert cli.main([path, "-o", str(out)]) == 0
    assert out.exists()

    kw = patched_cli["process_kwargs"]
    assert kw["method"] == "mosaic"
    assert kw["imgsz"] == 960
    assert kw["detect_every"] == 1
    assert kw["interp"] is True
    assert kw["keep_audio"] is True


def test_options_are_forwarded(patched_cli, make_video, tmp_path):
    path, _, _ = make_video(frames=6)
    out = tmp_path / "o.mp4"
    code = cli.main([
        path, "-o", str(out), "--method", "box", "--imgsz", "1280",
        "--conf", "0.15", "--detect-every", "2", "--batch-size", "4",
        "--linger", "9", "--no-audio", "--device", "cpu",
    ])
    assert code == 0
    kw = patched_cli["process_kwargs"]
    assert kw["method"] == "box"
    assert (kw["imgsz"], kw["conf"]) == (1280, 0.15)
    assert (kw["detect_every"], kw["batch_size"]) == (2, 4)
    assert kw["linger"] == 9
    assert kw["keep_audio"] is False
    assert patched_cli["detector_kwargs"]["device"] == "cpu"


def test_default_output_path(patched_cli, make_video):
    path, _, _ = make_video(name="clip.mp4", frames=5)
    assert cli.main([path]) == 0
    import os
    assert os.path.exists(path.replace(".mp4", "_anon.mp4"))


def test_missing_input_returns_1(patched_cli, tmp_path, capsys):
    assert cli.main([str(tmp_path / "nope.mp4")]) == 1
    assert "error:" in capsys.readouterr().err


def test_detect_every_without_interp_is_rejected(patched_cli, make_video):
    """SystemExit(2) — argparse.error 경로."""
    path, _, _ = make_video(frames=4)
    with pytest.raises(SystemExit) as e:
        cli.main([path, "--detect-every", "3", "--no-interp"])
    assert e.value.code == 2


def test_bad_method_is_rejected_by_argparse(make_video):
    path, _, _ = make_video(frames=3)
    with pytest.raises(SystemExit) as e:
        cli.main([path, "--method", "pixelate"])
    assert e.value.code == 2


def test_version_flag():
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0


def test_half_tri_state():
    p = cli.build_parser()
    assert p.parse_args(["x.mp4"]).half is None          # 미지정 → 자동
    assert p.parse_args(["x.mp4", "--half"]).half is True
    assert p.parse_args(["x.mp4", "--no-half"]).half is False


# ------------------------------------------------------------------ 진행률 바

def test_progress_bar_silent_when_not_a_tty():
    buf = io.StringIO()                 # isatty() False
    bar = cli.ProgressBar(True, stream=buf)
    for i in range(10):
        bar("detect", i, 10)
    assert buf.getvalue() == ""


def test_progress_bar_writes_when_tty():
    class Tty(io.StringIO):
        def isatty(self):
            return True

    buf = Tty()
    bar = cli.ProgressBar(True, stream=buf)
    bar("detect", 5, 10)
    bar("detect", 10, 10)
    text = buf.getvalue()
    assert "50%" in text and "100%" in text


def test_progress_bar_handles_zero_total():
    class Tty(io.StringIO):
        def isatty(self):
            return True

    cli.ProgressBar(True, stream=Tty())("detect", 0, 0)      # ZeroDivision 금지
