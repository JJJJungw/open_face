"""`.env` 읽기 테스트.

설정 파일은 조용히 틀리면 제일 나쁘다 — 값이 안 먹었는데 서버는 잘 뜬다.
그래서 '무엇이 이기는가' 를 여기서 못 박는다.
"""

import os

from face_anonymizer import env


def test_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    """export 로 준 값을 파일이 덮으면, 왜 안 바뀌는지 한참 찾게 된다."""
    f = tmp_path / ".env"
    f.write_text("FA_CRF=23\nFA_NEW_ONE=hello\n", encoding="utf-8")
    monkeypatch.setenv("FA_CRF", "19")

    applied = env.load(str(f))

    assert os.environ["FA_CRF"] == "19"          # 손으로 준 것이 이긴다
    assert os.environ["FA_NEW_ONE"] == "hello"   # 빈 자리는 채운다
    assert applied == ["FA_NEW_ONE"]


def test_override_can_be_forced(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("FA_CRF=23\n", encoding="utf-8")
    monkeypatch.setenv("FA_CRF", "19")

    env.load(str(f), override=True)

    assert os.environ["FA_CRF"] == "23"


def test_missing_file_is_not_an_error(tmp_path):
    assert env.load(str(tmp_path / "없는파일")) == []


def test_parses_comments_quotes_and_export():
    text = '\n'.join([
        "# 주석",
        "",
        "FA_A=1",
        "  FA_B = two  ",
        'FA_C="세 번째"',
        "FA_D='네 번째'",
        "export FA_E=5",
        "이건 = 아닌 줄이 아니다",     # '=' 가 있으므로 통과한다
        "쓰레기줄",
        "=값만있음",
    ])
    got = env.parse(text)
    assert got["FA_A"] == "1"
    assert got["FA_B"] == "two"           # 좌우 공백 제거
    assert got["FA_C"] == "세 번째"       # 따옴표 제거
    assert got["FA_D"] == "네 번째"
    assert got["FA_E"] == "5"             # export 접두 허용
    assert "쓰레기줄" not in got
    assert "" not in got


def test_explicit_path_wins_over_search(tmp_path, monkeypatch):
    f = tmp_path / "다른이름.env"
    f.write_text("FA_FROM_EXPLICIT=1\n", encoding="utf-8")
    monkeypatch.setenv("FA_ENV_FILE", str(f))
    assert env.find() == str(f)


def test_env_file_pointing_at_nothing_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("FA_ENV_FILE", str(tmp_path / "없다"))
    assert env.find() is None
