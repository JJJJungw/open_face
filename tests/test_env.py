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


def test_trailing_comment_is_not_part_of_the_value():
    """.env.example 을 그대로 복사해 쓰라고 해놓고 파싱이 안 되면 안 된다.

    실제로 겪었다 — FA_OUTPUT_HEIGHT 값이 '720          # 짧은 변 상한...' 이
    되어 int() 에서 터졌다.
    """
    got = env.parse("FA_OUTPUT_HEIGHT=720          # 짧은 변 상한. 0 이면 원본 유지")
    assert got["FA_OUTPUT_HEIGHT"] == "720"


def test_hash_inside_a_value_survives():
    """비밀번호 같은 데 그냥 붙어 있는 # 은 주석이 아니다."""
    assert env.parse("FA_X=abc#def")["FA_X"] == "abc#def"


def test_quotes_protect_everything_inside():
    got = env.parse('FA_X="값 # 안의 우물정"   # 이건 주석')
    assert got["FA_X"] == "값 # 안의 우물정"


def test_the_shipped_example_actually_parses():
    """.env.example 의 모든 활성 줄이 읽히는지 — 복사해서 쓰는 파일이다."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, ".env.example"), encoding="utf-8") as fh:
        got = env.parse(fh.read())
    assert got["FA_OUTPUT_HEIGHT"] == "720"
    assert got["FA_TARGET_BITRATE"] == "3500k"
    assert got["FA_MAX_BITRATE"] == "4000k"
    assert got["FA_S3_ROOT_PREFIX"] == "v1/input/"
    for key, value in got.items():
        assert "#" not in value, f"{key} 에 주석이 섞였다: {value!r}"
