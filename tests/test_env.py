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
    # **납품 대역(3000~3500 kbps)을 못 박는 자리는 여기 하나다.** 다른 테스트는
    # params.DEFAULTS 를 보고, 여기만 실제 숫자를 본다 — 대역이 바뀌면 계약이
    # 바뀐 것이므로 테스트 하나가 반드시 걸려야 한다.
    assert got["FA_TARGET_BITRATE"] == "3200k"      # CBR 목표 = 대역 한가운데
    assert got["FA_MIN_BITRATE"] == "3000k"
    assert got["FA_MAX_BITRATE"] == "3500k"
    assert got["FA_S3_ROOT_PREFIX"] == "v1/input/"
    for key, value in got.items():
        assert "#" not in value, f"{key} 에 주석이 섞였다: {value!r}"


# ── 문서와 코드가 갈라지지 않게 ─────────────────────────────────────────────

def test_every_setting_is_documented():
    """**코드가 읽는데 `.env.example` 에 없으면 아무도 그 값을 모른다.**

    설정은 늘어나기만 하고 문서는 손으로 따라가야 해서 반드시 갈라진다. 실제로
    여섯 개가 빠져 있었다(docs/issues/015). 사람이 기억하는 대신 여기서 센다.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    used = {}
    for base in ("face_anonymizer", "tools", "scripts"):
        p = root / base
        # 자리가 옮겨 가면 rglob 이 조용히 빈 목록을 준다 — 그러면 이 테스트가
        # 통과하면서 아무것도 안 센다. 없으면 여기서 터뜨린다.
        assert p.exists(), f"훑을 자리가 없다: {base}"
        files = [p] if p.is_file() else [f for f in p.rglob("*.py")
                                         if "__pycache__" not in str(f)]
        for f in files:
            for m in re.finditer(r'["\'](FA_[A-Z0-9_]+)["\']',
                                 f.read_text(encoding="utf-8")):
                used.setdefault(m.group(1), f.relative_to(root))

    doc = set(re.findall(r"^#?(FA_[A-Z0-9_]+)=",
                         (root / ".env.example").read_text(encoding="utf-8"), re.M))

    # 이 파일을 찾는 데 쓰이는 값이라 이 파일에 못 적는다 — 닭과 달걀이다.
    missing = {k: v for k, v in used.items() if k not in doc and k != "FA_ENV_FILE"}
    assert not missing, "\n".join(f"  {k}  ({v})" for k, v in sorted(missing.items()))

    # 반대쪽도 본다. 안 읽는 값이 문서에 남아 있으면 넣어도 아무 일이 없다.
    stale = doc - set(used) - {"FA_ENV_FILE"}
    assert not stale, sorted(stale)


def test_the_python_version_is_written_in_one_voice():
    """`pyproject.toml` 과 `.python-version` 이 같은 말을 하는가.

    같은 사실이 두 곳에 적혀 있으면 언젠가 갈라진다 — 이 저장소가 그걸로 이미
    여러 번 물렸다(docs/issues/014 · 023). 여기서는 특히 나쁜데, 갈라져도
    **아무 증상이 없다가** 배포한 파이썬이 개발한 파이썬과 달라지는 것으로만
    드러난다. 그게 docs/issues/026 이었다.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    pin = (root / ".python-version").read_text(encoding="utf-8").strip()
    spec = re.search(r'^requires-python\s*=\s*"([^"]+)"',
                     (root / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    assert spec, "requires-python 을 찾지 못했다"
    lo = re.search(r">=\s*(\d+\.\d+)", spec.group(1))
    hi = re.search(r"<\s*(\d+)\.(\d+)", spec.group(1))
    assert lo and hi, f"하한과 상한이 둘 다 있어야 한다: {spec.group(1)}"
    assert lo.group(1) == pin, f".python-version={pin} 인데 하한은 {lo.group(1)}"
    # 상한은 바로 다음 마이너여야 한다 — 3.11 을 고정한다는 뜻이다.
    major, minor = pin.split(".")
    assert (hi.group(1), hi.group(2)) == (major, str(int(minor) + 1)), spec.group(1)
