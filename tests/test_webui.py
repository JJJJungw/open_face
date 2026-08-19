"""웹 화면 — **버킷에서 온 이름이 코드가 되지 않는가.**

파일 이름은 우리가 만드는 값이 아니다. 버킷에 있는 것을 그대로 받아 그린다.
그런데 그 값을 `onclick="toggle('<이름>')"` 처럼 **JS 문자열 안에** 끼워 넣고
있었다. 그러면 이름이 곧 코드가 된다.

먼저 밟히는 건 보안이 아니라 기능이다. `KBS 뉴스 O'Brien 인터뷰.mp4` 처럼
아포스트로피가 든 정상적인 이름이 들어오면 구문이 깨져서 **체크박스가 눌려도
목록에 안 들어간다.** 사용자는 제출 버튼이 왜 계속 비활성인지 알 수 없다.

여기서 지키는 것은 둘이다. 이스케이프가 실제로 다섯 글자를 다 바꾸는 것과,
버킷에서 값이 오는 렌더 경로에 인라인 핸들러가 없는 것.
"""

import json
import pathlib
import re
import shutil
import subprocess

import pytest

HTML = (pathlib.Path(__file__).resolve().parent.parent
        / "face_anonymizer" / "service" / "static" / "index.html")
node_only = pytest.mark.skipif(not shutil.which("node"), reason="node 없음")


def source():
    return HTML.read_text(encoding="utf-8")


def scripts():
    return "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", source(), re.S))


@node_only
def test_esc_covers_every_character_that_can_break_out():
    """`'` 를 빼먹으면 onclick="f('...')" 안에서 그대로 뚫린다.

    예전에는 `& < > "` 넷만 바꿨다. 지금은 인라인 핸들러를 걷어냈지만,
    이스케이프가 좁으면 다음에 누가 하나 추가할 때 같은 구멍이 다시 생긴다.
    """
    fn = re.search(r"function esc\(s\) \{.*?\n\}", scripts(), re.S)
    assert fn, "esc() 를 찾지 못했다"
    hostile = """<img src=x onerror=alert(1)>'"&"""
    out = subprocess.run(
        ["node", "-e", fn.group(0) + f"\nconsole.log(esc({json.dumps(hostile)}))"],
        capture_output=True, text=True, check=True).stdout.strip()

    # `&` 는 빼고 본다 — 이스케이프 **결과**가 `&lt;` 라 당연히 들어 있다.
    for ch in "<>\"'":
        assert ch not in out, f"{ch!r} 가 그대로 남았다: {out}"
    assert "&lt;img" in out and "&#39;" in out


@node_only
def test_a_hostile_filename_makes_no_tag_and_no_attribute():
    """이스케이프한 이름은 **속성 값과 텍스트로만** 산다.

    문자열로 `onmouseover=` 가 들어 있는 것은 상관없다 — 그건 파일 이름의 일부다.
    문제는 그게 **실제 속성이 되는 것**이라, 문자열을 찾지 말고 파서에게 물어본다.
    """
    from html.parser import HTMLParser

    fn = re.search(r"function esc\(s\) \{.*?\n\}", scripts(), re.S).group(0)
    name = """v1/input/<script>alert(1)</script>'onmouseover='x.mp4"""
    js = fn + f"""
const k = {json.dumps(name)};
console.log(`<input type="checkbox" data-pick="${{esc(k)}}">` +
            `<span title="${{esc(k)}}">${{esc(k)}}</span>`);
"""
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True,
                         check=True).stdout

    seen, text = [], []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            seen.append((tag, sorted(k for k, _ in attrs)))

        def handle_data(self, data):
            text.append(data)

    P().feed(out)

    assert seen == [("input", ["data-pick", "type"]), ("span", ["title"])], seen
    # 이름은 통째로 **텍스트**로 살아 있어야 한다 — 가려지지도, 잘리지도 않는다.
    assert name in "".join(text)


# 버킷에서 값이 흘러드는 렌더 경로. 여기에는 인라인 핸들러를 두지 않는다.
BUCKET_FACING = ("crumbs", "renderBrowser", "renderProgress", "loadLogBatches")


def _body(name, text):
    """`function name(...) {` 부터 짝이 맞는 `}` 까지."""
    m = re.search(r"function %s\s*\([^)]*\)\s*\{" % re.escape(name), text)
    assert m, f"{name}() 를 찾지 못했다"
    depth, i = 0, m.end() - 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[m.start():i + 1]
        i += 1
    raise AssertionError(f"{name}() 의 끝을 못 찾았다")


@pytest.mark.parametrize("fname", BUCKET_FACING)
def test_bucket_facing_renderers_use_data_attributes(fname):
    """**이름을 코드가 아니라 데이터로 넘긴다.**

    `data-*` 는 브라우저가 속성 값으로만 다루므로, 이름이 무엇이든 HTML 문법에
    영향을 못 준다. 인라인 핸들러는 그 반대다 — 값이 실행 문맥에 들어간다.
    """
    body = _body(fname, scripts())
    inline = re.findall(r'\bon[a-z]+\s*=\s*"[^"]*"', body)
    assert not inline, (
        f"{fname}() 에 인라인 핸들러가 남아 있다: {inline}\n"
        "버킷에서 오는 값은 data-* 로 넘기고 이벤트 위임으로 받아야 한다.")
