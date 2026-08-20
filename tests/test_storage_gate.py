"""첫 관문 — **순서가 곧 안내다.**

깡통 서버에 이걸 처음 올린 사람이 보는 화면이다. 예전 순서는 버킷 이름이
첫 칸이고, 자격 증명은 일곱 번째 줄에 상태만 한 줄, 키 칸은 맨 밑이었다.
그러면 위에서부터 채우다가 열쇠 없이 저장을 누르게 되고, 돌아오는 말은
"리전 설정과 네트워크 상태를 확인해 주세요" 였다 — 리전도 망도 멀쩡한데.

여기서 지키는 것은 셋이다.

1. **자격 증명 칸이 버킷 칸보다 위에 있다.** 붙을 수 있어야 주소가 의미가 있다.
2. **안내는 제공자가 선언한 사실이다.** 화면이 AWS 를 특별대우하지 않는다 —
   키 없이 붙는 길이 AWS 에만 있다는 게 사실일 뿐이고, NCP 를 고르면 화면이
   "키가 있어야 붙습니다" 로 바뀐다.
3. **자격 증명이 이미 잡혀 있어도 키 칸을 숨기지 않는다.** 잡힌 것이 고른
   제공자의 것이라는 보장이 없다.
"""

import json
import pathlib
import re
import shutil
import subprocess

import pytest

from face_anonymizer.storage import providers

HTML = (pathlib.Path(__file__).resolve().parent.parent
        / "face_anonymizer" / "service" / "static" / "index.html")
node_only = pytest.mark.skipif(not shutil.which("node"), reason="node 없음")

# 관문을 그리는 데 필요한 것 전부. 화면 전체를 브라우저에 띄우지 않고 이
# 함수들만 떼어 실제로 돌린다 — 문자열을 눈으로 읽는 검사는 순서가 바뀌어도
# 통과한다.
NEEDED = ("esc", "stepLabel", "credHint", "credHomes", "credHomeList",
          "credHelp", "credFields", "weightsLine", "storageForm")


def _js():
    src = HTML.read_text(encoding="utf-8")
    js = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", src, re.S))
    out = []
    for name in NEEDED:
        m = re.search(r"^function %s\(.*?^\}" % name, js, re.S | re.M)
        assert m, f"{name}() 를 찾지 못했다 — 이름이 바뀌면 이 검사가 조용히 죽는다"
        out.append(m.group(0))
    return "\n".join(out)


def render(provider="s3", present=False, secure=True):
    """관문 하나를 실제로 그려서 HTML 문자열로 돌려준다."""
    d = {
        "current": {"provider": provider, "name": providers.get(provider)["name"],
                    "bucket": "", "region": None, "endpoint": None,
                    "root_prefix": "", "output_prefix": "v1/results/face/"},
        "providers": providers.listing(),
        "credentials": {"source": "~/.aws/credentials" if present else "없습니다",
                        "present": present},
        "weights": {"present": True, "size_mb": 40},
        "first_run": True,
    }
    harness = f"""
globalThis.window = {{ isSecureContext: {json.dumps(secure)} }};
{_js()}
const d = {json.dumps(d, ensure_ascii=False)};
let html = storageForm(d, {{ gate: true }});
// fillProvider 가 나중에 채우는 자리들이다. 여기서는 직접 넣어 같은 화면을 만든다.
const p = d.providers.find(x => x.id === {json.dumps(provider)});
html = html.replace('<div class="meta" id="stcredhint" style="margin-top:4px"></div>',
  '<div class="meta" id="stcredhint">' + credHint(p, d.credentials) + '</div>');
html = html.replace('<div id="stcredhelp"></div>',
  '<div id="stcredhelp">' + credHelp(p, d.credentials) + '</div>');
console.log(html);
"""
    r = subprocess.run(["node", "-e", harness],
                       capture_output=True, text=True, check=True)
    return r.stdout


@node_only
def test_credentials_come_before_the_bucket():
    """**이게 이 변경의 전부다.** 순서가 되돌아가면 여기서 걸린다."""
    html = render()
    cred = html.index('id="stcredhint"')
    keys = html.index('id="stak"')
    bucket = html.index('id="stbucket"')
    assert cred < keys < bucket, (
        "자격 증명이 버킷 칸보다 뒤에 있다 — 처음 받은 사람은 버킷부터 채우고 "
        "열쇠 없이 저장을 누른다")


@node_only
def test_the_three_steps_are_numbered_in_order():
    """번호가 없으면 사람은 그냥 위에서부터 채운다."""
    html = render()
    # 단계 번호만 센다. 안내판 안에도 번호가 붙은 항목이 있어서, 그냥
    # `>(\d)</b>` 로 훑으면 그것까지 섞인다 — 처음에 그렇게 짰다가 걸렸다.
    order = re.findall(r'color:var\(--faint\)">(\d)</b>', html)
    assert order == ["1", "2", "3"], order


@node_only
def test_aws_says_keyless_is_possible_and_ncp_says_it_is_not():
    """**제공자가 선언한 사실을 그대로 그린다.**

    AWS 만 무키 경로가 있다. 그건 우리 편향이 아니라 그 제공자의 사실이고,
    NCP 를 고른 사람에게 IAM 역할 이야기를 하면 그건 그냥 틀린 말이다.
    """
    aws, ncp = render("s3"), render("ncp")
    assert "IAM 역할" in aws
    assert "키 없이 붙는 길이 없습니다" in ncp
    assert "IAM 역할" not in ncp, "NCP 화면에 AWS 이야기가 남아 있다"
    # 발급처도 제공자마다 다르다.
    assert "IAM 콘솔" in aws and "인증키 관리" in ncp


@node_only
def test_key_fields_stay_open_even_when_credentials_are_already_found():
    """잡힌 열쇠가 **고른 제공자의 것이라는 보장이 없다.**

    `~/.aws/credentials` 가 있는 기계에서 NCP 를 고르면 예전 화면은 "이미
    있어서 따로 넣지 않으셔도 됩니다" 를 띄웠다. 그대로 저장하면 거절당하고,
    그때 사람은 이미 "안 넣어도 된다" 는 말을 들은 뒤라 열쇠를 마지막에 의심한다.
    """
    html = render("ncp", present=True)
    assert 'id="stak"' in html and 'id="stsk"' in html
    assert "따로 넣지 않으셔도" not in html
    assert "(선택)" in html, "이미 있을 때는 선택이라는 표시가 있어야 한다"


@node_only
def test_plain_http_still_refuses_to_take_keys():
    """평문에서 열쇠 칸을 여는 것은 안전하게 다뤘다는 인상만 준다."""
    html = render(secure=False)
    assert 'id="stak"' not in html
    assert "평문(http)" in html
    # 그래도 **다음에 할 일**은 말해 준다 — 제공자별 안내는 그대로 있다.
    assert 'id="stcredhint"' in html


@node_only
def test_the_panel_says_where_a_key_goes_to_stay():
    """**대가를 나중에 알게 하지 않는다.**

    화면에 넣은 열쇠는 메모리에만 있다. 그 사실이 필요해지는 것은 서버를 다시
    띄운 뒤인데, 그때 알림창은 이미 닫혀 있다. 그래서 넣기 전에, 닫히지 않는
    자리에 적어 둔다.
    """
    html = render("s3")
    assert "서버에 계속 남기려면" in html
    assert "AWS_ACCESS_KEY_ID=" in html
    assert "aws_secret_access_key = " in html


@node_only
def test_the_panel_opens_only_when_there_is_nothing_yet():
    """이미 잡힌 사람에게는 소음이다. 없는 사람에게만 펼쳐 준다."""
    assert "<details open" in render("s3", present=False)
    assert "<details open" not in render("s3", present=True)


@node_only
def test_the_credentials_file_is_not_advertised_as_aws_only():
    """`~/.aws/credentials` 는 **제공자와 무관하게** boto3 가 읽는다.

    파일 이름 때문에 AWS 전용으로 보인다. 이걸 안 적으면 NCP·R2 를 쓰는 사람은
    환경 변수 말고는 길이 없는 줄 안다.
    """
    ncp = render("ncp")
    assert "~/.aws/credentials" in ncp
    assert "제공자와 무관하게" in ncp


@node_only
def test_the_keyless_path_is_listed_only_where_it_exists():
    """AWS 는 셋, 나머지는 둘. 없는 길을 적으면 그건 그냥 틀린 말이다."""
    def homes(html):
        return re.findall(r'<b>(\d)</b> &nbsp;', html)

    assert homes(render("s3")) == ["1", "2", "3"]
    assert homes(render("ncp")) == ["1", "2"]
    assert "키를 두지 않는 길" not in render("ncp")


# ── 표 자체 ────────────────────────────────────────────────────────────────

def test_every_provider_declares_its_credential_facts():
    """새 제공자를 붙이면서 이 칸을 빠뜨려도 화면이 안 깨져야 한다."""
    for p in providers.listing():
        cred = p["credentials"]
        assert set(cred) == {"where", "url", "ambient", "env"}, p["id"]
        if p["supported"]:
            assert cred["where"], f"{p['id']} — 키를 어디서 받는지가 비어 있다"
            assert cred["env"], f"{p['id']} — 환경 변수 이름이 비어 있다"
        else:
            # 붙을 수 없는 것에 열쇠 이야기를 적으면 될 것처럼 보인다.
            assert not cred["where"] and not cred["ambient"], p["id"]


def test_only_aws_claims_a_keyless_path():
    """무키 경로는 컴퓨트와 저장소가 같은 클라우드일 때만 성립한다.

    여기 있는 나머지는 전부 남의 클라우드에서 붙는 경로라, 키가 없으면 붙을
    방법 자체가 없다. 이 목록이 늘어난다면 그건 사실이 바뀐 것이어야 한다.
    """
    keyless = [p["id"] for p in providers.listing() if p["credentials"]["ambient"]]
    assert keyless == ["s3"], keyless


def test_credential_facts_carry_no_actual_keys():
    """표에 적는 것은 **어디서 받나** 지 열쇠 자체가 아니다."""
    blob = json.dumps(providers.listing(), ensure_ascii=False)
    for leak in ("AKIA", "aws_secret_access_key", "secret_key"):
        assert leak not in blob, leak
