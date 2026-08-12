"""데이터셋 파일 이름 규칙 테스트.

    C_NNNNN_SS_STARTMS_ENDMS[_STATE].ext

실제 버킷의 입력은 대문자 카테고리에 상태 토큰이 없다(M_00297_00_0000000_
0194281.mp4). 처음 규칙은 소문자에 _raw 를 요구해서 실제 입력이 전부 규칙 밖
예비 경로로 떨어지고 있었다. 데이터에 규칙을 맞췄다.

비식별화는 정체성 필드(번호·세그먼트·구간)를 건드리지 않는다. 여기가 틀리면
결과물이 어느 원본의 어느 구간인지 추적할 수 없게 된다.
"""

import pytest

from face_anonymizer.storage import naming


def test_parses_the_convention():
    c = naming.parse("f_00123_07_0012500_0098750_raw.mp4")
    assert (c.category, c.number, c.segment) == ("f", 123, 7)
    assert (c.start_ms, c.end_ms) == (12500, 98750)
    assert c.state == "raw" and c.ext == ".mp4"
    assert c.duration_ms == 86250


def test_parse_ignores_directories():
    assert naming.parse("videos/2026-08/f_00001_00_0000000_0042000_raw.mp4") is not None


def test_round_trip_is_stable():
    name = "f_00001_00_0000000_0042000_raw.mp4"
    assert naming.parse(name).format() == name


def test_output_only_flips_state():
    """번호·세그먼트·구간은 그대로여야 한다."""
    assert naming.output_name("f_00123_07_0012500_0098750_raw.mp4") \
        == "f_00123_07_0012500_0098750_deid.mp4"


def test_output_extension_is_always_mp4():
    """입력이 mov 여도 결과물은 H.264/mp4 로 다시 뜬다."""
    assert naming.output_name("f_00001_00_0000000_0042000_raw.mov") \
        == "f_00001_00_0000000_0042000_deid.mp4"


@pytest.mark.parametrize("bad", [
    "f_1_00_0000000_0042000_raw.mp4",          # 번호 자릿수 부족
    "f_00001_0_0000000_0042000_raw.mp4",       # 세그먼트 자릿수 부족
    "f_00001_00_000000_0042000_raw.mp4",       # 시작 ms 자릿수 부족
    "f_000011_00_0000000_0042000_raw.mp4",     # 번호 자릿수 초과
    "ff_00001_00_0000000_0042000_raw.mp4",     # 카테고리 두 글자
    "f_00001_00_0000000_0042000_raw_x.mp4",    # STATE 뒤에 덧붙음
    "f_00001_00_0042000_0000000_raw.mp4",      # 구간이 뒤집힘
    "clip.mp4",
    "",
])
def test_rejects_off_convention(bad):
    assert naming.parse(bad) is None


def test_off_convention_still_gets_an_output_name():
    """규칙 밖 파일도 처리는 되어야 한다 (직접 업로드 등)."""
    assert naming.output_name("clip.mov") == "clip_deid.mp4"
    assert naming.output_name("f_1_00_0000000_0042000_raw.mp4") \
        == "f_1_00_0000000_0042000_raw_deid.mp4"


def test_reprocessing_a_deid_file_is_idempotent():
    """이미 deid 인 파일을 다시 돌려도 이름이 늘어나지 않는다."""
    once = naming.output_name("f_00001_00_0000000_0042000_raw.mp4")
    assert naming.output_name(once) == once


def test_zero_padding_survives_large_values():
    c = naming.parse("f_99999_99_9999999_9999999_raw.mp4")
    assert c.with_state("deid").format() == "f_99999_99_9999999_9999999_deid.mp4"


def test_is_output_recognises_deid_names():
    assert naming.is_output("f_00001_00_0000000_0042000_deid.mp4")
    assert not naming.is_output("f_00001_00_0000000_0042000_raw.mp4")
    # 규칙 밖 이름도 접미사로 판별한다 (직접 업로드분의 결과물)
    assert naming.is_output("some/path/clip_deid.mp4")
    assert not naming.is_output("clip.mp4")
    assert not naming.is_output("undeid.mp4")     # 접미사가 아니라 단어 일부


# ── 실제 버킷 형태 ───────────────────────────────────────────────────────────

def test_uppercase_category_without_state_is_the_real_input_form():
    """버킷의 입력이 이 모양이다. 여기서 None 이 나오면 규칙이 아무 일도 안 한다."""
    c = naming.parse("M_00297_00_0000000_0194281.mp4")
    assert c is not None
    assert (c.category, c.number, c.segment) == ("M", 297, 0)
    assert (c.start_ms, c.end_ms) == (0, 194281)
    assert c.state == ""                       # 입력은 상태 토큰이 없다


def test_output_appends_state_to_a_stateless_input():
    assert naming.output_name("K_00000_00_0000000_0034342.mp4") \
        == "K_00000_00_0000000_0034342_deid.mp4"


def test_stateless_name_round_trips():
    name = "M_00297_00_0000000_0194281.mp4"
    assert naming.parse(name).format() == name


def test_is_output_on_the_real_form():
    assert naming.is_output("M_00297_00_0000000_0194281_deid.mp4")
    assert not naming.is_output("M_00297_00_0000000_0194281.mp4")


def test_reprocessing_the_real_form_is_idempotent():
    once = naming.output_name("K_00000_00_0000000_0034342.mp4")
    assert naming.output_name(once) == once
