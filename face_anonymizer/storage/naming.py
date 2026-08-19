"""데이터셋 파일 이름 규칙.

    C_NNNNN_SS_STARTMS_ENDMS[_STATE].ext

    C         카테고리 한 글자          K(kbs) · M(mbc) · S(sbs) · f …
    NNNNN     원본 영상 번호 5자리      00297
    SS        세그먼트 2자리            00      (한 영상에서 여러 클립일 때)
    STARTMS   클립 시작 ms 7자리        0000000
    ENDMS     클립 끝 ms 7자리          0194281
    STATE     생략(입력) / deid(비식별 출력)

**STATE 는 없을 수 있다.** 실제 버킷의 입력 파일이 상태 토큰 없이 들어 있고
(``M_00297_00_0000000_0194281.mp4``), 대문자 카테고리를 쓴다. 처음에는 소문자에
``_raw`` 를 요구했는데 그러면 실제 입력이 전부 규칙 밖으로 떨어져, 규칙을
지켜서 도는 게 아니라 예비 경로로 도는 상태가 된다. 데이터에 규칙을 맞춘다.

비식별화는 **정체성 필드를 그대로 두고 STATE 만 붙인다.** 클립을 자르거나 합치지
않으므로 번호·세그먼트·구간은 입력 그대로여야 한다.

    M_00297_00_0000000_0194281.mp4      ->  M_00297_00_0000000_0194281_deid.mp4
    f_00001_00_0000000_0042000_raw.mp4  ->  f_00001_00_0000000_0042000_deid.mp4

규칙에 맞지 않는 이름도 처리는 된다(직접 업로드한 임의 파일 등). 그때는
``<이름>_deid.mp4`` 로 떨어지고, 호출자는 ``parse()`` 가 None 인 것으로 판별할 수
있다 — 규칙 밖 파일이 결과 폴더에 섞이는 것을 조용히 넘기지 않기 위해서다.
"""

import os
import re
from dataclasses import dataclass, replace

STATE_DEID = "deid"
STATE_NONE = ""                 # 입력 파일은 상태 토큰이 없다
DEFAULT_EXT = ".mp4"

# 자릿수는 규칙 그대로 고정한다. 느슨하게 받으면 정렬이 깨지고, 잘못 붙은 이름이
# 결과 폴더에 그대로 남는다. 반대로 카테고리 대소문자와 STATE 유무는 실제
# 데이터가 그렇게 생겼으므로 둘 다 받는다.
PATTERN = re.compile(
    r"^(?P<category>[A-Za-z])"
    r"_(?P<number>\d{5})"
    r"_(?P<segment>\d{2})"
    r"_(?P<start_ms>\d{7})"
    r"_(?P<end_ms>\d{7})"
    r"(?:_(?P<state>[A-Za-z]+))?$"
)


@dataclass(frozen=True)
class ClipName:
    category: str
    number: int
    segment: int
    start_ms: int
    end_ms: int
    state: str = STATE_NONE
    ext: str = DEFAULT_EXT

    @property
    def duration_ms(self):
        return self.end_ms - self.start_ms

    def with_state(self, state, ext=None):
        return replace(self, state=state, ext=ext or self.ext)

    def format(self):
        state = f"_{self.state}" if self.state else ""
        return (f"{self.category}"
                f"_{self.number:05d}"
                f"_{self.segment:02d}"
                f"_{self.start_ms:07d}"
                f"_{self.end_ms:07d}"
                f"{state}{self.ext}")

    def __str__(self):
        return self.format()


def parse(filename):
    """규칙에 맞으면 ClipName, 아니면 None.

    경로가 섞여 들어와도 파일명만 본다.
    """
    stem, ext = os.path.splitext(os.path.basename(filename or ""))
    m = PATTERN.match(stem)
    if not m:
        return None
    g = m.groupdict()
    start, end = int(g["start_ms"]), int(g["end_ms"])
    if end < start:
        return None                     # 구간이 뒤집힌 이름은 규칙 위반으로 본다
    return ClipName(category=g["category"], number=int(g["number"]),
                    segment=int(g["segment"]), start_ms=start, end_ms=end,
                    state=g["state"] or STATE_NONE, ext=ext or DEFAULT_EXT)


def output_name(filename, state=STATE_DEID, ext=DEFAULT_EXT):
    """입력 파일명 -> 결과 파일명.

    규칙에 맞으면 STATE 만 바꾸고, 아니면 ``<이름>_<state><ext>`` 로 떨어진다.
    확장자는 항상 결과물 컨테이너(mp4)를 따른다 — 입력이 mov 여도 H.264/mp4 로
    다시 뜨기 때문이다.
    """
    parsed = parse(filename)
    if parsed is not None:
        return parsed.with_state(state, ext).format()
    stem = os.path.splitext(os.path.basename(filename or "output"))[0]
    return f"{stem}_{state}{ext}"


def is_output(filename):
    """이미 비식별화된 결과물 이름인가.

    폴더를 통째로 제출할 때 결과물이 입력 목록에 섞이면 모자이크가 두 번
    올라간다. ``skip_processed`` 로는 못 막는다 — deid 파일의
    ``output_name()`` 은 자기 자신이라, "결과물이 이미 있다" 판정에 걸리지
    않는다. 이름으로 먼저 걸러 낸다.
    """
    parsed = parse(filename)
    if parsed is not None:
        return parsed.state.lower() == STATE_DEID
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    return stem.endswith("_" + STATE_DEID)
