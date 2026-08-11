"""데이터셋 파일 이름 규칙.

    f_NNNNN_SS_STARTMS_ENDMS_STATE.ext

    f         카테고리 (face)
    NNNNN     원본 영상 번호 5자리      00001
    SS        세그먼트 2자리            00      (한 영상에서 여러 클립일 때)
    STARTMS   클립 시작 ms 7자리        0000000
    ENDMS     클립 끝 ms 7자리          0042000
    STATE     raw(입력) / deid(비식별 출력)

비식별화는 **정체성 필드를 그대로 두고 STATE 만 바꾼다.** 클립을 자르거나 합치지
않으므로 번호·세그먼트·구간은 입력 그대로여야 한다.

    f_00001_00_0000000_0042000_raw.mp4  ->  f_00001_00_0000000_0042000_deid.mp4

규칙에 맞지 않는 이름도 처리는 된다(직접 업로드한 임의 파일 등). 그때는
``<이름>_deid.mp4`` 로 떨어지고, 호출자는 ``parse()`` 가 None 인 것으로 판별할 수
있다 — 규칙 밖 파일이 결과 폴더에 섞이는 것을 조용히 넘기지 않기 위해서다.
"""

import os
import re
from dataclasses import dataclass, replace

STATE_RAW = "raw"
STATE_DEID = "deid"
DEFAULT_EXT = ".mp4"

# 자릿수는 규칙 그대로 고정한다. 느슨하게 받으면 정렬이 깨지고, 잘못 붙은 이름이
# 결과 폴더에 그대로 남는다.
PATTERN = re.compile(
    r"^(?P<category>[a-z])"
    r"_(?P<number>\d{5})"
    r"_(?P<segment>\d{2})"
    r"_(?P<start_ms>\d{7})"
    r"_(?P<end_ms>\d{7})"
    r"_(?P<state>[a-z]+)$"
)


@dataclass(frozen=True)
class ClipName:
    category: str
    number: int
    segment: int
    start_ms: int
    end_ms: int
    state: str
    ext: str = DEFAULT_EXT

    @property
    def duration_ms(self):
        return self.end_ms - self.start_ms

    def with_state(self, state, ext=None):
        return replace(self, state=state, ext=ext or self.ext)

    def format(self):
        return (f"{self.category}"
                f"_{self.number:05d}"
                f"_{self.segment:02d}"
                f"_{self.start_ms:07d}"
                f"_{self.end_ms:07d}"
                f"_{self.state}{self.ext}")

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
                    state=g["state"], ext=ext or DEFAULT_EXT)


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
