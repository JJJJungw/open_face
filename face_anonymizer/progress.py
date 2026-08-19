"""진행률 — **두 얼굴이 같은 자로 잰다.**

api 는 화면에, msa 는 하트비트에 진행률을 싣는다. 목적이 같으므로 계산도 하나여야
한다. 여기 없이 각자 계산하면 같은 영상이 화면에서는 46%, 저쪽 화면에서는 12%로
보이는 일이 생긴다.

왜 단계마다 몫을 주나
---------------------
**단계별 퍼센트를 그대로 쓰면 안 된다.** 검출이 100%까지 찼다가 렌더가 시작되면서
0%로 떨어지는데, 보는 사람에게는 그냥 되감긴 것으로 보인다.

더 나쁜 건 **남은 시간이 터진다**는 것이다. 남은 시간은 보통

    남은 시간 = 지금까지 걸린 시간 × (100 - 진행률) / 진행률

로 되짚는데, 분자의 '걸린 시간' 은 **작업이 시작된 순간부터** 재고 분모의
'진행률' 은 **일부 단계만** 센다면 둘의 기준이 어긋난다. 전사에 25초를 쓰고
검출이 막 2%를 채운 순간, 25 × 98 / 2 = 1225초 — 40초짜리 영상에 **20분**이 뜬다.
실제로 그렇게 떴다.

그래서 모든 단계를 한 자 위에 올린다. 몫은 L40S 실측(인제스트 12.6 · 검출 13.6 ·
렌더 13.0 · 최종 0.8초)에서 왔다. 인스턴스가 바뀌면 비율도 조금 달라지지만,
진행률은 **줄지만 않으면** 쓸 만하다.
"""

STAGE_SPAN = (                      # (단계, 시작 지점, 차지하는 몫)
    ("download",  0.00, 0.08),
    ("transcode", 0.08, 0.22),
    ("detect",    0.30, 0.30),
    ("track",     0.60, 0.02),
    ("render",    0.62, 0.30),
    ("upload",    0.92, 0.08),
)
SPAN = {name: (base, width) for name, base, width in STAGE_SPAN}

# 화면에 띄울 단계 이름. 코드 이름을 그대로 보여 주면 사용자가 읽을 말이 아니다.
STAGE_LABEL = {"download": "원본 받는 중", "transcode": "읽을 수 있게 변환 중",
               "detect": "얼굴 찾는 중", "track": "추적 잇는 중",
               "render": "가리는 중", "upload": "결과 올리는 중"}

# 이 아래에서는 추정이 요동친다 — "남은 시간 47분" 이 떴다가 곧 3분이 된다.
# 모르는 구간에서는 아예 안 내놓는 편이 낫다. 화면은 '계산 중' 을 띄우면 된다.
ETA_FLOOR = 5.0


def label(stage):
    return STAGE_LABEL.get(stage or "", "")


def overall(stage, done, total, floor=0.0):
    """전체 대비 진행률(0~100). ``floor`` 아래로는 내려가지 않는다.

    되감기지 않게 하는 것이 핵심이다. 단계가 통째로 건너뛰어질 수 있고(h264
    원본이면 전사가 없다) 콜백이 늦게 도착할 수도 있는데, 그때 뒤로 가느니 조금
    부정확한 편이 낫다 — 앞으로만 가는 진행률은 읽히지만 뒤로 가는 진행률은
    고장으로 읽힌다.
    """
    base, width = SPAN.get(stage or "", (None, None))
    if base is None:
        return round(floor, 1)
    frac = 0.0
    if total:
        try:
            frac = min(1.0, max(0.0, float(done or 0) / float(total)))
        except (TypeError, ValueError, ZeroDivisionError):
            frac = 0.0
    return round(max(floor, (base + width * frac) * 100), 1)


def eta(elapsed_s, percent):
    """남은 시간(초). 아직 못 믿을 구간이면 ``None``.

    ``elapsed_s`` 는 **작업이 시작된 순간부터** 재야 한다. percent 가 전 단계를
    덮으므로 분모와 분자의 기준이 그때 맞는다.
    """
    try:
        p = float(percent)
        e = float(elapsed_s)
    except (TypeError, ValueError):
        return None
    if p < ETA_FLOOR or p >= 100 or e <= 0:
        return None
    return round(e * (100 - p) / p)
