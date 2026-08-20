"""GPU 메모리 — **얼마나 남았나.**

OOM 은 나고 나서는 원인을 못 본다. 터진 순간의 여유는 이미 사라졌고, 로그에는
"CUDA out of memory" 한 줄만 남는다. 그래서 **터지기 전부터 계속 남긴다** —
기동할 때 한 번, 잡이 시작하고 끝날 때, 그리고 도는 동안 하트비트마다.

그래야 나중에 이 질문들에 답할 수 있다.

- 이 영상이 유난히 컸나, 아니면 원래 여유가 없었나
- 다른 프로세스가 같은 카드를 쓰고 있었나
- batch 를 줄인 것이 실제로 효과가 있었나

**torch 를 최상위에서 임포트하지 않는다.** 이 모듈은 서버 기동 경로에 있고,
torch 는 2GB 짜리라 없는 환경(테스트·CPU 전용 도구)에서도 열려야 한다.
"""

import logging

log = logging.getLogger(__name__)

_UNAVAILABLE = {"available": False}


def snapshot(device=None):
    """지금 GPU 메모리. 없으면 ``{"available": False}``.

    ``mem_get_info`` 는 **드라이버가 보는 실제 여유**다. 같은 카드를 쓰는 다른
    프로세스까지 반영된다 — `torch.cuda.memory_allocated` 는 우리 텐서만 세므로
    "우리는 조금 쓰는데 왜 터지지" 를 설명하지 못한다.
    """
    try:
        import torch                                # noqa: PLC0415 — 지연 임포트
    except Exception:                               # noqa: BLE001
        return dict(_UNAVAILABLE)
    try:
        if not torch.cuda.is_available():
            return dict(_UNAVAILABLE)
        free, total = torch.cuda.mem_get_info(device)
    except Exception as e:                          # noqa: BLE001
        # 드라이버가 답을 안 주는 경우까지 서버를 죽이지 않는다. 여유를 못 재는
        # 것은 불편이고, 그것 때문에 처리가 멈추는 것은 사고다.
        log.debug("GPU 메모리를 읽지 못했습니다: %s", e)
        return dict(_UNAVAILABLE)
    used = total - free
    return {"available": True,
            "free_mb": round(free / 1048576),
            "total_mb": round(total / 1048576),
            "used_mb": round(used / 1048576),
            # 가용률. 사람이 보는 숫자는 이것 하나면 된다.
            "free_pct": round(free / total * 100, 1) if total else None,
            "name": _name()}


def _name():
    try:
        import torch                                # noqa: PLC0415
        return torch.cuda.get_device_name(0)
    except Exception:                               # noqa: BLE001
        return None


def fields():
    """기록에 실을 **지금** 여유. 못 재면 빈 dict — 줄을 늘리지 않는다.

    이름에 `vram_` 을 붙여 두는 이유는, 이 값들이 여러 이벤트에 섞여 들어가기
    때문이다. `free_mb` 는 디스크 여유와 헷갈린다(같은 저널에 둘 다 있다).
    """
    s = snapshot()
    if not s["available"]:
        return {}
    return {"vram_free_mb": s["free_mb"], "vram_free_pct": s["free_pct"],
            "vram_total_mb": s["total_mb"]}


class Watch:
    """도는 동안의 **최저 여유**를 지킨다.

    끝난 뒤에 한 번 재는 것은 거의 쓸모가 없다 — 그때는 텐서가 다 반납된
    뒤라 항상 넉넉해 보인다. **아슬아슬했는지는 도는 중에만 보인다.**

    파이프라인의 진행 콜백은 프레임마다 불린다. 그때마다 드라이버에 물어보면
    낭비라서 **시간으로 눌러서** 잰다(기본 2초).
    """

    def __init__(self, every_s=2.0):
        self.every = max(0.2, float(every_s))
        self.min_free = None
        self.total = None
        self.name = None
        self._last = 0.0
        self.sample(force=True)

    def sample(self, force=False):
        import time                                 # noqa: PLC0415
        now = time.monotonic()
        if not force and now - self._last < self.every:
            return
        self._last = now
        s = snapshot()
        if not s["available"]:
            return
        self.total, self.name = s["total_mb"], s["name"]
        if self.min_free is None or s["free_mb"] < self.min_free:
            self.min_free = s["free_mb"]

    def result(self):
        """기록에 실을 것. **못 쟀으면 빈 dict** — 줄을 늘리지 않는다."""
        if self.min_free is None or not self.total:
            return {}
        return {"vram_min_free_mb": self.min_free,
                "vram_min_free_pct": round(self.min_free / self.total * 100, 1),
                "vram_total_mb": self.total,
                "vram_name": self.name}


def line(prefix="GPU"):
    """로그 한 줄. 못 재면 그 사실을 그대로 적는다 — 조용히 빠지면 나중에
    "이때는 왜 GPU 줄이 없지" 가 된다."""
    s = snapshot()
    if not s["available"]:
        return f"{prefix}: 없음(CPU)"
    return (f"{prefix}: {s['free_mb']}MB 남음 / {s['total_mb']}MB "
            f"({s['free_pct']}% 여유)")
