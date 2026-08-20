"""여러 클라우드를 **동시에 들고 있는 자리.**

`storage/` 는 처음부터 어댑터 모양이었다 — `base.py` 에 계약이 있고,
`providers.py` 가 등록표고, `S3Store` 하나가 엔드포인트만 바꿔 AWS·NCP·R2·
MinIO·Wasabi 를 전부 단다. 빠져 있던 것은 **여러 개를 한꺼번에 들고 있는 자리**
하나였다. 전역이 하나뿐이라 "지금 붙어 있는 곳" 말고는 물어볼 수가 없었다.

    .env  ─▶  slots()   제공자별 설정을 읽어 어댑터를 만든다
              probe()   실제로 붙나 (읽기·쓰기 따로)  ← 카드에 불이 들어오는 근거
              active()  지금 쓰는 것 **하나**
              activate() 바꾼다

## 불이 들어오는 것과 활성인 것은 다르다

**불은 여럿 들어올 수 있고 활성은 항상 하나다.** 그래서 잡은 어느 클라우드인지
말할 필요가 없다 — `input_key` 만 오면 활성인 곳으로 처리한다. 부르는 쪽에
제공자를 실어 보내게 하면 그쪽 파이프라인이 우리 사정을 알아야 하고, 그건
경계를 잘못 그은 것이다.

## 둘 이상 켜져 있는데 활성이 안 정해졌으면 **아무것도 고르지 않는다**

여기서 임의로 하나 고르면 결과가 엉뚱한 버킷에 조용히 쌓인다. 900건을 돌리고
나서야 아는 종류의 사고이고, 그때는 되돌릴 수도 없다. 그래서 고르지 않고
사유를 말한다.

## 환경 변수 — 제공자 id 가 곧 접두어다

    FA_STORAGE_ACTIVE=s3          지금 쓰는 것. 하나만 설정돼 있으면 생략 가능

    FA_S3_BUCKET / FA_S3_REGION / FA_S3_ENDPOINT
    FA_S3_ROOT_PREFIX / FA_S3_OUTPUT_PREFIX
    FA_S3_ACCESS_KEY / FA_S3_SECRET_KEY        없으면 boto3 기본 체인

    FA_NCP_BUCKET / FA_NCP_ACCESS_KEY / ...
    FA_S3COMPAT_BUCKET / FA_S3COMPAT_ENDPOINT / ...

`FA_S3_*` 는 **예전부터 쓰던 이름 그대로**다. AWS 만 쓰던 설정은 한 글자도 안
고치고 그대로 돈다 — 접두어 규칙이 우연히 맞은 게 아니라, 맞도록 규칙을 골랐다.
"""

import logging
import os
import threading
import time

from . import providers

log = logging.getLogger(__name__)

# 붙는지 다시 재기까지의 간격. 카드 화면이 폴링해도 매번 클라우드를 치지 않게.
PROBE_TTL = float(os.environ.get("FA_CLOUD_PROBE_TTL", 30))

_LOCK = threading.Lock()
_probes = {}                     # id -> (잰 시각, 결과)


# 제공자마다 읽는 이름들. **여기가 정본이다.**
#
# 이 값들은 `FA_ + 제공자id + _ + 이름` 으로 **만들어져서** 코드에 문자열로
# 안 나타난다. 그래서 "코드가 읽는 값이 .env.example 에 다 적혀 있나" 를 보는
# 검사가 이 목록을 제공자 수만큼 곱해서 대조한다(tests/test_env.py).
PER_PROVIDER = ("BUCKET", "REGION", "ENDPOINT", "ROOT_PREFIX", "OUTPUT_PREFIX",
                "ACCESS_KEY", "SECRET_KEY", "SESSION_TOKEN")


def env_names():
    """이 모듈이 읽을 수 있는 모든 환경 변수 이름."""
    return {prefix_of(pid) + n
            for pid in providers.PROVIDERS for n in PER_PROVIDER}


def prefix_of(pid):
    """제공자 id → 환경 변수 접두어. `s3compat` → `FA_S3COMPAT_`."""
    return f"FA_{pid.strip().upper()}_"


def _env(pid, name):
    v = os.environ.get(prefix_of(pid) + name)
    return v.strip() if isinstance(v, str) else v


def config_of(pid):
    """이 제공자의 설정. **버킷이 없으면 설정이 없는 것으로 본다.**

    버킷 없이 리전만 있는 설정은 아무것도 할 수 없다. 그걸 "설정됨" 으로 세면
    카드가 켜졌다 꺼졌다 하고, 사람은 왜 안 되는지 모른다.
    """
    bucket = _env(pid, "BUCKET")
    if not bucket:
        return None
    return providers.StorageConfig(
        provider=pid,
        bucket=bucket,
        region=_env(pid, "REGION"),
        endpoint=_env(pid, "ENDPOINT"),
        root_prefix=_env(pid, "ROOT_PREFIX") or "",
        output_prefix=_env(pid, "OUTPUT_PREFIX"),
        # 구현을 갈아 끼우는 것은 **띄우는 사람의 일**이다. 요청에서 받지
        # 않는 이유는 그 값이 그대로 import_module() 에 들어가기 때문이다.
        store=os.environ.get("FA_STORAGE_STORE") or None,
    )


def creds_of(pid):
    """이 제공자의 열쇠. 없으면 None — 그때는 boto3 기본 체인이 답한다.

    **열쇠는 설정 객체에 안 넣는다.** `StorageConfig.as_dict()` 는 화면으로
    나가는 값이라, 거기 섞이는 순간 언젠가 응답에 실린다.
    """
    ak, sk = _env(pid, "ACCESS_KEY"), _env(pid, "SECRET_KEY")
    if not (ak and sk):
        return None
    out = {"aws_access_key_id": ak, "aws_secret_access_key": sk}
    token = _env(pid, "SESSION_TOKEN")
    if token:
        out["aws_session_token"] = token
    return out


def slots():
    """등록표 순서대로 (id, 설정, 열쇠). 설정이 없는 제공자는 설정이 None."""
    return [(pid, config_of(pid), creds_of(pid)) for pid in providers.PROVIDERS]


def configured():
    """설정이 있는 제공자 id 들."""
    return [pid for pid, cfg, _ in slots() if cfg is not None]


def store_for(pid):
    """이 제공자의 스토어. **전역을 안 건드린다** — 재 보기용이다."""
    from . import s3 as s3mod                       # 순환 방지

    cfg = config_of(pid)
    if cfg is None or not cfg.supported:
        return None
    return cfg.store_class(config=cfg,
                           client=s3mod.make_client(cfg, creds=creds_of(pid)))


def probe(pid, force=False):
    """실제로 붙나. **읽기와 쓰기를 따로 본다** — 읽기만 되는 열쇠가 흔하다.

    Returns:
        dict: {ok, read, write, detail, problem}. 설정이 없으면 ok=None.
    """
    now = time.time()
    with _LOCK:
        hit = _probes.get(pid)
        if hit and not force and now - hit[0] < PROBE_TTL:
            return hit[1]

    out = _probe_now(pid)
    with _LOCK:
        _probes[pid] = (now, out)
    return out


def _probe_now(pid):
    from . import s3 as s3mod

    cfg = config_of(pid)
    if cfg is None:
        return {"ok": None, "read": None, "write": None,
                "detail": f"{prefix_of(pid)}BUCKET 이 설정되지 않았습니다"}
    if not cfg.supported:
        return {"ok": False, "read": False, "write": False,
                "detail": f"{cfg.info['name']} 는 아직 지원하지 않습니다"}
    store = store_for(pid)
    if store is None:
        return {"ok": False, "read": False, "write": False,
                "detail": "스토어를 만들지 못했습니다"}
    try:
        store.check()
    except s3mod.S3Error as e:
        p = getattr(e, "problem", None)
        # 읽기에서 막혔는지 쓰기에서 막혔는지는 메시지가 안다 — `check()` 가
        # 두 단계에 서로 다른 말을 붙인다.
        read_ok = "결과물을 쓰지" in str(e)
        return {"ok": False, "read": read_ok, "write": False,
                "detail": str(e),
                "problem": {"code": p.code, "title": p.title, "hint": p.hint}
                if p is not None else None}
    except Exception as e:                          # noqa: BLE001
        return {"ok": False, "read": False, "write": False,
                "detail": f"{type(e).__name__}: {e}"}
    return {"ok": True, "read": True, "write": True,
            "detail": "읽기와 쓰기 모두 확인했습니다"}


def invalidate(pid=None):
    with _LOCK:
        _probes.pop(pid, None) if pid else _probes.clear()


# ── 활성 ──────────────────────────────────────────────────────────────────

def wanted():
    """활성으로 정해진 것. (id, 사유) — 못 정하면 id 가 None.

    1. `FA_STORAGE_ACTIVE` 가 있으면 그것
    2. 없는데 설정이 하나뿐이면 그것
    3. 없는데 둘 이상이면 **아무것도 안 고른다**
    """
    named = (os.environ.get("FA_STORAGE_ACTIVE") or "").strip().lower()
    if named:
        if named not in providers.PROVIDERS:
            return None, f"FA_STORAGE_ACTIVE={named} 는 모르는 제공자입니다"
        if config_of(named) is None:
            return None, (f"FA_STORAGE_ACTIVE={named} 인데 "
                          f"{prefix_of(named)}BUCKET 이 없습니다")
        return named, ""
    have = configured()
    if len(have) == 1:
        return have[0], ""
    if not have:
        return None, "설정된 클라우드가 없습니다"
    names = " · ".join(providers.get(p)["name"] for p in have)
    return None, (f"붙을 수 있는 곳이 여럿입니다({names}). 어느 쪽으로 처리할지 "
                  "정해지지 않았습니다 — FA_STORAGE_ACTIVE 를 정하거나 화면에서 "
                  "골라 주세요.")


ACTIVE = None                    # 지금 활성인 제공자 id (없으면 None)


def activate(pid):
    """활성을 바꾼다. **전역 설정과 열쇠를 같이 갈아 끼운다.**

    갈아 끼우는 자리는 여기 하나다 — `s3.get_store()` 를 지나는 열다섯 군데는
    자기가 어느 클라우드를 보고 있는지 모른 채 그대로 돈다.
    """
    from . import s3 as s3mod

    global ACTIVE
    cfg = config_of(pid)
    if cfg is None:
        raise ValueError(f"{prefix_of(pid)}BUCKET 이 설정되지 않았습니다")
    creds = creds_of(pid)
    s3mod.set_credentials(**({"access_key": creds["aws_access_key_id"],
                              "secret_key": creds["aws_secret_access_key"],
                              "session_token": creds.get("aws_session_token")}
                             if creds else {}))
    s3mod.reconfigure(cfg)
    ACTIVE = pid
    log.info("활성 클라우드: %s / %s", cfg.info["name"], cfg.bucket)
    return cfg


def resolve():
    """기동 때 한 번. 정해지면 활성으로 걸고, 못 정하면 사유를 남긴다."""
    pid, why = wanted()
    if pid is None:
        if why:
            log.warning("활성 클라우드가 정해지지 않았습니다 — %s", why)
        return None, why
    activate(pid)
    return pid, ""


def listing():
    """카드 화면이 그릴 것. **불이 들어오는 근거가 여기 다 있다.**"""
    pid_active, why = (ACTIVE, "") if ACTIVE else wanted()
    out = []
    for pid, cfg, creds in slots():
        info = providers.get(pid)
        row = {"id": pid, "name": info["name"],
               "supported": providers.is_supported(pid),
               "configured": cfg is not None,
               "active": pid == ACTIVE,
               "credentials": ("환경 변수 (%sACCESS_KEY)" % prefix_of(pid))
               if creds else None}
        if cfg is not None:
            row.update(bucket=cfg.bucket, region=cfg.region,
                       endpoint=cfg.endpoint, output_prefix=cfg.output_prefix)
        out.append(row)
    return {"clouds": out, "active": ACTIVE, "wanted": pid_active,
            "reason": why}
