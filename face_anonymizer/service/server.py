"""HTTP API + 웹 UI — 앱 조립과 라우트.

이 파일은 **HTTP 만** 다룬다. 무엇을 조절할 수 있는지는 ``config``, 작업이
어떻게 남는지는 ``jobs``, 실제 처리는 ``worker`` 에 있다.

    config  ← jobs  ← worker  ← server
                              (라우트는 아래 셋을 쓰고, 아래 셋은 라우트를 모른다)

의존이 한 방향이라 워커를 갈아끼울 수 있다 — 나중에 AWS Batch 를 붙일 때
``worker`` 만 바꾸면 라우트도 상태 코드도 그대로다.

**제출 진입점은 POST /api/jobs 하나다.** 한 건이든 여러 건이든 폴더든 같은
요청, 같은 응답. 진입점을 나누면 클라이언트가 경우마다 분기해야 하고 화면에도
버튼이 그만큼 늘어난다.

실행
    uvicorn face_anonymizer.service.server:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import Body, FastAPI, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from .. import events, logsetup, timefmt
from ..core.anonymize import METHODS
from ..core.pipeline import parse_bitrate
from ..storage import naming, providers
from ..storage import s3 as s3mod
from . import config, errors, jobs, metrics, worker
from .config import JOB_DEFAULTS
from .webui import INDEX_HTML

log = logging.getLogger(__name__)

_sweeper = None


@asynccontextmanager
async def lifespan(_app):
    global _sweeper
    # 여기서 안 하면 우리 INFO 는 전부 버려진다 — uvicorn 은 루트를 안 건드린다.
    logsetup.setup()
    os.makedirs(config.JOBS_DIR, exist_ok=True)
    log.info("서버 기동 — 작업 폴더 %s", config.JOBS_DIR)
    events.emit("server.started", jobs_dir=config.JOBS_DIR)
    worker.resume_orphans()
    if config.SWEEP_SEC > 0 and _sweeper is None:
        _sweeper = threading.Thread(target=jobs.sweep_loop, daemon=True,
                                    name="sweeper")
        _sweeper.start()
    if config.PRELOAD:
        # 여기서 죽이지 않고 사유를 남긴다. 크래시 루프로 재시작하면 로그가
        # 흘러가서 왜 안 뜨는지 알기 어렵다. health 가 503 으로 이유를 알려준다.
        try:
            worker.get_anonymizer()
        except Exception as e:                  # noqa: BLE001
            worker.model_error = f"{type(e).__name__}: {e}"
            log.exception("모델 로드 실패 — 준비되지 않은 상태로 뜬다")
    yield


app = FastAPI(title="face-anonymizer", version="0.3.0", lifespan=lifespan)
errors.install(app)


@app.get("/api/problems")
def problems():
    """이 서비스가 낼 수 있는 오류 목록.

    호출하는 쪽이 code 별 대응(재시도/전환/사람 호출)을 미리 짜 둘 수 있게
    한곳에서 노출한다.
    """
    return {"problems": [p.as_dict() for p in errors.CATALOG.values()]}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/api/status")
def status():
    """폴링 한 번에 필요한 전부. **디스크를 훑지 않는다.**

    ``ready``/``busy``/``queued`` 는 오케스트레이터가 보는 값이고,
    ``counts``·``running``·``recent`` 는 화면이 보는 값이다. 한 응답에 같이
    담는 이유는 화면이 0.7초마다 폴링하기 때문이다 — 엔드포인트를 나누면
    그만큼 왕복이 는다.

    **이 값들은 목록(GET /api/jobs)과 독립이다.** 목록은 잘려도 여기 숫자는
    전체를 센다(docs/issues/006).
    """
    return {"ready": worker.is_ready(), "busy": worker.is_busy(),
            "queued": jobs.queue_depth(), "free_mb": jobs.free_mb(),
            "model_error": worker.model_error,
            "counts": jobs.counts(), "running": jobs.running_snapshot(),
            "recent": jobs.recent_stats(), "next_up": jobs.next_up(),
            # 큐 화면이 쓰는 값들(대기 지연·처리량·재시도). GPU 는 여기서 안
            # 본다 — nvidia-smi 는 프로세스를 띄우는 일이라 0.7초 폴링에 못 얹는다.
            "queue": metrics.queue_metrics(jobs.memory_jobs()),
            "list_limit": config.LIST_LIMIT, "page_size": config.PAGE_SIZE}


@app.get("/api/metrics")
def metrics_endpoint():
    """현황 화면용. 큐 지표 + 자원 상태.

    작업 큐 대시보드가 공통으로 띄우는 것들이다(metrics.py 주석 참고).
    특히 ``latency`` 는 깊이만으로 못 보는 정체를 잡아 준다 — 3건이 두 시간째
    안 빠지는 것과 100건이 1분 만에 빠지는 것은 다른 상황이다.
    """
    m = metrics.queue_metrics(jobs.all_jobs())
    m.update({
        "ready": worker.is_ready(), "model_error": worker.model_error,
        "max_attempts": config.MAX_ATTEMPTS,
        "free_mb": jobs.free_mb(), "min_free_mb": config.MIN_FREE_MB,
        "gpu": metrics.gpu_status(),
    })
    return m


@app.get("/api/s3/progress")
def s3_progress(prefix: list[str] = Query(default=[])):
    """고른 폴더의 진척률. 큐가 아니라 **버킷** 기준이다.

    큐 지표는 지금 들어와 있는 것만 안다. 데이터셋을 통째로 돌리는 작업에서
    정작 궁금한 건 전체 중 얼마나 남았는지이고, 그건 결과 버킷에 있다.

    **고르지 않으면 아무것도 세지 않는다.** 예전에는 인자 없이 부르면 설정에
    박힌 루트 프리픽스 밑을 통째로 훑어서, 버킷에 있는 모든 폴더가 화면에
    나왔다. 그 범위를 화면에서 바꿀 방법이 없었고 — 즉 사람이 고른 적이 없는
    숫자였다. 지금 작업하는 폴더가 무엇인지는 사람만 안다.

    폴더는 여럿 고를 수 있다(``?prefix=a/&prefix=b/``). 한 번에 여러 방송사를
    돌리는 일이 흔해서, 하나만 고르게 하면 화면을 오가며 봐야 한다.
    """
    store = s3mod.get_store()
    if store is None:
        raise errors.S3_NOT_CONFIGURED()
    # 인자가 없으면 **지금까지 제출한 폴더들**을 본다. 예전에는 설정에 박힌
    # 루트 밑을 통째로 훑어서, 한 번도 돌린 적 없는 폴더까지 화면에 나왔다.
    picked = [p for p in (prefix or []) if p] or jobs.tracked_prefixes()
    for p in picked:
        if ".." in p:
            raise errors.INVALID_KEY(p)
    rows = []
    try:
        for p in picked:
            rows += metrics.folder_progress(store, p)
    except s3mod.S3Error as e:
        raise (e.problem or errors.S3_UPSTREAM)(str(e)) from e
    # 같은 폴더를 두 번 고르거나 상위·하위를 같이 골라도 한 줄만 남긴다.
    seen, uniq = set(), []
    for r in rows:
        if r["prefix"] not in seen:
            seen.add(r["prefix"])
            uniq.append(r)
    uniq.sort(key=lambda x: -x["total"])
    return {"prefixes": picked, "output_prefix": store.output_prefix,
            "root_prefix": store.root_prefix, "folders": uniq,
            "total": sum(r["total"] for r in uniq),
            "done": sum(r["done"] for r in uniq)}


@app.delete("/api/s3/progress", status_code=204)
def untrack(prefix: str):
    """진척률 목록에서 폴더 하나를 뺀다. **버킷은 건드리지 않는다.**

    다 끝난 폴더를 계속 띄워 두면 지금 돌고 있는 것이 안 보인다. 다시 제출하면
    자동으로 돌아온다.
    """
    jobs.untrack_prefix(prefix)
    return Response(status_code=204)


@app.get("/api/health")
def health(response: Response):
    ready = worker.is_ready()
    if not ready:
        # 준비 전에는 컨테이너를 healthy 로 보이게 하면 안 된다. 오케스트레이터가
        # 트래픽을 보내고, 그 요청이 모델 로딩을 통째로 기다리게 된다.
        response.status_code = 503
    info = {"status": "ok" if ready else "not-ready",
            "ready": ready, "busy": worker.is_busy(), "queued": jobs.queue_depth(),
            "free_mb": jobs.free_mb(),
            "model_loaded": worker._anonymizer is not None, "model_error": worker.model_error,
            "device": config.DEVICE or "auto", "imgsz": config.IMGSZ,
            "methods": list(METHODS)}
    loaded = worker._anonymizer is not None
    if loaded:
        # 검출기는 주입 가능하므로 FaceDetector 의 속성이 있다고 단정하지 않는다.
        d = worker._anonymizer.detector
        info.update(device=str(getattr(d, "device", "?")),
                    half=getattr(d, "half", None),
                    stride=getattr(d, "stride", None),
                    detector=type(d).__name__)
    rows = jobs.all_jobs()
    info["jobs"] = {"total": len(rows),
                    "running": sum(1 for j in rows if j.status == "running"),
                    "queued": sum(1 for j in rows if j.status == "queued")}
    info["pid"] = os.getpid()
    return info


@app.get("/api/defaults")
def defaults():
    """서비스가 쓰는 처리 파라미터 기본값.

    UI 가 컨트롤 초깃값을 여기서 받아 간다 — 화면에 값을 박아 두면 서버 설정을
    바꿔도 화면은 옛 값을 보내서 둘이 조용히 어긋난다.
    """
    return dict(JOB_DEFAULTS)


@app.get("/api/storage")
def storage_info():
    """지금 어디에 붙어 있나, 그리고 붙을 수 있는 곳들.

    **자격 증명은 여기 없다.** 우리가 애초에 들고 있지 않다 — boto3 기본
    체인(EC2 인스턴스 역할 또는 환경 변수)을 쓴다. 화면에서 키를 받는 설정은
    **API 인증을 정한 뒤**의 일이다. 지금 상태로 열면 서버에 닿을 수 있는
    누구나 키를 심거나 볼 수 있다.
    """
    # **실제로 붙은 것**을 말한다. 모듈 설정만 보면, 스토어가 다른 값으로
    # 만들어졌을 때 화면과 실제가 다른 말을 한다.
    # get_store() 는 갈아 끼울 수 있다(테스트·주입). 설정을 안 들고 있는
    # 스토어가 와도 화면이 깨지면 안 된다.
    store = s3mod.get_store()
    current = getattr(store, "config", None) or s3mod.CONFIG
    return {"current": current.as_dict(),
            "providers": providers.listing(),
            "reason": s3mod.unavailable_reason(),
            "editable": False,
            "note": "설정은 환경 변수로 정한다(.env). 화면에서 바꾸려면 "
                    "API 인증이 먼저 필요하다."}


@app.post("/api/storage/test")
def storage_test():
    """지금 설정으로 **실제로 붙는지** 확인한다.

    잘못된 버킷에 900건을 넣고 나서 아는 것보다 넣기 전에 아는 편이 낫다.
    읽기와 쓰기를 따로 본다 — 읽기만 되는 자격 증명이 흔하다.
    """
    store = s3mod.get_store()
    if store is None:
        raise errors.S3_NOT_CONFIGURED(s3mod.unavailable_reason())
    try:
        store.check()
    except s3mod.S3Error as e:
        raise (e.problem or errors.S3_UPSTREAM)(str(e)) from e
    return {"ok": True, "bucket": store.bucket,
            "endpoint": s3mod.CONFIG.endpoint,
            "provider": s3mod.CONFIG.provider,
            "detail": "읽기와 쓰기 모두 확인했습니다"}


@app.get("/api/s3/objects")
def s3_objects(prefix: str = ""):
    """버킷을 한 단계씩 나열한다 (S3 콘솔과 같은 방식).

    설정 전이면 404. UI 는 그때 안내만 띄우고 직접 업로드로 쓴다.
    """
    store = s3mod.get_store()
    if store is None:
        raise errors.S3_NOT_CONFIGURED()
    if ".." in prefix:
        raise errors.INVALID_KEY(prefix)
    try:
        folders, objects = store.list(prefix)
    except s3mod.S3Error as e:
        raise (e.problem or errors.S3_UPSTREAM)(str(e)) from e
    done = store.processed_keys()
    for o in objects:
        o["processed"] = store.output_key(o["key"]) in done
    return {"bucket": store.bucket, "prefix": prefix or store.root_prefix,
            "folders": folders, "objects": objects,
            "output_prefix": store.output_prefix}


def _coerce(key, value):
    """폼은 전부 문자열로 온다. 기본값의 타입에 맞춰 되돌린다."""
    ref = JOB_DEFAULTS.get(key)
    if isinstance(ref, bool):
        return str(value).strip().lower() in ("1", "true", "on", "yes")
    try:
        if isinstance(ref, int):
            return int(float(value))
        if isinstance(ref, float):
            return float(value)
    except (TypeError, ValueError) as e:
        raise errors.INVALID_INPUT(f"{key} 값이 숫자가 아닙니다: {value!r}",
                                   field=key) from e
    return value


def check_admission():
    """받을 수 있는 상태인지. 못 받으면 ProblemError."""
    if not worker.is_ready():
        raise (errors.MODEL_LOAD_FAILED(worker.model_error) if worker.model_error
               else errors.NOT_READY("서버가 기동 중입니다"))
    free = jobs.free_mb()
    if config.MIN_FREE_MB and free is not None and free < config.MIN_FREE_MB:
        raise errors.INSUFFICIENT_STORAGE(f"남은 공간 {free}MB, 최소 {config.MIN_FREE_MB}MB 가 필요합니다",
                                          free_mb=free, retry_after=config.RETRY_AFTER)


def resolve_params(given):
    """보낸 것만 덮고 나머지는 서비스 기본값. 검증도 여기서."""
    params = {**JOB_DEFAULTS,
              **{k: v for k, v in given.items() if v is not None}}
    if params["method"] not in METHODS:
        raise errors.INVALID_INPUT(f"모르는 익명화 방식입니다: {params['method']}",
                                   field="method", allowed=list(METHODS))
    if not 0 < params["conf"] < 1:
        raise errors.INVALID_INPUT(
            f"conf 는 0 과 1 사이여야 합니다 (받으신 값 {params['conf']})", field="conf")
    params["imgsz"] = max(320, min(int(params["imgsz"]), 2048))
    h = int(params["height"] or 0)
    if h and not 144 <= h <= 4320:
        raise errors.INVALID_INPUT(
            f"height 는 0(원본 유지) 이거나 144~4320 이어야 합니다 (받으신 값 {h})",
            field="height")
    params["height"] = h
    for k in ("bitrate", "max_bitrate"):
        v = params[k]
        if v not in ("", None) and parse_bitrate(v) is None:
            raise errors.INVALID_INPUT(
                f"{k} 값을 읽지 못했습니다: {v!r} — 3500k · 3.5M · 3500000 형태로 보내 주세요",
                field=k)
    return params


def check_video_name(name):
    ext = os.path.splitext(name)[1].lower()
    if ext not in config.VIDEO_EXT:
        raise errors.UNSUPPORTED_MEDIA(f"확장자가 {ext or '없습니다'}",
                                       supported=sorted(config.VIDEO_EXT))
    return ext


def check_s3_key(key):
    if s3mod.get_store() is None:
        raise errors.S3_NOT_CONFIGURED()
    if ".." in key or key.startswith("/"):
        raise errors.INVALID_KEY(key)
    return check_video_name(os.path.basename(key))


@app.post("/api/jobs", status_code=202)
async def create_jobs(request: Request):
    """**제출은 여기 하나다.** 한 건이든 여러 건이든 폴더든 같은 요청, 같은 응답.

    진입점을 나누면 클라이언트가 경우마다 분기해야 하고, 화면에도 버튼이 그만큼
    늘어난다. 입력이 무엇이냐만 다르고 나머지는 전부 같다.

    받는 형태::

        multipart/form-data   file=@clip.mp4                     # 업로드 한 건
        application/json      {"s3_keys": ["a.mp4", "b.mp4"]}    # 고른 파일들
        application/json      {"s3_prefix": "kbs/"}              # 폴더 하나
        application/json      {"s3_prefix": ["kbs/", "mbc/"]}    # 폴더 여럿

    **파일과 폴더는 같이 보낼 수 있다.** 화면에서 파일 두 개와 폴더 하나를
    한꺼번에 체크하는 게 자연스럽기 때문이다. 펼친 결과가 겹치면 한 번만
    넣는다. 업로드(``file``)만 S3 선택과 같이 못 보낸다 — 올라오는 바이트와
    버킷의 키는 아예 다른 경로다.

    옵션은 JSON 이면 ``params``, multipart 면 폼 필드로 준다. 안 주면 서비스
    기본값(GET /api/defaults).

    폴더 제출은 ``recursive``(하위 폴더까지, 기본 false)와
    ``skip_processed``(이미 결과물이 있는 건 건너뛰기, 기본 false)를 받는다.
    이름이 ``_deid`` 로 끝나는 결과물은 어느 쪽이든 입력에서 뺀다.

    응답은 항상 같다::

        {"accepted": [{"id", "name", "s3_key"}],
         "rejected": [{"s3_key", "error": {...}}],
         "queued": 3}

    **한 건이 거절돼도 나머지는 받는다.** 수백 건에서 키 하나가 오타라고 전체를
    되돌리면 호출하는 쪽이 무엇이 들어갔는지 알 수 없다.
    """
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip()
    upload = None
    keys, prefixes, recursive, skip_processed = [], [], False, False

    if ctype == "application/json":
        try:
            body = await request.json()
        except Exception as e:                      # noqa: BLE001
            raise errors.INVALID_INPUT("본문을 JSON 으로 읽지 못했습니다") from e
        if not isinstance(body, dict):
            raise errors.INVALID_INPUT("본문은 JSON 객체여야 합니다")
        keys = body.get("s3_keys") or []
        prefixes = body.get("s3_prefix") or body.get("s3_prefixes") or []
        if isinstance(prefixes, str):               # 한 개는 문자열로도 받는다
            prefixes = [prefixes]
        if not isinstance(prefixes, list):
            raise errors.INVALID_INPUT("s3_prefix 는 문자열이거나 배열이어야 합니다",
                                       field="s3_prefix")
        recursive = bool(body.get("recursive"))
        skip_processed = bool(body.get("skip_processed"))
        given = body.get("params") or {}
        if not isinstance(keys, list):
            raise errors.INVALID_INPUT("s3_keys 는 배열이어야 합니다", field="s3_keys")
    else:
        form = await request.form()
        upload = form.get("file")
        if isinstance(upload, str):                 # 파일이 아니라 문자열이면 무시
            upload = None
        keys = [v for v in form.getlist("s3_keys") if v]
        one = form.get("s3_key")
        if one:
            keys.append(one)
        prefixes = [v for v in form.getlist("s3_prefix") if v]
        recursive = str(form.get("recursive", "")).lower() in ("1", "true", "on")
        skip_processed = str(form.get("skip_processed", "")).lower() in (
            "1", "true", "on")
        given = {k: form.get(k) for k in JOB_DEFAULTS if form.get(k) is not None}
        given = {k: _coerce(k, v) for k, v in given.items()}

    uploaded = upload is not None and bool(getattr(upload, "filename", ""))
    if not uploaded and not keys and not prefixes:
        raise errors.MISSING_INPUT()
    if uploaded and (keys or prefixes):
        raise errors.CONFLICTING_INPUT(
            "업로드 파일과 S3 선택은 같이 보내실 수 없습니다")

    params = resolve_params(given)

    # 폴더는 여기서 펼친다. 클라이언트가 목록을 먼저 받아 오게 하면 그 사이에
    # 파일이 늘거나 줄 수 있고, 왕복도 한 번 더 든다.
    if prefixes:
        store = s3mod.get_store()
        if store is None:
            raise errors.S3_NOT_CONFIGURED()
        expanded = []
        for prefix in prefixes:
            if ".." in prefix:
                raise errors.INVALID_KEY(prefix)
            try:
                objs = (store.list_all(prefix) if recursive
                        else store.list(prefix)[1])
            except s3mod.S3Error as e:
                raise (e.problem or errors.S3_UPSTREAM)(str(e)) from e
            expanded += [o["key"] for o in objs
                         if os.path.splitext(o["key"])[1].lower() in config.VIDEO_EXT
                         and not naming.is_output(o["key"])]
        if not expanded and not keys:
            raise errors.BATCH_EMPTY(
                f"{' · '.join(prefixes)} 안에 처리할 영상이 없습니다")
        keys = keys + expanded

    # 폴더를 펼친 결과가 따로 고른 파일과 겹칠 수 있다. 순서는 유지한다 —
    # 화면에 보인 차례대로 큐에 들어가야 진행 상황이 읽힌다.
    keys = list(dict.fromkeys(keys))

    # '처리된 건 건너뛰기' 는 폴더든 낱개든 똑같이 적용한다. 화면의 체크박스
    # 하나가 폴더에서는 먹고 파일을 골랐을 때는 안 먹으면, 그건 설정이 아니라
    # 함정이다. 그리고 조용히 빼지 않고 건별 사유로 돌려준다 — 눌렀는데 아무
    # 일도 안 일어나는 것이 사용자가 겪을 수 있는 최악이다.
    skipped = []
    if skip_processed and keys:
        store = s3mod.get_store()
        if store is not None:
            try:
                done = store.processed_keys() - jobs.rejected_inputs()
            except s3mod.S3Error as e:
                raise (e.problem or errors.S3_UPSTREAM)(str(e)) from e
            fresh = []
            for k in keys:
                (skipped if store.output_key(k) in done else fresh).append(k)
            keys = fresh

    if config.BATCH_MAX and len(keys) > config.BATCH_MAX:
        raise errors.BATCH_TOO_LARGE(f"{len(keys)}건을 보내셨습니다 (상한 {config.BATCH_MAX}건)",
                                     limit=config.BATCH_MAX)
    check_admission()

    accepted, rejected = [], []
    for k in skipped:
        rejected.append({"s3_key": k,
                         "error": errors.ALREADY_PROCESSED(k).body()})

    if uploaded:
        name = os.path.basename(upload.filename)
        ext = check_video_name(name)
        jid = worker.new_job_id()
        workdir = os.path.join(config.JOBS_DIR, jid)
        os.makedirs(workdir, exist_ok=True)
        src = os.path.join(workdir, "input" + ext)
        size = 0
        try:
            with open(src, "wb") as f:
                while True:
                    chunk = await upload.read(config.CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > config.MAX_BYTES:
                        raise errors.PAYLOAD_TOO_LARGE(
                            f"상한 {config.MAX_BYTES // 1048576} MB")
                    f.write(chunk)
        except Exception:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        if size == 0:
            shutil.rmtree(workdir, ignore_errors=True)
            raise errors.EMPTY_FILE()
        _job, snap = worker.enqueue(name, params, jid=jid, workdir=workdir)
        accepted.append({"id": snap["id"], "name": name, "s3_key": None})
    else:
        for key in keys:
            try:
                if not isinstance(key, str):
                    raise errors.INVALID_KEY(str(key))
                check_s3_key(key)
                name = os.path.basename(key)
                _job, snap = worker.enqueue(name, dict(params), s3_key=key,
                                            batch=batch_of(key, prefixes))
                accepted.append({"id": snap["id"], "name": name, "s3_key": key})
                # **제출한 폴더가 곧 진척률 대상이다.** 어느 폴더를 돌릴지는
                # 여기서 이미 골랐다 — 진척률 화면에서 또 고르게 하면 두 번
                # 고르는 셈이고, 둘이 어긋나면 엉뚱한 폴더의 숫자가 뜬다.
                jobs.track_prefix(key)
            except errors.ProblemError as e:
                rejected.append({"s3_key": key, "error": e.body()})

    if not accepted and rejected:
        # 하나도 못 받았으면 202 를 줄 수 없다. 단건 제출이면 그 사유가 곧
        # 응답 코드가 되고(예: 415), 여러 건이면 항목별 사유를 함께 준다.
        codes = {r["error"].get("code") for r in rejected}
        problem = (errors.CATALOG.get(codes.pop()) if len(codes) == 1
                   else errors.INVALID_INPUT)
        raise (problem or errors.INVALID_INPUT)(
            rejected[0]["error"].get("detail", "") if len(rejected) == 1
            else f"{len(rejected)}건 전부 거절됐다",
            rejected=rejected)
    return {"accepted": accepted, "rejected": rejected, "queued": jobs.queue_depth()}


# 검수는 **사람을 기다리는 일**이라 대기보다 위다. 아래로 내리면 300건짜리
# 배치에서 검수 두 건이 완료 기록에 파묻혀 영영 안 보인다.
_LIST_RANK = {"running": 0, "review": 1, "queued": 2}


def list_key(j):
    """목록 정렬 기준 — **실행 순서**.

    수행중이 맨 위, 그 다음이 대기(오래된 것 = 다음 차례가 위), 끝난 것은
    최신순으로 뒤에 붙는다.

    만들어진 순으로만 자르면 안 되는 이유가 있다. 워커는 오래된 것부터
    처리하는데 목록은 최신순 100건이라, 300건짜리 배치에서는 지금 돌고 있는
    작업이 창 밖으로 밀려난다 — 화면이 '유휴' 라고 말한다(docs/issues/006).
    """
    rank = _LIST_RANK.get(j.status, 3)
    if rank == 3:                       # done·failed·cancelled — 최근 것부터
        return (rank, -(j.finished or j.created))
    return (rank, j.created)            # 수행중·검수·대기 — 먼저 들어온 것부터


def batch_of(key, prefixes):
    """이 키가 어느 묶음에 속하나. 폴더로 제출한 것만 묶음 이름을 갖는다.

    파일을 골라 넣은 것까지 묶으면 "kbs 폴더 12분" 같은 문장이 실제 폴더 처리와
    달라진다. 묶음은 **폴더를 통으로 넣었을 때만** 만든다.
    """
    for p in prefixes or ():
        if key.startswith(p):
            return p.rstrip("/").split("/")[-1] or p
    return ""


@app.get("/api/batches")
def list_batches():
    """폴더별 시작·종료·진척. 사람이 묻는 단위가 파일이 아니라 폴더다."""
    return {"batches": jobs.batches()}


@app.get("/api/events")
def list_events(job: str = None, batch: str = None, event: str = None,
                mode: str = None, since: float = None, before: float = None,
                q: str = None, limit: int = 200, text: bool = True,
                full: bool = False, from_day: str = None, to_day: str = None):
    """이벤트 저널. **로그가 아니라 기록이다.**

    로그 문장은 읽기 좋게 계속 바뀌므로 파싱 대상이 아니다. 기계가 볼 것은
    처음부터 따로 남긴다(face_anonymizer/events.py).

    ``full=true`` 면 저널 줄을 통째로 준다. 기본은 목록이 그리는 값만이다.

    ``text=true`` 면 화면이 바로 그릴 수 있게 라벨·문장·색조를 붙여 준다.
    **문장을 서버가 만드는 이유**는 로그 파일과 화면이 다른 말을 하면 안 되기
    때문이다. 기계로 읽을 거면 ``text=false`` 로 원본 줄만 받는다.

    ``before`` 는 '더 보기' 커서다. 받은 마지막 줄의 ``ts`` 를 넣으면 그 아래로
    이어진다. offset 을 쓰지 않는 이유는 읽는 사이에도 줄이 계속 쌓여서 기준이
    밀리기 때문이다 — 같은 줄을 두 번 보거나 통째로 건너뛴다.
    """
    # 한 줄 더 읽어 본다. 그게 있으면 뒤가 남았다는 뜻이다 — 전체를 세지 않고도
    # '더 보기' 를 띄울지 정할 수 있다. 저널은 계속 자라서 총 건수가 의미 없다.
    want = max(1, min(int(limit or 200), events.READ_MAX))
    # 목록은 **그리는 값만** 받는다. 단계별 소요나 경고 원문처럼 펼쳐야 보이는
    # 것까지 줄마다 붙어 오면 한 쪽에 몇 배가 실린다 — 상세는 펼칠 때 따로 온다.
    rows = events.read(job=job, batch=batch, event=event, mode=mode,
                       since=since, before=before, q=q, limit=want + 1,
                       from_day=from_day, to_day=to_day,
                       fields=None if full else events.LIST_FIELDS)
    more = len(rows) > want
    rows = rows[:want]
    return {"events": [events.decorate(r) for r in rows] if text else rows,
            "has_more": more,
            "cursor": rows[-1].get("ts") if rows else None,
            "mode": events.MODE}


# 내보내기 열 — **화면에서 보던 것과 같은 순서.** 파일을 열었을 때 화면과 다른
# 순서면 대조가 안 된다. 사건 종류가 섞이므로 해당 없는 칸은 비워 둔다.
EXPORT_COLUMNS = (
    ("시각", lambda r: r.get("at") or ""),
    ("경로", lambda r: r.get("mode") or ""),
    ("사건", lambda r: r.get("label") or r.get("event") or ""),
    ("파일명", lambda r: r.get("name") or ""),
    ("폴더", lambda r: r.get("batch") or ""),
    ("요약", lambda r: r.get("text") or ""),
    ("소요(초)", lambda r: r.get("seconds") if r.get("seconds") is not None
     else r.get("elapsed_s", "")),
    ("프레임", lambda r: r.get("frames", "")),
    ("검출 프레임", lambda r: r.get("detected_frames", "")),
    ("검출률(%)", lambda r: round(r["detection_rate"] * 100, 2)
     if isinstance(r.get("detection_rate"), (int, float)) else ""),
    ("원본 코덱", lambda r: r.get("source_codec") or ""),
    ("전사", lambda r: "예" if r.get("transcoded") else ""),
    ("시도", lambda r: r.get("attempts", "")),
    ("검수 필요", lambda r: "예" if r.get("review_needed") else ""),
    ("단계", lambda r: r.get("stage") or ""),
    ("작업 id", lambda r: r.get("job") or ""),
)


@app.get("/api/events/detail")
def event_detail(ts: float, job: str = None, event: str = None):
    """줄 하나를 **원본 그대로**. 로그 화면에서 펼쳤을 때 쓴다.

    목록에 상세까지 실어 보내지 않는 대신, 사람이 펼친 그 한 줄만 가져온다.
    60줄 중 사람이 펼치는 건 보통 한둘이다.
    """
    row = events.detail_of(job=job, ts=ts, event=event)
    if row is None:
        raise errors.JOB_NOT_FOUND(f"ts={ts}")
    return {"event": events.decorate(row), "raw": row}


@app.get("/api/events/days")
def event_days():
    """저널이 있는 날짜들. 날짜 고르기가 **있는 날만** 고르게 한다."""
    return {"days": events.days()}


@app.get("/api/events/batches")
def event_batches(from_day: str = None, to_day: str = None):
    """로그 화면의 폴더 필터 목록. **저널에서, 기간을 따라** 뽑는다.

    저널은 지워지지 않아서 없어진 폴더 이름이 영원히 남는다. 기간을 좁히면
    그 기간에 실제로 돈 폴더만 나온다(events.batches 주석 참고).
    """
    return {"batches": events.batches(from_day=from_day, to_day=to_day)}


@app.get("/api/export.csv")
def export_csv(job: str = None, batch: list[str] = Query(default=[]),
               event: str = None, mode: str = None, since: float = None,
               before: float = None, q: str = None, limit: int = 5000,
               from_day: str = None, to_day: str = None):
    """지금 화면에 걸린 조건 그대로 내보낸다.

    **보이는 것과 받는 것이 같아야 한다.** 내보내기 전용 조건을 따로 두면 화면
    에서 거른 것과 파일에 담긴 것이 달라지고, 그걸 알아채는 것은 파일을 연 뒤다.

    폴더는 여럿 줄 수 있다(``?batch=kbs&batch=mbc``). 아무것도 안 주면 전부다.

    ``from_day``/``to_day`` 는 ``2026-08-18`` 형식이고 **양 끝을 포함**한다.
    날짜 해석은 서버가 한다 — 화면이 브라우저 타임존으로 계산하면 다른 지역에서
    열었을 때 저널의 그 날짜와 다른 구간을 가리킨다.

    **UTF-8 BOM 을 붙인다.** 안 붙이면 한국어 윈도우 엑셀이 파일명을 깨뜨린다 —
    받아서 열었을 때 깨져 있으면 그게 첫인상이 된다.
    """
    import csv
    import io

    picked = [b for b in (batch or []) if b]
    rows = []
    if picked:
        # 폴더별로 따로 읽고 합친다. events.read 는 폴더 하나만 받는다.
        for b in picked:
            rows += events.read(job=job, batch=b, event=event, mode=mode,
                                since=since, before=before, q=q, limit=limit,
                                from_day=from_day, to_day=to_day)
        rows.sort(key=lambda r: -(r.get("ts") or 0))
        rows = rows[:limit]
    else:
        rows = events.read(job=job, event=event, mode=mode, since=since,
                           before=before, q=q, limit=limit,
                           from_day=from_day, to_day=to_day)

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")     # 엑셀은 CRLF 를 기대한다
    w.writerow([name for name, _get in EXPORT_COLUMNS])
    for raw in rows:
        r = events.decorate(raw)
        w.writerow([get(r) for _name, get in EXPORT_COLUMNS])

    bits = []
    if picked and len(picked) <= 3:
        bits += picked
    if from_day or to_day:
        bits.append(f"{from_day or '처음'}_{to_day or '지금'}")
    if not bits:
        bits.append((timefmt.iso(time.time()) or "")[:10])
    name = "face-anonymizer-log-" + "-".join(bits) + ".csv"
    return Response(
        content="\ufeff" + buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition":
                 f'attachment; filename="{name}"; '
                 f"filename*=UTF-8''{quote(name)}"})


@app.get("/api/jobs")
def list_jobs(limit: int = config.LIST_LIMIT, offset: int = 0, status: str = None):
    """작업 목록. 실행 순서, 한 페이지 ``limit`` 건.

    **거르는 일도 페이지를 나누는 일도 서버가 한다.** 잘라 보낸 뒤 화면에서
    거르면, 완료가 창 밖에 있을 때 '완료' 탭이 비어 보인다.

    ``status`` 는 실제 상태 하나이거나 ``active`` 다. ``active`` 는
    **수행중 + 대기**, 즉 '아직 처리할 일' 이고 화면의 기본 목록이다. 끝난 것을
    같이 섞으면 수백 건짜리 배치에서 지금 할 일이 완료 기록에 파묻힌다.

    전체 건수는 여기서 주지 않는다. 페이지를 넘기는 쪽은 ``GET /api/status`` 의
    ``counts`` 를 쓴다 — 세는 일과 보여 주는 일은 다른 경로다(docs/issues/006).

    대기 순번은 한 번에 계산한다 — 작업마다 전체를 다시 훑으면 O(N^2) 이고,
    전체 수행으로 수백 건을 넣으면 목록 한 번에 수십만 번 반복하게 된다.
    거르기 전 전체에서 매기므로 좁혀 봐도, 페이지를 넘겨도 순번은 그대로다.
    """
    rows = jobs.all_jobs()
    order = sorted((j for j in rows if j.status == "queued"),
                   key=lambda x: x.created)
    ahead = {j.id: i for i, j in enumerate(order)}
    if status == "active":
        rows = [j for j in rows if j.status in ("running", "queued")]
    elif status:
        rows = [j for j in rows if j.status == status]
    rows.sort(key=list_key)
    start = max(0, offset)
    return [jobs.snapshot(j, ahead.get(j.id, 0))
            for j in rows[start:start + max(0, limit)]]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = jobs.find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    return jobs.snapshot(j, jobs.queued_ahead_of(j))


@app.post("/api/jobs/cancel-all")
def cancel_all_jobs():
    """진행중인 작업을 한 번에 멈춘다. 대기는 즉시, 수행중은 몇 초 안에.

    폴더를 잘못 넣었거나 파라미터가 틀렸을 때 필요한 건 '전부 멈춰' 다. 화면이
    한 건씩 500번 호출하게 두면 그 사이에도 워커가 큐를 꺼내서, 취소하는 동안
    새 작업이 시작된다. 표시는 서버가 락 한 번에 한다(jobs.cancel_all).

    **S3 는 건드리지 않는다.** 원본도 이미 올라간 결과물도 그대로다. 여기서
    사라지는 것은 아직 하지 않은 일과, 하던 일의 중간 산물뿐이다.

    이 경로는 ``/api/jobs/{jid}/cancel`` 보다 먼저 선언해야 한다 — 뒤에 두면
    ``cancel-all`` 이 작업 id 로 잡힌다.
    """
    queued, running = jobs.cancel_all()
    log.info("전체 취소 — 대기 %d건 취소, 수행중 %d건에 중단 요청", queued, running)
    return {"cancelled": queued + running, "queued": queued, "running": running,
            "counts": jobs.counts()}


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    """대기 중이면 즉시, 수행 중이면 다음 진행 보고에서 끊는다.

    잘못 넣은 배치를 끝날 때까지 기다릴 이유가 없다. 수행 중인 작업은 진행
    콜백에서만 안전하게 끊을 수 있어(프레임 경계) 표시만 남기고 워커가 처리한다.
    """
    j = jobs.find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    if j.status == "review":
        # 취소를 허용해도 상태는 그대로다(취소는 queued 만 즉시 바꾼다). 아무 일도
        # 안 일어나는데 200 이 나가고, 사람은 취소했다고 믿는다.
        raise errors.JOB_IN_REVIEW(f"status={j.status}", status=j.status)
    if j.status in ("done", "failed", "cancelled"):
        raise errors.JOB_NOT_CANCELLABLE(f"status={j.status}", status=j.status)
    with jobs.LOCK:
        j.cancel = True
        if j.status == "queued":
            j.status, j.finished = "cancelled", time.time()
    jobs.save_job(j)
    return jobs.snapshot(j, jobs.queued_ahead_of(j))


@app.post("/api/jobs/{jid}/review")
def review_job(jid: str, body: dict = Body(default={})):
    """검수 판정 — **여기서만 완료가 된다.**

    처리가 끝나도 사람이 봐야 하는 사유가 있으면 상태가 ``review`` 로 멈춘다.
    얼굴이 하나도 안 잡힌 영상은 원본이 그대로 나간 것인데, 그게 얼굴 없는
    영상이라 정당한 0 인지 설정이 틀려서 0 인지 **코드가 구분할 수 없다**
    (docs/issues/008). 그 판단을 사람이 내리는 자리다.

    ``approve`` 면 완료로, ``reject`` 면 실패로 넘어간다. 어느 쪽이든 누가 언제
    무슨 사유로 넘겼는지가 작업 기록과 저널에 남는다 — 나중에 "이 영상 왜
    완료로 되어 있냐" 에 답할 수 있어야 한다.

    **반려해도 버킷의 결과물은 지우지 않는다.** 지우는 것은 되돌릴 수 없고,
    반려 사유가 "다시 처리하면 될 것" 일 수도 있어서다. 대신 결과물 키를
    응답과 저널에 남기므로 납품 폴더에서 빼는 판단을 사람이 할 수 있다.
    """
    action = str((body or {}).get("action") or "").strip().lower()
    note = str((body or {}).get("note") or "").strip()[:500]
    if action not in ("approve", "reject"):
        raise errors.REVIEW_ACTION_INVALID(f"action={action or '(없음)'}")
    j = jobs.find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)

    now = time.time()
    # **확인과 변경을 한 락 안에서 한다.** 밖에서 확인하면 탭 두 개에서 동시에
    # 누를 때 둘 다 통과해, 나중 것이 앞의 판정을 덮어쓴다 — 승인해 둔 건이
    # 조용히 실패로 바뀔 수 있다.
    with jobs.LOCK:
        if j.status != "review":
            raise errors.JOB_NOT_IN_REVIEW(f"status={j.status}", status=j.status)
        j.status = "done" if action == "approve" else "failed"
        j.finished = now
        j.reviewed = {"action": action, "note": note,
                      "at": round(now, 3), "at_iso": timefmt.iso(now)}
        if action == "reject":
            # 화면이 실패 카드와 같은 모양으로 그릴 수 있게 오류 형식을 맞춘다.
            j.error = {"code": "review_rejected", "title": "검수에서 반려되었습니다",
                       "detail": note or " / ".join(i["message"] for i in j.review),
                       "hint": "결과물은 버킷에 남아 있습니다. 납품 폴더에서 "
                               "빼거나 설정을 바꿔 다시 처리해 주세요.",
                       "retryable": False, "policy": "permanent"}
    jobs.save_job(j)
    log.info("%s  %s  %s", "☑ 검수 승인" if action == "approve" else "☒ 검수 반려",
             j.name, note or "")
    events.emit("job.reviewed", job=j.id, name=j.name, batch=j.batch or None,
                action=action, note=note or None,
                codes=[i.get("code") for i in (j.review or [])],
                s3_output=j.s3_output or None)
    return jobs.snapshot(j, 0)


@app.get("/api/jobs/{jid}/result")
def job_result(jid: str):
    """결과물이 어디 있는지 알려준다. **파일은 우리가 흘려보내지 않는다.**

    S3 작업이면 presigned URL 을 준다 — GPU 서버가 파일 전송까지 떠안을 이유가
    없고, 로컬 사본이 보관 기간에 정리돼도 S3 원본은 남아 있다. 예전에는
    ``/download`` 로 직접 내보내기도 했는데, 그건 테스트용으로 둔 것이었고
    **상태가 바뀌는 시점과 로컬 정리 사이의 틈**에서 500 이 나는 경쟁까지
    있었다(정리 뒤에 읽으면 파일이 없다). 지금은 위치만 알려 준다.
    """
    j = jobs.find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    # 검수 중에도 알려 준다 — **보지 않고는 판정할 수 없다.** download 만 열어
    # 두면 presigned URL 로 보려는 쪽이 막힌다(둘은 같은 목적의 두 경로다).
    if j.status not in ("done", "review"):
        raise (errors.JOB_FAILED(j.error.get("detail", ""))
               if j.status == "failed"
               else errors.JOB_NOT_FINISHED(f"status={j.status}"))
    out = {"id": j.id, "name": naming.output_name(j.name),
           "status": j.status, "review": list(j.review or ()),
           "s3_key": j.s3_output or None}
    store = s3mod.get_store()
    if j.s3_output and store is not None:
        out["download_url"] = store.presigned_url(j.s3_output)
        out["expires_in"] = s3mod.URL_TTL
        out["via"] = "s3"
    else:
        # 직접 업로드분은 버킷에 없다. 결과물은 서버 작업 폴더에 있고, 그건
        # 운영하는 사람이 가져갈 몫이다 — 우리가 스트리밍하지 않는다.
        if not j.output or not os.path.exists(j.output):
            # 보관 기간이 지나 정리됐다. 버킷 사본도 없으니 정말 없는 것이다.
            raise errors.RESULT_EXPIRED(jid)
        out["download_url"] = None
        out["via"] = "local"
        out["hint"] = "S3 로 제출한 작업이 아니라 서버 작업 폴더에만 있습니다."
    return out


@app.delete("/api/jobs/{jid}", status_code=204)
def delete_job(jid: str):
    j = jobs.find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    if j.status == "review":
        # 화면은 검수 카드에 삭제 버튼을 안 그리지만 API 는 열려 있었다.
        # 판정 없이 지우면 "왜 걸렸나" 가 같이 사라진다.
        raise errors.JOB_IN_REVIEW("판정 먼저", status=j.status)
    if j.status in ("queued", "running"):
        raise errors.JOB_NOT_CANCELLABLE(
            "진행 중이다. /cancel 로 먼저 취소하라", status=j.status)
    with jobs.LOCK:
        jobs.JOBS.pop(jid, None)
    shutil.rmtree(j.workdir or os.path.join(config.JOBS_DIR, jid), ignore_errors=True)
