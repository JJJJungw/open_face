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

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from ..core.anonymize import METHODS
from ..core.pipeline import parse_bitrate
from ..storage import naming
from ..storage import s3 as s3mod
from . import config, errors, jobs, metrics, worker
from .config import JOB_DEFAULTS
from .webui import INDEX_HTML

log = logging.getLogger(__name__)

_sweeper = None


@asynccontextmanager
async def lifespan(_app):
    global _sweeper
    os.makedirs(config.JOBS_DIR, exist_ok=True)
    jobs.recover_orphans()
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
    """오케스트레이터용 최소 응답. 디스크를 훑지 않는다."""
    return {"ready": worker.is_ready(), "busy": worker.is_busy(),
            "queued": jobs.queue_depth(), "free_mb": jobs.free_mb(),
            "model_error": worker.model_error}


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
def s3_progress(prefix: str = ""):
    """입력 폴더별 진척률. 큐가 아니라 **버킷** 기준이다.

    큐 지표는 지금 들어와 있는 것만 안다. 데이터셋을 통째로 돌리는 작업에서
    정작 궁금한 건 전체 중 얼마나 남았는지이고, 그건 결과 버킷에 있다.
    """
    store = s3mod.get_store()
    if store is None:
        raise errors.S3_NOT_CONFIGURED()
    if ".." in prefix:
        raise errors.INVALID_KEY(prefix)
    try:
        rows = metrics.folder_progress(store, prefix or store.root_prefix)
    except s3mod.S3Error as e:
        raise (e.problem or errors.S3_UPSTREAM)(str(e)) from e
    return {"prefix": prefix, "output_prefix": store.output_prefix,
            "folders": rows,
            "total": sum(r["total"] for r in rows),
            "done": sum(r["done"] for r in rows)}


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
                done = store.processed_keys()
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
                _job, snap = worker.enqueue(name, dict(params), s3_key=key)
                accepted.append({"id": snap["id"], "name": name, "s3_key": key})
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


@app.get("/api/jobs")
def list_jobs(limit: int = config.LIST_LIMIT, status: str = None):
    """작업 목록. 최신순, 기본 100건.

    대기 순번은 한 번에 계산한다 — 작업마다 전체를 다시 훑으면 O(N^2) 이고,
    전체 수행으로 수백 건을 넣으면 목록 한 번에 수십만 번 반복하게 된다.
    """
    rows = jobs.all_jobs()
    if status:
        rows = [j for j in rows if j.status == status]
    order = sorted((j for j in rows if j.status == "queued"),
                   key=lambda x: x.created)
    ahead = {j.id: i for i, j in enumerate(order)}
    return [jobs.snapshot(j, ahead.get(j.id, 0)) for j in rows[:max(0, limit)]]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = jobs.find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    return jobs.snapshot(j, jobs.queued_ahead_of(j))


@app.get("/api/jobs/{jid}/download")
def download(jid: str):
    j = jobs.find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    if j.status != "done":
        raise (errors.JOB_FAILED(j.error.get("detail", "") if isinstance(j.error, dict) else j.error)
           if j.status == "failed" else errors.JOB_NOT_FINISHED(f"status={j.status}"))
    if not j.output or not os.path.exists(j.output):
        # 로컬 사본은 정리됐어도 S3 원본은 남아 있다.
        store = s3mod.get_store()
        if j.s3_output and store is not None:
            return RedirectResponse(store.presigned_url(j.s3_output),
                                    status_code=302)
        raise errors.RESULT_EXPIRED(jid)
    name = naming.output_name(j.name)
    return FileResponse(j.output, media_type="video/mp4", filename=name)


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    """대기 중이면 즉시, 수행 중이면 다음 진행 보고에서 끊는다.

    잘못 넣은 배치를 끝날 때까지 기다릴 이유가 없다. 수행 중인 작업은 진행
    콜백에서만 안전하게 끊을 수 있어(프레임 경계) 표시만 남기고 워커가 처리한다.
    """
    j = jobs.find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    if j.status in ("done", "failed", "cancelled"):
        raise errors.JOB_NOT_CANCELLABLE(f"status={j.status}", status=j.status)
    with jobs.LOCK:
        j.cancel = True
        if j.status == "queued":
            j.status, j.finished = "cancelled", time.time()
    jobs.save_job(j)
    return jobs.snapshot(j, jobs.queued_ahead_of(j))


@app.get("/api/jobs/{jid}/result")
def job_result(jid: str):
    """결과물 받는 방법을 알려준다.

    S3 작업이면 **presigned URL** 을 준다 — GPU 서버가 파일 전송까지 떠안을
    이유가 없고, 로컬 사본이 보관 기간에 정리돼도 S3 원본은 남아 있다.
    """
    j = jobs.find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    if j.status != "done":
        raise (errors.JOB_FAILED(j.error.get("detail", ""))
               if j.status == "failed"
               else errors.JOB_NOT_FINISHED(f"status={j.status}"))
    out = {"id": j.id, "name": naming.output_name(j.name),
           "s3_key": j.s3_output or None}
    store = s3mod.get_store()
    if j.s3_output and store is not None:
        out["download_url"] = store.presigned_url(j.s3_output)
        out["expires_in"] = s3mod.URL_TTL
        out["via"] = "s3"
    else:
        out["download_url"] = f"/api/jobs/{j.id}/download"
        out["via"] = "server"
    return out


@app.delete("/api/jobs/{jid}", status_code=204)
def delete_job(jid: str):
    j = jobs.find_job(jid)
    if j is None:
        raise errors.JOB_NOT_FOUND(jid)
    if j.status in ("queued", "running"):
        raise errors.JOB_NOT_CANCELLABLE(
            "진행 중이다. /cancel 로 먼저 취소하라", status=j.status)
    with jobs.LOCK:
        jobs.JOBS.pop(jid, None)
    shutil.rmtree(j.workdir or os.path.join(config.JOBS_DIR, jid), ignore_errors=True)
