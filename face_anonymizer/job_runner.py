"""무DB 잡 러너 — 큐 프로토콜만 아는 워커의 본체.

**이 모듈은 우리 서버(service/)를 모른다.** 입력은 잡 페이로드(dict) 하나이고
바깥세상과의 접점은 셋뿐이다.

    ① 입력  presigned GET  으로 받는다
    ② 산출  presigned PUT  으로 올린다
    ③ 진행  호출자가 준 on_heartbeat 콜백으로 알린다

DB 도, 버킷 이름도, 자격 증명도, 우리 큐도 없다. 그래서 이 러너를 드는 컨테이너는
이미지·클러스터·계정을 바꿔도 설정이 없다.

왜 이 모양인가
--------------
붙을 곳(RebornStudio)이 이미 이 프로토콜을 1호(트랜스코더)로 검증해 두었고,
GPU 워크로드(deidentify·cliper·infer)를 같은 형판으로 찍겠다고 문서에 적어 두었다
(packages/reborn-contracts/queues.md). 우리가 새 계약을 제안하는 것보다 이미
돌아가는 계약에 맞추는 편이 붙이는 비용이 훨씬 싸다.

**상태는 우리가 갖지 않는다.** 리스·펜싱 토큰·재시도 횟수·상한은 잡을 준 쪽이
DB 로 관리한다. 우리는 실패를 일시(transient)와 영구로 **1차 분류만** 해서
돌려준다 — 몇 번까지 다시 해볼지는 우리가 알 수 있는 정보가 아니다.

그래서 service/worker.py 의 재시도 백오프(docs/issues/003)와 재시작 복구
(docs/issues/002)는 **이 경로에서 쓰지 않는다.** 버리는 게 아니라, 우리가 큐를
소유하는 단독 운영에서만 쓴다. 같은 일을 두 곳에서 하면 서로를 방해한다.

잡 페이로드
-----------
::

    {
      "video_id": "uuid",
      "token": "펜싱 토큰 — 완료 보고에 그대로 되돌린다",
      "input_url": "presigned GET (TTL 4h — 만료 403 은 일시로 분류한다)",
      "targets": [
        {"label": "deid-720p",
         "height": 720,                  # null 이면 원본 해상도 유지 (업스케일 금지)
         "bitrate": "3500k", "max_bitrate": "4000k",
         "method": "mosaic", "conf": 0.25, "imgsz": 1280,
         "put_url": "presigned PUT", "content_type": "video/mp4"}
      ],
      "weights_url": "(선택) 서명된 GET — FA_WEIGHTS_SOURCE=url 일 때만 본다",
      "heartbeat_every_s": 60
    }

산출 키는 잡을 준 쪽이 결정론적으로 정하므로 완료 보고에 되돌리지 않는다.
"""

import logging
import os
import tempfile
import time

from . import params

log = logging.getLogger(__name__)

# 진행 보고가 이만큼 멎으면 환경 문제로 보고 일시 실패로 끊는다.
STALL_S = float(os.environ.get("FA_STALL_S", 300))

# 내려받기·인코딩·올리기가 전체에서 차지하는 몫. 하트비트에 실어 보내는 '진행 초'
# 는 정확할 필요가 없다 — 저쪽은 리스 연장에만 쓴다.
_DOWNLOAD_SHARE = 0.15


# ── 검수 딱지 ────────────────────────────────────────────────────────────────
#
# **실패가 아니라 딱지다.** 얼굴이 없는 풍경 영상은 검출 0 이 정당한 결과이므로
# 오류로 던지면 안 된다. 그런데 가중치 손상·회전된 영상·잘못된 imgsz·HDR 톤매핑
# 실패도 결과가 똑같이 0 이고, 그때는 **원본이 그대로 나간다.**
#
# 둘을 코드가 구분할 방법은 없다. 그래서 판단을 사람에게 넘기되 **사실은 반드시
# 같이 보낸다** — "얼굴이 하나도 안 잡혔는데, 얼굴이 없는 영상이 맞나요?"
#
# 이걸 안 하면 비식별화가 하나도 안 된 원본이 '비식별화 완료' 로 납품된다.
# 화질 문제가 아니라 개인정보 사고다.
REVIEW = {
    "no-detections":
        "얼굴이 하나도 검출되지 않았습니다. 원본이 그대로 나갔습니다 — "
        "얼굴이 없는 영상이 맞는지 한 번 확인해 주세요.",
    "low-detection-rate":
        "얼굴이 잡힌 프레임이 매우 적습니다. 영상 회전·해상도·임계값 때문에 "
        "놓쳤을 수 있어 한 번 봐 주시면 좋겠습니다.",
    "decode-partial":
        "영상을 끝까지 읽지 못했습니다. 뒷부분이 결과물에서 빠졌을 수 있습니다.",
    "decode-short":
        "읽어 낸 프레임이 원본에 적힌 수보다 적습니다. 결과물 길이를 확인해 "
        "주세요.",
}

# 알아 두면 좋지만 사람을 부를 일은 아닌 것들.
NOTICE = {
    "decode-unverified": "프레임 수를 확인할 수 없어 완결성 검사를 건너뛰었습니다.",
    "audio": "오디오를 원본 그대로 옮기지 못했습니다.",
}


def review_of(warnings):
    """파이프라인 경고 → 검수 딱지. 사람이 봐야 하는 것만 골라 낸다.

    경고 문자열은 ``low-detection-rate: 0.50%`` 처럼 뒤에 수치가 붙는다.
    앞의 코드로 가르고 원문은 ``detail`` 에 그대로 남긴다 — 요약만 남기면
    나중에 "얼마나 낮았는데?" 에 답할 수 없다.
    """
    out = []
    for w in warnings or ():
        code = str(w).split(":")[0].strip()
        text = REVIEW.get(code)
        if text:
            out.append({"code": code, "detail": str(w), "message": text})
    return out


STAGES = ("ingest", "detect", "track", "render", "audio", "total")


def timing_of(result):
    """파이프라인 단계별 시간(초). 없으면 빈 dict.

    **``audio`` 는 이름과 다르다.** 오디오만이 아니라 ``finalize_output()`` 전체를
    잰다 — H.264 재인코딩 + 스케일 + 오디오 합성 + 프레임 수 검증. 이 값이 크다고
    오디오를 파면 엉뚱한 데를 뒤지게 된다. 필드 이름은 job.json 호환 때문에 그대로
    두고, 읽는 쪽(tools/msa_smoke.py)이 제대로 된 이름으로 표시한다.

    소수점 첫째 자리로 반올림하지 않는다 — 짧은 클립에서는 단계 시간이 전부
    0.0 이 되어 어디가 느린지 안 보인다.
    """
    t = getattr(result, "timing", None)
    if t is None:
        return {}
    return {s: round(getattr(t, s, 0.0) or 0.0, 3) for s in STAGES
            if hasattr(t, s)}


def notices_of(warnings):
    return [{"code": c, "detail": str(w), "message": NOTICE[c]}
            for w in warnings or ()
            for c in [str(w).split(":")[0].strip()] if c in NOTICE]


class JobError(RuntimeError):
    """잡 실패. ``transient`` 면 재큐잉 대상(시도 횟수 미소모)이다."""

    def __init__(self, message, *, transient, stage=""):
        super().__init__(message)
        self.transient = transient
        self.stage = stage

    def as_dict(self):
        """화면에 그대로 띄울 수 있는 모양.

        ``str(e)`` 만 보내면 UI 에 "presign 만료" 같은 **우리 내부 문구**가 뜬다.
        코드는 기계가, 제목·힌트는 사람이 읽는다 — service/errors.py 가 HTTP 쪽에
        하는 일과 같은 분담이다. 그쪽을 임포트하지 않는 이유는 fastapi 를 끌고
        오기 때문이다. 이 컨테이너에는 웹 프레임워크가 들어가지 않는다.
        """
        f = STAGE_FACE.get(self.stage or "", STAGE_FACE["unknown"])
        return {"code": self.stage or "unknown", "title": f[0], "hint": f[1],
                "detail": str(self), "retryable": bool(self.transient)}


# 실패 단계 → 사람이 읽을 제목과 다음에 할 일. 화면이 이걸 그대로 띄운다.
STAGE_FACE = {
    "payload": ("잡 내용이 비어 있습니다", "보낸 쪽에서 targets 를 확인해 주세요"),
    "download": ("원본을 내려받지 못했습니다",
                 "서명 URL 이 만료됐을 수 있습니다. 다시 발급하면 됩니다"),
    "weights": ("모델 가중치를 받지 못했습니다", "잠시 뒤 자동으로 다시 시도합니다"),
    "model": ("검출기를 올리지 못했습니다", "잠시 뒤 자동으로 다시 시도합니다"),
    "process": ("영상을 처리하지 못했습니다",
                "같은 파일로 다시 해도 결과가 같습니다. 원본을 확인해 주세요"),
    "stall": ("처리가 도중에 멎었습니다", "잠시 뒤 자동으로 다시 시도합니다"),
    "upload": ("결과를 올리지 못했습니다",
               "서명 URL 이 만료됐을 수 있습니다. 다시 발급하면 됩니다"),
    "unknown": ("알 수 없는 오류입니다", "잠시 뒤 자동으로 다시 시도합니다"),
}

# ---------------------------------------------------------------------------
# 진행률 — **이 값은 우리만 안다.**
#
# 목록도 순번도 남은 건수도 잡을 준 쪽이 안다. 우리만 아는 것은 딱 하나,
# "지금 손에 든 이 영상이 어디쯤 가고 있나" 다. 그래서 하트비트에 실어 보낸다.
# 안 실어 보내면 화면은 스피너까지밖에 못 그린다.
#
# **단계별 퍼센트를 그대로 주면 안 된다.** 검출 100% → 렌더 0% 로 떨어지는데,
# 보는 사람에게는 그냥 되감긴 것으로 보인다. 그래서 단계마다 전체에서 차지하는
# 몫을 주고 **한 줄로 이어 붙인다.**
#
# 몫은 L40S 실측(인제스트 12.6 · 검출 13.6 · 렌더 13.0 · 최종 0.8초)에서 왔다.
# 인스턴스가 바뀌면 비율도 조금 바뀌지만, 진행률은 **줄지만 않으면** 쓸 만하다.
STAGE_SPAN = (                      # (단계, 시작 지점, 차지하는 몫)
    ("download",  0.00, 0.08),
    ("transcode", 0.08, 0.22),
    ("detect",    0.30, 0.30),
    ("track",     0.60, 0.02),
    ("render",    0.62, 0.30),
    ("upload",    0.92, 0.08),
)
_SPAN = {name: (base, width) for name, base, width in STAGE_SPAN}

# 화면에 띄울 단계 이름. 코드 이름을 그대로 보여 주면 사용자가 읽을 말이 아니다.
STAGE_LABEL = {"download": "원본 받는 중", "transcode": "읽을 수 있게 변환 중",
               "detect": "얼굴 찾는 중", "track": "추적 잇는 중",
               "render": "가리는 중", "upload": "결과 올리는 중"}


class _Beat:
    """하트비트와 정체 감시를 한 곳에서 본다.

    파이프라인의 진행 콜백은 프레임마다 불린다. 그때마다 큐로 메시지를 보내면
    초당 수십 건이 되므로 **시간으로 눌러서** 보낸다.

    정체 감시를 별도 스레드로 두지 않은 것은 선택이다. 스레드를 쓰면 멎은
    ffmpeg 도 끊을 수 있지만, 데몬 스레드를 남긴 채 임시 디렉터리를 지우게 되어
    더 나쁜 상태가 된다. 콜백이 아예 안 오는 종류의 멈춤(ffmpeg 내부 행)은 저쪽
    **리스 만료 → 회수 → 재전달**이 처리하고, 뒤늦게 끝난 우리 보고는 펜싱
    토큰이 걸러 낸다. 그쪽 설계가 이미 그 경우를 전제한다.
    """

    def __init__(self, on_heartbeat, every_s):
        self.on_heartbeat = on_heartbeat
        self.every = max(1.0, float(every_s or 60))
        self.t0 = time.monotonic()
        self.last_beat = self.t0
        self.last_move = self.t0
        self.mark = None
        self.stage = None
        self.percent = 0.0          # 절대 되돌아가지 않는다 (아래 참고)

    def __call__(self, position, stage=None, done=None, total=None):
        """``position`` 은 아무 단조 증가 값이면 된다(프레임 수·바이트 수).

        ``stage``/``done``/``total`` 은 화면용이다. 없어도 정체 감시는 돈다.
        """
        now = time.monotonic()
        if position != self.mark:
            self.mark, self.last_move = position, now
        elif now - self.last_move > STALL_S:
            raise JobError(f"진행이 {int(STALL_S)}초 동안 멎었습니다",
                           transient=True, stage="stall")
        if stage:
            self.stage = stage
            self._advance(stage, done, total)
        if now - self.last_beat >= self.every:
            self.last_beat = now
            if self.on_heartbeat is not None:
                self.on_heartbeat(self.snapshot())

    def _advance(self, stage, done, total):
        """전체 진행률을 갱신한다. **줄어들지 않는다.**

        단계가 건너뛰어질 수 있다 — h264 원본이면 전사가 통째로 없다. 그때
        진행률이 뒤로 가면 화면이 되감긴 것처럼 보이므로, 새 값이 더 작으면
        버린다. 앞으로만 가는 진행률은 조금 부정확해도 읽히지만, 뒤로 가는
        진행률은 고장으로 읽힌다.
        """
        base, width = _SPAN.get(stage, (self.percent, 0.0))
        frac = 0.0
        if total:
            frac = min(1.0, max(0.0, float(done or 0) / float(total)))
        self.percent = max(self.percent, round((base + width * frac) * 100, 1))

    def snapshot(self):
        """하트비트에 실을 것. 화면이 이걸로 진행바와 한 줄 설명을 그린다."""
        elapsed = round(time.monotonic() - self.t0, 1)
        eta = None
        # 5% 아래에서는 추정이 요동쳐서 "남은 시간 47분" 같은 값이 나온다.
        # 모르는 구간에서는 아예 안 보내는 편이 낫다.
        if self.percent >= 5:
            eta = round(elapsed * (100 - self.percent) / self.percent)
        return {"elapsed_s": elapsed, "percent": self.percent,
                "stage": self.stage,
                "stage_label": STAGE_LABEL.get(self.stage or ""),
                "eta_s": eta}


def target_params(t):
    """잡의 타깃 사양 → 파이프라인 인자.

    저쪽 트랜스코더는 화질을 ``crf`` 로 말하고 우리 납품 기준은 타깃 비트레이트다
    (720p / 3500~4000 kbps). 둘 다 받아 준다 — 어휘를 하나로 강제하면 붙이는
    쪽이 자기 파이프라인을 고쳐야 한다.
    """
    # **잡이 말하지 않은 것은 서비스 기본값으로 채운다.**
    #
    # 전에는 안 채웠다. 그러면 파이프라인 시그니처 기본값(batch_size=1,
    # imgsz=960)으로 떨어지는데, 그건 "안전한 최소값" 이지 우리가 튜닝한 값이
    # 아니다. L40S 에서 GPU 를 20% 만 쓰고 한 편에 49.5초를 썼다
    # (docs/issues/009). 잡 페이로드는 **다르게 하고 싶은 것만** 적는 자리다.
    p = {k: v for k, v in params.DEFAULTS.items() if k in params.JOB_OVERRIDABLE}
    for k in params.JOB_OVERRIDABLE:
        if t.get(k) is not None:
            p[k] = t[k]

    # height 는 **키가 있는데 값이 null** 인 것이 뜻을 갖는다 — 저쪽 규약에서
    # "스케일 생략(업스케일 금지)" 이다. 없는 것과 구분해야 해서 따로 본다.
    if "height" in t:
        p["height"] = t["height"] or 0            # 0 = 해상도 그대로

    # 화질 정책은 둘 중 하나만 걸어야 한다(docs/issues 의 AV1 상한 사고). 타깃이
    # crf 로 말했으면 우리 기본 타깃 비트레이트를 비워서 CRF 쪽으로 넘긴다.
    #
    # 판정은 **잡이 무엇을 말했는지**(t)로 한다. 기본값을 채운 뒤(p)로 보면
    # 둘 다 늘 들어 있어서 이 분기가 영영 안 걸린다.
    if t.get("crf") is not None and t.get("bitrate") is None:
        p["bitrate"] = p["max_bitrate"] = ""
    return p


def is_oom(exc):
    """CUDA 메모리 부족인가. torch 를 임포트하지 않고 판정한다.

    ``torch.cuda.OutOfMemoryError`` 로 잡으면 torch 가 없는 환경(테스트·CPU
    전용)에서 이 모듈이 통째로 못 뜬다. 메시지로 보는 것이 지저분해 보여도,
    여기서 torch 를 끌고 오는 대가보다 싸다.
    """
    if type(exc).__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    s = str(exc).lower()
    return "out of memory" in s or "cuda oom" in s


def run_target(anonymizer, src, out, target, *, progress=None, note=None):
    """타깃 하나를 처리한다. **메모리가 부족하면 배치를 줄여 다시 해 본다.**

    운영 인스턴스는 개발기보다 작다. 개발기가 45GB 짜리라 batch 32 가 여유롭다고
    그 값을 기본으로 박아 두면, 인스턴스를 줄이는 순간 CUDA OOM 이 난다. 그리고
    OOM 은 파이프라인 예외라 **영구 실패로 분류돼 큐 전체가 재시도 없이 죽는다.**

    메모리 부족은 이 영상의 문제가 아니라 **환경의 문제**다. 그래서 실패로 던지기
    전에 배치를 절반씩 줄여 가며 다시 해 본다. 1까지 내려가도 안 되면 그때는
    일시 실패다 — 더 작은 배치로 다시 시도할 여지가 남아 있지 않고, 다른(더 큰)
    워커에서는 될 수 있기 때문이다.
    """
    p = target_params(target)
    batch = int(p.get("batch_size") or 1)
    first = None
    while True:
        p["batch_size"] = batch
        try:
            return anonymizer.process(src, out, progress=progress, **p)
        except JobError:
            raise
        except Exception as e:                      # noqa: BLE001
            if not is_oom(e):
                raise
            first = first or e
            if batch <= params.BATCH_MIN:
                raise JobError(
                    f"GPU 메모리가 부족합니다 (batch {batch} 까지 낮췄습니다). "
                    f"더 큰 워커에서는 될 수 있습니다: {e}",
                    transient=True, stage="oom") from first
            batch = max(params.BATCH_MIN, batch // 2)
            log.warning("GPU 메모리 부족 — 배치를 %d 로 낮춰 다시 시도한다", batch)
            if note is not None:
                note(batch)
            _free_cuda()


def _free_cuda():
    """다시 시도하기 전에 캐시를 비운다. torch 가 없으면 아무것도 안 한다."""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                               # noqa: BLE001
        pass


def _weights_ready(job):
    """검출기를 만들기 전에 가중치를 갖춘다.

    조달처는 ``FA_WEIGHTS_SOURCE`` 가 정한다(s3 | baked | url). 지금 기본은 s3 이고,
    저쪽 규약(자격 증명 0)으로 갈 때는 url 로 바꾼다 — **이 함수만 그 사실을 안다.**
    미결 D1: docs/integration/rebornstudio.md
    """
    from .core.paths import DEFAULT_WEIGHTS
    from .storage import weights as weights_store

    try:
        weights_store.ensure(DEFAULT_WEIGHTS, url=job.get("weights_url"))
    except weights_store.WeightsUnavailable as e:
        # 모델을 못 갖춘 것은 이 영상의 문제가 아니다. 다른 워커·다음 배포에서는
        # 될 수 있으므로 일시로 분류한다 — 영구로 두면 큐 전체가 상한까지 태워진다.
        raise JobError(str(e), transient=True, stage="weights") from e


def run_job(job, *, on_heartbeat=None, anonymizer=None):
    """잡 1건 실행. ``{"elapsed_s": ..., "targets": [...]}`` 를 돌려준다.

    Args:
        on_heartbeat: ``fn(진행_초)``. 호출자가 리스 연장 메시지로 바꾼다.
        anonymizer:   테스트·재사용을 위한 주입구. 없으면 만들어 쓴다.

    Raises:
        JobError: 실패 사유. ``transient`` 로 재시도 가치를 표시한다.
    """
    started = time.monotonic()
    beat = _Beat(on_heartbeat, job.get("heartbeat_every_s"))
    targets = job.get("targets") or []
    if not targets:
        raise JobError("targets 가 비어 있습니다", transient=False, stage="payload")

    from .storage import transfer

    with tempfile.TemporaryDirectory(prefix="fa-job-") as tmp:
        src = os.path.join(tmp, "input.mp4")
        try:
            seen, size = [0], [0]

            def on_chunk(n):
                seen[0] += n
                beat(seen[0], "download", seen[0], size[0])

            transfer.fetch(job["input_url"], src, callback=on_chunk,
                           on_total=lambda n: size.__setitem__(0, n or 0))
        except transfer.TransferError as e:
            raise JobError(str(e), transient=e.transient, stage="download") from e

        if anonymizer is None:
            _weights_ready(job)
            from .core.pipeline import VideoAnonymizer
            try:
                anonymizer = VideoAnonymizer()
            except Exception as e:                  # noqa: BLE001 — 모델 로드 전 구간
                raise JobError(f"검출기를 올리지 못했습니다: {e}",
                               transient=True, stage="model") from e

        # **인코딩이 전부 끝난 뒤에 올린다.** 중간에 실패했는데 반쪽 산출이 이미
        # 올라가 있으면, 재시도가 성공할 때까지 버킷에 잘못된 결과가 남는다.
        done, shrunk = [], []
        for t in targets:
            out = os.path.join(tmp, f"{t.get('label') or 'out'}.mp4")
            try:
                res = run_target(anonymizer, src, out, t,
                                 progress=lambda s, d, n: beat((s, d), s, d, n),
                                 note=shrunk.append)
            except JobError:
                raise                               # 정체 감시가 던진 것 — 그대로
            except Exception as e:                  # noqa: BLE001
                # 같은 입력으로 다시 해도 같은 결과다. 재시도로 큐를 태우지 않는다.
                raise JobError(f"{t.get('label')}: {type(e).__name__}: {e}",
                               transient=False, stage="process") from e
            done.append((t, getattr(res, "output", out), res))

        for i, (t, path, _res) in enumerate(done):
            beat(("upload", i), "upload", i, len(done))
            try:
                transfer.put(t["put_url"], path,
                             t.get("content_type", "video/mp4"))
            except transfer.TransferError as e:
                raise JobError(str(e), transient=e.transient, stage="upload") from e

    elapsed = round(time.monotonic() - started, 1)

    # 검수 딱지는 타깃별이 아니라 **영상 단위**로 모은다. 저쪽이 판정하는 단위가
    # video_id 이고, 같은 영상의 타깃 둘에 같은 딱지가 두 번 붙어 봐야 사람이
    # 볼 것은 하나다.
    review, notices, seen = [], [], set()
    for _t, _p, r in done:
        for item in review_of(getattr(r, "warnings", ())):
            if item["detail"] not in seen:
                seen.add(item["detail"])
                review.append(item)
        for item in notices_of(getattr(r, "warnings", ())):
            if item["detail"] not in seen:
                seen.add(item["detail"])
                notices.append(item)
    if shrunk:
        # 조용히 줄이면 "왜 이 인스턴스에서만 느리지" 가 원인 불명으로 남는다.
        notices.append({"code": "batch-reduced",
                        "detail": f"batch_size → {min(shrunk)}",
                        "message": "GPU 메모리가 모자라 배치를 줄여 처리했습니다. "
                                   "이 워커에는 이 영상이 버겁습니다."})
    if review:
        log.warning("검수 필요 video_id=%s: %s", job.get("video_id"),
                    " / ".join(i["code"] for i in review))

    log.info("잡 완료: video_id=%s targets=%d %.1fs%s",
             job.get("video_id"), len(done), elapsed,
             " (검수 필요)" if review else "")
    return {
        "elapsed_s": elapsed,
        # 처리는 성공했지만 사람이 한 번 봐야 한다. **실패가 아니다.**
        "review_needed": bool(review),
        "review": review,
        "notices": notices,
        "targets": [{"label": t.get("label"),
                     "frames": getattr(r, "frames", None),
                     "detected_frames": getattr(r, "detected_frames", None),
                     "detection_rate": round(getattr(r, "detection_rate", 0) or 0, 4),
                     "realtime_factor": round(getattr(r, "realtime_factor", 0) or 0, 2),
                     # 단계별 시간이 없으면 "느리다" 까지만 알고 **어디가** 느린지
                     # 모른다. 검출이 대부분이면 GPU 를 늘리는 수밖에 없고,
                     # 렌더·인제스트가 대부분이면 GPU 가 놀고 있다는 뜻이라 대응이
                     # 정반대다. 워커를 몇 대 붙일지가 이 한 줄로 갈린다.
                     "timing": timing_of(r),
                     # 인제스트가 오래 걸릴 때 그게 '디코딩이 느린 것' 인지
                     # '코덱이 안 맞아 통째로 전사한 것' 인지 구분할 근거.
                     # 대응이 전혀 다르다.
                     "source_codec": getattr(r, "source_codec", None),
                     "transcoded": bool(getattr(r, "transcoded", False)),
                     "warnings": list(getattr(r, "warnings", ()))}
                    for t, _p, r in done],
    }
