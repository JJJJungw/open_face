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

    def __call__(self, position):
        """``position`` 은 아무 단조 증가 값이면 된다(프레임 수·바이트 수)."""
        now = time.monotonic()
        if position != self.mark:
            self.mark, self.last_move = position, now
        elif now - self.last_move > STALL_S:
            raise JobError(f"진행이 {int(STALL_S)}초 동안 멎었습니다",
                           transient=True, stage="stall")
        if now - self.last_beat >= self.every:
            self.last_beat = now
            if self.on_heartbeat is not None:
                self.on_heartbeat(round(now - self.t0, 1))


def target_params(t):
    """잡의 타깃 사양 → 파이프라인 인자.

    저쪽 트랜스코더는 화질을 ``crf`` 로 말하고 우리 납품 기준은 타깃 비트레이트다
    (720p / 3500~4000 kbps). 둘 다 받아 준다 — 어휘를 하나로 강제하면 붙이는
    쪽이 자기 파이프라인을 고쳐야 한다.
    """
    p = {}
    for k in ("method", "conf", "imgsz", "batch_size", "keep_audio",
              "bitrate", "max_bitrate", "crf"):
        if t.get(k) is not None:
            p[k] = t[k]

    # height 는 **키가 있는데 값이 null** 인 것이 뜻을 갖는다 — 저쪽 규약에서
    # "스케일 생략(업스케일 금지)" 이다. 없는 것과 구분해야 해서 따로 본다.
    if "height" in t:
        p["height"] = t["height"] or 0            # 0 = 해상도 그대로

    # 화질 정책은 둘 중 하나만 걸어야 한다(docs/issues 의 AV1 상한 사고). 타깃이
    # crf 로 말했으면 우리 기본 타깃 비트레이트를 비워서 CRF 쪽으로 넘긴다.
    if "crf" in p and "bitrate" not in p:
        p["bitrate"] = p["max_bitrate"] = ""
    return p


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
            seen = [0]

            def on_chunk(n):
                seen[0] += n
                beat(seen[0])

            transfer.fetch(job["input_url"], src, callback=on_chunk)
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
        done = []
        for t in targets:
            out = os.path.join(tmp, f"{t.get('label') or 'out'}.mp4")
            try:
                res = anonymizer.process(src, out, progress=lambda s, d, n: beat((s, d)),
                                         **target_params(t))
            except JobError:
                raise                               # 정체 감시가 던진 것 — 그대로
            except Exception as e:                  # noqa: BLE001
                # 같은 입력으로 다시 해도 같은 결과다. 재시도로 큐를 태우지 않는다.
                raise JobError(f"{t.get('label')}: {type(e).__name__}: {e}",
                               transient=False, stage="process") from e
            done.append((t, getattr(res, "output", out), res))

        for t, path, _res in done:
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
