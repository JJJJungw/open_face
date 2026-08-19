"""디코딩 완결성 · 검출 신뢰도 검증 회귀 테스트.

두 사고 유형을 막는다.

* 디코딩이 중간에 끊겨도 성공하던 것 — 영상 뒷부분이 결과물에 통째로 없는데
  예외가 안 났다. 1·2차 패스 프레임 수 비교는 둘 다 같은 지점에서 끊기면
  통과하므로 이걸 못 잡는다.
* 검출 0건도 정상 성공이던 것 — 가중치 손상·회전 영상·잘못된 imgsz 등 원인이
  무엇이든 결과가 똑같이 조용했다.
"""

import pytest

from conftest import FakeDetector

from face_anonymizer import VideoAnonymizer, probe
from face_anonymizer.core import pipeline as P


class NoFaceDetector:
    """아무것도 못 찾는 검출기 (가중치 손상 / 회전 영상 상황 재현)."""

    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        return [[] for _ in frames]


def run(src, out, size, detector=None, **kw):
    a = VideoAnonymizer(detector=detector or FakeDetector(size))
    kw.setdefault("keep_audio", False)
    return a.process(str(src), str(out), batch_size=8, **kw)


# ── 디코딩 완결성 ────────────────────────────────────────────────────────────

def test_probe_uses_video_stream_duration(make_video):
    """기대 프레임 수는 비디오 스트림 길이 x fps 로 잡는다."""
    src, n, _ = make_video(frames=30)
    info = probe(src)
    assert info.count_source in ("duration", "container")
    assert info.frame_count == n


def test_truncated_decode_fails(tmp_path, make_video, monkeypatch):
    """디코딩이 절반만 되면 실패해야 한다 (예전에는 조용히 성공했다)."""
    src, n, size = make_video(frames=30)
    real = P.probe
    monkeypatch.setattr(
        P, "probe",
        lambda p: P.VideoInfo(**{**vars(real(p)), "frame_count": n * 2,
                                 "count_source": "packets"}))

    with pytest.raises(P.DecodeIncompleteError) as e:
        run(src, tmp_path / "out.mp4", size)
    assert f"{n}/{n * 2}" in str(e.value)


def test_allow_partial_downgrades_to_warning(tmp_path, make_video, monkeypatch):
    src, n, size = make_video(frames=30)
    real = P.probe
    monkeypatch.setattr(
        P, "probe",
        lambda p: P.VideoInfo(**{**vars(real(p)), "frame_count": n * 2,
                                 "count_source": "packets"}))

    res = run(src, tmp_path / "out.mp4", size, allow_partial=True)
    assert any(w.startswith("decode-partial") for w in res.warnings)


def test_small_count_mismatch_is_tolerated(tmp_path, make_video, monkeypatch):
    """마지막 GOP 처리 차이로 한두 장 어긋나는 건 정상으로 본다."""
    src, n, size = make_video(frames=30)
    real = P.probe
    monkeypatch.setattr(
        P, "probe",
        lambda p: P.VideoInfo(**{**vars(real(p)), "frame_count": n + 2,
                                 "count_source": "packets"}))

    res = run(src, tmp_path / "out.mp4", size)
    assert not [w for w in res.warnings if w.startswith("decode")]


def test_unknown_count_skips_check(tmp_path, make_video, monkeypatch):
    """프레임 수를 모르면 검사를 건너뛰되 그 사실을 남긴다."""
    src, n, size = make_video(frames=20)
    real = P.probe
    monkeypatch.setattr(
        P, "probe",
        lambda p: P.VideoInfo(**{**vars(real(p)), "frame_count": 0,
                                 "count_source": "unknown"}))

    res = run(src, tmp_path / "out.mp4", size)
    assert "decode-unverified" in res.warnings


# ── 검출 신뢰도 ──────────────────────────────────────────────────────────────

def test_zero_detections_is_flagged_not_silent(tmp_path, make_video):
    """검출 0건은 실패는 아니지만 반드시 드러나야 한다."""
    src, n, size = make_video(frames=20)
    res = run(src, tmp_path / "out.mp4", size, detector=NoFaceDetector())

    assert res.raw_boxes == 0
    assert "no-detections" in res.warnings
    assert res.detection_rate == 0.0


def test_min_detection_rate_can_force_failure(tmp_path, make_video):
    """얼굴이 반드시 있는 영상을 도는 파이프라인은 실패로 만들 수 있다."""
    src, n, size = make_video(frames=20)
    with pytest.raises(P.DetectionSanityError):
        run(src, tmp_path / "out.mp4", size, detector=NoFaceDetector(),
            min_detection_rate=0.5)


def test_normal_run_has_no_warnings(tmp_path, make_video):
    src, n, size = make_video(frames=20)
    res = run(src, tmp_path / "out.mp4", size)

    assert res.warnings == ()
    assert res.detection_rate == 1.0
    assert res.detected_frames == n


def test_detection_rate_counts_frames_not_boxes(tmp_path, make_video):
    """미검출 프레임이 있으면 비율에 그대로 반영된다."""
    src, n, size = make_video(frames=20)
    res = run(src, tmp_path / "out.mp4", size,
              detector=FakeDetector(size, miss_frames={0, 1, 2, 3}))

    assert res.detected_frames == n - 4
    assert res.detection_rate == pytest.approx((n - 4) / n)


# ── 실제 파일에서 오탐이 나지 않는가 ─────────────────────────────────────────
#
# 검증이 지나치게 빡빡하면 정상 영상을 거부한다. 서비스에서는 그게 더 큰 사고라,
# 아래 두 형태는 반드시 통과해야 한다.

import shutil as _shutil
import subprocess as _sp

ffmpeg_only = pytest.mark.skipif(
    not (_shutil.which("ffmpeg") and _shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe 없음")


@ffmpeg_only
def test_trimmed_video_is_not_rejected(tmp_path, make_video):
    """앞부분을 잘라낸 영상(edit list)을 거부하면 안 된다.

    컨테이너에 패킷은 그대로 남고 재생 대상만 줄어든다. 실측: 앞 0.5초를 자른
    파일이 패킷 30개 / 실제 디코딩 22프레임. 패킷 수를 기준으로 삼으면 멀쩡한
    영상이 '8장 누락'으로 보인다. 아이폰·편집 앱을 거친 영상 상당수가 이 형태다.
    """
    src, n, size = make_video(frames=30, fps=15.0)
    h264 = tmp_path / "src264.mp4"
    trimmed = tmp_path / "trimmed.mp4"
    _sp.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264",
             "-g", "250", "-crf", "23", "-preset", "veryfast", str(h264)],
            check=True)
    _sp.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(h264),
             "-c", "copy", str(trimmed)], check=True)

    res = run(trimmed, tmp_path / "out.mp4", size)

    assert not [w for w in res.warnings if w.startswith("decode")], res.warnings


@ffmpeg_only
def test_audio_longer_than_video_is_not_rejected(tmp_path, make_video):
    """오디오가 영상보다 길어도 거부하면 안 된다.

    format.duration 은 모든 스트림 중 가장 긴 값이라, 그걸 기준으로 삼으면
    영상 1.33초 + 오디오 3초 파일에서 20프레임을 45프레임으로 계산한다.
    마이크가 늦게 끊긴 녹화물에서 흔하다.
    """
    src, n, size = make_video(name="clip.mp4", frames=20, fps=15.0)
    withaudio = tmp_path / "with_audio.mp4"
    _sp.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-f", "lavfi", "-t", "3.0", "-i", "sine=frequency=440",
             "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
             str(withaudio)], check=True)

    res = run(withaudio, tmp_path / "out.mp4", size, keep_audio=True)

    assert not [w for w in res.warnings if w.startswith("decode")], res.warnings
    assert res.frames == n


@ffmpeg_only
def test_moderate_gap_warns_but_does_not_fail(tmp_path, make_video, monkeypatch):
    """애매한 차이는 실패가 아니라 경고다 (정상 영상 거부 방지)."""
    src, n, size = make_video(frames=100)
    real = P.probe
    monkeypatch.setattr(                      # 10% 부족하게 보이도록
        P, "probe",
        lambda p: P.VideoInfo(**{**vars(real(p)),
                                 "frame_count": int(n / 0.9),
                                 "count_source": "duration"}))

    res = run(src, tmp_path / "out.mp4", size)
    assert any(w.startswith("decode-short") for w in res.warnings)


@ffmpeg_only
def test_output_is_h264(tmp_path, make_video):
    """다운로드 결과물은 H.264 여야 한다 (mp4v 는 같은 화질에 약 9.5배)."""
    if P.pick_encoder() is None:
        pytest.skip("H.264 인코더 없음")
    src, n, size = make_video(frames=20)
    out = tmp_path / "out.mp4"

    run(src, out, size)

    codec = _sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                     str(out)], capture_output=True, text=True).stdout.strip()
    assert codec == "h264", f"코덱이 {codec}"


# ── 비트레이트 상한 ─────────────────────────────────────────────────────────
#
# 실측(1080p AV1 632 kbps): 상한을 원본 그대로 걸어 639 kbps 로 뽑으니 평평한
# 면이 전부 블록으로 깨졌다. 같은 원본을 상한 없이 뽑으면 2.16 Mbps 다.

def test_bitrate_cap_accounts_for_source_codec(monkeypatch):
    """AV1 632 kbps 를 H.264 632 kbps 상한으로 받으면 안 된다."""
    monkeypatch.setattr(P, "video_bitrate", lambda p, t=None: 632_561)

    monkeypatch.setattr(P, "video_codec", lambda p, t=None: "av1")
    av1 = P.bitrate_cap("x.mp4", 1.0)

    monkeypatch.setattr(P, "video_codec", lambda p, t=None: "h264")
    h264 = P.bitrate_cap("x.mp4", 1.0)

    assert h264 == 632_561              # 같은 코덱이면 원본 그대로
    assert av1 > 1_200_000              # AV1 이면 넉넉하게
    assert av1 == pytest.approx(h264 * 2.0)


def test_bitrate_cap_is_off_when_ratio_is_zero(monkeypatch):
    monkeypatch.setattr(P, "video_bitrate", lambda p, t=None: 632_561)
    monkeypatch.setattr(P, "video_codec", lambda p, t=None: "av1")
    assert P.bitrate_cap("x.mp4", 0) is None


def test_unknown_codec_falls_back_to_plain_ratio(monkeypatch):
    monkeypatch.setattr(P, "video_bitrate", lambda p, t=None: 1_000_000)
    monkeypatch.setattr(P, "video_codec", lambda p, t=None: "weirdcodec")
    assert P.bitrate_cap("x.mp4", 1.0) == 1_000_000


# ── 납품 스펙(해상도 · 비트레이트) ──────────────────────────────────────────
#
# 값만 바꿔서 대응할 수 있어야 한다. 기준이 바뀔 때 코드를 고치게 하면 안 된다.

@pytest.mark.parametrize("value,expect", [
    ("3500k", 3_500_000), ("3.5M", 3_500_000), ("3500000", 3_500_000),
    (3_500_000, 3_500_000), ("", None), (None, None), (0, None),
    ("abc", None),
])
def test_parse_bitrate_accepts_the_usual_spellings(value, expect):
    assert P.parse_bitrate(value) == expect


@pytest.mark.parametrize("w,h,target,expect", [
    (1920, 1080, 720, "scale=-2:720"),     # 가로 1080p -> 720p
    (1280, 720, 720, None),                # 이미 720p — 건드리지 않는다
    (640, 360, 720, None),                 # 더 작다 — 확대하지 않는다
    (1080, 1920, 720, "scale=720:-2"),     # 세로: 짧은 변 기준
    (1920, 1080, 0, None),                 # 0 이면 원본 유지
    (0, 0, 720, None),                     # 크기를 모르면 손대지 않는다
])
def test_scale_filter_uses_the_short_side_and_never_upscales(w, h, target, expect):
    assert P.scale_filter(w, h, target) == expect


# ── 납품 비트레이트 대역 (3000~3500 kbps) ────────────────────────────────────
#
# **이 대역은 지켜지는 게 아니라 강제해야 하는 값이다.** 예전 설정
# (`-b:v 3500k -maxrate 4000k`)은 양쪽으로 다 샜다 — 단순한 장면 14 kbps,
# 복잡한 장면 3915 kbps. 그런데 그걸 잡아 줄 테스트가 하나도 없었다.
# parse_bitrate 의 철자와 bitrate_cap 의 산수만 봤지, **실제로 인코딩된 파일을
# 재는 테스트가 없었다.** 900건을 납품한 뒤 검수에서 알게 될 종류의 구멍이다.

def test_rate_args_turns_on_stuffing_per_encoder():
    """`-b:v` 는 목표 평균이지 하한이 아니다. 스터핑을 켜야 아래끝이 선다.

    x264 는 `-minrate` 를 무시한다(실측: minrate 3200k 인데 14 kbps). 인코더마다
    켜는 방법이 달라서, 그걸 여기서 못 박는다.
    """
    x264 = P.rate_args("libx264", "3200k", "3500k")
    assert "nal-hrd=cbr" in x264, "x264 는 HRD 스터핑이 없으면 아래끝이 샌다"
    # CBR 이려면 목표·최대·버퍼가 한 값이어야 한다.
    assert x264[x264.index("-b:v") + 1] == x264[x264.index("-maxrate") + 1]
    assert x264[x264.index("-bufsize") + 1] == x264[x264.index("-b:v") + 1]

    nv = P.rate_args("h264_nvenc", "3200k", "3500k")
    assert nv[:2] == ["-rc", "cbr"], "NVENC 는 -rc cbr 로 켠다"
    # 후보 표의 `-rc vbr` 뒤에 붙어야 이긴다 — 순서가 뒤집히면 조용히 VBR 이다.
    assert "vbr" not in nv

    assert P.rate_args("libx264", "") is None      # 목표가 없으면 CRF 경로로


def test_rate_args_still_caps_an_unknown_encoder():
    """모르는 인코더라도 상한은 건다. 아래끝은 결과물 검사가 잡는다."""
    other = P.rate_args("libx265", "3200k", "3500k")
    assert other[other.index("-maxrate") + 1] == str(3_500_000)


def _clip(tmp_path, name, seconds=10, kind="noise", crf=16):
    """짧으면 컨테이너 오버헤드가 비율로 커져 위끝을 스친다(4초에서 3513 kbps).
    납품 클립은 분 단위라 그 영역이 정상이다 — 10초면 그 효과가 가라앉는다."""
    src = tmp_path / name
    lavfi = (f"nullsrc=s=640x480:r=30:d={seconds},geq=random(1)*255:128:128"
             if kind == "noise" else
             f"color=c=navy:s=640x480:r=30:d={seconds}")
    _sp.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", lavfi,
             "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
             str(src)], check=True)
    return src


@ffmpeg_only
def test_a_high_bitrate_source_is_brought_down_into_the_band(tmp_path):
    """원본이 대역 위면 위끝에서 자른다 — 이게 정상적인 다운코딩이다."""
    src = _clip(tmp_path, "big.mp4", kind="noise")
    assert P.video_bitrate(str(src)) > 3_500_000, "실험 전제가 깨졌다"

    out = tmp_path / "out.mp4"
    res = run(src, out, (640, 480))

    got = P.file_bitrate(str(out))
    assert got <= 3_500_000, f"대역 위끝을 넘었다: {got // 1000} kbps"
    assert got >= 3_000_000, f"대역 아래로 떨어졌다: {got // 1000} kbps"
    assert not [w for w in res.warnings if str(w).startswith("bitrate")]


@ffmpeg_only
def test_a_low_bitrate_source_is_never_inflated(tmp_path):
    """**올려 담지 않는다.**

    원본이 312 kbps 인데 3200 으로 뽑으면 압축 열화를 고화질로 보존할 뿐이다.
    실측(720p 방송 클립): 원본 그대로가 PSNR 47.4 dB — 재인코딩으로는 사실상
    무손실인데, 대역까지 올리면 10배를 쓰고 SSIM 0.9937 → 0.9984 를 산다.
    둘 다 사람 눈 구분 한계 위다. 900건이면 6 GB 와 60 GB 의 차이가 된다.
    """
    src = _clip(tmp_path, "small.mp4", kind="flat", crf=30)
    source = P.video_bitrate(str(src))
    assert source < 3_000_000, "실험 전제가 깨졌다"

    out = tmp_path / "out.mp4"
    res = run(src, out, (640, 480))
    got = P.file_bitrate(str(out))

    assert got < 3_000_000, f"대역 아래 원본을 올려 담았다: {got // 1000} kbps"
    # 원본 근처여야 한다. 모자이크가 들어가 정확히 같지는 않다.
    assert got < source * 3, f"원본 {source // 1000}k 대비 과하다: {got // 1000}k"
    # 사람을 부르지는 않되, 그런 파일이었다는 기록은 남는다.
    codes = [str(w).split(":")[0] for w in res.warnings]
    assert "source-below-band" in codes, res.warnings
    assert "bitrate-out-of-band" not in codes


@ffmpeg_only
def test_a_low_source_does_not_call_a_human(tmp_path):
    """저품질 원본은 의도한 결과다 — 검수로 올리면 900건 중 수백 건이 걸린다."""
    from face_anonymizer import job_runner

    src = _clip(tmp_path, "small.mp4", kind="flat", crf=30)
    res = run(src, tmp_path / "out.mp4", (640, 480))

    codes = [r["code"] for r in job_runner.review_of(res.warnings)]
    assert "source-below-band" not in codes
    # 대신 알림 쪽에는 있어야 한다 — 조용히 사라지면 안 된다.
    assert "source-below-band" in job_runner.NOTICE


def test_delivery_target_never_goes_above_the_source(monkeypatch):
    """`목표 = min(원본, 납품 목표)`."""
    monkeypatch.setattr(P, "video_bitrate", lambda p, t=None: 312_000)
    monkeypatch.setattr(P, "video_codec", lambda p, t=None: "h264")
    assert P.delivery_target("x.mp4", "3200k") == 312_000      # 낮으면 그대로

    monkeypatch.setattr(P, "video_bitrate", lambda p, t=None: 9_000_000)
    assert P.delivery_target("x.mp4", "3200k") == 3_200_000    # 높으면 목표에서


def test_delivery_target_respects_codec_efficiency(monkeypatch):
    """AV1 632 kbps 를 H.264 632 kbps 로 받는 것은 '그대로' 가 아니다."""
    monkeypatch.setattr(P, "video_bitrate", lambda p, t=None: 632_000)
    monkeypatch.setattr(P, "video_codec", lambda p, t=None: "av1")
    assert P.delivery_target("x.mp4", "3200k") == 1_264_000    # 632k x 2.0


def test_delivery_target_falls_back_to_the_ceiling(monkeypatch):
    """원본을 못 재면 규격을 맞추는 쪽으로 간다 — 모르면서 미달을 내지 않는다."""
    monkeypatch.setattr(P, "video_bitrate", lambda p, t=None: None)
    assert P.delivery_target("x.mp4", "3200k") == 3_200_000


@ffmpeg_only
def test_a_clip_too_short_to_measure_is_not_flagged(tmp_path, make_video):
    """CBR 은 버퍼가 차야 목표에 붙는다. 0.7초짜리를 미달로 부르면 오탐이다."""
    src, n, size = make_video(frames=20)           # 0.7초
    res = run(src, tmp_path / "out.mp4", size)
    assert not [w for w in res.warnings if str(w).startswith("bitrate")]


def test_file_bitrate_reads_by_name_not_by_position(tmp_path):
    """ffprobe 는 요청한 순서가 아니라 자기 고정 순서로 찍는다.

    duration 이 bit_rate 보다 먼저 나오는데 값만 받아 자리로 세면 **길이를
    비트레이트로 읽는다.** 실제로 그렇게 틀렸다.
    """
    seen = {}

    class P_:
        returncode = 0
        stdout = "duration=10.000000\nbit_rate=3200000\n"

    def fake_run(cmd, timeout=None):
        seen["cmd"] = cmd
        return P_()

    import face_anonymizer.core.pipeline as mod
    old = mod._run
    mod._run = fake_run
    try:
        assert mod.file_bitrate("x.mp4") == 3_200_000
    finally:
        mod._run = old
    assert "default=nw=1" in seen["cmd"], "키를 지우면 자리로 세게 된다"
