"""커맨드라인 인터페이스.

예:
    face-anonymize input.mp4
    face-anonymize input.mp4 -o out.mp4 --method mosaic --conf 0.20 --imgsz 1280
    face-anonymize input.mp4 --method box
    face-anonymize input.mp4 --batch-size 16                    # GPU 처리량 우선

종료 코드
    0    성공
    1    처리 실패 (영상을 못 열거나 인코더 문제 등)
    2    잘못된 인자
    130  사용자 중단(Ctrl-C)
"""

import argparse
import logging
import os
import sys
import time

from . import __version__
from .storage import naming
from .core.pipeline import (DetectionSanityError, VideoOpenError,
                            VideoWriteError)


def build_parser():
    p = argparse.ArgumentParser(
        prog="face-anonymize",
        description="YOLO-FaceV2 + ByteTrack 기반 영상 얼굴 비식별화",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="입력 영상 경로")
    p.add_argument("-o", "--output", default=None,
                   help="출력 경로 (기본: 데이터셋 규칙에 따라 *_deid.mp4)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    g = p.add_argument_group("익명화")
    g.add_argument("--method", default="mosaic", choices=["mosaic", "blur", "box"],
                   help="익명화 방식")
    g.add_argument("--mosaic-scale", type=float, default=0.06,
                   help="모자이크 블록 크기. 낮출수록 더 강하게 가림")
    g.add_argument("--pad", type=float, default=0.15, help="박스 확장 비율")

    g = p.add_argument_group("검출")
    g.add_argument("-w", "--weights", default=None, help="YOLO-FaceV2 가중치 경로")
    g.add_argument("--imgsz", type=int, default=960,
                   help="추론 해상도. 720p 작은 얼굴이면 1280 권장")
    g.add_argument("--conf", type=float, default=0.25,
                   help="검출 임계값. 낮출수록 재현율↑(누출↓)·오탐↑")
    g.add_argument("--iou", type=float, default=0.45, help="NMS IoU 임계값")
    g.add_argument("--device", default=None, help="'cuda:0' / 'cpu' (기본: 자동)")
    g.add_argument("--half", action="store_true", default=None,
                   help="FP16 추론 (기본: CUDA 에서 자동 활성화)")
    g.add_argument("--no-half", dest="half", action="store_false",
                   help="FP16 강제 해제")

    g = p.add_argument_group("처리량")
    g.add_argument("--batch-size", type=int, default=1, metavar="N",
                   help="한 번에 모델에 넣을 프레임 수 (GPU 에서 클수록 빠름)")

    g = p.add_argument_group("누출 방지")
    g.add_argument("--linger", type=int, default=5,
                   help="트랙 소실 후 박스 유지 프레임 수")
    g.add_argument("--no-interp", action="store_true", help="트랙 보간 끄기")

    g = p.add_argument_group("검증")
    g.add_argument("--allow-partial", action="store_true",
                   help="디코딩이 중간에 끊겨도 진행 (기본: 실패 처리)")
    g.add_argument("--min-detection-rate", type=float, default=None, metavar="R",
                   help="검출된 프레임 비율이 R 미만이면 실패 (예: 0.5)")
    g.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="입력을 시계방향으로 회전 (메타데이터 위에 추가로 적용)")

    g = p.add_argument_group("출력")
    g.add_argument("--crf", type=int, default=None, metavar="N",
                   help="H.264 품질 (낮을수록 고화질/큰 파일, 기본 23)")
    g.add_argument("--bitrate-ratio", type=float, default=None, metavar="R",
                   help="출력 비트레이트 상한 = 원본 x R (기본 1.0, 0 이면 무제한)")
    g.add_argument("--no-audio", action="store_true", help="원본 오디오를 합성하지 않음")
    g.add_argument("-q", "--quiet", action="store_true", help="경고 이상만 출력")
    g.add_argument("-v", "--verbose", action="store_true", help="디버그 로그 출력")
    return p


def fmt_dur(sec):
    """초 → 'M:SS' (한 시간 넘으면 'H:MM:SS')."""
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class ProgressBar:
    """의존성 없는 한 줄 진행률 표시. 파이프로 넘길 때는 자동으로 조용해진다.

    단계가 바뀌면 타이머를 리셋한다. 검출(GPU)과 렌더(CPU+디스크)는 속도가
    몇 배씩 차이 나므로, 합쳐서 평균 내면 남은 시간 추정이 무의미해진다.
    """

    def __init__(self, enabled=True, stream=None):
        self.stream = stream or sys.stderr
        self.enabled = bool(enabled and self.stream.isatty())
        self.stage = None
        self.t0 = 0.0
        self.last = -1

    def __call__(self, stage, done, total):
        if not self.enabled or not total:
            return
        if stage != self.stage:
            if self.stage is not None and self.last < 100:
                self.stream.write("\n")     # 이전 단계 줄을 덮어쓰지 않는다
            self.stage, self.t0, self.last = stage, time.perf_counter(), -1
        pct = int(100 * min(1.0, done / total))
        if pct == self.last:
            return
        self.last = pct
        elapsed = time.perf_counter() - self.t0
        fps = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / fps if fps > 0 and done < total else 0.0
        filled = pct * 30 // 100
        self.stream.write(
            f"\r{stage:>7} [{'#' * filled}{'.' * (30 - filled)}] {pct:3d}% "
            f"{fps:7.1f}f/s  {fmt_dur(elapsed)} 경과  ETA {fmt_dur(eta)}   "
        )
        self.stream.flush()
        if pct >= 100:
            self.stream.write("\n")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose
               else logging.WARNING if args.quiet
               else logging.INFO),
        format="%(message)s",
    )

    # 데이터셋 규칙(naming.py)을 따른다. 규칙 밖 이름이면 <이름>_deid.mp4.
    out = args.output or os.path.join(
        os.path.dirname(args.input), naming.output_name(args.input))

    detector_kwargs = {"device": args.device, "half": args.half, "imgsz": args.imgsz}
    if args.weights:
        detector_kwargs["weights"] = args.weights

    try:
        from .pipeline import VideoAnonymizer
        anonymizer = VideoAnonymizer(**detector_kwargs)
        res = anonymizer.process(
            args.input, out,
            method=args.method, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            pad=args.pad, mosaic_scale=args.mosaic_scale, linger=args.linger,
            interp=not args.no_interp, batch_size=args.batch_size,
            keep_audio=not args.no_audio,
            allow_partial=args.allow_partial,
            min_detection_rate=args.min_detection_rate,
            rotate=args.rotate,
            **({"crf": args.crf} if args.crf is not None else {}),
            **({"bitrate_ratio": args.bitrate_ratio}
               if args.bitrate_ratio is not None else {}),
            progress=ProgressBar(not args.quiet),
        )
    except (FileNotFoundError, VideoOpenError, VideoWriteError,
            DetectionSanityError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    if not args.quiet:
        t = res.timing
        dur = res.frames / res.video.fps if res.video and res.video.fps else 0
        print(f"{res.output}  frames={res.frames} "
              f"boxes={res.raw_boxes}(+{res.filled_boxes} 보간) audio={res.audio}")
        print(f"  영상 {fmt_dur(dur)} | 처리 {fmt_dur(t.total)} "
              f"({res.fps:.1f} fps, 실시간 대비 {res.realtime_factor:.2f}x)")
        print(f"  검출 {fmt_dur(t.detect)} ({res.detect_fps:.1f} fps) · "
              f"추적 {fmt_dur(t.track)} · 렌더 {fmt_dur(t.render)} · "
              f"오디오 {fmt_dur(t.audio)}")
    # 결과를 그대로 믿으면 안 되는 사유는 stdout 요약이 아니라 stderr 로,
    # 눈에 띄게 낸다. 조용히 지나가면 파이프라인에 그대로 흘러든다.
    for w in res.warnings:
        if w == "no-detections":
            print("warning: 얼굴이 하나도 검출되지 않았다 — 원본이 그대로 "
                  "출력됐다. conf/imgsz/가중치/영상 회전을 확인하라.",
                  file=sys.stderr)
        elif w.startswith("audio:"):
            print(f"warning: 오디오를 합성하지 못했다 ({w[7:]}). "
                  "영상은 익명화된 상태로 정상 출력됐다.", file=sys.stderr)
        else:
            print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
