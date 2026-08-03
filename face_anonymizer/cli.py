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

from . import __version__
from .pipeline import VideoOpenError, VideoWriteError


def build_parser():
    p = argparse.ArgumentParser(
        prog="face-anonymize",
        description="YOLO-FaceV2 + ByteTrack 기반 영상 얼굴 비식별화",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="입력 영상 경로")
    p.add_argument("-o", "--output", default=None, help="출력 경로 (기본: *_anon.mp4)")
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

    g = p.add_argument_group("출력")
    g.add_argument("--no-audio", action="store_true", help="원본 오디오를 합성하지 않음")
    g.add_argument("-q", "--quiet", action="store_true", help="경고 이상만 출력")
    g.add_argument("-v", "--verbose", action="store_true", help="디버그 로그 출력")
    return p


class ProgressBar:
    """의존성 없는 한 줄 진행률 표시. 파이프로 넘길 때는 자동으로 조용해진다."""

    def __init__(self, enabled=True, stream=None):
        self.stream = stream or sys.stderr
        self.enabled = bool(enabled and self.stream.isatty())
        self.last = -1

    def __call__(self, stage, done, total):
        if not self.enabled or not total:
            return
        pct = int(100 * min(1.0, done / total))
        if pct == self.last:
            return
        self.last = pct
        filled = pct * 30 // 100
        self.stream.write(
            f"\r{stage:>7} [{'#' * filled}{'.' * (30 - filled)}] {pct:3d}%"
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

    out = args.output or (os.path.splitext(args.input)[0] + "_anon.mp4")

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
            progress=ProgressBar(not args.quiet),
        )
    except (FileNotFoundError, VideoOpenError, VideoWriteError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    if not args.quiet:
        print(f"{res.output}  frames={res.frames} "
              f"boxes={res.raw_boxes}(+{res.filled_boxes} 보간) audio={res.audio}")
    if res.audio.startswith("ffmpeg-"):
        print(f"warning: 오디오를 합성하지 못했다 ({res.audio}). 영상은 정상 출력됐다.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
