"""커맨드라인 인터페이스.

예:
    face-anonymize input.mp4
    face-anonymize input.mp4 -o out.mp4 --method mosaic --conf 0.20 --imgsz 1280
    face-anonymize input.mp4 --method box
"""

import argparse
import os

from .detector import DEFAULT_WEIGHTS
from .pipeline import VideoAnonymizer


def build_parser():
    p = argparse.ArgumentParser(
        prog="face-anonymize",
        description="YOLO-FaceV2 + ByteTrack 기반 영상 얼굴 비식별화",
    )
    p.add_argument("input", help="입력 영상 경로")
    p.add_argument("-o", "--output", default=None, help="출력 경로 (기본: *_anon.mp4)")
    p.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS, help="YOLO-FaceV2 가중치 경로")
    p.add_argument("--method", default="mosaic", choices=["mosaic", "blur", "box"],
                   help="익명화 방식 (기본: mosaic)")
    p.add_argument("--imgsz", type=int, default=960,
                   help="추론 해상도. 720p 작은 얼굴이면 1280 권장")
    p.add_argument("--conf", type=float, default=0.25,
                   help="검출 임계값. 낮출수록 재현율↑(누출↓)·오탐↑")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU 임계값")
    p.add_argument("--pad", type=float, default=0.15, help="박스 확장 비율")
    p.add_argument("--mosaic-scale", type=float, default=0.06,
                   help="모자이크 블록 크기. 낮출수록 더 강하게 가림")
    p.add_argument("--linger", type=int, default=5,
                   help="트랙 소실 후 박스 유지 프레임 수")
    p.add_argument("--no-interp", action="store_true", help="트랙 보간 끄기")
    p.add_argument("--device", default=None, help="'cuda:0' / 'cpu' (기본: 자동)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = args.output or (os.path.splitext(args.input)[0] + "_anon.mp4")

    anonymizer = VideoAnonymizer(weights=args.weights, device=args.device)
    anonymizer.process(
        args.input, out,
        method=args.method, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        pad=args.pad, mosaic_scale=args.mosaic_scale, linger=args.linger,
        interp=not args.no_interp,
    )


if __name__ == "__main__":
    main()
