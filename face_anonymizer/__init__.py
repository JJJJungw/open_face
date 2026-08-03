"""face_anonymizer — YOLO-FaceV2 + ByteTrack 기반 영상 얼굴 비식별화."""

from .pipeline import (
    Cancelled,
    Result,
    VideoAnonymizer,
    VideoInfo,
    VideoOpenError,
    VideoWriteError,
    probe,
)

__all__ = [
    "VideoAnonymizer", "FaceDetector", "Result", "VideoInfo", "probe",
    "VideoOpenError", "VideoWriteError", "Cancelled",
]
__version__ = "0.2.0"


def __getattr__(name):
    """FaceDetector 는 torch 를 끌고 오므로 실제로 접근할 때만 임포트한다.

    덕분에 검출기를 주입해서 쓰는 쪽(테스트, 후처리 전용)은 torch 없이도
    ``from face_anonymizer import VideoAnonymizer`` 가 된다.
    """
    if name == "FaceDetector":
        from .detector import FaceDetector
        return FaceDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
