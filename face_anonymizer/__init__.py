"""face_anonymizer — YOLO-FaceV2 + ByteTrack 기반 영상 얼굴 비식별화."""

# .env 는 다른 무엇보다 먼저 읽어야 한다. 설정 모듈들이 임포트 시점에
# os.environ 을 읽으므로, 그 뒤에 채우면 아무 효과가 없다.
from . import env as _env

_env.load()

from .core.pipeline import (  # noqa: E402
    Result,
    VideoAnonymizer,
    VideoInfo,
    VideoOpenError,
    VideoWriteError,
    probe,
)

# FaceDetector 는 일부러 빼 둔다. `from face_anonymizer import *` 는 __all__ 의
# 모든 이름을 getattr 하므로, 여기 넣으면 아래 __getattr__ 가 발동해 torch 를
# 끌고 온다 — "torch 없이 파이프라인만 쓴다" 는 약속이 별표 임포트에서 깨진다.
# 명시적 접근(`from face_anonymizer import FaceDetector`)은 그대로 동작한다.
__all__ = [
    "VideoAnonymizer", "Result", "VideoInfo", "probe",
    "VideoOpenError", "VideoWriteError",
]
__version__ = "0.2.0"


def __getattr__(name):
    """FaceDetector 는 torch 를 끌고 오므로 실제로 접근할 때만 임포트한다.

    덕분에 검출기를 주입해서 쓰는 쪽(테스트, 후처리 전용)은 torch 없이도
    ``from face_anonymizer import VideoAnonymizer`` 가 된다.
    """
    if name == "FaceDetector":
        from .core.detector import FaceDetector
        return FaceDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
