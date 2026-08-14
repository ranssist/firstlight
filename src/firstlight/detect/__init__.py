"""연기 탐지 — 프레임에서 연기 후보를 찾는다.

`ultralytics` 패키지를 쓰지 않는다. 추론은 onnxruntime 또는 openvino 로만
하고 전/후처리는 직접 구현한다. 이유와 남은 라이선스 쟁점은 README 참조.
"""

from firstlight.detect.detector import Detection, Detector
from firstlight.detect.backend import OnnxDetector, load_detector
from firstlight.detect.tiling import TiledDetector

__all__ = ["Detection", "Detector", "OnnxDetector", "TiledDetector", "load_detector"]
