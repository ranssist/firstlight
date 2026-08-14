"""ONNX / OpenVINO 추론 백엔드.

모델 실측 (`models/pyronear/model.onnx`, YOLO11s):
    입력  images   [batch, 3, height, width]  — 동적 크기, RGB, 0~1 정규화
    출력  output0  [batch, 5, anchors]        — (cx, cy, w, h, score) 전치 배치
    imgsz 1024, stride 32, 단일 클래스(`single_cls: true`), **NMS 미포함**

NMS 가 모델 밖에 있으므로 후처리는 우리 몫이다.

가속: 이 환경의 onnxruntime 빌드에는 OpenVINO 실행 공급자가 없다
(`['AzureExecutionProvider', 'CPUExecutionProvider']`). 따라서 Intel 가속은
openvino 런타임을 직접 쓴다. 실측 가용 장치: CPU(Core Ultra 7 258V),
GPU(Arc 140V iGPU), NPU(AI Boost).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from firstlight.detect.detector import Detection, nms

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "pyronear" / "model.onnx"

# 모델 카드 권장값. iou 가 유난히 낮은데(0.01), 연기는 하나의 큰 덩어리라
# 겹치는 박스를 공격적으로 합치는 편이 맞기 때문이다.
DEFAULT_IMGSZ = 1024
DEFAULT_CONF = 0.20
DEFAULT_IOU = 0.20


def letterbox(
    image: np.ndarray, target: int, pad_value: int = 114
) -> tuple[np.ndarray, float, float, float]:
    """종횡비를 유지한 채 정사각 캔버스에 얹는다.

    Returns:
        (캔버스, 배율, x오프셋, y오프셋). 오프셋·배율은 박스를 원본 좌표로
        되돌리는 데 쓴다.
    """
    h, w = image.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    import cv2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), pad_value, dtype=image.dtype)
    dx = (target - new_w) // 2
    dy = (target - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized
    return canvas, scale, float(dx), float(dy)


def _decode(
    raw: np.ndarray,
    conf_threshold: float,
    scale: float,
    dx: float,
    dy: float,
    label: str = "smoke",
) -> list[Detection]:
    """(5, anchors) 원시 출력을 원본 좌표계 탐지 목록으로.

    5 = cx, cy, w, h, score. 단일 클래스라 별도 objectness 가 없다.
    """
    preds = raw[0] if raw.ndim == 3 else raw          # (5, anchors)
    scores = preds[4]
    keep = scores >= conf_threshold
    if not keep.any():
        return []

    cx, cy, bw, bh = preds[0][keep], preds[1][keep], preds[2][keep], preds[3][keep]
    scores = scores[keep]

    # 레터박스 되돌리기: 패딩 제거 후 배율 복원.
    x1 = (cx - bw / 2 - dx) / scale
    y1 = (cy - bh / 2 - dy) / scale
    x2 = (cx + bw / 2 - dx) / scale
    y2 = (cy + bh / 2 - dy) / scale

    return [
        Detection(float(a), float(b), float(c), float(d), float(s), label)
        for a, b, c, d, s in zip(x1, y1, x2, y2, scores)
    ]


class OnnxDetector:
    """단일 프레임 연기 탐지기.

    Args:
        model_path: .onnx 경로.
        device: "cpu" | "gpu" | "npu" | "auto".
            "cpu" 는 onnxruntime, 나머지는 openvino 런타임을 쓴다.
        imgsz: 정사각 입력 변. 32의 배수여야 한다 (stride).
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL,
        device: str = "cpu",
        imgsz: int = DEFAULT_IMGSZ,
        conf_threshold: float = DEFAULT_CONF,
        iou_threshold: float = DEFAULT_IOU,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"모델이 없다: {self.model_path}\n"
                f"  uv run python scripts/fetch_model.py"
            )
        if imgsz % 32:
            raise ValueError(f"imgsz 는 32의 배수여야 한다 (stride=32), 받은 값 {imgsz}")

        self.device = device.lower()
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.last_inference_ms: float = float("nan")

        if self.device == "cpu":
            self._init_onnxruntime()
        else:
            self._init_openvino()

    # ------------------------------------------------------------- 백엔드

    def _init_onnxruntime(self) -> None:
        import onnxruntime as ort

        self._backend = "onnxruntime"
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path), opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def _init_openvino(self) -> None:
        import openvino as ov

        self._backend = "openvino"
        core = ov.Core()
        available = core.available_devices
        requested = {"gpu": "GPU", "npu": "NPU", "auto": "AUTO"}.get(self.device)
        if requested is None:
            raise ValueError(f"알 수 없는 device: {self.device}")
        if requested != "AUTO" and requested not in available:
            raise RuntimeError(
                f"장치 '{requested}' 를 쓸 수 없다. 사용 가능: {available}"
            )

        model = core.read_model(str(self.model_path))
        # 동적 축을 고정하면 컴파일러가 훨씬 나은 커널을 고른다.
        model.reshape({0: [1, 3, self.imgsz, self.imgsz]})
        self._compiled = core.compile_model(model, requested)
        self._output_port = self._compiled.output(0)

    # -------------------------------------------------------------- 추론

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        if self._backend == "onnxruntime":
            raw = self._session.run(None, {self._input_name: tensor})[0]
        else:
            raw = self._compiled(tensor)[self._output_port]
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        return np.asarray(raw)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """BGR 프레임에서 연기 후보를 찾는다 (원본 픽셀 좌표)."""
        canvas, scale, dx, dy = letterbox(frame_bgr, self.imgsz)
        rgb = canvas[:, :, ::-1]                                  # BGR → RGB
        tensor = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        )
        raw = self._infer(tensor)
        dets = _decode(raw, self.conf_threshold, scale, dx, dy)
        return nms(dets, self.iou_threshold)

    def __repr__(self) -> str:
        return (
            f"OnnxDetector(backend={self._backend}, device={self.device}, "
            f"imgsz={self.imgsz}, conf={self.conf_threshold})"
        )


def load_detector(
    device: str = "cpu",
    model_path: Path | str = DEFAULT_MODEL,
    **kwargs,
) -> OnnxDetector:
    """탐지기를 만든다. 가속 장치가 없으면 CPU 로 조용히 물러난다."""
    try:
        return OnnxDetector(model_path, device=device, **kwargs)
    except (RuntimeError, ImportError) as exc:
        if device == "cpu":
            raise
        import warnings

        warnings.warn(f"{device} 사용 불가 ({exc}) — CPU 로 전환한다", stacklevel=2)
        return OnnxDetector(model_path, device="cpu", **kwargs)
