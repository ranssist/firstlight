"""탐지 인터페이스 — 백엔드와 파이프라인을 분리한다.

파이프라인은 이 프로토콜만 알면 되므로, 부트스트랩 가중치를 자체 학습
모델로 교체할 때(라이선스 정리 시 반드시 하게 된다) 파이프라인은 손대지
않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    """단일 프레임의 연기 후보 하나.

    좌표는 **원본 프레임 픽셀** 기준이다. 레터박스·타일 오프셋은 백엔드가
    되돌려 놓는다 — 지오레퍼런싱이 원본 픽셀을 기대하기 때문이다.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "smoke"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centroid(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    @property
    def base_point(self) -> tuple[float, float]:
        """연기 기저 대용점 — 박스 하단 중앙. 지오레퍼런싱이 쓰는 점이다."""
        return (0.5 * (self.x1 + self.x2), self.y2)

    def iou(self, other: Detection) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Detector(Protocol):
    """프레임 하나를 받아 탐지 목록을 돌려준다."""

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        ...


# --------------------------------------------------------------------------


def nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """신뢰도 내림차순 탐욕적 NMS.

    모델이 `nms: False` 로 내보내졌으므로 우리가 해야 한다.
    """
    keep: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(det.iou(kept) <= iou_threshold for kept in keep):
            keep.append(det)
    return keep
