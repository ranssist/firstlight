"""슬라이스 추론 — 멀리 있는 작은 연기를 살린다.

왜 필요한가:
    1920x1080 프레임을 1024 로 레터박스하면 배율이 0.53 이다. 원본에서
    40px 짜리 초기 연기는 21px 로 줄어든다. 탐지기는 이 크기에서 급격히
    나빠진다. 그런데 소개서가 노리는 것은 정확히 그 "초기" 단계다 —
    다 커진 연기는 굳이 AI 가 아니어도 보인다.

    프레임을 겹치는 타일로 잘라 원해상도로 추론하면 그 40px 가 그대로
    40px 로 들어간다.

비용:
    1920x1080 을 1024 타일로 자르면 3x2 = 6장이고, 전역 맥락을 위한
    전체 프레임 1장을 더하면 프레임당 7회 추론이다. CPU 에서는 감당이
    안 되므로 기본값은 꺼둔다. 순찰 영상은 실시간일 필요가 없고
    (녹화 후 배치 처리), 실시간이 필요하면 iGPU/NPU 로 돌린다.
"""

from __future__ import annotations

import numpy as np

from firstlight.detect.detector import Detection, Detector, nms


def _spans(size: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    """한 축의 타일 구간 [(시작, 끝), ...]. 전 범위를 빠짐없이 덮는다."""
    if size <= tile:
        return [(0, size)]

    stride = max(1, int(round(tile * (1.0 - overlap))))
    n = int(np.ceil((size - tile) / stride)) + 1
    starts = np.linspace(0, size - tile, n).round().astype(int)
    return [(int(s), int(s) + tile) for s in dict.fromkeys(starts.tolist())]


class TiledDetector:
    """탐지기를 감싸 겹치는 타일에 각각 추론한다.

    Args:
        detector: 감쌀 탐지기.
        tile: 타일 한 변 픽셀. 보통 탐지기 imgsz 와 같게 둔다.
        overlap: 인접 타일 겹침 비율. 경계에 걸친 연기를 위해 필요하다.
        include_full_frame: 전체 프레임도 한 번 본다. 타일보다 큰 연기를
            놓치지 않기 위한 것이다.
        iou_threshold: 타일 간 중복 제거용 전역 NMS 임계값.
    """

    def __init__(
        self,
        detector: Detector,
        tile: int = 1024,
        overlap: float = 0.25,
        include_full_frame: bool = True,
        iou_threshold: float = 0.4,
    ) -> None:
        self.detector = detector
        self.tile = tile
        self.overlap = overlap
        self.include_full_frame = include_full_frame
        self.iou_threshold = iou_threshold
        self.last_tile_count: int = 0

    def tiles_for(self, height: int, width: int) -> list[tuple[int, int, int, int]]:
        """(x1, y1, x2, y2) 타일 목록."""
        return [
            (x1, y1, x2, y2)
            for x1, x2 in _spans(width, self.tile, self.overlap)
            for y1, y2 in _spans(height, self.tile, self.overlap)
        ]

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        collected: list[Detection] = []

        boxes = self.tiles_for(h, w)
        for x1, y1, x2, y2 in boxes:
            crop = frame_bgr[y1:y2, x1:x2]
            for det in self.detector.detect(crop):
                collected.append(
                    Detection(
                        det.x1 + x1, det.y1 + y1, det.x2 + x1, det.y2 + y1,
                        det.confidence, det.label,
                    )
                )

        n_infer = len(boxes)
        if self.include_full_frame and (w > self.tile or h > self.tile):
            collected.extend(self.detector.detect(frame_bgr))
            n_infer += 1

        self.last_tile_count = n_infer
        return nms(collected, self.iou_threshold)

    def __repr__(self) -> str:
        return (
            f"TiledDetector(tile={self.tile}, overlap={self.overlap}, "
            f"full_frame={self.include_full_frame})"
        )
