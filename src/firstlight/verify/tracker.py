"""탐지 연결 — 프레임별 박스를 트랙으로 잇는다.

시간축 특징은 전부 "같은 대상을 여러 프레임에 걸쳐 봤다"를 전제한다.
연결이 틀리면 특징이 통째로 무의미해지므로 여기가 먼저다.

IoU 기반 탐욕 연결을 쓴다. 칼만 필터를 쓰지 않는 이유:
    연기는 강체가 아니라 계속 형태가 변하고 커진다. 등속 운동 모델이
    맞지 않고, 60초 간격 데이터(FIgLib)에서는 예측 자체가 무의미하다.
    대신 IoU 문턱을 낮게 두고 중심거리를 보조로 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from firstlight.detect.detector import Detection


@dataclass
class TrackObservation:
    """한 프레임에서의 관측."""

    timestamp: float
    frame_index: int
    detection: Detection
    # 특징 계산용 축소 그레이스케일 패치 (없을 수도 있다).
    patch: np.ndarray | None = None
    # 자기운동 보정 후 마스크 내부 잔차 광류 통계 (dense 모드에서만).
    flow: dict | None = None


@dataclass
class Track:
    """여러 프레임에 걸친 하나의 연기 후보."""

    track_id: int
    observations: list[TrackObservation] = field(default_factory=list)
    misses: int = 0

    # ------------------------------------------------------------ 조회

    @property
    def last(self) -> TrackObservation:
        return self.observations[-1]

    @property
    def first(self) -> TrackObservation:
        return self.observations[0]

    @property
    def hits(self) -> int:
        return len(self.observations)

    @property
    def duration_s(self) -> float:
        if len(self.observations) < 2:
            return 0.0
        return self.last.timestamp - self.first.timestamp

    @property
    def frame_span(self) -> int:
        if len(self.observations) < 2:
            return 1
        return self.last.frame_index - self.first.frame_index + 1

    @property
    def peak_confidence(self) -> float:
        return max(o.detection.confidence for o in self.observations)

    def times(self) -> np.ndarray:
        return np.array([o.timestamp for o in self.observations], dtype=float)

    def areas(self) -> np.ndarray:
        return np.array([o.detection.area for o in self.observations], dtype=float)

    def centroids(self) -> np.ndarray:
        return np.array([o.detection.centroid for o in self.observations], dtype=float)

    def confidences(self) -> np.ndarray:
        return np.array([o.detection.confidence for o in self.observations], dtype=float)

    def window(self, seconds: float) -> Track:
        """최근 `seconds` 구간만 담은 얕은 사본."""
        if not self.observations:
            return self
        cutoff = self.last.timestamp - seconds
        recent = [o for o in self.observations if o.timestamp >= cutoff]
        return Track(track_id=self.track_id, observations=recent, misses=self.misses)


class Tracker:
    """프레임 간 탐지 연결기.

    Args:
        iou_threshold: 연결로 인정할 최소 IoU. 연기는 형태가 빠르게 변하고
            60초 간격에서는 겹침이 작아지므로 낮게 잡는다.
        max_misses: 이 횟수만큼 연속으로 못 찾으면 트랙을 끝낸다.
        max_center_distance_factor: IoU 가 0 이어도 중심이 박스 크기의 이
            배수 이내면 같은 대상으로 본다. 희소 프레임에서 필요하다.
    """

    def __init__(
        self,
        iou_threshold: float = 0.10,
        max_misses: int = 2,
        max_center_distance_factor: float = 1.5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.max_center_distance_factor = max_center_distance_factor
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 0

    # ------------------------------------------------------------------

    def _affinity(self, track: Track, det: Detection) -> float:
        """연결 점수. 클수록 같은 대상일 가능성이 높다. 0 이면 불가."""
        prev = track.last.detection
        iou = prev.iou(det)
        if iou >= self.iou_threshold:
            return 1.0 + iou              # IoU 매칭을 항상 우선한다

        # IoU 가 없어도 중심이 충분히 가까우면 후보로 본다.
        px, py = prev.centroid
        cx, cy = det.centroid
        distance = float(np.hypot(cx - px, cy - py))
        scale = max(np.sqrt(max(prev.area, 1.0)), 1.0)
        if distance <= self.max_center_distance_factor * scale:
            return 1.0 - distance / (self.max_center_distance_factor * scale)
        return 0.0

    def update(
        self,
        detections: list[Detection],
        timestamp: float,
        frame_index: int,
        patches: list[np.ndarray | None] | None = None,
        flows: list[dict | None] | None = None,
    ) -> list[Track]:
        """한 프레임을 반영하고 현재 활성 트랙을 돌려준다."""
        patches = patches or [None] * len(detections)
        flows = flows or [None] * len(detections)

        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.active):
            for di, det in enumerate(detections):
                score = self._affinity(track, det)
                if score > 0:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for score, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            track = self.active[ti]
            track.observations.append(
                TrackObservation(timestamp, frame_index, detections[di],
                                 patches[di], flows[di])
            )
            track.misses = 0
            used_tracks.add(ti)
            used_dets.add(di)

        # 매칭 안 된 트랙은 미스 누적, 한도 넘으면 종료.
        still_active: list[Track] = []
        for ti, track in enumerate(self.active):
            if ti not in used_tracks:
                track.misses += 1
                if track.misses > self.max_misses:
                    self.finished.append(track)
                    continue
            still_active.append(track)

        # 매칭 안 된 탐지는 새 트랙.
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            track = Track(track_id=self._next_id)
            self._next_id += 1
            track.observations.append(
                TrackObservation(timestamp, frame_index, det, patches[di], flows[di])
            )
            still_active.append(track)

        self.active = still_active
        return self.active

    def close(self) -> list[Track]:
        """남은 활성 트랙을 모두 종료하고 전체 트랙을 돌려준다."""
        self.finished.extend(self.active)
        self.active = []
        return self.finished

    @property
    def all_tracks(self) -> list[Track]:
        return self.finished + self.active
