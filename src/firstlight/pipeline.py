"""파이프라인 — 프레임 하나가 경보가 되기까지.

    프레임 + 텔레메트리
        → 탐지          (detect/)
        → 트랙 연결      (verify/tracker)
        → 시퀀스 검증    (verify/features + scorer)
        → 좌표 산출      (geo/solver)
        → 등급 판정      (verify/scorer.tier — 좌표 실패는 여기서 강등된다)
        → 저장·통지      (events/router)

소개서 §3 시나리오의 "5초 안에 사람 개입 없이"가 이 함수 하나다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from firstlight.detect.detector import Detection, Detector
from firstlight.events.models import Event
from firstlight.events.router import AlertRouter, Notification
from firstlight.geo.camera import CameraIntrinsics
from firstlight.geo.pose import CameraPose
from firstlight.geo.raycast import GeoFix
from firstlight.geo.solver import GeoSolver
from firstlight.geo.wind import Wind
from firstlight.verify.criteria import (
    CriteriaConfig,
    SequenceCriteria,
    evaluate_criteria,
)
from firstlight.verify.egomotion import EgoMotionCompensator, residual_flow_stats
from firstlight.verify.features import TrackFeatures, VerifierMode, extract_features
from firstlight.verify.scorer import SequenceScorer, Tier
from firstlight.verify.tracker import Track, Tracker


@dataclass
class Verdict:
    """한 트랙에 대한 이번 프레임의 판단.

    `criteria` 가 등급을 정한다. `score` 는 같은 등급 안의 우선순위와
    재학습 환류에 쓰는 보조 신호다 (verify/criteria.py 참조).
    """

    track: Track
    features: TrackFeatures
    criteria: SequenceCriteria
    score: float
    tier: Tier
    fix: GeoFix | None
    event: Event | None = None
    notification: Notification | None = None


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    detections: list[Detection]
    verdicts: list[Verdict] = field(default_factory=list)
    egomotion_ok: bool = False


class Pipeline:
    """탐지부터 통지까지를 묶는다.

    Args:
        detector: 프레임 탐지기.
        scorer: 시퀀스 검증 스코어러.
        geo_solver: 지오레퍼런싱. None 이면 좌표를 내지 않고 모든 FLARE 가
            GLOW 로 강등된다 (좌표 없는 자동 경보는 보내지 않는다는 원칙).
        router: 저장·통지. None 이면 판단만 하고 아무것도 남기지 않는다.
        frame_interval_s: 프레임 간격. dense/sparse 모드를 가른다.
        use_egomotion: 드론처럼 카메라가 움직이면 켠다. 고정 카메라는 끈다.
    """

    def __init__(
        self,
        detector: Detector,
        scorer: SequenceScorer | None = None,
        geo_solver: GeoSolver | None = None,
        router: AlertRouter | None = None,
        tracker: Tracker | None = None,
        frame_interval_s: float = 1.0,
        use_egomotion: bool = False,
        site: str = "",
        camera: str = "",
        wind: Wind | None = None,
        criteria_config: CriteriaConfig | None = None,
        snapshot_dir: Path | None = None,
    ) -> None:
        self.detector = detector
        self.snapshot_dir = snapshot_dir
        self.mode = VerifierMode.from_interval(frame_interval_s)
        self.scorer = scorer or SequenceScorer(mode=self.mode)
        self.criteria_config = criteria_config or CriteriaConfig()
        self.geo_solver = geo_solver
        self.router = router
        self.tracker = tracker or Tracker()
        self.frame_interval_s = frame_interval_s
        self.use_egomotion = use_egomotion
        self.site = site
        self.camera = camera
        self.wind = wind

        self._ego = EgoMotionCompensator() if use_egomotion else None
        self._prev_gray: np.ndarray | None = None

    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        frame_index: int,
        pose: CameraPose | None = None,
    ) -> FrameResult:
        """프레임 하나를 처리한다."""
        import cv2

        detections = self.detector.detect(frame_bgr)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        patches = [_patch(frame_bgr, d) for d in detections]
        flows, ego_ok = self._flow_stats(gray, detections)

        active = self.tracker.update(
            detections, timestamp, frame_index, patches=patches, flows=flows
        )

        result = FrameResult(frame_index, timestamp, detections, egomotion_ok=ego_ok)
        for track in active:
            result.verdicts.append(self._judge(track, timestamp, pose, frame_bgr))

        self._prev_gray = gray
        return result

    # ------------------------------------------------------------------

    def _flow_stats(
        self, gray: np.ndarray, detections: list[Detection]
    ) -> tuple[list[dict | None], bool]:
        """dense 모드에서만 잔차 광류를 계산한다."""
        if self.mode is not VerifierMode.DENSE or self._prev_gray is None:
            return [None] * len(detections), False

        homography = None
        if self._ego is not None:
            homography = self._ego.estimate(self._prev_gray, gray, detections)

        stats = [
            residual_flow_stats(self._prev_gray, gray, det.bbox, homography)
            for det in detections
        ]
        # 자기운동 보정을 켰는데 추정에 실패했다면 그 사실을 알려야 한다.
        return stats, (homography is not None) if self._ego else True

    def _judge(
        self,
        track: Track,
        timestamp: float,
        pose: CameraPose | None,
        frame_bgr: np.ndarray | None = None,
    ) -> Verdict:
        features = extract_features(track, self.frame_interval_s, mode=self.mode)
        criteria = evaluate_criteria(track, features, self.criteria_config)
        score = self.scorer.score(features)

        fix = None
        if self.geo_solver is not None and pose is not None:
            fix = self.geo_solver.solve_bbox(pose, track.last.detection.bbox, wind=self.wind)

        # 등급은 **만족한 조건 수**로 정해진다 (작품설명서 Ⅱ-1).
        # 좌표 미발행·프레임 부족은 FLARE 를 GLOW 로 강등시킬 뿐 버리지 않는다.
        geo_ok = bool(fix is not None and fix.ok)
        tier = self.scorer.tier_from_criteria(criteria, geo_ok=geo_ok)

        verdict = Verdict(
            track=track,
            features=features,
            criteria=criteria,
            score=score,
            tier=tier,
            fix=fix,
        )

        if self.router is not None:
            # 탐지 시점 크롭을 남긴다 — 관제 요원의 1클릭 판정에 필요한
            # "그 순간 그 자리의 그림"이다 (events/snapshot.py 참조).
            snapshot = None
            if frame_bgr is not None and self.snapshot_dir is not None:
                from firstlight.events.snapshot import save_snapshot

                snapshot = save_snapshot(
                    frame_bgr,
                    track.last.detection.bbox,
                    self.snapshot_dir,
                    # 이벤트마다 별도 파일 — 트랙 단위로 덮어쓰면 과거
                    # 이벤트가 나중 프레임의 그림을 가리킨다.
                    name=(
                        f"{self.site or 'site'}"
                        f"-t{track.track_id}-f{track.last.frame_index}"
                    ),
                    tier=tier.value,
                )

            event = Event.from_verdict(
                track_id=track.track_id,
                tier=tier,
                score=score,
                timestamp=timestamp,
                bbox=track.last.detection.bbox,
                confidence=track.last.detection.confidence,
                features=features,
                fix=fix,
                explanation=self.scorer.explain(features).contributions,
                site=self.site,
                camera=self.camera,
                n_observations=track.hits,
                criteria=criteria,
            )
            event.snapshot = snapshot
            verdict.event = event
            verdict.notification = self.router.route(event, now=timestamp)

        return verdict

    def close(self) -> list[Track]:
        return self.tracker.close()


def _patch(frame: np.ndarray, det: Detection, size: int = 48) -> np.ndarray | None:
    """특징 계산용 축소 그레이스케일 패치."""
    import cv2

    h, w = frame.shape[:2]
    x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
    x2, y2 = min(w, int(np.ceil(det.x2))), min(h, int(np.ceil(det.y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    ) / 255.0


def build_pipeline(
    site_name: str,
    camera_name: str = "generic_wide",
    device: str = "gpu",
    conf: float = 0.20,
    frame_interval_s: float = 1.0,
    db_path: str = "data/events.db",
    scorer_path: str | None = None,
    use_egomotion: bool = True,
    synthetic_dem_fallback: bool = False,
    snapshot_dir: Path | str | None = "data/snapshots",
) -> tuple[Pipeline, CameraIntrinsics]:
    """설정 파일에서 파이프라인을 조립한다."""
    from firstlight.config import CameraConfig, SiteConfig
    from firstlight.detect import OnnxDetector
    from firstlight.events.router import AlertRouter
    from firstlight.events.store import EventStore
    from firstlight.geo.dem import synthetic_dem

    site = SiteConfig.load(site_name)
    camera = CameraConfig.load(camera_name)

    try:
        dem = site.load_dem()
    except FileNotFoundError:
        if not synthetic_dem_fallback:
            raise
        dem = synthetic_dem(site.bbox, resolution_deg=1 / 1200, seed=7)

    scorer = (
        SequenceScorer.load(Path(scorer_path))
        if scorer_path and Path(scorer_path).exists()
        else SequenceScorer(mode=VerifierMode.from_interval(frame_interval_s))
    )

    store = EventStore(db_path)
    pipeline = Pipeline(
        detector=OnnxDetector(device=device, conf_threshold=conf),
        scorer=scorer,
        geo_solver=GeoSolver(dem, camera.intrinsics),
        router=AlertRouter(store, response_units=site.response_units),
        frame_interval_s=frame_interval_s,
        use_egomotion=use_egomotion,
        site=site.name,
        camera=camera.name,
        snapshot_dir=Path(snapshot_dir) if snapshot_dir else None,
    )
    return pipeline, camera.intrinsics
