"""이벤트 자료형."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum

from firstlight.geo.raycast import GeoFix
from firstlight.verify.features import TrackFeatures
from firstlight.verify.scorer import Tier


class EventLabel(str, Enum):
    """관제 요원의 판정. 그대로 재학습 라벨이 된다."""

    UNLABELLED = "unlabelled"
    CONFIRMED = "confirmed"        # 실제 연기였다
    FALSE_POSITIVE = "false_positive"


@dataclass
class Event:
    """단일 연기 후보에 대한 시스템의 판단.

    좌표 필드는 `geo_ok` 가 True 일 때만 의미가 있다. False 면 지오레퍼런싱이
    좌표 발행을 거절한 것이고 `geo_reject_reason` 에 이유가 들어간다.
    """

    track_id: int
    tier: Tier
    score: float
    timestamp: float                     # 관측 시각 (영상 기준 초)
    site: str = ""
    camera: str = ""

    # 탐지
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    confidence: float = 0.0
    n_observations: int = 0

    # 지오레퍼런싱
    geo_ok: bool = False
    lat: float | None = None
    lon: float | None = None
    elevation_m: float | None = None
    range_m: float | None = None
    depression_deg: float | None = None
    cep50_m: float | None = None
    cep90_m: float | None = None
    geo_reject_reason: str | None = None

    # 시퀀스 검증 2조건 판정 (작품설명서 Ⅱ-1). 등급의 직접적 근거다.
    flow_ok: bool = False
    area_ok: bool = False
    criteria: dict = field(default_factory=dict)

    # 환류
    label: EventLabel = EventLabel.UNLABELLED
    features: dict = field(default_factory=dict)
    explanation: dict = field(default_factory=dict)

    event_id: int | None = None
    created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------

    @classmethod
    def from_verdict(
        cls,
        track_id: int,
        tier: Tier,
        score: float,
        timestamp: float,
        bbox: tuple[float, float, float, float],
        confidence: float,
        features: TrackFeatures,
        fix: GeoFix | None,
        explanation: dict | None = None,
        site: str = "",
        camera: str = "",
        n_observations: int = 0,
        criteria=None,
    ) -> Event:
        event = cls(
            track_id=track_id,
            tier=tier,
            score=score,
            timestamp=timestamp,
            site=site,
            camera=camera,
            bbox=bbox,
            confidence=confidence,
            n_observations=n_observations,
            features=asdict(features) if features else {},
            explanation=explanation or {},
        )
        if criteria is not None:
            event.flow_ok = bool(criteria.flow_ok)
            event.area_ok = bool(criteria.area_ok)
            event.criteria = {
                "flow_ok": bool(criteria.flow_ok),
                "area_ok": bool(criteria.area_ok),
                "n_satisfied": criteria.n_satisfied,
                "enough_frames": bool(criteria.enough_frames),
                "flow_from_proxy": bool(criteria.flow_from_proxy),
                "flow_upward_ratio": criteria.flow_upward_ratio,
                "flow_divergence": criteria.flow_divergence,
                "rise_rate": criteria.rise_rate,
                "area_growth_rate": criteria.area_growth_rate,
                "area_monotonicity": criteria.area_monotonicity,
                "reasons": criteria.reasons(),
            }
        if fix is not None:
            event.depression_deg = fix.depression_deg
            event.cep50_m = fix.cep50_m
            event.cep90_m = fix.cep90_m
            if fix.ok:
                event.geo_ok = True
                event.lat = fix.lat
                event.lon = fix.lon
                event.elevation_m = fix.elevation_m
                event.range_m = fix.range_m
            else:
                event.geo_reject_reason = (
                    fix.rejected.value if fix.rejected else "unknown"
                )
        return event

    # ------------------------------------------------------------------

    def to_row(self) -> dict:
        """sqlite 행으로. dataclass 필드를 평탄화한다."""
        return {
            "track_id": self.track_id,
            "tier": self.tier.value,
            "score": self.score,
            "timestamp": self.timestamp,
            "site": self.site,
            "camera": self.camera,
            "bbox": json.dumps(list(self.bbox)),
            "confidence": self.confidence,
            "n_observations": self.n_observations,
            "geo_ok": int(self.geo_ok),
            "lat": self.lat,
            "lon": self.lon,
            "elevation_m": self.elevation_m,
            "range_m": self.range_m,
            "depression_deg": self.depression_deg,
            "cep50_m": self.cep50_m,
            "cep90_m": self.cep90_m,
            "geo_reject_reason": self.geo_reject_reason,
            "flow_ok": int(self.flow_ok),
            "area_ok": int(self.area_ok),
            "criteria": json.dumps(self.criteria, ensure_ascii=False),
            "label": self.label.value,
            "features": json.dumps(self.features, ensure_ascii=False),
            "explanation": json.dumps(self.explanation, ensure_ascii=False),
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row) -> Event:
        return cls(
            event_id=row["id"],
            track_id=row["track_id"],
            tier=Tier(row["tier"]),
            score=row["score"],
            timestamp=row["timestamp"],
            site=row["site"] or "",
            camera=row["camera"] or "",
            bbox=tuple(json.loads(row["bbox"])),
            confidence=row["confidence"],
            n_observations=row["n_observations"],
            geo_ok=bool(row["geo_ok"]),
            lat=row["lat"],
            lon=row["lon"],
            elevation_m=row["elevation_m"],
            range_m=row["range_m"],
            depression_deg=row["depression_deg"],
            cep50_m=row["cep50_m"],
            cep90_m=row["cep90_m"],
            geo_reject_reason=row["geo_reject_reason"],
            flow_ok=bool(row["flow_ok"]),
            area_ok=bool(row["area_ok"]),
            criteria=json.loads(row["criteria"] or "{}"),
            label=EventLabel(row["label"]),
            features=json.loads(row["features"] or "{}"),
            explanation=json.loads(row["explanation"] or "{}"),
            created_at=row["created_at"],
        )

    def to_public_dict(self) -> dict:
        """대시보드로 나가는 형태."""
        data = asdict(self)
        data["tier"] = self.tier.value
        data["label"] = self.label.value
        data["tier_colour"] = self.tier.colour
        data["tier_label_ko"] = self.tier.label_ko
        data["bbox"] = list(self.bbox)
        return data
