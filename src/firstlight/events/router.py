"""등급 라우팅 — 무엇을 누구에게 언제 보낼지.

소개서 §4③ 의 문장이 이 모듈의 설계 원칙이다:
    "안전 시스템은 놓치는 것보다 무시당하는 것이 더 위험하다."

같은 불에 대해 경보를 40번 보내면 담당자는 41번째를 안 본다. 그래서
탐지와 통지를 분리한다 — 트랙은 매 프레임 재평가하되, **통지는 상태가
실제로 바뀔 때만** 나간다.

억제 규칙 셋:
    1. 트랙 단위  — 같은 트랙은 등급이 **올라갈 때만** 다시 통지한다.
    2. 공간 단위  — 이미 통지한 지점 근처의 새 트랙은 억제한다. 하나의
                    불이 연기 기둥 여러 개로 쪼개져 탐지되는 경우다.
    3. 시간 단위  — 쿨다운 안에서는 같은 지점을 다시 울리지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from firstlight.events.models import Event
from firstlight.events.store import EventStore
from firstlight.geo.frame import lonlat_to_enu
from firstlight.verify.scorer import Tier

# 등급 서열. 올라갈 때만 재통지한다.
_RANK = {Tier.SPARK: 0, Tier.GLOW: 1, Tier.FLARE: 2}


@dataclass
class RouterConfig:
    """억제 파라미터.

    `spatial_radius_m` 는 CEP 와 같은 자릿수여야 한다. 너무 작으면 하나의
    불이 여러 경보로 쪼개지고, 너무 크면 인접한 별개의 발화를 삼킨다.
    실측 CEP50 이 부각 20도에서 34m 이므로 300m 는 넉넉한 편이다.
    """

    spatial_radius_m: float = 300.0
    cooldown_s: float = 600.0
    notify_tiers: tuple[Tier, ...] = (Tier.FLARE, Tier.GLOW)
    # SPARK 는 통지하지 않지만 저장은 한다 — 재학습 데이터이기 때문이다.
    persist_tiers: tuple[Tier, ...] = (Tier.FLARE, Tier.GLOW, Tier.SPARK)


@dataclass
class Notification:
    """실제로 나가는 통지."""

    event: Event
    reason: str                       # "new" | "escalated"
    nearest_unit_km: float | None = None


@dataclass
class _Recent:
    lat: float
    lon: float
    tier: Tier
    at: float


class AlertRouter:
    """등급 판정 결과를 저장·통지로 옮긴다."""

    def __init__(
        self,
        store: EventStore,
        config: RouterConfig | None = None,
        response_units: list[dict] | None = None,
    ) -> None:
        self.store = store
        self.config = config or RouterConfig()
        self.response_units = response_units or []
        self._track_tier: dict[tuple[str, int], Tier] = {}
        self._recent: list[_Recent] = []
        self.notifications: list[Notification] = []

    # ------------------------------------------------------------------

    def _suppressed_spatially(self, event: Event, now: float) -> bool:
        """이미 통지한 지점 근처인가."""
        if not event.geo_ok or event.lat is None:
            return False
        cfg = self.config
        self._recent = [r for r in self._recent if now - r.at <= cfg.cooldown_s]
        for recent in self._recent:
            east, north = lonlat_to_enu(event.lon, event.lat, recent.lat, recent.lon)
            if float(np.hypot(east, north)) <= cfg.spatial_radius_m:
                # 등급이 올라갔다면 억제하지 않는다.
                if _RANK[event.tier] > _RANK[recent.tier]:
                    return False
                return True
        return False

    def _nearest_unit_km(self, event: Event) -> float | None:
        if not event.geo_ok or not self.response_units:
            return None
        best = None
        for unit in self.response_units:
            east, north = lonlat_to_enu(
                event.lon, event.lat, float(unit["lat"]), float(unit["lon"])
            )
            distance = float(np.hypot(east, north)) / 1000.0
            best = distance if best is None else min(best, distance)
        return best

    def route(self, event: Event, now: float | None = None) -> Notification | None:
        """이벤트를 저장하고, 통지가 필요하면 만들어 돌려준다."""
        now = now if now is not None else time.time()
        cfg = self.config

        if event.tier in cfg.persist_tiers:
            self.store.add(event)

        if event.tier not in cfg.notify_tiers:
            return None

        key = (event.site, event.track_id)
        previous = self._track_tier.get(key)
        if previous is not None and _RANK[event.tier] <= _RANK[previous]:
            return None                       # 이미 같거나 더 높은 등급으로 알렸다
        reason = "escalated" if previous is not None else "new"

        if self._suppressed_spatially(event, now):
            self._track_tier[key] = event.tier
            return None

        self._track_tier[key] = event.tier
        if event.geo_ok and event.lat is not None:
            self._recent.append(_Recent(event.lat, event.lon, event.tier, now))

        notification = Notification(
            event=event, reason=reason, nearest_unit_km=self._nearest_unit_km(event)
        )
        self.notifications.append(notification)
        return notification

    # ------------------------------------------------------------------

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for note in self.notifications:
            counts[note.event.tier.value] = counts.get(note.event.tier.value, 0) + 1
        return {
            "notifications": len(self.notifications),
            "by_tier": counts,
            "tracks_seen": len(self._track_tier),
        }

    def format_alert(self, note: Notification) -> str:
        """소개서 §3 시나리오와 같은 형태의 한 줄 경보문."""
        event = note.event
        head = f"▲ {event.tier.value} 경보 — {event.site or '현장'}"
        if event.geo_ok:
            body = (
                f"{event.lat:.4f}N {event.lon:.4f}E "
                f"(오차반경 {event.cep50_m:.0f}m, 사거리 {event.range_m:,.0f}m)"
            )
        else:
            body = f"좌표 미발행 ({event.geo_reject_reason}) — 영상 확인 필요"
        tail = (
            f" / 최근접 진화대 {note.nearest_unit_km:.1f}km"
            if note.nearest_unit_km is not None
            else ""
        )
        return f"{head}\n  {body}{tail}"
