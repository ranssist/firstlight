"""이벤트 저장과 등급 라우팅.

라우터의 본체는 억제 로직이다. "같은 불에 40번 울리지 않는다"가 지켜지지
않으면 시스템은 무시당하고, 무시당하는 안전 시스템은 없는 것보다 나쁘다.
"""

import pytest

from firstlight.events.models import Event, EventLabel
from firstlight.events.router import AlertRouter, RouterConfig
from firstlight.events.store import EventStore
from firstlight.geo.raycast import GeoFix, RejectReason
from firstlight.verify.features import TrackFeatures
from firstlight.verify.scorer import Tier

LAT, LON = 36.4127, 128.7043


@pytest.fixture
def store():
    s = EventStore(":memory:")
    yield s
    s.close()


def make_event(
    track_id: int = 1,
    tier: Tier = Tier.FLARE,
    lat: float = LAT,
    lon: float = LON,
    geo_ok: bool = True,
    score: float = 0.9,
    timestamp: float = 0.0,
) -> Event:
    fix = (
        GeoFix(lat=lat, lon=lon, elevation_m=300.0, range_m=800.0,
               depression_deg=25.0, cep50_m=34.0, cep90_m=69.0)
        if geo_ok
        else GeoFix.reject(RejectReason.GRAZING, depression_deg=3.0)
    )
    return Event.from_verdict(
        track_id=track_id, tier=tier, score=score, timestamp=timestamp,
        bbox=(100, 100, 160, 160), confidence=0.7,
        features=TrackFeatures(), fix=fix, site="uiseong", camera="generic_wide",
    )


# ------------------------------------------------------------------- 저장소


def test_add_and_get_roundtrip(store):
    event = store.add(make_event())
    assert event.event_id is not None

    loaded = store.get(event.event_id)
    assert loaded is not None
    assert loaded.tier is Tier.FLARE
    assert loaded.lat == pytest.approx(LAT)
    assert loaded.geo_ok is True
    assert loaded.bbox == (100, 100, 160, 160)


def test_rejected_geo_survives_roundtrip(store):
    event = store.add(make_event(geo_ok=False, tier=Tier.GLOW))
    loaded = store.get(event.event_id)
    assert loaded.geo_ok is False
    assert loaded.lat is None
    assert loaded.geo_reject_reason == "grazing"
    # 거절해도 부각은 남아야 한다 — 왜 거절됐는지 설명할 수 있어야 한다.
    assert loaded.depression_deg == pytest.approx(3.0)


def test_filter_by_tier_and_label(store):
    store.add(make_event(track_id=1, tier=Tier.FLARE))
    store.add(make_event(track_id=2, tier=Tier.GLOW))
    store.add(make_event(track_id=3, tier=Tier.SPARK))

    assert len(store.list(tier="FLARE")) == 1
    assert len(store.list()) == 3
    assert store.counts_by_tier() == {"FLARE": 1, "GLOW": 1, "SPARK": 1}


def test_labelling_feeds_retraining(store):
    """관제 요원의 판정이 재학습 입력으로 나와야 한다 (§4③ 환류 루프)."""
    a = store.add(make_event(track_id=1))
    store.add(make_event(track_id=2))

    assert store.set_label(a.event_id, EventLabel.FALSE_POSITIVE)
    labelled = store.labelled_events()
    assert len(labelled) == 1
    assert labelled[0].label is EventLabel.FALSE_POSITIVE


def test_set_label_on_missing_event(store):
    assert store.set_label(9999, EventLabel.CONFIRMED) is False


def test_features_survive_roundtrip(store):
    """특징이 보존돼야 재학습이 가능하다."""
    feats = TrackFeatures(persistence=0.8, area_growth_rate=1.5e-3)
    event = Event.from_verdict(
        track_id=1, tier=Tier.GLOW, score=0.5, timestamp=0.0,
        bbox=(0, 0, 10, 10), confidence=0.5, features=feats, fix=None,
    )
    loaded = store.get(store.add(event).event_id)
    assert loaded.features["persistence"] == pytest.approx(0.8)
    assert loaded.features["area_growth_rate"] == pytest.approx(1.5e-3)


# -------------------------------------------------------------------- 라우터


def test_first_flare_notifies(store):
    router = AlertRouter(store)
    note = router.route(make_event(), now=0.0)
    assert note is not None and note.reason == "new"


def test_same_track_does_not_renotify(store):
    """같은 트랙에 40번 울리면 담당자가 알림을 꺼버린다."""
    router = AlertRouter(store)
    assert router.route(make_event(track_id=1), now=0.0) is not None
    for t in range(1, 20):
        assert router.route(make_event(track_id=1), now=float(t)) is None
    assert len(router.notifications) == 1


def test_escalation_from_glow_to_flare_notifies(store):
    """등급이 올라가면 다시 알려야 한다 — 상황이 실제로 바뀐 것이다."""
    router = AlertRouter(store)
    assert router.route(make_event(track_id=1, tier=Tier.GLOW), now=0.0) is not None
    assert router.route(make_event(track_id=1, tier=Tier.FLARE), now=5.0) is not None
    assert len(router.notifications) == 2


def test_downgrade_does_not_notify(store):
    router = AlertRouter(store)
    router.route(make_event(track_id=1, tier=Tier.FLARE), now=0.0)
    assert router.route(make_event(track_id=1, tier=Tier.GLOW), now=5.0) is None


def test_nearby_new_track_is_suppressed(store):
    """하나의 불이 연기 기둥 여러 개로 쪼개져 탐지되는 경우."""
    router = AlertRouter(store, RouterConfig(spatial_radius_m=300.0))
    assert router.route(make_event(track_id=1), now=0.0) is not None
    # 약 100m 북쪽 — 같은 불로 본다.
    assert router.route(make_event(track_id=2, lat=LAT + 0.0009), now=10.0) is None


def test_distant_new_track_still_notifies(store):
    """멀리 떨어진 별개의 발화까지 삼키면 안 된다."""
    router = AlertRouter(store, RouterConfig(spatial_radius_m=300.0))
    router.route(make_event(track_id=1), now=0.0)
    # 약 1.1km 북쪽.
    assert router.route(make_event(track_id=2, lat=LAT + 0.01), now=10.0) is not None


def test_suppression_expires_after_cooldown(store):
    router = AlertRouter(store, RouterConfig(spatial_radius_m=300.0, cooldown_s=600.0))
    router.route(make_event(track_id=1), now=0.0)
    assert router.route(make_event(track_id=2, lat=LAT + 0.0009), now=100.0) is None
    assert router.route(make_event(track_id=3, lat=LAT + 0.0009), now=700.0) is not None


def test_spark_is_stored_but_not_notified(store):
    """SPARK 는 알리지 않되 저장은 한다 — 재학습 데이터이기 때문이다."""
    router = AlertRouter(store)
    assert router.route(make_event(track_id=1, tier=Tier.SPARK), now=0.0) is None
    assert len(store.list(tier="SPARK")) == 1


def test_event_without_coordinate_still_notifies_as_glow(store):
    """좌표가 없어도 사람은 봐야 한다."""
    router = AlertRouter(store)
    note = router.route(make_event(track_id=1, tier=Tier.GLOW, geo_ok=False), now=0.0)
    assert note is not None
    assert "좌표 미발행" in router.format_alert(note)


def test_nearest_unit_distance(store):
    units = [{"name": "안평", "lat": LAT + 0.03, "lon": LON}]     # 약 3.3km
    router = AlertRouter(store, response_units=units)
    note = router.route(make_event(), now=0.0)
    assert note.nearest_unit_km == pytest.approx(3.3, abs=0.3)
    assert "진화대" in router.format_alert(note)


def test_alert_text_has_coordinate_and_error(store):
    router = AlertRouter(store)
    text = router.format_alert(router.route(make_event(), now=0.0))
    assert "FLARE" in text and "36.4127" in text and "오차반경" in text


def test_summary_counts(store):
    router = AlertRouter(store)
    router.route(make_event(track_id=1, tier=Tier.FLARE, lat=LAT), now=0.0)
    router.route(make_event(track_id=2, tier=Tier.GLOW, lat=LAT + 0.02), now=1.0)
    summary = router.summary()
    assert summary["notifications"] == 2
    assert summary["by_tier"] == {"FLARE": 1, "GLOW": 1}
