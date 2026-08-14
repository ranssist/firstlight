"""대응 상태 — 작품설명서 「이력 타임라인: 경보 이력 및 대응 상태(접수-출동-진화)」.

판정(EventLabel)과 다른 축이라는 것, 그리고 되돌릴 수 없다는 것이 핵심이다.
이 이력은 대응 지연을 재기 위한 기록이라 순서가 뒤집히면 측정이 무의미해진다.
"""

import pytest

from firstlight.events.models import Event, EventLabel, ResponseStatus
from firstlight.events.store import EventStore
from firstlight.geo.raycast import GeoFix
from firstlight.verify.features import TrackFeatures
from firstlight.verify.scorer import Tier

LAT, LON = 36.4127, 128.7043


@pytest.fixture
def store():
    s = EventStore(":memory:")
    yield s
    s.close()


def make_event() -> Event:
    return Event.from_verdict(
        track_id=1, tier=Tier.FLARE, score=0.9, timestamp=0.0,
        bbox=(10, 10, 60, 60), confidence=0.8, features=TrackFeatures(),
        fix=GeoFix(LAT, LON, 300.0, 800.0, 25.0, cep50_m=30.0, cep90_m=60.0),
        site="uiseong",
    )


# ------------------------------------------------------------------- 자료형


def test_order_follows_response_sequence():
    assert ResponseStatus.NONE.order == 0
    assert ResponseStatus.RECEIVED.order < ResponseStatus.DISPATCHED.order
    assert ResponseStatus.DISPATCHED.order < ResponseStatus.SUPPRESSED.order


def test_korean_labels():
    assert ResponseStatus.RECEIVED.label_ko == "접수"
    assert ResponseStatus.DISPATCHED.label_ko == "출동"
    assert ResponseStatus.SUPPRESSED.label_ko == "진화"


def test_new_event_starts_unresponded():
    assert make_event().response is ResponseStatus.NONE


# ------------------------------------------------------------------- 저장소


def test_set_response_records_timestamp(store):
    event = store.add(make_event())
    assert store.set_response(event.event_id, ResponseStatus.RECEIVED, at=1000.0)

    loaded = store.get(event.event_id)
    assert loaded.response is ResponseStatus.RECEIVED
    assert loaded.response_history == [{"status": "received", "at": 1000.0}]


def test_history_accumulates_each_step(store):
    """접수→출동 간격을 재려면 각 단계의 시각이 남아야 한다."""
    event = store.add(make_event())
    store.set_response(event.event_id, ResponseStatus.RECEIVED, at=100.0)
    store.set_response(event.event_id, ResponseStatus.DISPATCHED, at=340.0)
    store.set_response(event.event_id, ResponseStatus.SUPPRESSED, at=900.0)

    history = store.get(event.event_id).response_history
    assert [h["status"] for h in history] == ["received", "dispatched", "suppressed"]
    assert history[1]["at"] - history[0]["at"] == 240.0      # 접수→출동 4분


def test_response_is_independent_of_label(store):
    """오탐으로 판정한 건에도 접수 이력이 남을 수 있어야 한다."""
    event = store.add(make_event())
    store.set_response(event.event_id, ResponseStatus.RECEIVED, at=1.0)
    store.set_label(event.event_id, EventLabel.FALSE_POSITIVE)

    loaded = store.get(event.event_id)
    assert loaded.label is EventLabel.FALSE_POSITIVE
    assert loaded.response is ResponseStatus.RECEIVED


def test_set_response_on_missing_event(store):
    assert store.set_response(9999, ResponseStatus.RECEIVED) is False


# ----------------------------------------------------------------------- API


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from firstlight.api.main import create_app

    app = create_app(str(tmp_path / "events.db"), tmp_path / "s.json")
    client = TestClient(app)
    app.state.store.add(make_event())
    return client


def test_advance_through_all_steps(client):
    for status in ("received", "dispatched", "suppressed"):
        response = client.post("/api/events/1/response", json={"response": status})
        assert response.status_code == 200, response.text
        assert response.json()["response"] == status

    event = client.get("/api/events/1").json()
    assert event["response"] == "suppressed"
    assert event["response_label_ko"] == "진화"
    assert len(event["response_history"]) == 3


def test_cannot_go_backwards(client):
    """되돌린 이력은 대응 지연 측정을 무의미하게 만든다."""
    client.post("/api/events/1/response", json={"response": "dispatched"})
    response = client.post("/api/events/1/response", json={"response": "received"})
    assert response.status_code == 400
    assert "되돌릴 수 없다" in response.json()["detail"]


def test_cannot_repeat_same_step(client):
    client.post("/api/events/1/response", json={"response": "received"})
    assert client.post(
        "/api/events/1/response", json={"response": "received"}
    ).status_code == 400


def test_unknown_status_rejected(client):
    assert client.post(
        "/api/events/1/response", json={"response": "정리중"}
    ).status_code == 400


def test_missing_event_is_404(client):
    assert client.post(
        "/api/events/9999/response", json={"response": "received"}
    ).status_code == 404
