"""파이프라인 종단 동작과 관제 API.

무거운 탐지 모델 대신 가짜 탐지기를 쓴다. 여기서 검증할 것은 탐지 성능이
아니라 **배선**이다 — 좌표 실패가 등급을 낮추는가, 억제가 통지를 줄이는가,
라벨이 재학습으로 이어지는가.
"""

import numpy as np
import pytest

from firstlight.detect.detector import Detection
from firstlight.events.models import EventLabel
from firstlight.events.router import AlertRouter
from firstlight.events.store import EventStore
from firstlight.geo.camera import CameraIntrinsics
from firstlight.geo.dem import synthetic_dem
from firstlight.geo.pose import CameraPose
from firstlight.geo.solver import GeoSolver, GeoSolverConfig
from firstlight.geo.uncertainty import NoiseModel
from firstlight.pipeline import Pipeline
from firstlight.verify.scorer import SequenceScorer, Tier

LAT0, LON0 = 36.4127, 128.7043
BBOX = (LON0 - 0.35, LAT0 - 0.35, LON0 + 0.35, LAT0 + 0.35)


class GrowingSmokeDetector:
    """프레임마다 커지고 올라가는 박스 하나를 낸다."""

    def __init__(self, x=900.0, y0=700.0, size0=60.0, growth=0.10, rise=6.0):
        self.i = 0
        self.x, self.y0, self.size0, self.growth, self.rise = x, y0, size0, growth, rise
        self.conf_threshold = 0.2

    def detect(self, frame_bgr):
        size = self.size0 * (1.0 + self.growth) ** self.i
        cy = self.y0 - self.rise * self.i
        self.i += 1
        return [
            Detection(self.x - size / 2, cy - size / 2,
                      self.x + size / 2, cy + size / 2, 0.75)
        ]


class SilentDetector:
    conf_threshold = 0.2

    def detect(self, frame_bgr):
        return []


@pytest.fixture(scope="module")
def dem():
    return synthetic_dem(BBOX, resolution_deg=1 / 1200, seed=7)


@pytest.fixture(scope="module")
def intrinsics():
    return CameraIntrinsics.from_fov(1920, 1080, hfov_deg=73.7)


@pytest.fixture
def store():
    s = EventStore(":memory:")
    yield s
    s.close()


def make_pose(dem, pitch_deg=-40.0):
    alt = float(dem.elevation(LON0, LAT0)) + 400.0
    return CameraPose(LAT0, LON0, alt, yaw_deg=15.0, pitch_deg=pitch_deg)


def blank_frame():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def make_solver(dem, intrinsics, **kw):
    cfg = GeoSolverConfig(noise=NoiseModel.consumer_gnss(), mc_trials=24,
                          step_m=10.0, **kw)
    return GeoSolver(dem, intrinsics, cfg, rng=np.random.default_rng(0))


# ---------------------------------------------------------------- 파이프라인


def test_growing_smoke_reaches_flare(dem, intrinsics, store):
    pipeline = Pipeline(
        detector=GrowingSmokeDetector(),
        geo_solver=make_solver(dem, intrinsics),
        router=AlertRouter(store),
        frame_interval_s=60.0,
        site="uiseong",
    )
    pose = make_pose(dem)

    tiers = []
    for i in range(12):
        result = pipeline.process_frame(blank_frame(), float(i * 60), i, pose=pose)
        tiers.append(result.verdicts[0].tier)

    assert Tier.FLARE in tiers, f"성장하는 연기가 FLARE 에 도달하지 못했다: {tiers}"


def test_flare_requires_minimum_observations(dem, intrinsics, store):
    """첫 두 프레임에서는 FLARE 가 나오면 안 된다."""
    pipeline = Pipeline(
        detector=GrowingSmokeDetector(),
        geo_solver=make_solver(dem, intrinsics),
        router=AlertRouter(store),
        frame_interval_s=60.0,
    )
    pose = make_pose(dem)
    for i in range(2):
        result = pipeline.process_frame(blank_frame(), float(i * 60), i, pose=pose)
        assert result.verdicts[0].tier is not Tier.FLARE


def test_grazing_geometry_blocks_flare(dem, intrinsics, store):
    """부각이 낮아 좌표를 못 내면 아무리 점수가 높아도 GLOW 여야 한다."""
    pipeline = Pipeline(
        detector=GrowingSmokeDetector(),
        geo_solver=make_solver(dem, intrinsics),
        router=AlertRouter(store),
        frame_interval_s=60.0,
    )
    pose = make_pose(dem, pitch_deg=-1.0)          # 거의 수평

    for i in range(12):
        result = pipeline.process_frame(blank_frame(), float(i * 60), i, pose=pose)
        verdict = result.verdicts[0]
        assert verdict.tier is not Tier.FLARE
        assert verdict.fix is not None and not verdict.fix.ok


def test_no_geo_solver_means_no_flare(dem, intrinsics, store):
    """지오레퍼런싱이 아예 없으면 자동 경보를 울리지 않는다."""
    pipeline = Pipeline(
        detector=GrowingSmokeDetector(),
        geo_solver=None,
        router=AlertRouter(store),
        frame_interval_s=60.0,
    )
    for i in range(12):
        result = pipeline.process_frame(blank_frame(), float(i * 60), i)
        assert result.verdicts[0].tier is not Tier.FLARE


def test_suppression_collapses_many_verdicts_into_few_alerts(dem, intrinsics, store):
    """판정은 매 프레임, 통지는 상태가 바뀔 때만."""
    router = AlertRouter(store)
    pipeline = Pipeline(
        detector=GrowingSmokeDetector(),
        geo_solver=make_solver(dem, intrinsics),
        router=router,
        frame_interval_s=60.0,
        site="uiseong",
    )
    pose = make_pose(dem)

    n_verdicts = 0
    for i in range(30):
        n_verdicts += len(pipeline.process_frame(blank_frame(), float(i * 60), i, pose=pose).verdicts)

    assert n_verdicts >= 25
    # GLOW 1건 + FLARE 승격 1건 정도여야 한다.
    assert len(router.notifications) <= 3, f"통지가 너무 많다: {len(router.notifications)}"


def test_empty_detections_produce_nothing(store):
    pipeline = Pipeline(detector=SilentDetector(), router=AlertRouter(store),
                        frame_interval_s=1.0)
    result = pipeline.process_frame(blank_frame(), 0.0, 0)
    assert result.detections == [] and result.verdicts == []


def test_events_are_persisted_with_features(dem, intrinsics, store):
    pipeline = Pipeline(
        detector=GrowingSmokeDetector(),
        geo_solver=make_solver(dem, intrinsics),
        router=AlertRouter(store),
        frame_interval_s=60.0,
        site="uiseong",
    )
    pose = make_pose(dem)
    for i in range(6):
        pipeline.process_frame(blank_frame(), float(i * 60), i, pose=pose)

    events = store.list()
    assert events
    assert events[0].features, "특징이 저장되지 않으면 재학습을 할 수 없다"
    assert events[0].explanation, "설명이 없으면 판단 근거를 보여줄 수 없다"


def test_dense_mode_selected_for_video_rate(store):
    pipeline = Pipeline(detector=SilentDetector(), frame_interval_s=1 / 25)
    assert pipeline.mode.value == "dense"


# ---------------------------------------------------------------------- API


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from firstlight.api.main import create_app

    return TestClient(create_app(str(tmp_path / "events.db"),
                                 tmp_path / "scorer.json"))


def _seed(client, tmp_path_db=None):
    """API 가 읽을 이벤트를 하나 넣는다."""
    from firstlight.events.models import Event
    from firstlight.geo.raycast import GeoFix
    from firstlight.verify.features import TrackFeatures

    store = client.app.state.store
    return store.add(
        Event.from_verdict(
            track_id=1, tier=Tier.GLOW, score=0.5, timestamp=0.0,
            bbox=(10, 10, 60, 60), confidence=0.6,
            features=TrackFeatures(persistence=0.9),
            fix=GeoFix(LAT0, LON0, 300.0, 800.0, 25.0, cep50_m=30.0, cep90_m=60.0),
            site="uiseong",
        )
    )


def test_index_serves_dashboard(client):
    """빌드가 있으면 React 앱을, 없으면 빌드 안내를 준다.

    어느 쪽이든 200 이어야 한다 — 프런트엔드가 빌드되지 않았다고 API 까지
    죽으면 파이프라인 검증을 못 한다.
    """
    from firstlight.api.main import DIST_DIR

    response = client.get("/")
    assert response.status_code == 200
    assert "FIRSTLIGHT" in response.text

    if (DIST_DIR / "index.html").exists():
        # Vite 산출물은 해시가 붙은 모듈 스크립트를 물고 있다.
        assert 'type="module"' in response.text
        assert "/assets/" in response.text
    else:
        assert "npm run build" in response.text


def test_events_endpoint(client):
    _seed(client)
    body = client.get("/api/events").json()
    assert len(body) == 1
    assert body[0]["tier"] == "GLOW"
    assert body[0]["tier_colour"] == "#F4A261"
    assert body[0]["tier_label_ko"] == "어른거림"


def test_label_endpoint_closes_the_loop(client):
    event = _seed(client)
    response = client.post(f"/api/events/{event.event_id}/label",
                           json={"label": "false_positive"})
    assert response.status_code == 200

    assert client.get("/api/summary").json()["labelled"] == 1
    assert client.get(f"/api/events/{event.event_id}").json()["label"] == "false_positive"


def test_label_rejects_unknown_value(client):
    event = _seed(client)
    assert client.post(f"/api/events/{event.event_id}/label",
                       json={"label": "maybe"}).status_code == 400


def test_label_missing_event_is_404(client):
    assert client.post("/api/events/9999/label",
                       json={"label": "confirmed"}).status_code == 404


def test_retrain_refuses_insufficient_labels(client):
    """표본이 모자라면 조용히 퇴화한 모델을 만드는 대신 거절해야 한다."""
    event = _seed(client)
    client.post(f"/api/events/{event.event_id}/label", json={"label": "confirmed"})
    response = client.post("/api/scorer/retrain")
    assert response.status_code == 400
    assert "최소 8개" in response.json()["detail"]


def test_summary_reports_scorer_state(client):
    body = client.get("/api/summary").json()
    assert body["scorer"]["fitted"] is False
    assert body["scorer"]["mode"] == "sparse"


def test_site_endpoint_exposes_response_units(client):
    """지도에 최근접 진화대를 겹쳐 표시하려면 좌표가 나와야 한다."""
    body = client.get("/api/site").json()
    assert "response_units" in body


def test_websocket_pushes_events(client):
    """작품설명서 표 Ⅱ-1 「WebSocket」 — 폴링이 아니라 서버가 밀어준다.

    이 테스트가 있는 이유: `from __future__ import annotations` 때문에
    FastAPI 가 WebSocket 주석을 해석하지 못해 연결이 조용히 끊긴 적이 있다.
    라우트가 등록돼 있어도 실제로 붙어서 데이터가 오는지를 봐야 잡힌다.
    """
    _seed(client)
    with client.websocket_connect("/api/ws") as socket:
        payload = socket.receive_json()

    assert payload["type"] == "events"
    assert len(payload["events"]) == 1
    assert payload["events"][0]["tier"] == "GLOW"


def test_websocket_event_carries_criteria(client):
    """등급 근거(2조건)가 스트림에도 실려야 대시보드가 설명할 수 있다."""
    _seed(client)
    with client.websocket_connect("/api/ws") as socket:
        event = socket.receive_json()["events"][0]

    assert "flow_ok" in event and "area_ok" in event
    assert "criteria" in event
