"""폐루프 자체검증 — 이 저장소에서 가장 중요한 테스트.

알려진 지상점을 알려진 자세로 투영해 픽셀을 얻고, 그 픽셀을 지오레퍼런싱
엔진에 되먹여 원래 좌표가 복원되는지 본다. **정답을 아는 검증**이므로
드론도 영상도 필요 없고, 소개서가 주장하는 "오차 50m 이내"의 근거가 된다.

여기가 통과하지 않으면 CEP 표의 숫자는 전부 의미가 없다.
"""

import numpy as np
import pytest

from firstlight.geo.camera import CameraIntrinsics
from firstlight.geo.dem import synthetic_dem
from firstlight.geo.frame import geodetic_to_enu
from firstlight.geo.pose import CameraPose
from firstlight.geo.raycast import RejectReason
from firstlight.geo.solver import GeoSolver, GeoSolverConfig, project_ground_point
from firstlight.geo.uncertainty import NoiseModel
from firstlight.geo.wind import Wind

LAT0, LON0 = 36.4127, 128.7043
BBOX = (LON0 - 0.35, LAT0 - 0.35, LON0 + 0.35, LAT0 + 0.35)


@pytest.fixture(scope="module")
def dem():
    return synthetic_dem(BBOX, seed=7, relief_m=450.0, base_m=180.0)


@pytest.fixture(scope="module")
def intrinsics():
    return CameraIntrinsics.from_fov(1920, 1080, hfov_deg=73.7, name="test")


def make_solver(dem, intrinsics, **cfg_kwargs):
    cfg = GeoSolverConfig(
        noise=NoiseModel.perfect(), mc_trials=32, step_m=5.0, **cfg_kwargs
    )
    return GeoSolver(dem, intrinsics, cfg, rng=np.random.default_rng(0))


def ground_distance_m(lat_a, lon_a, lat_b, lon_b) -> float:
    east, north, _ = geodetic_to_enu(lon_b, lat_b, 0.0, lat_a, lon_a, 0.0)
    return float(np.hypot(east, north))


# ------------------------------------------------------------------ 폐루프


@pytest.mark.parametrize("yaw_deg", [0.0, 47.0, 135.0, 213.0, 305.0])
@pytest.mark.parametrize("pitch_deg", [-70.0, -45.0, -25.0, -15.0])
def test_closed_loop_recovers_ground_point(dem, intrinsics, yaw_deg, pitch_deg):
    """지상점 → 픽셀 → 해석 → 같은 지상점.

    무노이즈이므로 왕복 오차는 이산화 한계(< 1m)여야 한다.
    """
    pose = CameraPose(lat=LAT0, lon=LON0, alt_msl=1400.0,
                      yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=0.0)
    solver = make_solver(dem, intrinsics)

    # 먼저 광축을 쏘아 지형 위의 실재하는 점 하나를 얻는다.
    seed_fix = solver.solve(pose, intrinsics.cx, intrinsics.cy, with_uncertainty=False)
    assert seed_fix.ok, f"기준점 확보 실패: {seed_fix.rejected}"

    # 그 점을 다시 투영해 픽셀을 만들고,
    uv = project_ground_point(intrinsics, pose, seed_fix.lat, seed_fix.lon,
                              seed_fix.elevation_m)
    assert uv is not None
    assert uv[0] == pytest.approx(intrinsics.cx, abs=0.5)
    assert uv[1] == pytest.approx(intrinsics.cy, abs=0.5)

    # 그 픽셀로 다시 풀면 원래 점이 나와야 한다.
    fix = solver.solve(pose, uv[0], uv[1], with_uncertainty=False)
    assert fix.ok
    assert ground_distance_m(seed_fix.lat, seed_fix.lon, fix.lat, fix.lon) < 1.0


@pytest.mark.parametrize("u_frac,v_frac", [(0.2, 0.8), (0.5, 0.65), (0.85, 0.9), (0.1, 0.95)])
def test_closed_loop_across_image_plane(dem, intrinsics, u_frac, v_frac):
    """화면 중앙뿐 아니라 주변부 픽셀에서도 왕복이 성립해야 한다."""
    pose = CameraPose(lat=LAT0, lon=LON0, alt_msl=1500.0,
                      yaw_deg=30.0, pitch_deg=-35.0, roll_deg=8.0)
    solver = make_solver(dem, intrinsics)

    u = u_frac * intrinsics.width
    v = v_frac * intrinsics.height
    fix = solver.solve(pose, u, v, with_uncertainty=False)
    if not fix.ok:
        pytest.skip(f"이 픽셀은 거절됨: {fix.rejected}")

    uv = project_ground_point(intrinsics, pose, fix.lat, fix.lon, fix.elevation_m)
    assert uv is not None
    assert uv[0] == pytest.approx(u, abs=0.5)
    assert uv[1] == pytest.approx(v, abs=0.5)


def test_hit_lies_on_the_terrain_surface(dem, intrinsics):
    """해가 실제로 지형면 위의 점이어야 한다 (공중이나 지하가 아니라)."""
    pose = CameraPose(LAT0, LON0, 1400.0, yaw_deg=100.0, pitch_deg=-40.0)
    solver = make_solver(dem, intrinsics)
    fix = solver.solve(pose, intrinsics.cx, intrinsics.cy, with_uncertainty=False)
    assert fix.ok
    assert fix.elevation_m == pytest.approx(float(dem.elevation(fix.lon, fix.lat)), abs=0.1)


# ------------------------------------------------------------------ 거절 경로


def test_grazing_ray_is_rejected(dem, intrinsics):
    """수평에 가까운 광선은 좌표를 내지 않아야 한다."""
    pose = CameraPose(LAT0, LON0, 1400.0, yaw_deg=0.0, pitch_deg=-2.0)
    fix = make_solver(dem, intrinsics).solve(pose, intrinsics.cx, intrinsics.cy)
    assert not fix.ok
    assert fix.rejected is RejectReason.GRAZING
    assert np.isnan(fix.lat)


def test_upward_ray_is_rejected(dem, intrinsics):
    pose = CameraPose(LAT0, LON0, 1400.0, yaw_deg=0.0, pitch_deg=+20.0)
    fix = make_solver(dem, intrinsics).solve(pose, intrinsics.cx, intrinsics.cy)
    assert not fix.ok
    assert fix.rejected is RejectReason.GRAZING


def test_cep_limit_rejects_noisy_geometry(dem, intrinsics):
    """오차반경이 상한을 넘으면 좌표를 발행하지 않는다."""
    pose = CameraPose(LAT0, LON0, 1400.0, yaw_deg=0.0, pitch_deg=-12.0)
    sloppy = NoiseModel(gps_horizontal_m=5.0, gps_vertical_m=10.0, yaw_deg=15.0,
                        pitch_deg=8.0, roll_deg=8.0, pixel_px=20.0, dem_vertical_m=15.0)
    cfg = GeoSolverConfig(noise=sloppy, mc_trials=128, step_m=10.0, max_cep90_m=50.0)
    fix = GeoSolver(dem, intrinsics, cfg, rng=np.random.default_rng(1)).solve(
        pose, intrinsics.cx, intrinsics.cy
    )
    assert not fix.ok
    assert fix.rejected is RejectReason.CEP_TOO_LARGE
    # 거절해도 왜 거절했는지는 남겨야 한다.
    assert fix.cep90_m is not None and fix.cep90_m > 50.0


def test_rejected_fix_reports_not_ok(dem, intrinsics):
    """호출부가 좌표를 먼저 읽지 못하도록 NaN 이어야 한다."""
    pose = CameraPose(LAT0, LON0, 1400.0, yaw_deg=0.0, pitch_deg=0.0)
    fix = make_solver(dem, intrinsics).solve(pose, intrinsics.cx, intrinsics.cy)
    assert not fix.ok and np.isnan(fix.lat) and np.isnan(fix.lon)


# ------------------------------------------------------------------ 오차·부가


def test_cep_grows_with_noise(dem, intrinsics):
    """노이즈가 커지면 CEP 도 커져야 한다 (몬테카를로가 실제로 동작하는지)."""
    pose = CameraPose(LAT0, LON0, 1400.0, yaw_deg=0.0, pitch_deg=-45.0)

    def cep(noise):
        cfg = GeoSolverConfig(noise=noise, mc_trials=256, step_m=10.0, max_cep90_m=1e9)
        f = GeoSolver(dem, intrinsics, cfg, rng=np.random.default_rng(2)).solve(
            pose, intrinsics.cx, intrinsics.cy
        )
        assert f.ok
        return f.cep90_m

    assert cep(NoiseModel.perfect()) < cep(NoiseModel.rtk()) < cep(NoiseModel.consumer_gnss())


def test_perfect_noise_gives_zero_cep(dem, intrinsics):
    pose = CameraPose(LAT0, LON0, 1400.0, yaw_deg=15.0, pitch_deg=-50.0)
    fix = make_solver(dem, intrinsics).solve(pose, intrinsics.cx, intrinsics.cy)
    assert fix.ok
    assert fix.cep90_m == pytest.approx(0.0, abs=0.01)


def test_bbox_uses_centre_pixel_by_default(dem, intrinsics):
    """작품설명서 Ⅱ-1: "탐지된 연기 영역의 **중심 픽셀** (u, v)를 좌표로 변환"."""
    pose = CameraPose(LAT0, LON0, 1500.0, yaw_deg=0.0, pitch_deg=-40.0)
    solver = make_solver(dem, intrinsics)
    bbox = (800.0, 300.0, 1000.0, 700.0)

    from_bbox = solver.solve_bbox(pose, bbox, with_uncertainty=False)
    from_centre = solver.solve(pose, 900.0, 500.0, with_uncertainty=False)

    assert from_bbox.ok and from_centre.ok
    assert ground_distance_m(from_bbox.lat, from_bbox.lon,
                             from_centre.lat, from_centre.lon) < 0.01


def test_bbox_base_anchor_is_available(dem, intrinsics):
    """하단 중앙(연기 기저) 방식도 옵션으로 남아 있어야 한다.

    중심 픽셀은 연기 기둥 높이만큼 풍하로 밀리므로, 현장에서 두 방식을
    비교할 수 있어야 한다.
    """
    pose = CameraPose(LAT0, LON0, 1500.0, yaw_deg=0.0, pitch_deg=-40.0)
    solver = make_solver(dem, intrinsics)
    bbox = (800.0, 300.0, 1000.0, 700.0)

    centre = solver.solve_bbox(pose, bbox, with_uncertainty=False)
    base = solver.solve_bbox(pose, bbox, with_uncertainty=False, anchor="base")
    from_bottom = solver.solve(pose, 900.0, 700.0, with_uncertainty=False)

    assert ground_distance_m(base.lat, base.lon,
                             from_bottom.lat, from_bottom.lon) < 0.01
    # 두 방식이 실제로 다른 좌표를 내야 비교에 의미가 있다.
    assert ground_distance_m(centre.lat, centre.lon, base.lat, base.lon) > 10.0


def test_wind_correction_moves_upwind_and_keeps_original(dem, intrinsics):
    """풍향 보정은 풍상으로 움직이고, 보정 전 좌표를 남겨야 한다."""
    pose = CameraPose(LAT0, LON0, 1400.0, yaw_deg=0.0, pitch_deg=-50.0)
    cfg = GeoSolverConfig(noise=NoiseModel.perfect(), mc_trials=16, step_m=5.0,
                          wind_drift_seconds=10.0)
    solver = GeoSolver(dem, intrinsics, cfg, rng=np.random.default_rng(3))

    # 북서풍: 북서에서 불어온다 → 남동으로 흐른다 → 발화점은 북서쪽에 있다.
    wind = Wind(speed_ms=6.2, from_deg=315.0)
    fix = solver.solve(pose, intrinsics.cx, intrinsics.cy, wind=wind)
    assert fix.ok
    assert fix.lat_uncorrected is not None and fix.lon_uncorrected is not None

    east, north, _ = geodetic_to_enu(fix.lon, fix.lat, 0.0,
                                     fix.lat_uncorrected, fix.lon_uncorrected, 0.0)
    assert float(north) > 0 and float(east) < 0, "보정은 북서(풍상) 방향이어야 한다"
    assert float(np.hypot(east, north)) == pytest.approx(6.2 * 10.0, rel=0.01)


def test_wind_disabled_by_default(dem, intrinsics):
    """drift_seconds 기본값 0 이면 보정이 꺼져 있어야 한다."""
    pose = CameraPose(LAT0, LON0, 1400.0, yaw_deg=0.0, pitch_deg=-50.0)
    fix = make_solver(dem, intrinsics).solve(
        pose, intrinsics.cx, intrinsics.cy, wind=Wind(6.2, 315.0), with_uncertainty=False
    )
    assert fix.ok and fix.lat_uncorrected is None
