"""광선-지형 교차와 거절 로직.

평면 지형에서는 교차점을 손으로 풀 수 있으므로 해석해와 대조한다.
거절 경로는 하나하나 발동시켜 본다 — 거절이 이 모듈의 본체이고, 조용히
동작하지 않는 안전장치는 없는 것보다 나쁘기 때문이다.
"""

import numpy as np
import pytest

from firstlight.geo.dem import DEM, synthetic_dem
from firstlight.geo.frame import geodetic_to_enu
from firstlight.geo.pose import enu_from_camera, rays_to_enu
from firstlight.geo.raycast import raycast_to_ground

LAT0, LON0 = 36.4127, 128.7043


def flat_dem(elevation_m: float = 0.0, half_width_deg: float = 0.4) -> DEM:
    """일정 표고의 평면 지형. 해석해와 대조하기 위한 것."""
    n = 400
    grid = np.full((n, n), elevation_m, dtype=np.float32)
    return DEM(
        grid=grid,
        lon0=LON0 - half_width_deg,
        lat0=LAT0 + half_width_deg,
        dlon=2 * half_width_deg / n,
        dlat=-2 * half_width_deg / n,
    )


def ray_from_angles(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """카메라 광축 방향 ENU 광선 (픽셀 중심)."""
    R = enu_from_camera(yaw_deg, pitch_deg, 0.0)
    return rays_to_enu(np.array([[0.0, 0.0, 1.0]]), R)


# --------------------------------------------------------------- 해석해 대조


@pytest.mark.parametrize("pitch_deg", [-90.0, -60.0, -30.0, -15.0])
def test_flat_ground_range_matches_analytic(pitch_deg):
    """평면에서 사거리는 height / sin(부각) 이어야 한다."""
    height = 300.0
    hits = raycast_to_ground(LAT0, LON0, height, ray_from_angles(0.0, pitch_deg),
                             flat_dem(0.0), step_m=5.0)

    assert bool(hits.hit[0])
    expected = height / np.sin(np.radians(-pitch_deg))
    # 곡률 때문에 실제 사거리는 평면 해석해보다 아주 조금 짧다.
    assert hits.range_m[0] == pytest.approx(expected, rel=2e-3)


def test_nadir_lands_directly_below():
    """수직 하방 광선은 드론 바로 아래에 떨어져야 한다."""
    hits = raycast_to_ground(LAT0, LON0, 300.0, ray_from_angles(0.0, -90.0),
                             flat_dem(0.0), step_m=5.0)
    assert hits.lat[0] == pytest.approx(LAT0, abs=1e-9)
    assert hits.lon[0] == pytest.approx(LON0, abs=1e-9)
    assert hits.range_m[0] == pytest.approx(300.0, rel=1e-4)


@pytest.mark.parametrize(
    "yaw_deg,expect",
    [
        (0.0, "north"),
        (90.0, "east"),
        (180.0, "south"),
        (270.0, "west"),
    ],
)
def test_hit_lands_in_heading_direction(yaw_deg, expect):
    """방위각대로 떨어져야 한다. 부호가 뒤집히면 여기서 잡힌다."""
    hits = raycast_to_ground(LAT0, LON0, 500.0, ray_from_angles(yaw_deg, -20.0),
                             flat_dem(0.0), step_m=10.0)
    assert bool(hits.hit[0])
    east, north, _ = geodetic_to_enu(hits.lon[0], hits.lat[0], 0.0, LAT0, LON0, 0.0)

    if expect == "north":
        assert north > 100 and abs(east) < 1.0
    elif expect == "south":
        assert north < -100 and abs(east) < 1.0
    elif expect == "east":
        assert east > 100 and abs(north) < 1.0
    else:
        assert east < -100 and abs(north) < 1.0


def test_elevation_of_hit_matches_terrain():
    """교차점의 고도는 그 지점 표고와 같아야 한다."""
    dem = flat_dem(420.0)
    hits = raycast_to_ground(LAT0, LON0, 900.0, ray_from_angles(35.0, -25.0), dem, step_m=5.0)
    assert bool(hits.hit[0])
    assert hits.elevation_m[0] == pytest.approx(420.0, abs=0.05)


def test_rising_terrain_shortens_range():
    """더 높은 지형은 더 가까이서 만나야 한다."""
    low = raycast_to_ground(LAT0, LON0, 1000.0, ray_from_angles(0.0, -20.0),
                            flat_dem(0.0), step_m=5.0)
    high = raycast_to_ground(LAT0, LON0, 1000.0, ray_from_angles(0.0, -20.0),
                             flat_dem(600.0), step_m=5.0)
    assert high.range_m[0] < low.range_m[0]


# ------------------------------------------------------------------- 거절 경로


def test_upward_ray_never_intersects():
    """위를 향한 광선은 미교차여야 한다 (지평선 위)."""
    hits = raycast_to_ground(LAT0, LON0, 300.0, ray_from_angles(0.0, +10.0),
                             flat_dem(0.0), step_m=20.0)
    assert not bool(hits.hit[0])
    assert not bool(hits.out_of_dem[0])
    assert np.isnan(hits.lat[0])


def test_horizontal_ray_beyond_max_range_is_no_hit():
    """수평 광선은 최대거리 안에서 평지와 만나지 못한다."""
    hits = raycast_to_ground(LAT0, LON0, 300.0, ray_from_angles(0.0, 0.0),
                             flat_dem(0.0), max_range_m=15000.0, step_m=20.0)
    assert not bool(hits.hit[0])


def test_leaving_dem_is_flagged():
    """DEM 범위를 벗어나면 교차 실패로 두지 말고 사유를 구분해야 한다."""
    tiny = flat_dem(0.0, half_width_deg=0.002)      # 약 200m 반경
    hits = raycast_to_ground(LAT0, LON0, 2000.0, ray_from_angles(0.0, -5.0), tiny, step_m=20.0)
    assert not bool(hits.hit[0])
    assert bool(hits.out_of_dem[0])


def test_origin_below_terrain_is_flagged():
    """드론이 지형 아래면(고도·기준면 오류) 좌표를 내면 안 된다."""
    hits = raycast_to_ground(LAT0, LON0, 100.0, ray_from_angles(0.0, -30.0),
                             flat_dem(800.0), step_m=20.0)
    assert not bool(hits.hit[0])
    assert bool(hits.out_of_dem[0])


# ---------------------------------------------------------------- 벡터화·수렴


def test_batched_rays_match_individual():
    """N개 동시 처리 결과가 개별 처리와 같아야 한다 (몬테카를로 경로)."""
    dem = synthetic_dem((LON0 - 0.3, LAT0 - 0.3, LON0 + 0.3, LAT0 + 0.3), seed=3)
    yaws = np.array([0.0, 45.0, 120.0, 250.0])
    rays = np.vstack([ray_from_angles(y, -25.0) for y in yaws])

    batch = raycast_to_ground(LAT0, LON0, 1200.0, rays, dem, step_m=10.0)
    for i, yaw in enumerate(yaws):
        single = raycast_to_ground(LAT0, LON0, 1200.0, ray_from_angles(yaw, -25.0),
                                   dem, step_m=10.0)
        assert bool(batch.hit[i]) == bool(single.hit[0])
        if bool(single.hit[0]):
            assert batch.lat[i] == pytest.approx(single.lat[0], abs=1e-9)
            assert batch.lon[i] == pytest.approx(single.lon[0], abs=1e-9)


def test_origin_offsets_shift_the_hit():
    """원점 오프셋이 실제로 교차점을 움직여야 한다 (몬테카를로가 이걸 쓴다)."""
    dem = flat_dem(0.0)
    ray = ray_from_angles(0.0, -45.0)
    base = raycast_to_ground(LAT0, LON0, 500.0, ray, dem, step_m=5.0)

    rays2 = np.vstack([ray, ray])
    off = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])   # 두 번째만 동쪽 100m
    shifted = raycast_to_ground(LAT0, LON0, 500.0, rays2, dem, step_m=5.0,
                                origin_offsets_enu=off)

    assert shifted.lat[0] == pytest.approx(base.lat[0], abs=1e-9)
    east0, _, _ = geodetic_to_enu(shifted.lon[0], shifted.lat[0], 0.0, LAT0, LON0, 0.0)
    east1, _, _ = geodetic_to_enu(shifted.lon[1], shifted.lat[1], 0.0, LAT0, LON0, 0.0)
    assert float(east1 - east0) == pytest.approx(100.0, abs=0.5)


def test_bisection_converges_regardless_of_step():
    """행진 간격을 바꿔도 수렴한 교차점은 같아야 한다."""
    dem = synthetic_dem((LON0 - 0.3, LAT0 - 0.3, LON0 + 0.3, LAT0 + 0.3), seed=11)
    ray = ray_from_angles(20.0, -35.0)
    coarse = raycast_to_ground(LAT0, LON0, 1500.0, ray, dem, step_m=40.0)
    fine = raycast_to_ground(LAT0, LON0, 1500.0, ray, dem, step_m=5.0)
    assert bool(coarse.hit[0]) and bool(fine.hit[0])
    assert coarse.range_m[0] == pytest.approx(fine.range_m[0], abs=1.0)
