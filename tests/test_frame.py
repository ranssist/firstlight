"""ENU ↔ 측지 변환을 PROJ 의 독립 구현과 대조한다.

기준으로 PROJ 의 `+proj=topocentric` 파이프라인을 쓴다. 우리 코드와 완전히
별개의 구현이므로, 양쪽이 밀리미터 단위로 맞으면 규약과 수식이 모두 맞다는
뜻이다. 스스로 만든 근사를 스스로 검증하는 자기참조를 피하려는 것이다.

이 파일의 초기 버전은 해석 근사식의 13m 오차를 잡아냈다. 그래서 남겨둔다.
"""

import numpy as np
import pytest
from pyproj import Transformer

from firstlight.geo.frame import (
    enu_to_geodetic,
    enu_to_lonlat,
    geodetic_to_enu,
    lonlat_to_enu,
)

# 의성 일대 — 소개서 시나리오 기준점.
LAT0, LON0, H0 = 36.4127, 128.7043, 620.0


def _proj_reference(lat0: float, lon0: float, h0: float) -> Transformer:
    """측지(경도, 위도, 고도) → 국소 ENU. PROJ 의 독립 구현."""
    pipeline = (
        "+proj=pipeline "
        "+step +proj=unitconvert +xy_in=deg +xy_out=rad "
        "+step +proj=cart +ellps=WGS84 "
        f"+step +proj=topocentric +ellps=WGS84 +lat_0={lat0} +lon_0={lon0} +h_0={h0}"
    )
    return Transformer.from_pipeline(pipeline)


REF = _proj_reference(LAT0, LON0, H0)


@pytest.mark.parametrize("distance_m", [10.0, 500.0, 2_000.0, 8_000.0, 15_000.0])
@pytest.mark.parametrize("azimuth_deg", [0.0, 45.0, 90.0, 173.0, 250.0, 318.0])
def test_enu_to_geodetic_matches_proj(distance_m, azimuth_deg):
    """우리 순변환 결과를 PROJ 로 되돌리면 원래 ENU 가 나와야 한다."""
    east = distance_m * np.sin(np.radians(azimuth_deg))
    north = distance_m * np.cos(np.radians(azimuth_deg))
    up = 0.0

    lon, lat, h = enu_to_geodetic(east, north, up, LAT0, LON0, H0)
    e_ref, n_ref, u_ref = REF.transform(float(lon), float(lat), float(h))

    assert abs(e_ref - east) < 1e-3, f"east 오차 {e_ref - east:.6f}m"
    assert abs(n_ref - north) < 1e-3, f"north 오차 {n_ref - north:.6f}m"
    assert abs(u_ref - up) < 1e-3, f"up 오차 {u_ref - up:.6f}m"


@pytest.mark.parametrize("lat0", [33.2, 36.4, 38.6])
def test_accuracy_holds_across_korean_latitudes(lat0):
    """제주에서 휴전선까지 전 위도에서 성립해야 한다."""
    ref = _proj_reference(lat0, LON0, H0)
    for east, north, up in [
        (12_000.0, 0.0, 0.0),
        (0.0, 12_000.0, 0.0),
        (-9_000.0, 9_000.0, -400.0),
    ]:
        lon, lat, h = enu_to_geodetic(east, north, up, lat0, LON0, H0)
        e_ref, n_ref, u_ref = ref.transform(float(lon), float(lat), float(h))
        assert abs(e_ref - east) < 1e-3
        assert abs(n_ref - north) < 1e-3
        assert abs(u_ref - up) < 1e-3


CURVATURE_15KM_M = 15_000.0**2 / (2 * 6_371_000.0)   # d^2 / 2R ≈ 17.7m


def test_tangent_plane_rises_above_ellipsoid():
    """접평면 위(up=0)를 15km 가면 측지고가 약 18m **높아진다**.

    접평면은 평평한데 타원체면이 아래로 휘기 때문이다. 초안의 평면 근사는
    이 항 자체가 없어서 표고 비교가 그만큼 어긋났다.
    """
    _, _, h = enu_to_geodetic(0.0, 15_000.0, 0.0, LAT0, LON0, H0)
    rise = float(h) - H0
    assert 0.8 * CURVATURE_15KM_M < rise < 1.2 * CURVATURE_15KM_M, f"융기 {rise:.2f}m"


def test_constant_elevation_terrain_falls_below_tangent_plane():
    """같은 표고의 지형은 멀어질수록 접평면 아래로 내려간다 (운용상의 표현).

    광선-지형 교차가 실제로 겪는 형태다. 15km 밖의 표고 620m 지점은
    드론 접평면 기준으로 약 18m 아래에 있다.
    """
    _, _, up = geodetic_to_enu(*enu_to_geodetic(0.0, 15_000.0, 0.0, LAT0, LON0, H0)[:2],
                               H0, LAT0, LON0, H0)
    assert -1.2 * CURVATURE_15KM_M < float(up) < -0.8 * CURVATURE_15KM_M, f"up={float(up):.2f}m"


@pytest.mark.parametrize("up", [0.0, -300.0, 250.0])
def test_roundtrip_is_exact(up):
    """enu → 측지 → enu 가 밀리미터 이내로 제자리."""
    east = np.array([0.0, 1_000.0, -7_500.0, 14_000.0])
    north = np.array([0.0, -3_000.0, 11_000.0, -14_000.0])
    up_arr = np.full_like(east, up)

    lon, lat, h = enu_to_geodetic(east, north, up_arr, LAT0, LON0, H0)
    e_back, n_back, u_back = geodetic_to_enu(lon, lat, h, LAT0, LON0, H0)

    assert np.allclose(east, e_back, atol=1e-6)
    assert np.allclose(north, n_back, atol=1e-6)
    assert np.allclose(up_arr, u_back, atol=1e-6)


def test_origin_maps_to_itself():
    lon, lat, h = enu_to_geodetic(0.0, 0.0, 0.0, LAT0, LON0, H0)
    assert np.isclose(float(lon), LON0, atol=1e-11)
    assert np.isclose(float(lat), LAT0, atol=1e-11)
    assert np.isclose(float(h), H0, atol=1e-6)


def test_axis_directions():
    """+east 는 경도를, +north 는 위도를 늘려야 한다."""
    lon_e, lat_e = enu_to_lonlat(1_000.0, 0.0, LAT0, LON0)
    assert float(lon_e) > LON0
    lon_n, lat_n = enu_to_lonlat(0.0, 1_000.0, LAT0, LON0)
    assert float(lat_n) > LAT0


def test_horizontal_helpers_roundtrip():
    east = np.array([0.0, 2_500.0, -6_000.0])
    north = np.array([0.0, -1_200.0, 9_000.0])
    lon, lat = enu_to_lonlat(east, north, LAT0, LON0)
    e_back, n_back = lonlat_to_enu(lon, lat, LAT0, LON0)
    assert np.allclose(east, e_back, atol=1e-6)
    assert np.allclose(north, n_back, atol=1e-6)


def test_vectorised_shape_preserved():
    east = np.zeros((4, 7))
    north = np.ones((4, 7))
    up = np.full((4, 7), -50.0)
    lon, lat, h = enu_to_geodetic(east, north, up, LAT0, LON0, H0)
    assert lon.shape == (4, 7) and lat.shape == (4, 7) and h.shape == (4, 7)
