"""기상청 격자 변환과 폴백 동작.

격자 변환은 기상청이 공개한 알려진 대응값으로 검증한다.
API 자체는 키가 필요하므로 테스트하지 않되, **키가 없을 때 파이프라인이
멈추지 않는지**는 반드시 확인한다.
"""

import pytest

from firstlight.geo.wind import Wind
from firstlight.weather import fetch_wind, latlon_to_grid


@pytest.mark.parametrize(
    "lat,lon,nx,ny,name",
    [
        (37.5665, 126.9780, 60, 127, "서울시청"),
        (35.1796, 129.0756, 98, 76, "부산시청"),
        (33.4996, 126.5312, 53, 38, "제주시청"),
        (38.0, 126.0, 43, 136, "투영 기준점"),
    ],
)
def test_grid_conversion_matches_known_values(lat, lon, nx, ny, name):
    """기상청이 공개한 지점별 격자값과 일치해야 한다.

    마지막 항목은 투영의 정의상 기준점(_OLAT, _OLON → _XO, _YO)이라
    반드시 맞아야 한다.
    """
    grid = latlon_to_grid(lat, lon)
    assert (grid.nx, grid.ny) == (nx, ny), f"{name}: {grid} != ({nx}, {ny})"


def test_grid_axes_increase_in_the_right_direction():
    """동쪽으로 가면 nx 가, 북쪽으로 가면 ny 가 커져야 한다."""
    base = latlon_to_grid(36.4127, 128.7043)
    assert latlon_to_grid(36.4127, 129.2).nx > base.nx
    assert latlon_to_grid(36.9, 128.7043).ny > base.ny


def test_uiseong_grid_is_plausible():
    """의성 — 격자 범위(1~149, 1~253) 안에 들어야 한다."""
    grid = latlon_to_grid(36.4127, 128.7043)
    assert 1 <= grid.nx <= 149 and 1 <= grid.ny <= 253


def test_missing_key_falls_back(monkeypatch, tmp_path):
    """서비스키가 없다고 파이프라인이 멈추면 안 된다."""
    monkeypatch.delenv("KMA_SERVICE_KEY", raising=False)
    monkeypatch.chdir(tmp_path)          # configs/secrets.yaml 이 안 보이도록

    fallback = Wind(speed_ms=6.2, from_deg=315.0)
    wind, source = fetch_wind(36.4127, 128.7043, fallback=fallback)
    assert source == "fallback"
    assert wind is fallback


def test_missing_key_without_fallback_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv("KMA_SERVICE_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    wind, source = fetch_wind(36.4127, 128.7043)
    assert wind is None and source == "unavailable"


def test_wind_convention_matches_kma():
    """기상청 VEC 는 '불어오는 방향' — Wind.from_deg 와 같은 규약이어야 한다.

    북서풍(315도)은 남동쪽으로 흐른다.
    """
    east, north = Wind(speed_ms=6.2, from_deg=315.0).toward_enu
    assert east > 0 and north < 0, "북서풍은 남동으로 흘러야 한다"
