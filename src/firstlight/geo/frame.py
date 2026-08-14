"""국소 ENU 평면 ↔ 측지좌표 변환 (WGS84, 폐형식).

**근사하지 않는다.** 초안에서는 위도별 곡률반경을 쓴 해석 근사를 썼는데,
`tests/test_frame.py` 가 동서 15km 에서 13m 오차를 잡아냈다. CEP 를 수십 m
단위로 주장하는 파이프라인에서 좌표변환이 십수 m 를 흘리면 그 아래 모든
숫자가 무의미해진다.

지금은 ENU → ECEF → 측지 를 폐형식으로 계산한다 (역변환은 Bowring 법).
전부 numpy 벡터연산이라 근사식보다 눈에 띄게 느리지도 않다.

지구 곡률이 여기서 자동으로 처리된다:
    ENU 의 up 축은 원점에서의 **접평면** 기준이다. 15km 떨어진 지점은
    타원체면이 접평면보다 약 18m 아래로 내려간다 (d²/2R). 직선 광선을
    ENU 에서 행진시키고 z 를 그대로 표고와 비교하면 이 18m 를 통째로
    놓친다. 부각이 낮은 광선에서 18m 는 지상거리 수백 m 다.
    폐형식 변환은 각 표본점의 진짜 측지고를 주므로 이 오차가 원천적으로
    생기지 않는다.

모델링하지 않은 것 (알려진 한계):
    - **대기굴절.** 수평에 가까운 장거리 광선을 지구 곡률의 약 1/7 만큼
      되휘게 한다(부호는 반대). 15km 부각 3도에서 수 m 급. 부각 8도 이상만
      좌표를 발행하는 현재 정책에서는 CEP 대비 작지만, 장거리 운용으로
      확장하면 넣어야 한다.
    - **지오이드 경사.** 고도를 정표고(MSL)로 다루고 지오이드가 AOI 안에서
      타원체와 국소 평행하다고 본다. 한반도에서 15km 당 지오이드고 변화는
      1m 미만이라 CEP 대비 무시할 수 있다.
"""

from __future__ import annotations

import numpy as np

# WGS84
_A = 6378137.0
_F = 1.0 / 298.257223563
_B = _A * (1.0 - _F)
_E2 = _F * (2.0 - _F)                       # 제1이심률 제곱
_EP2 = (_A * _A - _B * _B) / (_B * _B)      # 제2이심률 제곱
_DEG = np.pi / 180.0


def _origin_ecef(lat0: float, lon0: float, h0: float):
    """원점의 ECEF 좌표와 ENU 회전에 필요한 삼각함수."""
    phi = lat0 * _DEG
    lam = lon0 * _DEG
    sp, cp = np.sin(phi), np.cos(phi)
    sl, cl = np.sin(lam), np.cos(lam)
    n0 = _A / np.sqrt(1.0 - _E2 * sp * sp)
    x0 = (n0 + h0) * cp * cl
    y0 = (n0 + h0) * cp * sl
    z0 = (n0 * (1.0 - _E2) + h0) * sp
    return (x0, y0, z0), (sp, cp, sl, cl)


def enu_to_geodetic(east, north, up, lat0: float, lon0: float, h0: float):
    """국소 ENU 오프셋(m) → (경도, 위도, 고도). 전부 벡터화.

    고도는 원점 고도 `h0` 와 같은 수직 기준면으로 나온다 (모듈 docstring 참조).
    """
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)
    up = np.asarray(up, dtype=float)

    (x0, y0, z0), (sp, cp, sl, cl) = _origin_ecef(lat0, lon0, h0)

    # ENU → ECEF 회전 (원점 접평면 기준).
    x = x0 - sl * east - sp * cl * north + cp * cl * up
    y = y0 + cl * east - sp * sl * north + cp * sl * up
    z = z0 + cp * north + sp * up

    return _ecef_to_geodetic(x, y, z)


def _ecef_to_geodetic(x, y, z):
    """ECEF → (경도, 위도, 고도). Bowring 폐형식 — 지표 부근에서 서브밀리미터."""
    p = np.hypot(x, y)
    theta = np.arctan2(z * _A, p * _B)
    st, ct = np.sin(theta), np.cos(theta)

    lat = np.arctan2(z + _EP2 * _B * st**3, p - _E2 * _A * ct**3)
    lon = np.arctan2(y, x)

    slat = np.sin(lat)
    n = _A / np.sqrt(1.0 - _E2 * slat * slat)
    # 극 근처에서 cos(lat)→0 이면 이 식이 불안정하지만, 운용 위도(33~39N)에서는
    # 문제되지 않는다.
    h = p / np.cos(lat) - n

    return lon / _DEG, lat / _DEG, h


def geodetic_to_enu(lon, lat, h, lat0: float, lon0: float, h0: float):
    """(경도, 위도, 고도) → 국소 ENU 오프셋(m). `enu_to_geodetic` 의 역변환."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    h = np.asarray(h, dtype=float)

    phi = lat * _DEG
    lam = lon * _DEG
    sp_t, cp_t = np.sin(phi), np.cos(phi)
    n = _A / np.sqrt(1.0 - _E2 * sp_t * sp_t)
    x = (n + h) * cp_t * np.cos(lam)
    y = (n + h) * cp_t * np.sin(lam)
    z = (n * (1.0 - _E2) + h) * sp_t

    (x0, y0, z0), (sp, cp, sl, cl) = _origin_ecef(lat0, lon0, h0)
    dx, dy, dz = x - x0, y - y0, z - z0

    east = -sl * dx + cl * dy
    north = -sp * cl * dx - sp * sl * dy + cp * dz
    up = cp * cl * dx + cp * sl * dy + sp * dz
    return east, north, up


# --------------------------------------------------------------------------
# 수평 전용 편의 함수. 고도가 상관없는 곳(풍향 보정, 산포 반경 계산)에서 쓴다.
# 접평면 up=0 을 가정하므로 장거리에서는 곡률만큼의 고도차가 생기지만,
# 수평 성분에는 영향이 없다.
# --------------------------------------------------------------------------


def enu_to_lonlat(east, north, lat0: float, lon0: float):
    """수평 ENU 오프셋 → (경도, 위도)."""
    lon, lat, _ = enu_to_geodetic(east, north, np.zeros_like(np.asarray(east, dtype=float)),
                                  lat0, lon0, 0.0)
    return lon, lat


def lonlat_to_enu(lon, lat, lat0: float, lon0: float):
    """(경도, 위도) → 수평 ENU 오프셋."""
    east, north, _ = geodetic_to_enu(lon, lat, np.zeros_like(np.asarray(lat, dtype=float)),
                                     lat0, lon0, 0.0)
    return east, north
