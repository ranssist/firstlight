"""광선-지형 교차 — 카메라 광선이 지표와 만나는 지점을 찾는다.

전략: 조밀 행진으로 부호변화 구간을 잡고 그 안에서 이분탐색.
전체가 벡터화되어 있어 N개 광선을 한 번에 처리한다. 몬테카를로 오차전파가
같은 함수를 200개 광선으로 한 번 호출하는 구조를 쓰기 때문에 이게 중요하다.

이 모듈에서 가장 중요한 부분은 교차 계산이 아니라 **거절 로직**이다.
좌표를 못 내는 상황에서 억지로 숫자를 내면 그 숫자가 진화대를 엉뚱한 곳으로
보낸다. 못 내면 못 낸다고 해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from firstlight.geo.dem import DEM
from firstlight.geo.frame import enu_to_geodetic


class RejectReason(str, Enum):
    """좌표를 발행하지 않는 사유."""

    NO_INTERSECTION = "no_intersection"      # 최대거리 내 지형과 만나지 않음 (지평선 위)
    GRAZING = "grazing"                      # 부각이 너무 낮아 오차가 폭발함
    OUT_OF_DEM = "out_of_dem"                # DEM 범위를 벗어남
    ORIGIN_BELOW_TERRAIN = "origin_below"     # 드론이 지형면 아래 (고도/기준면 오류)
    CEP_TOO_LARGE = "cep_too_large"          # 추정 오차반경이 상한 초과


@dataclass
class RaycastHits:
    """N개 광선의 교차 결과 (배열 단위)."""

    lat: np.ndarray          # (N,) 교차점 위도, 실패 시 NaN
    lon: np.ndarray          # (N,) 교차점 경도, 실패 시 NaN
    elevation_m: np.ndarray  # (N,) 교차점 표고
    range_m: np.ndarray      # (N,) 카메라로부터의 사거리
    hit: np.ndarray          # (N,) bool
    out_of_dem: np.ndarray   # (N,) bool — 교차 전에 DEM 을 벗어났는지

    def __len__(self) -> int:
        return int(self.hit.shape[0])


def raycast_to_ground(
    origin_lat: float,
    origin_lon: float,
    origin_alt_msl: float,
    rays_enu: np.ndarray,
    dem: DEM,
    max_range_m: float = 15000.0,
    step_m: float = 20.0,
    refine_iters: int = 20,
    origin_offsets_enu: np.ndarray | None = None,
) -> RaycastHits:
    """ENU 광선들을 지형면과 교차시킨다.

    Args:
        origin_*: 공칭 카메라 위치. 고도는 DEM 과 **같은 수직 기준면**이어야
            한다 (dem.geoid_offset_m 설명 참조).
        rays_enu: (N, 3) 단위 광선.
        max_range_m: 이 거리까지 못 만나면 미교차 처리.
        step_m: 행진 간격. 작을수록 얇은 능선을 놓칠 확률이 준다.
        refine_iters: 이분탐색 반복. 20회면 20m 구간이 밀리미터로 수렴한다.
        origin_offsets_enu: (N, 3) 광선별 원점 오프셋(m). 몬테카를로에서
            GPS 오차·수직기준면 편차를 넣는 통로다. None 이면 전부 공칭 위치.
    """
    rays = np.atleast_2d(np.asarray(rays_enu, dtype=float))
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    n_rays = rays.shape[0]

    if origin_offsets_enu is None:
        off = np.zeros((n_rays, 3), dtype=float)
    else:
        off = np.broadcast_to(
            np.atleast_2d(np.asarray(origin_offsets_enu, dtype=float)), (n_rays, 3)
        )

    n_steps = max(2, int(np.ceil(max_range_m / step_m)) + 1)
    t = np.linspace(0.0, max_range_m, n_steps)                    # (M,)

    # (N, M) 표본의 ENU 오프셋. 원점은 공칭 카메라 위치 + off.
    x = off[:, 0:1] + rays[:, 0:1] * t[None, :]
    y = off[:, 1:2] + rays[:, 1:2] * t[None, :]
    u = off[:, 2:3] + rays[:, 2:3] * t[None, :]

    # 폐형식 변환이므로 지구 곡률이 여기서 자동 반영된다. 평면 근사로
    # z 를 따로 계산하면 15km 에서 약 18m 를 놓친다 (geo/frame.py 참조).
    lon, lat, z = enu_to_geodetic(x, y, u, origin_lat, origin_lon, origin_alt_msl)
    terrain = dem.elevation(lon, lat)                              # (N, M), 범위 밖은 NaN

    # g > 0 이면 광선이 지형 위, g <= 0 이면 지형 아래/관통.
    gap = z - terrain

    valid = np.isfinite(gap)
    below = valid & (gap <= 0.0)

    # 최초 관통 인덱스. 없으면 -1.
    any_below = below.any(axis=1)
    first_below = np.where(any_below, below.argmax(axis=1), -1)

    # DEM 이탈 판정: 관통 지점보다 앞서 NaN 이 나왔다면 결과를 신뢰할 수 없다.
    invalid_idx = np.where(~valid, np.arange(n_steps)[None, :], n_steps)
    first_invalid = invalid_idx.min(axis=1)
    out_of_dem = first_invalid < np.where(any_below, first_below, n_steps)

    # 원점이 이미 지형 아래인 경우(고도 오류/기준면 불일치)는 별도로 표시.
    origin_below = valid[:, 0] & (gap[:, 0] <= 0.0)

    hit = any_below & ~out_of_dem & ~origin_below

    # ---- 이분탐색으로 브래킷 구간을 좁힌다 -------------------------------
    idx = np.clip(first_below, 1, n_steps - 1)
    t_lo = t[idx - 1].copy()
    t_hi = t[idx].copy()

    for _ in range(refine_iters):
        t_mid = 0.5 * (t_lo + t_hi)
        xm = off[:, 0] + rays[:, 0] * t_mid
        ym = off[:, 1] + rays[:, 1] * t_mid
        um = off[:, 2] + rays[:, 2] * t_mid
        lon_m, lat_m, zm = enu_to_geodetic(xm, ym, um, origin_lat, origin_lon, origin_alt_msl)
        gm = zm - dem.elevation(lon_m, lat_m)
        # gm > 0 (아직 지형 위) 이면 하한을 올리고, 아니면 상한을 내린다.
        still_above = np.isfinite(gm) & (gm > 0.0)
        t_lo = np.where(still_above, t_mid, t_lo)
        t_hi = np.where(still_above, t_hi, t_mid)

    t_hit = 0.5 * (t_lo + t_hi)
    x_hit = off[:, 0] + rays[:, 0] * t_hit
    y_hit = off[:, 1] + rays[:, 1] * t_hit
    u_hit = off[:, 2] + rays[:, 2] * t_hit
    lon_hit, lat_hit, z_hit = enu_to_geodetic(
        x_hit, y_hit, u_hit, origin_lat, origin_lon, origin_alt_msl
    )

    nan = np.full(n_rays, np.nan)
    return RaycastHits(
        lat=np.where(hit, lat_hit, nan),
        lon=np.where(hit, lon_hit, nan),
        elevation_m=np.where(hit, z_hit, nan),
        range_m=np.where(hit, t_hit, nan),
        hit=hit,
        out_of_dem=out_of_dem | origin_below,
    )


@dataclass
class GeoFix:
    """단일 탐지에 대한 지오레퍼런싱 결과.

    `rejected` 가 None 이 아니면 좌표 필드는 의미가 없다. 호출부는
    반드시 이 필드를 먼저 확인해야 한다 — 그러라고 좌표를 NaN 으로 둔다.
    """

    lat: float
    lon: float
    elevation_m: float
    range_m: float
    depression_deg: float
    rejected: RejectReason | None = None
    cep50_m: float | None = None
    cep90_m: float | None = None
    # 풍향 보정 전 좌표. 보정을 적용했을 때만 채워진다.
    lat_uncorrected: float | None = None
    lon_uncorrected: float | None = None

    @property
    def ok(self) -> bool:
        return self.rejected is None

    @classmethod
    def reject(cls, reason: RejectReason, depression_deg: float = float("nan")) -> GeoFix:
        return cls(
            lat=float("nan"),
            lon=float("nan"),
            elevation_m=float("nan"),
            range_m=float("nan"),
            depression_deg=depression_deg,
            rejected=reason,
        )
