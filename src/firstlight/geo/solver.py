"""지오레퍼런싱 최상위 — 탐지 픽셀 하나를 신고 가능한 좌표로 바꾼다.

파이프라인:
    픽셀 → 카메라광선 → ENU광선 → [부각 검사] → 지형교차
         → [몬테카를로 CEP] → [CEP 상한 검사] → 풍향보정 → GeoFix

대괄호가 거절 지점이다. 거절 로직이 이 모듈의 본체이며, 교차 계산 자체보다
중요하다. 좌표를 못 내는 상황에서 억지로 숫자를 만들면 그 숫자가 진화대를
엉뚱한 능선으로 보낸다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from firstlight.geo.camera import CameraIntrinsics
from firstlight.geo.dem import DEM
from firstlight.geo.frame import geodetic_to_enu
from firstlight.geo.pose import CameraPose, depression_deg, enu_from_camera, rays_to_enu
from firstlight.geo.raycast import GeoFix, RejectReason, raycast_to_ground
from firstlight.geo.uncertainty import NoiseModel, monte_carlo_cep
from firstlight.geo.wind import Wind, correct_for_wind


def project_ground_point(
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    lat: float,
    lon: float,
    elevation_m: float,
) -> tuple[float, float] | None:
    """지상 좌표 → 픽셀. `GeoSolver.solve` 의 역방향.

    폐루프 자체검증이 이 함수를 쓴다: 알려진 지상점을 투영해 픽셀을 얻고,
    그 픽셀을 solve 에 넣어 원래 좌표가 복원되는지 본다. 정답을 아는 검증이라
    드론도 영상도 필요 없다.

    카메라 뒤이거나 화각 밖이면 None.

    수직 성분을 `elevation_m - alt_msl` 로 직접 빼면 안 된다. 그건 지구
    곡률을 무시하는 평면 근사이고, 4km 거리에서 1.3m 의 고도 오차를 만든다.
    부각 15도에서 그 1.3m 는 지상거리 약 5m 가 된다 — 폐루프 테스트가
    정확히 이 값으로 실패했었다. `geodetic_to_enu` 가 세 성분을 함께
    정확히 계산한다.
    """
    east, north, up = geodetic_to_enu(
        lon, lat, elevation_m, pose.lat, pose.lon, pose.alt_msl
    )
    vec_enu = np.array([float(east), float(north), float(up)])

    R = enu_from_camera(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    ray_cam = R.T @ vec_enu          # 회전행렬의 역 = 전치
    if ray_cam[2] <= 0.0:
        return None                  # 광축 뒤

    uv = intrinsics.ray_to_pixel(ray_cam[None, :])
    if not bool(intrinsics.contains(uv)[0]):
        return None                  # 화각 밖
    return float(uv[0, 0]), float(uv[0, 1])


@dataclass
class GeoSolverConfig:
    """거절 임계값과 수치 파라미터.

    두 겹의 게이트를 둔다. `min_depression_deg` 는 값싼 사전 차단이고,
    `max_cep90_m` 가 실제 판정이다.

    실측 근거 (의성 실제 DEM, 고도 300m AGL, 200회 몬테카를로,
    `firstlight geo-selftest --site uiseong` 재현 가능):

        부각    사거리    CEP50 / CEP90 (일반 GNSS)    (RTK)
         5도    3,108m      169 / 744 m               51 / 295 m
         8도    2,145m       92 / 239 m               25 /  58 m
        15도    1,213m       47 / 102 m               15 /  34 m
        20도      865m       34 /  69 m               12 /  27 m
        45도      399m        9 /  19 m                4 /   7 m
        90도      300m        3 /   6 m                1 /   2 m

    읽는 법:
        - **거리가 아니라 부각이 지배한다.** 사거리가 10배 늘 때 CEP 는
          50배 넘게 는다. 수평에 가까운 광선에서 자세 오차가 지상거리로
          증폭되기 때문이다.
        - 소개서 §5 의 "50m 이내(CEP)" 는 일반 GNSS 기준 **부각 15도
          이상**에서 성립한다. 무조건 성립하는 수치가 아니다.
        - 기본 8도는 사전 차단용으로만 의미가 있다. 일반 GNSS 로 8도면
          CEP90 이 239m 라 어차피 CEP 게이트에서 걸린다(발행률 12%).
          RTK 를 도입하면 8도가 실질적인 하한이 된다.
    """

    max_range_m: float = 15000.0
    step_m: float = 20.0
    min_depression_deg: float = 8.0
    max_cep90_m: float = 150.0
    mc_trials: int = 200
    wind_drift_seconds: float = 0.0
    # "centre" = 작품설명서가 지정한 중심 픽셀, "base" = 박스 하단(연기 기저).
    bbox_anchor: str = "centre"
    noise: NoiseModel = field(default_factory=NoiseModel)


class GeoSolver:
    """DEM 과 카메라 제원을 물고 있는 재사용 가능한 해석기."""

    def __init__(
        self,
        dem: DEM,
        intrinsics: CameraIntrinsics,
        config: GeoSolverConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.dem = dem
        self.intrinsics = intrinsics
        self.config = config or GeoSolverConfig()
        self.rng = rng or np.random.default_rng(0)

    # ------------------------------------------------------------------

    def solve(
        self,
        pose: CameraPose,
        u: float,
        v: float,
        wind: Wind | None = None,
        with_uncertainty: bool = True,
    ) -> GeoFix:
        """탐지 픽셀 (u, v) 를 지상 좌표로 해석한다."""
        cfg = self.config

        ray_cam = self.intrinsics.pixel_to_ray(u, v)
        R = enu_from_camera(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
        ray_enu = rays_to_enu(ray_cam, R)
        depr = float(depression_deg(ray_enu)[0])

        # 거절 ①: 스침각. 교차를 계산하기 전에 잘라낸다.
        if not np.isfinite(depr) or depr < cfg.min_depression_deg:
            return GeoFix.reject(RejectReason.GRAZING, depression_deg=depr)

        hits = raycast_to_ground(
            pose.lat,
            pose.lon,
            pose.alt_msl,
            ray_enu,
            self.dem,
            max_range_m=cfg.max_range_m,
            step_m=cfg.step_m,
        )

        # 거절 ②: 교차 실패.
        if not bool(hits.hit[0]):
            reason = (
                RejectReason.OUT_OF_DEM
                if bool(hits.out_of_dem[0])
                else RejectReason.NO_INTERSECTION
            )
            return GeoFix.reject(reason, depression_deg=depr)

        lat = float(hits.lat[0])
        lon = float(hits.lon[0])
        fix = GeoFix(
            lat=lat,
            lon=lon,
            elevation_m=float(hits.elevation_m[0]),
            range_m=float(hits.range_m[0]),
            depression_deg=depr,
        )

        if with_uncertainty:
            mc = monte_carlo_cep(
                self.intrinsics,
                pose,
                u,
                v,
                self.dem,
                nominal_lat=lat,
                nominal_lon=lon,
                noise=cfg.noise,
                trials=cfg.mc_trials,
                rng=self.rng,
                max_range_m=cfg.max_range_m,
                step_m=cfg.step_m,
            )
            fix.cep50_m = mc.cep50_m
            fix.cep90_m = mc.cep90_m

            # 거절 ③: 오차반경이 상한을 넘으면 좌표를 발행하지 않는다.
            if not np.isfinite(mc.cep90_m) or mc.cep90_m > cfg.max_cep90_m:
                rejected = GeoFix.reject(RejectReason.CEP_TOO_LARGE, depression_deg=depr)
                rejected.cep50_m = mc.cep50_m
                rejected.cep90_m = mc.cep90_m
                return rejected

        # 풍향 보정. 보정 전 좌표를 반드시 함께 남긴다.
        if wind is not None and cfg.wind_drift_seconds > 0.0:
            lat_c, lon_c, moved = correct_for_wind(
                lat, lon, wind, drift_seconds=cfg.wind_drift_seconds
            )
            if moved > 0.0:
                fix.lat_uncorrected = lat
                fix.lon_uncorrected = lon
                fix.lat = lat_c
                fix.lon = lon_c

        return fix

    # ------------------------------------------------------------------

    def solve_bbox(
        self,
        pose: CameraPose,
        bbox: tuple[float, float, float, float],
        wind: Wind | None = None,
        with_uncertainty: bool = True,
        anchor: str | None = None,
    ) -> GeoFix:
        """탐지 박스 (x1, y1, x2, y2) 를 해석한다.

        기본 기준점은 **중심 픽셀**이다 — 작품설명서 Ⅱ-1 이 "탐지된 연기
        영역의 중심 픽셀 (u, v)를 좌표로 변환"이라고 명시한다.

        `anchor="base"` 를 주면 박스 **하단 중앙**을 쓴다. 연기 기저가 발화점
        바로 위라 표류가 가장 작은 지점이고, 중심을 쓰면 연기 기둥 높이만큼
        풍하 방향으로 밀린 좌표가 나온다. 명세는 중심을 쓰는 대신 그 편차를
        풍향 보정으로 처리하도록 설계돼 있으므로, 기본값은 명세를 따르고
        기저 방식은 현장 비교용으로 남겨 둔다.
        """
        anchor = anchor or self.config.bbox_anchor
        x1, y1, x2, y2 = bbox
        u = 0.5 * (x1 + x2)
        v = max(y1, y2) if anchor == "base" else 0.5 * (y1 + y2)
        return self.solve(pose, u=u, v=v, wind=wind,
                          with_uncertainty=with_uncertainty)
