"""지오레퍼런싱 정확도 — 폐루프 자체검증과 CEP 스윕.

소개서는 "발화 지점 위치 정확도 50m 이내(CEP)"라고 적었다. 이 모듈이
그 숫자가 어떤 조건에서 성립하고 어떤 조건에서 깨지는지를 표로 만든다.

두 가지를 잰다:

1. **왕복 정확도** — 알려진 지상점을 투영해 픽셀을 얻고, 그 픽셀을 다시
   풀어 원래 좌표가 나오는지. 노이즈가 없으므로 순수한 구현 오차만 남는다.
   여기서 몇 m 가 새면 아래 CEP 는 전부 의미가 없다.

2. **CEP 스윕** — 부각과 측위 등급을 바꿔가며 몬테카를로 오차반경을 잰다.
   결론은 언제나 같은 방향이다: **거리보다 부각이 지배한다.**
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from firstlight.geo.camera import CameraIntrinsics
from firstlight.geo.dem import DEM
from firstlight.geo.frame import geodetic_to_enu
from firstlight.geo.pose import CameraPose
from firstlight.geo.solver import GeoSolver, GeoSolverConfig, project_ground_point
from firstlight.geo.uncertainty import NoiseModel


@dataclass
class RoundTripResult:
    """무노이즈 왕복 오차 통계 (m)."""

    n_samples: int
    median_m: float
    p95_m: float
    max_m: float
    pixel_residual_px: float


@dataclass
class SweepRow:
    """부각 하나에 대한 집계 결과."""

    depression_deg: float
    n_solved: int
    n_attempted: int
    median_range_m: float
    cep50_m: float
    cep90_m: float
    publish_rate: float          # CEP 상한까지 통과해 좌표가 발행된 비율

    @property
    def solve_rate(self) -> float:
        return self.n_solved / self.n_attempted if self.n_attempted else 0.0


@dataclass
class GeoAccuracyReport:
    roundtrip: RoundTripResult
    sweeps: dict[str, list[SweepRow]] = field(default_factory=dict)
    altitude_agl_m: float = 300.0
    max_cep90_m: float = 150.0


def _ground_distance_m(lat_a, lon_a, lat_b, lon_b) -> float:
    east, north, _ = geodetic_to_enu(lon_b, lat_b, 0.0, lat_a, lon_a, 0.0)
    return float(np.hypot(east, north))


def _terrain_alt(dem: DEM, lat: float, lon: float) -> float:
    return float(dem.elevation(lon, lat))


# --------------------------------------------------------------------- 왕복


def measure_roundtrip(
    dem: DEM,
    intrinsics: CameraIntrinsics,
    site_lat: float,
    site_lon: float,
    altitude_agl_m: float = 300.0,
    azimuths: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
    depressions: tuple[float, ...] = (15.0, 25.0, 40.0, 60.0, 85.0),
    pixels: tuple[tuple[float, float], ...] = ((0.5, 0.5), (0.25, 0.7), (0.75, 0.35)),
) -> RoundTripResult:
    """지상점 → 픽셀 → 좌표 왕복 오차를 잰다 (노이즈 없음)."""
    cfg = GeoSolverConfig(noise=NoiseModel.perfect(), step_m=5.0, max_cep90_m=float("inf"))
    solver = GeoSolver(dem, intrinsics, cfg)
    alt = _terrain_alt(dem, site_lat, site_lon) + altitude_agl_m
    pose_kwargs = dict(lat=site_lat, lon=site_lon, alt_msl=alt)

    errors: list[float] = []
    pixel_residuals: list[float] = []

    for az in azimuths:
        for depr in depressions:
            pose = CameraPose(yaw_deg=az, pitch_deg=-depr, roll_deg=0.0, **pose_kwargs)
            for fu, fv in pixels:
                u = fu * intrinsics.width
                v = fv * intrinsics.height

                seed = solver.solve(pose, u, v, with_uncertainty=False)
                if not seed.ok:
                    continue

                uv = project_ground_point(
                    intrinsics, pose, seed.lat, seed.lon, seed.elevation_m
                )
                if uv is None:
                    continue
                pixel_residuals.append(float(np.hypot(uv[0] - u, uv[1] - v)))

                back = solver.solve(pose, uv[0], uv[1], with_uncertainty=False)
                if not back.ok:
                    continue
                errors.append(_ground_distance_m(seed.lat, seed.lon, back.lat, back.lon))

    if not errors:
        return RoundTripResult(0, float("nan"), float("nan"), float("nan"), float("nan"))

    arr = np.array(errors)
    return RoundTripResult(
        n_samples=len(errors),
        median_m=float(np.median(arr)),
        p95_m=float(np.percentile(arr, 95)),
        max_m=float(arr.max()),
        pixel_residual_px=float(np.max(pixel_residuals)) if pixel_residuals else float("nan"),
    )


# ----------------------------------------------------------------- CEP 스윕


def sweep_cep(
    dem: DEM,
    intrinsics: CameraIntrinsics,
    site_lat: float,
    site_lon: float,
    noise: NoiseModel,
    altitude_agl_m: float = 300.0,
    depressions: tuple[float, ...] = (5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0),
    azimuths: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
    trials: int = 200,
    max_cep90_m: float = 150.0,
    seed: int = 0,
) -> list[SweepRow]:
    """부각을 바꿔가며 CEP 를 잰다. 방위각 여러 개로 지형 다양성을 확보한다.

    `min_depression_deg` 게이트는 여기서 끈다 — 게이트가 왜 필요한지를
    보여주는 것이 목적이므로, 게이트에 걸릴 구간의 숫자도 봐야 한다.
    대신 CEP 상한 통과 여부를 `publish_rate` 로 따로 집계한다.
    """
    alt = _terrain_alt(dem, site_lat, site_lon) + altitude_agl_m
    rows: list[SweepRow] = []

    for depr in depressions:
        ceps50: list[float] = []
        ceps90: list[float] = []
        ranges: list[float] = []
        published = 0
        attempted = 0

        for i, az in enumerate(azimuths):
            attempted += 1
            cfg = GeoSolverConfig(
                noise=noise,
                mc_trials=trials,
                step_m=10.0,
                min_depression_deg=0.0,        # 게이트 해제: 나쁜 구간도 측정한다
                max_cep90_m=float("inf"),      # 거절 대신 값을 받아서 직접 판정
            )
            solver = GeoSolver(dem, intrinsics, cfg, rng=np.random.default_rng(seed + i))
            pose = CameraPose(site_lat, site_lon, alt, yaw_deg=az, pitch_deg=-depr)

            fix = solver.solve(pose, intrinsics.cx, intrinsics.cy)
            if not fix.ok or fix.cep90_m is None or not np.isfinite(fix.cep90_m):
                continue

            ceps50.append(fix.cep50_m)
            ceps90.append(fix.cep90_m)
            ranges.append(fix.range_m)
            if fix.cep90_m <= max_cep90_m:
                published += 1

        rows.append(
            SweepRow(
                depression_deg=depr,
                n_solved=len(ceps90),
                n_attempted=attempted,
                median_range_m=float(np.median(ranges)) if ranges else float("nan"),
                cep50_m=float(np.median(ceps50)) if ceps50 else float("nan"),
                cep90_m=float(np.median(ceps90)) if ceps90 else float("nan"),
                publish_rate=published / attempted if attempted else 0.0,
            )
        )

    return rows


def run_report(
    dem: DEM,
    intrinsics: CameraIntrinsics,
    site_lat: float,
    site_lon: float,
    altitude_agl_m: float = 300.0,
    trials: int = 200,
    max_cep90_m: float = 150.0,
) -> GeoAccuracyReport:
    """왕복 + 측위등급별 CEP 스윕을 한 번에 낸다."""
    report = GeoAccuracyReport(
        roundtrip=measure_roundtrip(
            dem, intrinsics, site_lat, site_lon, altitude_agl_m=altitude_agl_m
        ),
        altitude_agl_m=altitude_agl_m,
        max_cep90_m=max_cep90_m,
    )
    for label, noise in (
        ("일반 GNSS", NoiseModel.consumer_gnss()),
        ("RTK", NoiseModel.rtk()),
    ):
        report.sweeps[label] = sweep_cep(
            dem,
            intrinsics,
            site_lat,
            site_lon,
            noise=noise,
            altitude_agl_m=altitude_agl_m,
            trials=trials,
            max_cep90_m=max_cep90_m,
        )
    return report
