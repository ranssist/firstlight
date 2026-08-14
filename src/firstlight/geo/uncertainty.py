"""오차전파 — 몬테카를로로 좌표의 오차반경(CEP)을 추정한다.

소개서 §3 시나리오의 "오차반경 약 38m" 같은 숫자는 여기서 나와야 한다.
지어낸 숫자와 계산된 숫자의 차이가 이 모듈의 존재 이유다.

모델링한 오차원:
    - GNSS 수평/수직  → 광선 원점 이동
    - 자세각 (yaw/pitch/roll) → 광선 방향 회전
    - 탐지 픽셀 위치  → 광선 방향 미세 변화
    - DEM 수직오차    → 지형면 높이

DEM 오차를 화이트노이즈가 아니라 **공간상관된 편의(bias)** 로 모델링한 이유:
    GLO-30 의 수직오차는 인접 픽셀끼리 강하게 상관된다. 픽셀마다 독립인
    노이즈로 두면 광선 경로를 따라 평균되어 오차가 실제보다 작게 나온다.
    시행마다 지형 전체를 통째로 δ 만큼 올리는 편의 모델이 이 기하에서
    지배적이고 보수적인 모드다. 지형을 δ 올리는 것은 카메라를 δ 내리는
    것과 같으므로 원점 z 오프셋으로 흡수된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from firstlight.geo.camera import CameraIntrinsics
from firstlight.geo.dem import DEM
from firstlight.geo.frame import lonlat_to_enu
from firstlight.geo.pose import CameraPose, enu_from_camera, rays_to_enu
from firstlight.geo.raycast import raycast_to_ground


@dataclass(frozen=True)
class NoiseModel:
    """1σ 오차. 각도는 도, 길이는 m, 픽셀은 px.

    수평 GNSS 오차는 **축당** σ 다 (σ_east = σ_north = gps_horizontal_m).
    """

    gps_horizontal_m: float = 2.0
    gps_vertical_m: float = 3.0
    yaw_deg: float = 2.0
    pitch_deg: float = 0.5
    roll_deg: float = 0.5
    pixel_px: float = 5.0
    dem_vertical_m: float = 4.0

    @classmethod
    def consumer_gnss(cls) -> NoiseModel:
        """일반 소비자급 드론 (DJI 표준 GNSS + 짐벌)."""
        return cls()

    @classmethod
    def rtk(cls) -> NoiseModel:
        """RTK 측위 + 고급 IMU. 파일럿 단계에서 도달 가능한 수준."""
        return cls(
            gps_horizontal_m=0.10,
            gps_vertical_m=0.15,
            yaw_deg=0.5,
            pitch_deg=0.2,
            roll_deg=0.2,
            pixel_px=3.0,
            dem_vertical_m=4.0,
        )

    @classmethod
    def perfect(cls) -> NoiseModel:
        """무노이즈. 폐루프 왕복 정확도 확인용."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class MonteCarloResult:
    cep50_m: float
    cep90_m: float
    hit_rate: float          # 교차에 성공한 시행 비율
    lat_samples: np.ndarray
    lon_samples: np.ndarray
    bias_m: float            # 표본 중심과 공칭해의 거리


def monte_carlo_cep(
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    u: float,
    v: float,
    dem: DEM,
    nominal_lat: float,
    nominal_lon: float,
    noise: NoiseModel,
    trials: int = 200,
    rng: np.random.Generator | None = None,
    max_range_m: float = 15000.0,
    step_m: float = 20.0,
) -> MonteCarloResult:
    """공칭해 주변의 오차 산포를 추정한다.

    반환하는 CEP 는 "공칭해로부터 표본이 얼마나 흩어지는가"이며, 대칭성에
    의해 "보고한 좌표로부터 진값이 얼마나 떨어져 있을 수 있는가"의 근사다.
    """
    rng = rng or np.random.default_rng(0)
    n = int(trials)

    # 픽셀 섭동 → 광선 방향 변화
    du = rng.normal(0.0, noise.pixel_px, n)
    dv = rng.normal(0.0, noise.pixel_px, n)
    rays_cam = intrinsics.pixel_to_ray(u + du, v + dv)

    # 자세 섭동 → 회전
    R = enu_from_camera(
        pose.yaw_deg + rng.normal(0.0, noise.yaw_deg, n),
        pose.pitch_deg + rng.normal(0.0, noise.pitch_deg, n),
        pose.roll_deg + rng.normal(0.0, noise.roll_deg, n),
    )
    rays_enu = rays_to_enu(rays_cam, R)

    # 원점 섭동: 수평 GNSS + (수직 GNSS - DEM 편의)
    off = np.zeros((n, 3))
    off[:, 0] = rng.normal(0.0, noise.gps_horizontal_m, n)
    off[:, 1] = rng.normal(0.0, noise.gps_horizontal_m, n)
    off[:, 2] = rng.normal(0.0, noise.gps_vertical_m, n) - rng.normal(
        0.0, noise.dem_vertical_m, n
    )

    hits = raycast_to_ground(
        pose.lat,
        pose.lon,
        pose.alt_msl,
        rays_enu,
        dem,
        max_range_m=max_range_m,
        step_m=step_m,
        origin_offsets_enu=off,
    )

    ok = hits.hit
    hit_rate = float(ok.mean()) if n else 0.0
    if not ok.any():
        return MonteCarloResult(
            cep50_m=float("inf"),
            cep90_m=float("inf"),
            hit_rate=hit_rate,
            lat_samples=np.array([]),
            lon_samples=np.array([]),
            bias_m=float("nan"),
        )

    lat_s = hits.lat[ok]
    lon_s = hits.lon[ok]

    # 공칭해 기준 국소 평면에서 반경 거리를 잰다.
    dx, dy = lonlat_to_enu(lon_s, lat_s, nominal_lat, nominal_lon)
    radial = np.hypot(dx, dy)

    return MonteCarloResult(
        cep50_m=float(np.percentile(radial, 50)),
        cep90_m=float(np.percentile(radial, 90)),
        hit_rate=hit_rate,
        lat_samples=lat_s,
        lon_samples=lon_s,
        bias_m=float(np.hypot(dx.mean(), dy.mean())),
    )
