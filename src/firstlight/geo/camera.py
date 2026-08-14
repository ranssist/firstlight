"""카메라 내부 파라미터 — 픽셀 ↔ 카메라 광선 변환.

카메라 프레임은 x=우, y=하, z=전방(광축).
왜곡 모델은 Brown-Conrady (k1, k2, p1, p2, k3) 를 순수 numpy 로 구현한다.
cv2 에 의존하지 않는 이유: geo/ 는 M1 단계에서 무거운 CV 의존성 없이
단독으로 검증 가능해야 하기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 왜곡 역변환은 고정점 반복이라 선형 수렴한다. 실측(70도 화각, k1=-0.28):
#   5회 → 1.2px,  10회 → 1.1e-2px,  20회 → 9.2e-7px,  40회 → 기계정밀도.
# OpenCV 관례인 5회는 강한 통형 왜곡에서 1px 넘게 틀린다. 잔차를 보고
# 끊되 상한을 넉넉히 둔다 — 탐지 픽셀 몇 개만 변환하므로 비용은 무시할 만하다.
_UNDISTORT_MAX_ITERS = 40
_UNDISTORT_TOL = 1e-12          # 정규화 좌표 기준 (≈ 1e-9 px)


@dataclass(frozen=True)
class CameraIntrinsics:
    """핀홀 내부 파라미터 + 방사/접선 왜곡 계수."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # (k1, k2, p1, p2, k3). 전부 0 이면 왜곡 없음으로 취급하고 역변환을 건너뛴다.
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    name: str = "camera"

    # ---------------------------------------------------------------- 생성자

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        hfov_deg: float,
        vfov_deg: float | None = None,
        **kwargs,
    ) -> CameraIntrinsics:
        """수평 화각(+선택적 수직 화각)으로부터 생성.

        vfov 를 주지 않으면 정사각 픽셀(fy = fx)로 가정한다.
        """
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        fy = fx if vfov_deg is None else (height / 2.0) / np.tan(np.radians(vfov_deg) / 2.0)
        return cls(
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=width / 2.0,
            cy=height / 2.0,
            **kwargs,
        )

    @classmethod
    def from_sensor(
        cls,
        width: int,
        height: int,
        focal_mm: float,
        sensor_width_mm: float,
        sensor_height_mm: float | None = None,
        **kwargs,
    ) -> CameraIntrinsics:
        """물리 센서 제원으로부터 생성. 기체 스펙시트에서 바로 옮겨 적을 수 있다."""
        fx = focal_mm * width / sensor_width_mm
        if sensor_height_mm is None:
            fy = fx  # 정사각 픽셀 가정
        else:
            fy = focal_mm * height / sensor_height_mm
        return cls(
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=width / 2.0,
            cy=height / 2.0,
            **kwargs,
        )

    # ---------------------------------------------------------------- 파생값

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.fx))))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.height / (2.0 * self.fy))))

    @property
    def has_distortion(self) -> bool:
        return any(abs(c) > 1e-12 for c in self.dist)

    # ------------------------------------------------------------ 왜곡 모델

    def _distort_normalized(self, xn: np.ndarray, yn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """정규화 이미지 좌표에 왜곡을 **적용**한다 (이상 → 실측)."""
        k1, k2, p1, p2, k3 = self.dist
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        return xd, yd

    def _undistort_normalized(
        self, xd: np.ndarray, yd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """왜곡을 **제거**한다 (실측 → 이상). 고정점 반복."""
        if not self.has_distortion:
            return xd, yd
        xn, yn = xd.copy(), yd.copy()
        for _ in range(_UNDISTORT_MAX_ITERS):
            xe, ye = self._distort_normalized(xn, yn)
            rx, ry = xd - xe, yd - ye
            xn = xn + rx
            yn = yn + ry
            if max(np.max(np.abs(rx)), np.max(np.abs(ry))) < _UNDISTORT_TOL:
                break
        return xn, yn

    # ------------------------------------------------------------- 픽셀↔광선

    def pixel_to_ray(self, u, v) -> np.ndarray:
        """픽셀 (u, v) → 카메라 프레임 단위 광선. 스칼라·배열 모두 허용.

        반환 shape 은 (N, 3). 입력이 스칼라여도 (1, 3) 이다.
        """
        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))
        xd = (u - self.cx) / self.fx
        yd = (v - self.cy) / self.fy
        xn, yn = self._undistort_normalized(xd, yd)
        rays = np.stack([xn, yn, np.ones_like(xn)], axis=-1)
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    def ray_to_pixel(self, ray_cam: np.ndarray) -> np.ndarray:
        """카메라 프레임 광선 → 픽셀 (u, v). `pixel_to_ray` 의 역변환.

        폐루프 자체검증(eval/geo_accuracy.py)이 이 함수를 쓴다.
        광축 뒤(z <= 0)의 점은 NaN 을 반환한다.
        """
        rays = np.atleast_2d(np.asarray(ray_cam, dtype=float))
        z = rays[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = np.where(z > 0, rays[:, 0] / z, np.nan)
            yn = np.where(z > 0, rays[:, 1] / z, np.nan)
        xd, yd = self._distort_normalized(xn, yn)
        return np.stack([xd * self.fx + self.cx, yd * self.fy + self.cy], axis=-1)

    def contains(self, uv: np.ndarray) -> np.ndarray:
        """픽셀이 이미지 경계 안에 있는지."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        u, v = uv[:, 0], uv[:, 1]
        return (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
