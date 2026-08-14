"""카메라 내부 파라미터 — 픽셀 ↔ 광선 왕복."""

import numpy as np
import pytest

from firstlight.geo.camera import CameraIntrinsics

WIDTH, HEIGHT = 1920, 1080


def test_from_fov_recovers_fov():
    intr = CameraIntrinsics.from_fov(WIDTH, HEIGHT, hfov_deg=73.7)
    assert intr.hfov_deg == pytest.approx(73.7, abs=1e-9)


def test_from_sensor_matches_hand_calculation():
    """fx = f * W / sensor_width."""
    intr = CameraIntrinsics.from_sensor(WIDTH, HEIGHT, focal_mm=24.0, sensor_width_mm=13.2)
    assert intr.fx == pytest.approx(24.0 * WIDTH / 13.2)


def test_principal_ray_is_optical_axis():
    """주점 픽셀은 정확히 전방(0,0,1) 광선이어야 한다."""
    intr = CameraIntrinsics.from_fov(WIDTH, HEIGHT, 70.0)
    ray = intr.pixel_to_ray(intr.cx, intr.cy)[0]
    assert np.allclose(ray, [0.0, 0.0, 1.0], atol=1e-12)


def test_rays_are_unit_length():
    intr = CameraIntrinsics.from_fov(WIDTH, HEIGHT, 70.0)
    u = np.array([0.0, 500.0, 1919.0, 960.0])
    v = np.array([0.0, 200.0, 1079.0, 540.0])
    assert np.allclose(np.linalg.norm(intr.pixel_to_ray(u, v), axis=1), 1.0)


def test_pixel_axes_point_the_right_way():
    """u 증가는 카메라 x(우), v 증가는 카메라 y(하) 방향이어야 한다."""
    intr = CameraIntrinsics.from_fov(WIDTH, HEIGHT, 70.0)
    right = intr.pixel_to_ray(intr.cx + 400, intr.cy)[0]
    below = intr.pixel_to_ray(intr.cx, intr.cy + 300)[0]
    assert right[0] > 0 and abs(right[1]) < 1e-12
    assert below[1] > 0 and abs(below[0]) < 1e-12


def test_horizontal_edge_ray_matches_half_fov():
    """이미지 좌우 끝 광선의 각도는 hfov/2 여야 한다."""
    intr = CameraIntrinsics.from_fov(WIDTH, HEIGHT, 70.0)
    edge = intr.pixel_to_ray(WIDTH, intr.cy)[0]
    angle = np.degrees(np.arctan2(edge[0], edge[2]))
    assert angle == pytest.approx(35.0, abs=1e-9)


@pytest.mark.parametrize(
    "dist",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (-0.28, 0.09, 0.0, 0.0, 0.0),                # 통형 왜곡
        (0.15, -0.04, 0.001, -0.002, 0.01),          # 방사 + 접선
    ],
)
def test_pixel_ray_roundtrip(dist):
    """픽셀 → 광선 → 픽셀 이 서브픽셀로 돌아와야 한다 (왜곡 포함)."""
    intr = CameraIntrinsics.from_fov(WIDTH, HEIGHT, 70.0, dist=dist)
    u = np.array([10.0, 480.0, 960.0, 1400.0, 1910.0])
    v = np.array([10.0, 270.0, 540.0, 800.0, 1070.0])

    back = intr.ray_to_pixel(intr.pixel_to_ray(u, v))
    assert np.allclose(back[:, 0], u, atol=1e-3)
    assert np.allclose(back[:, 1], v, atol=1e-3)


def test_distortion_actually_changes_rays():
    """왜곡 계수가 결과에 영향을 줘야 한다 (무시되고 있지 않은지)."""
    plain = CameraIntrinsics.from_fov(WIDTH, HEIGHT, 70.0)
    warped = CameraIntrinsics.from_fov(WIDTH, HEIGHT, 70.0, dist=(-0.3, 0.1, 0.0, 0.0, 0.0))
    corner = (50.0, 50.0)
    assert not np.allclose(
        plain.pixel_to_ray(*corner), warped.pixel_to_ray(*corner), atol=1e-6
    )


def test_ray_behind_camera_is_nan():
    intr = CameraIntrinsics.from_fov(WIDTH, HEIGHT, 70.0)
    uv = intr.ray_to_pixel(np.array([[0.0, 0.0, -1.0]]))
    assert np.isnan(uv).all()


def test_contains_bounds():
    intr = CameraIntrinsics.from_fov(WIDTH, HEIGHT, 70.0)
    uv = np.array([[0.0, 0.0], [1919.9, 1079.9], [-1.0, 500.0], [500.0, 1080.0]])
    assert list(intr.contains(uv)) == [True, True, False, False]
