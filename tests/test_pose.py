"""자세 회전행렬 — 규약을 숫자로 못박는다.

회전 규약은 틀려도 조용히 틀린다. 부호 하나가 뒤집히면 좌표가 지형 반대편에
찍히는데, 값은 그럴듯하게 나온다. 그래서 각 기저벡터가 어느 방위를 가리켜야
하는지를 사람이 읽고 검증할 수 있는 형태로 적어둔다.
"""

import numpy as np
import pytest

from firstlight.geo.pose import depression_deg, enu_from_camera, rays_to_enu

EAST = np.array([1.0, 0.0, 0.0])
NORTH = np.array([0.0, 1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])


def _cols(R):
    """(우, 하, 전방) ENU 방향."""
    return R[:, 0], R[:, 1], R[:, 2]


def test_level_north_facing():
    """수평·북향·롤0 → 전방=북, 우=동, 하=아래."""
    right, down, forward = _cols(enu_from_camera(yaw_deg=0, pitch_deg=0))
    assert np.allclose(forward, NORTH, atol=1e-12)
    assert np.allclose(right, EAST, atol=1e-12)
    assert np.allclose(down, -UP, atol=1e-12)


def test_yaw_is_clockwise_from_north():
    """yaw=90 은 동쪽. 북→동이 시계방향이라는 규약."""
    _, _, forward = _cols(enu_from_camera(yaw_deg=90, pitch_deg=0))
    assert np.allclose(forward, EAST, atol=1e-12)

    _, _, forward = _cols(enu_from_camera(yaw_deg=180, pitch_deg=0))
    assert np.allclose(forward, -NORTH, atol=1e-12)


def test_right_vector_is_right_of_heading():
    """동쪽을 보면 오른쪽은 남쪽."""
    right, _, _ = _cols(enu_from_camera(yaw_deg=90, pitch_deg=0))
    assert np.allclose(right, -NORTH, atol=1e-12)


def test_negative_pitch_looks_down():
    """pitch=-90 이 수직 하방. DJI 짐벌 규약과 같아야 한다."""
    _, _, forward = _cols(enu_from_camera(yaw_deg=0, pitch_deg=-90))
    assert np.allclose(forward, -UP, atol=1e-12)

    _, _, forward = _cols(enu_from_camera(yaw_deg=0, pitch_deg=45))
    assert forward[2] > 0, "양수 pitch 는 위를 봐야 한다"


def test_nadir_image_down_is_south_when_heading_north():
    """수직 하방·북향일 때 화면 아래쪽은 남쪽이다."""
    _, down, _ = _cols(enu_from_camera(yaw_deg=0, pitch_deg=-90))
    assert np.allclose(down, -NORTH, atol=1e-12)


def test_roll_rotates_about_optical_axis():
    """롤은 전방벡터를 바꾸지 않는다."""
    _, _, f0 = _cols(enu_from_camera(30, -20, 0))
    _, _, f1 = _cols(enu_from_camera(30, -20, 35))
    assert np.allclose(f0, f1, atol=1e-12)


def test_positive_roll_turns_right_toward_down():
    """양수 롤은 우측벡터를 하방벡터 쪽으로 돌린다 (우측 기울임)."""
    right0, down0, _ = _cols(enu_from_camera(0, 0, 0))
    right, _, _ = _cols(enu_from_camera(0, 0, 90))
    assert np.allclose(right, down0, atol=1e-12)


@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [(0, 0, 0), (37, -22, 11), (-140, 5, -60), (270, -89, 180)],
)
def test_is_proper_rotation(yaw, pitch, roll):
    """정규직교이고 행렬식이 +1 이어야 한다 (반사가 섞이면 안 된다)."""
    R = enu_from_camera(yaw, pitch, roll)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_batched_matches_scalar():
    """배열 입력이 스칼라 입력과 같은 결과를 줘야 한다 (몬테카를로 경로)."""
    yaws = np.array([0.0, 45.0, 123.0])
    pitches = np.array([-10.0, -30.0, 5.0])
    rolls = np.array([0.0, 12.0, -7.0])

    batched = enu_from_camera(yaws, pitches, rolls)
    assert batched.shape == (3, 3, 3)
    for i in range(3):
        assert np.allclose(batched[i], enu_from_camera(yaws[i], pitches[i], rolls[i]))


def test_scalar_input_returns_single_matrix():
    assert enu_from_camera(10.0, -20.0, 0.0).shape == (3, 3)


def test_rays_to_enu_per_ray_rotation():
    """(N,3) 광선 x (N,3,3) 자세 대응이 맞는지."""
    rays = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    R = enu_from_camera(np.array([0.0, 90.0]), np.array([0.0, 0.0]))
    out = rays_to_enu(rays, R)
    assert np.allclose(out[0], NORTH, atol=1e-12)
    assert np.allclose(out[1], EAST, atol=1e-12)


def test_depression_sign():
    """아래를 향하면 양수, 위를 향하면 음수."""
    assert np.isclose(depression_deg(np.array([[0.0, 0.0, -1.0]]))[0], 90.0)
    assert np.isclose(depression_deg(np.array([[0.0, 1.0, 0.0]]))[0], 0.0, atol=1e-12)
    assert depression_deg(np.array([[0.0, 1.0, 1.0]]))[0] < 0
