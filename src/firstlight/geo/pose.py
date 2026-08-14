"""카메라 자세 — 카메라 프레임 광선을 ENU 월드 프레임으로 회전시킨다.

회전행렬을 오일러 시퀀스 문자열로 조립하지 않고 **기저벡터를 직접 구성**한다.
시퀀스 규약(내재/외재, 축 순서, 부호)은 틀리기 쉽고 틀려도 조용히 틀리는데,
기저벡터 방식은 각 열의 의미가 명시적이라 단위 테스트로 바로 못박을 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    """촬영 순간의 카메라 위치·자세.

    Attributes:
        lat, lon: WGS84 도.
        alt_msl: 평균해수면 기준 고도 m. 상대고도를 쓰면 안 된다.
        yaw_deg: 북에서 시계방향 (0=북, 90=동).
        pitch_deg: 수평이 0, 양수가 위. 수직 하방은 -90.
        roll_deg: 광축 기준, 양수가 우측 기울임.
        timestamp: epoch 초. 없으면 0.
    """

    lat: float
    lon: float
    alt_msl: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0
    timestamp: float = 0.0


def enu_from_camera(
    yaw_deg, pitch_deg, roll_deg=0.0
) -> np.ndarray:
    """카메라 → ENU 회전행렬을 만든다.

    스칼라를 주면 (3, 3), 배열을 주면 (N, 3, 3) 을 반환한다.
    배열 입력은 몬테카를로 오차전파에서 N개 자세를 한 번에 처리하기 위한 것이다.

    열의 의미:
        col 0 = 카메라 x(우)  의 ENU 방향
        col 1 = 카메라 y(하)  의 ENU 방향
        col 2 = 카메라 z(전방)의 ENU 방향
    """
    scalar_input = (
        np.ndim(yaw_deg) == 0 and np.ndim(pitch_deg) == 0 and np.ndim(roll_deg) == 0
    )
    yaw = np.atleast_1d(np.radians(np.asarray(yaw_deg, dtype=float)))
    pitch = np.atleast_1d(np.radians(np.asarray(pitch_deg, dtype=float)))
    roll = np.atleast_1d(np.radians(np.asarray(roll_deg, dtype=float)))
    yaw, pitch, roll = np.broadcast_arrays(yaw, pitch, roll)

    sy, cy = np.sin(yaw), np.cos(yaw)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sr, cr = np.sin(roll), np.cos(roll)

    # 전방: yaw/pitch 만으로 결정된다 (roll 은 광축을 돌리므로 전방을 바꾸지 않는다).
    forward = np.stack([cp * sy, cp * cy, sp], axis=-1)
    # 롤 적용 전 우측: 항상 수평면 안에 있다.
    right0 = np.stack([cy, -sy, np.zeros_like(cy)], axis=-1)
    # 롤 적용 전 하방: forward × right0 (우수계에서 y = z × x).
    down0 = np.cross(forward, right0)

    # 롤은 전방축 기준 회전. 양수 롤이 우측벡터를 하방벡터 쪽으로 돌린다.
    right = right0 * cr[..., None] + down0 * sr[..., None]
    down = -right0 * sr[..., None] + down0 * cr[..., None]

    R = np.stack([right, down, forward], axis=-1)  # (N, 3, 3), 열이 기저벡터
    return R[0] if scalar_input else R


def rays_to_enu(rays_cam: np.ndarray, R_enu_cam: np.ndarray) -> np.ndarray:
    """카메라 프레임 광선을 ENU 로 회전. (N,3) x (3,3) 또는 (N,3) x (N,3,3)."""
    rays_cam = np.atleast_2d(np.asarray(rays_cam, dtype=float))
    R = np.asarray(R_enu_cam, dtype=float)
    if R.ndim == 2:
        return rays_cam @ R.T
    # 자세마다 광선이 하나씩 대응하는 경우 (몬테카를로).
    return np.einsum("nij,nj->ni", R, rays_cam)


def depression_deg(rays_enu: np.ndarray) -> np.ndarray:
    """ENU 단위광선의 부각(수평면 아래로 기운 각, 도). 위를 향하면 음수.

    이 값이 작을수록(수평에 가까울수록) 자세 오차가 지상거리 오차로
    증폭된다. 지오레퍼런싱 거절 판정의 핵심 입력이다.
    """
    rays = np.atleast_2d(np.asarray(rays_enu, dtype=float))
    norms = np.linalg.norm(rays, axis=-1)
    return np.degrees(np.arcsin(np.clip(-rays[:, 2] / norms, -1.0, 1.0)))
