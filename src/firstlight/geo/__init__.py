"""지오레퍼런싱 — 탐지 픽셀을 지상 좌표로 변환한다.

좌표계 규약 (전 모듈 공통, 여기서 한 번만 정의한다):

카메라 프레임
    x = 우(right), y = 하(down), z = 전방(forward, 광축)
    일반적인 컴퓨터비전 규약이다.

월드 프레임
    ENU — x = 동(East), y = 북(North), z = 상(Up). 단위 m.
    드론 위치를 원점으로 하는 국소 평면(pyproj AEQD)을 쓴다.

자세각
    yaw   : 북에서 시계방향 도(0 = 북, 90 = 동)
    pitch : 수평이 0, **양수가 위**. 따라서 수직 하방은 -90.
            (DJI 짐벌 pitch 규약과 동일)
    roll  : 광축 기준 회전. 양수가 우측 기울임.

고도
    모두 평균해수면(MSL) 기준 m. 드론 텔레메트리가 상대고도를 준다면
    이륙지점 표고를 더해 MSL로 변환한 뒤 이 모듈에 넣어야 한다.
"""

from firstlight.geo.camera import CameraIntrinsics
from firstlight.geo.dem import DEM, synthetic_dem
from firstlight.geo.frame import enu_to_lonlat, lonlat_to_enu
from firstlight.geo.pose import CameraPose, depression_deg, enu_from_camera, rays_to_enu
from firstlight.geo.raycast import GeoFix, RejectReason, raycast_to_ground
from firstlight.geo.solver import GeoSolver, GeoSolverConfig
from firstlight.geo.uncertainty import NoiseModel, monte_carlo_cep
from firstlight.geo.wind import Wind

__all__ = [
    "CameraIntrinsics",
    "CameraPose",
    "DEM",
    "GeoFix",
    "GeoSolver",
    "GeoSolverConfig",
    "NoiseModel",
    "RejectReason",
    "Wind",
    "depression_deg",
    "enu_from_camera",
    "enu_to_lonlat",
    "lonlat_to_enu",
    "monte_carlo_cep",
    "rays_to_enu",
    "raycast_to_ground",
    "synthetic_dem",
]
