"""풍향 보정 — 연기 기둥 위치에서 발화점 위치를 되짚는다.

**이 체인에서 가장 약한 고리다.** 솔직하게 적어둔다.

문제: 카메라가 보는 것은 연기이지 불이 아니다. 연기는 바람을 타고 흘러가므로
연기의 어느 지점을 좌표화하느냐에 따라 발화점과 수백 m 어긋날 수 있다.

이 구현의 입장:
    1. 좌표화 대상을 **연기 마스크의 하단 중앙**으로 잡는다. 연기 기저는
       발화점 바로 위이므로 표류가 가장 작은 지점이다. 이게 보정의 8할이다.
    2. 남은 표류에 1차 보정을 넣는다: 풍상 방향으로 `speed * drift_seconds`.
       `drift_seconds` 는 연기가 기저에서 관측 지점까지 이동한 시간의 대용값인데,
       단일 시점 영상에서는 관측되지 않는 양이다. 따라서 이 항은 **경험적
       보정 계수**이며 현장 캘리브레이션 없이는 신뢰할 수 없다.
    3. 그래서 보정 전/후 좌표를 **둘 다** 반환한다. 관제 요원이 두 점 사이의
       거리를 보고 판단할 수 있어야 한다. 보정값 하나만 내놓고 그게 정답인
       척하는 것이 가장 위험하다.

`drift_seconds=0.0` 이면 2번 항이 꺼지고 1번만 남는다. 현장 데이터가 없는
지금 단계의 기본값으로는 이쪽이 정직하다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from firstlight.geo.frame import enu_to_lonlat


@dataclass(frozen=True)
class Wind:
    """기상 규약을 따른다 — `from_deg` 는 바람이 **불어오는** 방향이다.

    북풍(북에서 불어옴) = 0, 동풍 = 90. 기상청 API 의 WSD/VEC 와 같은 규약.
    소개서 §3 시나리오의 "북서풍 6.2m/s" 는 from_deg=315, speed_ms=6.2.
    """

    speed_ms: float
    from_deg: float

    @property
    def toward_enu(self) -> tuple[float, float]:
        """바람이 **향하는** 방향의 단위벡터 (east, north)."""
        bearing = math.radians(self.from_deg + 180.0)
        return (math.sin(bearing), math.cos(bearing))


def correct_for_wind(
    lat: float,
    lon: float,
    wind: Wind | None,
    drift_seconds: float = 0.0,
) -> tuple[float, float, float]:
    """연기 관측 좌표 → 추정 발화 좌표.

    Returns:
        (보정 위도, 보정 경도, 이동거리 m). 바람 정보가 없거나
        drift_seconds 가 0 이면 입력을 그대로 돌려주고 거리는 0.
    """
    if wind is None or drift_seconds <= 0.0 or wind.speed_ms <= 0.0:
        return lat, lon, 0.0

    east, north = wind.toward_enu
    distance = wind.speed_ms * drift_seconds
    # 발화점은 관측된 연기의 **풍상**(바람이 불어온 쪽)에 있다.
    lon_c, lat_c = enu_to_lonlat(-east * distance, -north * distance, lat, lon)
    return float(lat_c), float(lon_c), float(distance)
