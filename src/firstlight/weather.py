"""기상청 단기예보 API — 풍향·풍속을 가져온다.

풍향 보정(`geo/wind.py`)의 입력이다.

**서비스키가 필요하다.** 공공데이터포털(data.go.kr)에서 "기상청_단기예보
((구)_동네예보) 조회서비스"를 신청하면 발급된다. 키가 없으면 설정된 고정
풍향으로 동작하는 스텁으로 물러난다 — 키가 없다고 파이프라인 전체가 멈추면
안 되기 때문이다.

    환경변수 KMA_SERVICE_KEY 또는 configs/secrets.yaml 에 넣는다.

격자 변환:
    기상청 API 는 위경도가 아니라 자체 격자 좌표(nx, ny)를 받는다.
    5km 람베르트 정각원추도법이며 파라미터는 기상청이 공개한 값이다.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from firstlight.geo.wind import Wind

KST = timezone(timedelta(hours=9))
NCST_URL = (
    "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
)

# 기상청 격자 파라미터 (5km LCC).
_RE = 6371.00877      # 지구 반경 km
_GRID = 5.0           # 격자 간격 km
_SLAT1, _SLAT2 = 30.0, 60.0     # 표준위도
_OLON, _OLAT = 126.0, 38.0      # 기준점 경위도
_XO, _YO = 43, 136              # 기준점 격자 좌표


@dataclass(frozen=True)
class GridPoint:
    nx: int
    ny: int


def latlon_to_grid(lat: float, lon: float) -> GridPoint:
    """위경도 → 기상청 격자 (nx, ny)."""
    degrad = math.pi / 180.0
    re = _RE / _GRID
    slat1, slat2 = _SLAT1 * degrad, _SLAT2 * degrad
    olon, olat = _OLON * degrad, _OLAT * degrad

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf**sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro**sn)

    ra = math.tan(math.pi * 0.25 + lat * degrad * 0.5)
    ra = re * sf / (ra**sn)
    theta = lon * degrad - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    return GridPoint(
        nx=int(ra * math.sin(theta) + _XO + 0.5),
        ny=int(ro - ra * math.cos(theta) + _YO + 0.5),
    )


def _service_key() -> str | None:
    key = os.environ.get("KMA_SERVICE_KEY")
    if key:
        return key
    secrets = Path("configs/secrets.yaml")
    if secrets.exists():
        import yaml

        raw = yaml.safe_load(secrets.read_text(encoding="utf-8")) or {}
        return raw.get("kma_service_key")
    return None


def _base_datetime(now: datetime | None = None) -> tuple[str, str]:
    """초단기실황 기준시각. 매시 40분 이후에 해당 정시 자료가 공개된다."""
    now = now or datetime.now(KST)
    if now.minute < 40:
        now = now - timedelta(hours=1)
    return now.strftime("%Y%m%d"), now.strftime("%H00")


def fetch_wind(
    lat: float,
    lon: float,
    fallback: Wind | None = None,
    timeout: float = 10.0,
    now: datetime | None = None,
) -> tuple[Wind | None, str]:
    """해당 지점의 현재 풍향·풍속을 가져온다.

    Returns:
        (바람, 출처). 출처는 "kma" | "fallback" | "unavailable".
        호출부가 실측인지 가정값인지 구분할 수 있어야 하므로 문자열로 알린다.
    """
    key = _service_key()
    if not key:
        return fallback, "fallback" if fallback else "unavailable"

    import requests

    base_date, base_time = _base_datetime(now)
    grid = latlon_to_grid(lat, lon)
    try:
        response = requests.get(
            NCST_URL,
            params={
                "serviceKey": key,
                "numOfRows": 100,
                "pageNo": 1,
                "dataType": "JSON",
                "base_date": base_date,
                "base_time": base_time,
                "nx": grid.nx,
                "ny": grid.ny,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        items = response.json()["response"]["body"]["items"]["item"]
    except Exception:  # noqa: BLE001 — 기상 자료 실패로 탐지를 멈추지 않는다
        return fallback, "fallback" if fallback else "unavailable"

    values = {item["category"]: item["obsrValue"] for item in items}
    if "WSD" not in values or "VEC" not in values:
        return fallback, "fallback" if fallback else "unavailable"

    # VEC 는 기상 규약(불어오는 방향)이라 Wind.from_deg 와 같다.
    return Wind(speed_ms=float(values["WSD"]), from_deg=float(values["VEC"])), "kma"
