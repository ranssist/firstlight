"""이벤트 저장과 등급 라우팅 — 소개서 §4③.

저장은 표준 라이브러리 sqlite3 로 한다. 소개서에는 PostGIS 라고 썼지만,
Windows 노트북에서 Docker 를 요구하는 것은 MVP 진입장벽이다. `EventStore`
인터페이스 뒤에 두었으므로 파일럿 단계에서 PostGIS 어댑터로 갈아끼우면 된다.
공간 질의가 필요해지는 시점(위험지도, 광역 집계)이 그 교체 시점이다.
"""

from firstlight.events.models import Event, EventLabel
from firstlight.events.store import EventStore
from firstlight.events.router import AlertRouter, RouterConfig


def open_store(dsn: str):
    """DSN 으로 저장소를 연다 — SQLite 경로 또는 postgresql:// URL."""
    from firstlight.events.postgis_store import open_store as _open

    return _open(dsn)


__all__ = [
    "AlertRouter",
    "Event",
    "EventLabel",
    "EventStore",
    "RouterConfig",
    "open_store",
]
