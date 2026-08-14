"""PostGIS 저장소 — 작품설명서 표 Ⅱ-1 「PostgreSQL + PostGIS」.

⚠️ **아직 실행 검증되지 않았다.** 이 노트북에 PostgreSQL 서버가 없어
스키마와 질의를 실제로 돌려보지 못했다. 인터페이스는 `EventStore` 와
동일하므로 서버가 준비되면 `--db postgresql://...` 한 줄로 갈아끼울 수 있고,
그때 `tests/test_events.py` 를 이 구현으로 한 번 더 돌리면 검증된다.

SQLite 와의 차이는 좌표 열 하나다:

    SQLite  : lat REAL, lon REAL          — 애플리케이션이 거리 계산
    PostGIS : geom geometry(Point, 4326)  — DB 가 공간 인덱스로 계산

이 차이가 필요해지는 시점은 명확하다. 이벤트가 수만 건 쌓여
"이 폴리곤 안의 지난 시즌 발화점" 같은 질의를 하게 될 때다. 그전까지는
SQLite 로 충분하고, 그래서 기본값은 SQLite 로 둔다.
"""

from __future__ import annotations

import json
from typing import Any

from firstlight.events.models import Event, EventLabel

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS events (
    id                SERIAL PRIMARY KEY,
    track_id          INTEGER NOT NULL,
    tier              TEXT    NOT NULL,
    score             DOUBLE PRECISION NOT NULL,
    timestamp         DOUBLE PRECISION NOT NULL,
    site              TEXT,
    camera            TEXT,
    bbox              JSONB,
    confidence        DOUBLE PRECISION,
    n_observations    INTEGER,
    geo_ok            BOOLEAN NOT NULL DEFAULT FALSE,
    lat               DOUBLE PRECISION,
    lon               DOUBLE PRECISION,
    -- 공간 인덱스의 대상. lat/lon 은 조회 편의를 위해 함께 남긴다.
    geom              geometry(Point, 4326),
    elevation_m       DOUBLE PRECISION,
    range_m           DOUBLE PRECISION,
    depression_deg    DOUBLE PRECISION,
    cep50_m           DOUBLE PRECISION,
    cep90_m           DOUBLE PRECISION,
    geo_reject_reason TEXT,
    flow_ok           BOOLEAN NOT NULL DEFAULT FALSE,
    area_ok           BOOLEAN NOT NULL DEFAULT FALSE,
    criteria          JSONB,
    label             TEXT    NOT NULL DEFAULT 'unlabelled',
    features          JSONB,
    explanation       JSONB,
    created_at        DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_geom  ON events USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_events_tier  ON events (tier);
CREATE INDEX IF NOT EXISTS idx_events_label ON events (label);
CREATE INDEX IF NOT EXISTS idx_events_time  ON events (timestamp);
CREATE INDEX IF NOT EXISTS idx_events_track ON events (site, track_id);
"""

_COLUMNS = (
    "track_id, tier, score, timestamp, site, camera, bbox, confidence, "
    "n_observations, geo_ok, lat, lon, geom, elevation_m, range_m, "
    "depression_deg, cep50_m, cep90_m, geo_reject_reason, flow_ok, area_ok, "
    "criteria, label, features, explanation, created_at"
)


class PostgisEventStore:
    """`EventStore` 와 같은 인터페이스의 PostGIS 구현.

    Args:
        dsn: `postgresql://user:pass@host/dbname`
    """

    def __init__(self, dsn: str) -> None:
        import psycopg
        from psycopg.rows import dict_row

        self.conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA)

    # ------------------------------------------------------------------

    def add(self, event: Event) -> Event:
        point = (
            None
            if not event.geo_ok or event.lon is None
            else f"SRID=4326;POINT({event.lon} {event.lat})"
        )
        values = (
            event.track_id, event.tier.value, event.score, event.timestamp,
            event.site, event.camera, json.dumps(list(event.bbox)),
            event.confidence, event.n_observations, event.geo_ok,
            event.lat, event.lon, point, event.elevation_m, event.range_m,
            event.depression_deg, event.cep50_m, event.cep90_m,
            event.geo_reject_reason, event.flow_ok, event.area_ok,
            json.dumps(event.criteria, ensure_ascii=False), event.label.value,
            json.dumps(event.features, ensure_ascii=False),
            json.dumps(event.explanation, ensure_ascii=False), event.created_at,
        )
        placeholders = ", ".join(["%s"] * len(values))
        with self.conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO events ({_COLUMNS}) VALUES ({placeholders}) RETURNING id",
                values,
            )
            event.event_id = cur.fetchone()["id"]
        return event

    def get(self, event_id: int) -> Event | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM events WHERE id = %s", (event_id,))
            row = cur.fetchone()
        return _to_event(row) if row else None

    def list(
        self,
        tier: str | None = None,
        label: str | None = None,
        site: str | None = None,
        limit: int = 200,
    ) -> list[Event]:
        clauses, params = [], []
        for column, value in (("tier", tier), ("label", label), ("site", site)):
            if value:
                clauses.append(f"{column} = %s")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM events {where} ORDER BY id DESC LIMIT %s",
                (*params, limit),
            )
            rows = cur.fetchall()
        return [_to_event(r) for r in rows]

    def latest_for_track(self, site: str, track_id: int) -> Event | None:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM events WHERE site = %s AND track_id = %s "
                "ORDER BY id DESC LIMIT 1",
                (site, track_id),
            )
            row = cur.fetchone()
        return _to_event(row) if row else None

    def set_label(self, event_id: int, label: EventLabel) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE events SET label = %s WHERE id = %s", (label.value, event_id)
            )
            return cur.rowcount > 0

    def counts_by_tier(self, site: str | None = None) -> dict[str, int]:
        where = "WHERE site = %s" if site else ""
        params = (site,) if site else ()
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT tier, COUNT(*) AS n FROM events {where} GROUP BY tier",
                        params)
            return {r["tier"]: r["n"] for r in cur.fetchall()}

    def labelled_events(self) -> list[Event]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM events WHERE label <> 'unlabelled' ORDER BY id"
            )
            return [_to_event(r) for r in cur.fetchall()]

    # -------------------------------------------------- PostGIS 고유 기능

    def within_radius(self, lat: float, lon: float, radius_m: float) -> list[Event]:
        """반경 안의 이벤트. **SQLite 구현에는 없는 기능**이다.

        위험지도 정밀화(작품설명서 「데이터 자산화」)가 이 질의를 쓴다.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM events WHERE geom IS NOT NULL AND "
                "ST_DWithin(geom::geography, ST_MakePoint(%s, %s)::geography, %s) "
                "ORDER BY id DESC",
                (lon, lat, radius_m),
            )
            return [_to_event(r) for r in cur.fetchall()]

    def clear(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM events")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> PostgisEventStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _to_event(row: dict[str, Any]) -> Event:
    """psycopg 는 JSONB 를 파이썬 객체로 이미 풀어 준다."""
    from firstlight.verify.scorer import Tier

    return Event(
        event_id=row["id"],
        track_id=row["track_id"],
        tier=Tier(row["tier"]),
        score=row["score"],
        timestamp=row["timestamp"],
        site=row["site"] or "",
        camera=row["camera"] or "",
        bbox=tuple(row["bbox"]),
        confidence=row["confidence"],
        n_observations=row["n_observations"],
        geo_ok=bool(row["geo_ok"]),
        lat=row["lat"],
        lon=row["lon"],
        elevation_m=row["elevation_m"],
        range_m=row["range_m"],
        depression_deg=row["depression_deg"],
        cep50_m=row["cep50_m"],
        cep90_m=row["cep90_m"],
        geo_reject_reason=row["geo_reject_reason"],
        flow_ok=bool(row["flow_ok"]),
        area_ok=bool(row["area_ok"]),
        criteria=row["criteria"] or {},
        label=EventLabel(row["label"]),
        features=row["features"] or {},
        explanation=row["explanation"] or {},
        created_at=row["created_at"],
    )


def open_store(dsn: str):
    """DSN 을 보고 알맞은 저장소를 연다.

        data/events.db              → SQLite
        postgresql://user@host/db   → PostGIS
    """
    if dsn.startswith(("postgresql://", "postgres://")):
        return PostgisEventStore(dsn)
    from firstlight.events.store import EventStore

    return EventStore(dsn)
