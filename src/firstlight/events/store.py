"""이벤트 저장소 — sqlite3 (표준 라이브러리).

PostGIS 로 갈아끼울 자리를 남기려고 좁은 인터페이스만 노출한다.
공간 질의가 필요해지는 시점(위험지도, 광역 집계)이 교체 시점이다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from firstlight.events.models import Event, EventLabel

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id          INTEGER NOT NULL,
    tier              TEXT    NOT NULL,
    score             REAL    NOT NULL,
    timestamp         REAL    NOT NULL,
    site              TEXT,
    camera            TEXT,
    bbox              TEXT,
    confidence        REAL,
    n_observations    INTEGER,
    geo_ok            INTEGER NOT NULL DEFAULT 0,
    lat               REAL,
    lon               REAL,
    elevation_m       REAL,
    range_m           REAL,
    depression_deg    REAL,
    cep50_m           REAL,
    cep90_m           REAL,
    geo_reject_reason TEXT,
    flow_ok           INTEGER NOT NULL DEFAULT 0,
    area_ok           INTEGER NOT NULL DEFAULT 0,
    criteria          TEXT,
    label             TEXT    NOT NULL DEFAULT 'unlabelled',
    features          TEXT,
    explanation       TEXT,
    created_at        REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_tier      ON events(tier);
CREATE INDEX IF NOT EXISTS idx_events_label     ON events(label);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_track     ON events(site, track_id);
"""


class EventStore:
    """이벤트 영속화."""

    def __init__(self, path: Path | str = "data/events.db") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """앞선 실행으로 만들어진 DB 에 새 열을 채워 넣는다.

        스키마가 바뀔 때마다 사용자가 DB 를 지워야 한다면 그건 도구가
        일을 떠넘기는 것이다. 없는 열만 조용히 추가한다.
        """
        existing = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(events)")
        }
        additions = {
            "flow_ok": "INTEGER NOT NULL DEFAULT 0",
            "area_ok": "INTEGER NOT NULL DEFAULT 0",
            "criteria": "TEXT",
        }
        for column, spec in additions.items():
            if column not in existing:
                self.conn.execute(f"ALTER TABLE events ADD COLUMN {column} {spec}")

    # ------------------------------------------------------------------

    def add(self, event: Event) -> Event:
        row = event.to_row()
        columns = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        cursor = self.conn.execute(
            f"INSERT INTO events ({columns}) VALUES ({placeholders})", row
        )
        self.conn.commit()
        event.event_id = int(cursor.lastrowid)
        return event

    def get(self, event_id: int) -> Event | None:
        row = self.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return Event.from_row(row) if row else None

    def list(
        self,
        tier: str | None = None,
        label: str | None = None,
        site: str | None = None,
        limit: int = 200,
    ) -> list[Event]:
        clauses, params = [], []
        if tier:
            clauses.append("tier = ?")
            params.append(tier)
        if label:
            clauses.append("label = ?")
            params.append(label)
        if site:
            clauses.append("site = ?")
            params.append(site)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ?", (*params, limit)
        ).fetchall()
        return [Event.from_row(r) for r in rows]

    def latest_for_track(self, site: str, track_id: int) -> Event | None:
        row = self.conn.execute(
            "SELECT * FROM events WHERE site = ? AND track_id = ? ORDER BY id DESC LIMIT 1",
            (site, track_id),
        ).fetchone()
        return Event.from_row(row) if row else None

    def set_label(self, event_id: int, label: EventLabel) -> bool:
        cursor = self.conn.execute(
            "UPDATE events SET label = ? WHERE id = ?", (label.value, event_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def counts_by_tier(self, site: str | None = None) -> dict[str, int]:
        where = "WHERE site = ?" if site else ""
        params = (site,) if site else ()
        rows = self.conn.execute(
            f"SELECT tier, COUNT(*) AS n FROM events {where} GROUP BY tier", params
        ).fetchall()
        return {r["tier"]: r["n"] for r in rows}

    def labelled_events(self) -> list[Event]:
        """사람이 판정한 이벤트 — 재학습 입력."""
        rows = self.conn.execute(
            "SELECT * FROM events WHERE label != 'unlabelled' ORDER BY id"
        ).fetchall()
        return [Event.from_row(r) for r in rows]

    def clear(self) -> None:
        self.conn.execute("DELETE FROM events")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
