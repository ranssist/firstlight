"""공유용 정적 데모에 구울 데이터를 만든다.

아티팩트로 배포하는 대시보드는 백엔드가 없다. `/api/*` 응답을 미리 뽑아
번들에 넣고, 지도 배경은 보유한 DEM 으로 만든 음영기복도를 쓴다.

지도 배경을 OSM 타일 대신 지형으로 바꾼 이유는 두 가지다:
    1. 아티팩트 CSP 가 외부 호스트 요청을 전부 막는다 — 타일이 안 온다.
    2. 산불 조기탐지에서 중요한 배경은 도로망이 아니라 **능선과 계곡**이다.
       지오레퍼런싱이 광선을 꽂는 대상이 바로 그 지형면이다.

사용:
    uv run python scripts/build_demo_data.py
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

from firstlight.config import SiteConfig
from firstlight.events.store import EventStore

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "web" / "src" / "demo"

# 음영기복도 출력 크기. 너무 키우면 base64 가 커지고, 작으면 능선이 뭉갠다.
HILLSHADE_PX = 900


def build_hillshade(site: SiteConfig, bounds: tuple[float, float, float, float]) -> str:
    """DEM → 음영기복도 PNG → data URI.

    표준 Horn 방식. 북서 45도에서 빛이 든다고 가정한다 — 지도 관례이고,
    사람 눈이 그 방향에서 볼록/오목을 헷갈리지 않는다.
    """
    from PIL import Image

    dem = site.load_dem()
    min_lon, min_lat, max_lon, max_lat = bounds

    # 표시 범위만 잘라 격자를 다시 만든다.
    lons = np.linspace(min_lon, max_lon, HILLSHADE_PX)
    lats = np.linspace(max_lat, min_lat, HILLSHADE_PX)      # 위에서 아래로
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    elevation = dem.elevation(grid_lon, grid_lat)
    elevation = np.nan_to_num(elevation, nan=float(np.nanmin(elevation)))

    # 셀 크기(m). 위도 1도 ≈ 111,132m, 경도는 cos(위도) 배.
    mid_lat = 0.5 * (min_lat + max_lat)
    dy = (max_lat - min_lat) * 111_132.0 / HILLSHADE_PX
    dx = (max_lon - min_lon) * 111_320.0 * np.cos(np.radians(mid_lat)) / HILLSHADE_PX

    dz_dy, dz_dx = np.gradient(elevation, dy, dx)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)

    azimuth = np.radians(315.0)          # 북서
    altitude = np.radians(45.0)
    shade = np.sin(altitude) * np.cos(slope) + np.cos(altitude) * np.sin(slope) * np.cos(
        azimuth - aspect
    )
    shade = np.clip(shade, 0.0, 1.0)

    # 표고를 살짝 섞어 고도감을 준다 — 음영만 쓰면 평지와 고원이 구분되지 않는다.
    span = float(elevation.max() - elevation.min()) or 1.0
    height_tint = (elevation - elevation.min()) / span
    value = 0.72 * shade + 0.28 * height_tint

    # 관제 화면(심야 남색)에 얹을 것이므로 청회색 계열로 굽는다.
    rgb = np.stack(
        [
            18 + 150 * value,        # R
            24 + 155 * value,        # G
            48 + 150 * value,        # B
        ],
        axis=-1,
    ).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    print(f"  음영기복도 {HILLSHADE_PX}x{HILLSHADE_PX} · {len(encoded) / 1024:.0f} KB (base64)")
    print(f"    표고 {elevation.min():.0f}~{elevation.max():.0f} m")
    return f"data:image/png;base64,{encoded}"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    site = SiteConfig.load("uiseong")
    store = EventStore(ROOT / "data" / "events.db")

    events = store.list(limit=300)
    if not events:
        print("[실패] events.db 가 비어 있다. 먼저 파이프라인을 돌린다:")
        print("  uv run firstlight run --source data/figlib/<시퀀스> --reset")
        return 1

    payload_events = [e.to_public_dict() for e in events]
    counts = store.counts_by_tier()
    summary = {
        "counts": counts,
        "total": sum(counts.values()),
        "labelled": len(store.labelled_events()),
        "queue": counts.get("GLOW", 0),
        "scorer": {
            "fitted": False,
            "n_train": 0,
            "mode": "sparse",
            "tau_high": 0.7,
            "tau_low": 0.35,
        },
    }

    # 지도 표시 범위: 이벤트를 모두 담되 여백을 둔다.
    located = [e for e in payload_events if e["geo_ok"] and e["lat"] is not None]
    lats = [e["lat"] for e in located] + [u["lat"] for u in site.response_units]
    lons = [e["lon"] for e in located] + [u["lon"] for u in site.response_units]
    pad = 0.012
    bounds = (
        min(lons) - pad, min(lats) - pad,
        max(lons) + pad, max(lats) + pad,
    )

    print(f"이벤트 {len(payload_events)}건 · 좌표 발행 {len(located)}건")
    print(f"  등급 분포 {counts}")

    hillshade = build_hillshade(site, bounds)

    site_payload = {
        "name": site.name,
        "label": site.label,
        "lat": site.lat,
        "lon": site.lon,
        "bbox": list(site.bbox),
        "response_units": site.response_units,
    }

    (DEST / "events.json").write_text(
        json.dumps(payload_events, ensure_ascii=False), encoding="utf-8"
    )
    (DEST / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    (DEST / "site.json").write_text(
        json.dumps(site_payload, ensure_ascii=False), encoding="utf-8"
    )
    (DEST / "hillshade.json").write_text(
        json.dumps(
            {
                "image": hillshade,
                # Leaflet ImageOverlay 는 [[남,서],[북,동]] 순서를 받는다.
                "bounds": [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    store.close()
    print(f"\n저장 위치: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
