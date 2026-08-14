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

# 음영기복도 출력 크기.
#
# 이 값이 "화질"을 정하지는 않는다 — 원본 DEM 이 Copernicus GLO-30(30m)이라
# 9km 범위에 표본이 300개뿐이다. 3000px 로 뽑아도 셀당 10px 로 늘린 것이지
# 없던 지형이 생기지는 않는다.
#
# 그럼에도 올리는 이유는 **확대했을 때 보간 면(facet)이 드러나지 않게** 하기
# 위해서다. 900px 에서는 셀당 3px 라 Leaflet 이 확대하는 순간 이중선형 보간의
# 각진 경계가 그대로 보였다. 고해상도로 뽑고 표고를 미리 매끄럽게 만들면
# 확대해도 연속적인 지형으로 읽힌다.
#
# 진짜 디테일이 필요하면 작품설명서가 지정한 국토지리정보원 5m DEM 을
# `configs/sites/*.yaml` 의 `dem.local_dir` 에 연결하면 된다 — 같은 범위에
# 표본이 1,800개가 되어 6배 세밀해진다.
HILLSHADE_PX = 3000

# JPEG 품질. 음영기복도는 부드러운 연속 계조라 JPEG 가 PNG 보다 훨씬 작다
# (같은 3000px 에서 PNG 약 9MB → JPEG 약 1MB). 아티팩트 16MB 한도 안에
# 스냅샷까지 함께 넣으려면 이 차이가 필요하다.
HILLSHADE_QUALITY = 88

# 수직 과장 배율. 의성 일대는 9km 범위에 표고차가 355m 뿐이라 실제 비율로는
# 음영이 거의 생기지 않아 평면처럼 보인다. 지형도에서 관례적으로 쓰는 과장이며,
# 목적이 경사도 측정이 아니라 **능선과 계곡의 형태를 읽히게** 하는 것이므로
# 정당하다. 좌표 계산에는 전혀 쓰이지 않는다 — 순수하게 배경 그림용이다.
Z_FACTOR = 3.0


def build_hillshade(site: SiteConfig, bounds: tuple[float, float, float, float]) -> str:
    """DEM → 음영기복도 JPEG → data URI.

    Horn 방식 음영기복이되 두 가지를 더 한다:

    1. **표고 평활화** — DEM 을 10배로 늘려 뽑으면 이중선형 보간 때문에 격자
       셀 경계에서 기울기가 계단처럼 튄다. 음영은 기울기의 함수라 그 계단이
       각진 면으로 그대로 보인다. 표고를 미리 가우시안으로 부드럽게 만들면
       기울기가 연속이 되어 확대해도 면이 드러나지 않는다.

    2. **다방향 광원** — 단일 광원은 빛과 평행한 능선을 통째로 삼킨다.
       주광(북서)에 보조광(북동)을 약하게 섞으면 그 방향의 능선도 읽힌다.
    """
    from PIL import Image
    from scipy.ndimage import gaussian_filter

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

    # 출력 픽셀 하나가 DEM 셀의 몇 분의 일인지. 이 배율만큼 평활화한다.
    dem_cell_m = 30.0
    upsample = dem_cell_m / max(dy, 1e-9)
    sigma = max(0.8, upsample * 0.35)
    smooth = gaussian_filter(elevation.astype(np.float32), sigma=sigma)
    print(f"    표고 평활화: 업샘플 {upsample:.1f}배 → 시그마 {sigma:.1f} px")

    # 수직 과장. 이 일대는 9km 에 표고차 355m 라 경사가 완만해서, 실제
    # 비율로 음영을 계산하면 거의 평면처럼 보인다. 지형도에서 표준적으로
    # 쓰는 z-factor 로 기복을 읽을 수 있게 만든다. 지형의 **형태**를 보이는
    # 것이 목적이지 경사도를 재는 것이 아니므로 과장이 정당하다.
    dz_dy, dz_dx = np.gradient(smooth * Z_FACTOR, dy, dx)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)

    def shade_from(azimuth_deg: float, altitude_deg: float) -> np.ndarray:
        azimuth = np.radians(azimuth_deg)
        altitude = np.radians(altitude_deg)
        return np.clip(
            np.sin(altitude) * np.cos(slope)
            + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect),
            0.0,
            1.0,
        )

    # 주광 북서 45° + 보조광 북동 30°. 지도 관례상 북서가 주광이어야
    # 사람 눈이 볼록/오목을 헷갈리지 않는다.
    shade = 0.75 * shade_from(315.0, 45.0) + 0.25 * shade_from(45.0, 30.0)

    # 대비 확장. 음영 값이 실제로 쓰는 구간은 좁아서(완만한 지형일수록 더),
    # 그대로 쓰면 회색 덩어리가 된다. 상하위 2% 를 잘라 전 구간으로 편다.
    low, high = np.percentile(shade, [2, 98])
    if high - low > 1e-6:
        shade = np.clip((shade - low) / (high - low), 0.0, 1.0)
    print(f"    음영 대비 확장: [{low:.3f}, {high:.3f}] → [0, 1]")

    # 표고를 섞어 고도감을 준다 — 음영만 쓰면 평지와 고원이 구분되지 않는다.
    span = float(smooth.max() - smooth.min()) or 1.0
    height_tint = (smooth - smooth.min()) / span
    value = np.clip(0.70 * shade + 0.30 * height_tint, 0.0, 1.0)

    # 관제 화면(심야 남색)에 얹을 청회색 계열.
    #
    # 밝기 상한을 일부러 낮게 잡는다. 지형은 **배경**이지 주인공이 아니다 —
    # 배경이 밝으면 그 위에 얹히는 등급 색(FLARE 빨강 / GLOW 주황)이 묻힌다.
    # 기복의 형태는 위에서 대비를 폈으므로 이미 읽히고, 여기서는 전체를
    # 한 단계 어둡게 눌러 마커가 앞으로 나오게 한다.
    rgb = np.stack(
        [
            12 + 120 * value,        # R
            18 + 126 * value,        # G
            40 + 128 * value,        # B
        ],
        axis=-1,
    ).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(
        buffer, format="JPEG", quality=HILLSHADE_QUALITY, optimize=True, progressive=True
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    print(
        f"  음영기복도 {HILLSHADE_PX}x{HILLSHADE_PX} JPEG q{HILLSHADE_QUALITY} · "
        f"{len(encoded) / 1024:.0f} KB (base64)"
    )
    print(f"    표고 {elevation.min():.0f}~{elevation.max():.0f} m")
    return f"data:image/jpeg;base64,{encoded}"


def embed_snapshots(events: list[dict], snapshot_dir: Path | None = None) -> None:
    """스냅샷 파일명을 data URI 로 바꾼다 (제자리 수정).

    정적 배포에는 파일을 내려줄 서버가 없다. 57건 × 약 20KB ≈ 1.1MB 로
    아티팩트 한도 안에 충분히 들어간다.
    """
    snapshot_dir = snapshot_dir or (ROOT / "data" / "snapshots")
    embedded = missing = 0
    total_bytes = 0

    for event in events:
        name = event.get("snapshot")
        if not name:
            missing += 1
            continue
        path = snapshot_dir / name
        if not path.is_file():
            event["snapshot"] = None
            missing += 1
            continue
        raw = path.read_bytes()
        total_bytes += len(raw)
        # data URI 로 덮어쓰기 전에 원래 파일명을 남긴다 — 시퀀스 GIF 를
        # 트랙에 매칭하는 키가 파일명에 들어 있다.
        event["_snapshot_name"] = name
        event["snapshot"] = (
            "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        )
        embedded += 1

    print(f"  스냅샷 {embedded}건 임베드 · {total_bytes / 1024:.0f} KB (원본)")
    if missing:
        print(f"    스냅샷 없는 이벤트 {missing}건 — 카드에서 이미지 자리가 비워진다")


def embed_sequence_gifs(events: list[dict], gif_dir: Path | None = None) -> dict[str, str]:
    """시퀀스 GIF 를 별도 사전으로 내보내고, 이벤트에는 **키만** 붙인다.

    GIF 는 트랙 단위라 한 트랙의 이벤트 수십 건이 같은 파일을 공유한다.
    data URI 를 이벤트마다 복사하면 40건짜리 트랙 하나가 335KB × 40 = 13MB 가
    된다 (실제로 그렇게 만들었다가 번들이 24MB 로 튀어 한도를 넘겼다).
    키로 참조하면 GIF 당 한 번만 실린다.

    Returns:
        {트랙 키: data URI}
    """
    import json
    import re

    gif_dir = gif_dir or (ROOT / "data" / "gifs")
    manifest_path = gif_dir / "manifest.json"
    if not manifest_path.is_file():
        print("  시퀀스 GIF 없음 — scripts/build_sequence_gifs.py 를 먼저 돌린다")
        return {}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gifs: dict[str, str] = {}
    total = 0
    for key, filename in manifest.items():
        path = gif_dir / filename
        if not path.is_file():
            continue
        raw = path.read_bytes()
        total += len(raw)
        gifs[key] = "data:image/gif;base64," + base64.b64encode(raw).decode("ascii")

    pattern = re.compile(r"^(?P<key>.+-t\d+)-f\d+$")
    attached = 0
    for event in events:
        name = event.get("_snapshot_name")
        if not name:
            continue
        match = pattern.match(Path(name).stem)
        if match and match.group("key") in gifs:
            event["sequence_gif_key"] = match.group("key")
            attached += 1

    print(
        f"  시퀀스 GIF {len(gifs)}개 → 이벤트 {attached}건이 참조 · "
        f"{total / 1024:.0f} KB (중복 없음)"
    )
    return gifs


def seed_response_states(events: list[dict]) -> None:
    """데모에 대응 이력을 몇 건 심는다.

    전부 '미대응'이면 타임라인이 빈 채로만 보여서 기능이 있는지조차 알 수
    없다. 실제 운용에서 나올 법한 상태 — FLARE 는 접수·출동까지 갔고
    나머지는 아직 — 를 만들어 화면이 무엇을 하는지 보이게 한다.
    """
    import time

    now = time.time()
    flares = [e for e in events if e["tier"] == "FLARE"]
    if not flares:
        return

    plan = [
        (flares[0], ["received", "dispatched", "suppressed"]),
        (flares[1] if len(flares) > 1 else None, ["received", "dispatched"]),
        (flares[2] if len(flares) > 2 else None, ["received"]),
    ]
    ko = {"received": "접수", "dispatched": "출동", "suppressed": "진화"}

    seeded = 0
    for event, steps in plan:
        if event is None:
            continue
        history = [
            {"status": s, "at": now - (len(steps) - i) * 240}
            for i, s in enumerate(steps)
        ]
        event["response"] = steps[-1]
        event["response_label_ko"] = ko[steps[-1]]
        event["response_history"] = history
        seeded += 1

    print(f"  대응 이력 {seeded}건 시드 (데모용 예시 상태)")


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
    embed_snapshots(payload_events)
    sequence_gifs = embed_sequence_gifs(payload_events)
    seed_response_states(payload_events)
    for event in payload_events:
        event.pop("_snapshot_name", None)      # 내부 키는 내보내지 않는다
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
    (DEST / "gifs.json").write_text(
        json.dumps(sequence_gifs, ensure_ascii=False), encoding="utf-8"
    )

    store.close()
    print(f"\n저장 위치: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
