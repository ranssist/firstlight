"""시퀀스 GIF 생성 — 작품설명서 「이벤트 카드: … 연기 스냅샷, **시퀀스 GIF**, 등급」.

정지 이미지 한 장으로는 "형태 확산 + 상승 운동"을 보일 수 없다. 그런데 그
두 가지가 바로 시퀀스 검증이 판정 근거로 삼는 것이다 — 요원이 판정을
확인하려면 시스템이 본 것과 같은 움직임을 봐야 한다.

**고정 크롭**을 쓰는 것이 핵심이다. 프레임마다 박스를 따라가며 자르면 연기가
화면 가운데 머물러 "커진다"가 보이지 않는다. 트랙 전체 박스의 합집합을 잡아
고정해 두면 그 안에서 연기가 자라 올라가는 것이 그대로 읽힌다.

스냅샷(`events/snapshot.py`)과 달리 트랙 전체를 알아야 하므로 사후 처리다.
실제 운용에서는 트랙 종료 시 배경 작업으로 돌리면 된다.

사용:
    uv run python scripts/build_sequence_gifs.py --source data/figlib/<시퀀스>
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "gifs"

# GIF 한 변. 카드 썸네일 위에 겹쳐 재생하므로 스냅샷보다 작아도 된다.
GIF_PX = 200
# 최대 프레임 수. 전부 넣으면 용량이 커지고, 너무 적으면 움직임이 안 보인다.
MAX_FRAMES = 14
# 프레임 간 표시 시간(ms). FIgLib 은 60초 간격이라 실시간 재생은 무의미하고,
# "무엇이 변했는가"를 읽히게 하는 속도로 맞춘다.
FRAME_MS = 320

TIER_BGR = {
    "FLARE": (70, 57, 230),
    "GLOW": (97, 162, 244),
    "SPARK": (174, 153, 141),
}

# 스냅샷 파일명 규약: {site}-t{track}-f{frame}.jpg
SNAPSHOT_NAME = re.compile(r"^(?P<site>.+)-t(?P<track>\d+)-f(?P<frame>\d+)$")


def stable_crop_box(
    boxes: list[tuple[float, float, float, float]],
    width: int,
    height: int,
    margin: float = 0.9,
) -> tuple[int, int, int]:
    """트랙 전체를 담는 정사각 크롭. (left, top, size)

    합집합을 쓰는 이유: 연기는 자라므로 마지막 박스가 가장 크다. 첫 박스에
    맞추면 나중 프레임에서 연기가 크롭 밖으로 나가고, 매 프레임 따라가면
    "커진다"가 사라진다.
    """
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin)
    side = max(side, 120.0)
    side = min(side, float(min(width, height)))

    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    left = int(round(min(max(cx - side / 2, 0), width - side)))
    top = int(round(min(max(cy - side / 2, 0), height - side)))
    return left, top, int(round(side))


def build_gif(
    frames: list[tuple[Path, tuple[float, float, float, float], str]],
    dest: Path,
) -> int | None:
    """프레임 목록 → GIF. 반환값은 바이트 수."""
    import cv2
    from PIL import Image

    if len(frames) < 2:
        return None

    # 균등 간격으로 솎아낸다 — 앞부분만 쓰면 성장이 안 보인다.
    if len(frames) > MAX_FRAMES:
        idx = np.linspace(0, len(frames) - 1, MAX_FRAMES).round().astype(int)
        frames = [frames[i] for i in idx]

    first = cv2.imread(str(frames[0][0]))
    if first is None:
        return None
    height, width = first.shape[:2]
    left, top, size = stable_crop_box([b for _, b, _ in frames], width, height)

    images: list[Image.Image] = []
    for path, bbox, tier in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        crop = img[top : top + size, left : left + size].copy()
        if crop.size == 0:
            continue
        cv2.rectangle(
            crop,
            (int(bbox[0] - left), int(bbox[1] - top)),
            (int(bbox[2] - left), int(bbox[3] - top)),
            TIER_BGR.get(tier, TIER_BGR["GLOW"]),
            max(1, size // 110),
        )
        crop = cv2.resize(crop, (GIF_PX, GIF_PX), interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(crop[:, :, ::-1]))

    if len(images) < 2:
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        dest,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    return dest.stat().st_size


def main() -> int:
    import json

    from firstlight.events.store import EventStore

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="원본 프레임 디렉터리")
    ap.add_argument("--db", default="data/events.db")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"[실패] 원본 디렉터리가 없다: {source}")
        return 1

    # 프레임 인덱스 → 파일. `firstlight run` 과 같은 정렬(시각 오름차순)을 쓴다.
    from firstlight.cli import _open_source

    reader = _open_source(str(source))
    frame_paths = [path for path, _ in reader.frames]     # type: ignore[attr-defined]

    store = EventStore(args.db)
    events = store.list(limit=2000)

    # 트랙별로 (프레임 인덱스, 박스, 등급) 을 모은다.
    by_track: dict[tuple[str, int], list[tuple[int, tuple, str]]] = defaultdict(list)
    for event in events:
        if not event.snapshot:
            continue
        match = SNAPSHOT_NAME.match(Path(event.snapshot).stem)
        if not match:
            continue
        key = (match.group("site"), int(match.group("track")))
        by_track[key].append(
            (int(match.group("frame")), tuple(event.bbox), event.tier.value)
        )

    manifest: dict[str, str] = {}
    total = 0
    for (site, track), items in sorted(by_track.items()):
        items.sort()
        frames = [
            (frame_paths[i], bbox, tier)
            for i, bbox, tier in items
            if 0 <= i < len(frame_paths)
        ]
        name = f"{site}-t{track}.gif"
        size = build_gif(frames, DEST / name)
        if size is None:
            print(f"  트랙 {track}: 프레임 부족 — 건너뜀")
            continue
        total += size
        manifest[f"{site}-t{track}"] = name
        print(f"  트랙 {track}: {len(frames)}프레임 → {name} ({size / 1024:.0f} KB)")

    (DEST / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    store.close()
    print(f"\nGIF {len(manifest)}개 · 합계 {total / 1024:.0f} KB → {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
