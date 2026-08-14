"""pyro-sdis 검증셋 일부를 내려받아 로컬에 캐시한다.

전체 데이터셋은 33,636장(수 GB)이라 베이스라인 측정에는 과하다.
HuggingFace datasets-server 의 rows API 로 필요한 만큼만 가져온다.

주의 — 이 데이터는 **지상 감시탑 시점**이다. 드론 시점이 아니다.
여기서 나오는 수치는 파이프라인이 동작한다는 증거이지 드론 운용 성능이
아니다 (README 한계 참조).

사용:
    uv run python scripts/fetch_pyro_sdis.py --split val --count 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

API = "https://datasets-server.huggingface.co/rows"
DATASET = "pyronear/pyro-sdis"
DEST = Path(__file__).resolve().parents[1] / "data" / "pyro_sdis"
PAGE = 100          # rows API 한 번에 가져올 수 있는 최대치


def fetch_page(split: str, offset: int, length: int) -> list[dict]:
    resp = requests.get(
        API,
        params={
            "dataset": DATASET,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("rows", [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    img_dir = DEST / args.split / "images"
    lbl_dir = DEST / args.split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    saved = skipped = failed = 0
    offset = 0

    while saved + skipped < args.count:
        want = min(PAGE, args.count - (saved + skipped))
        rows = fetch_page(args.split, offset, want)
        if not rows:
            print(f"더 이상 행이 없다 (offset={offset})")
            break
        offset += len(rows)

        for row in rows:
            rec = row["row"]
            name = Path(rec["image_name"]).stem
            img_path = img_dir / f"{name}.jpg"
            lbl_path = lbl_dir / f"{name}.txt"

            if img_path.exists() and lbl_path.exists() and not args.force:
                skipped += 1
            else:
                try:
                    src = rec["image"]["src"]
                    blob = requests.get(src, timeout=120)
                    blob.raise_for_status()
                    img_path.write_bytes(blob.content)
                    # 주석은 YOLO 정규화 형식: "cls cx cy w h" 줄바꿈 구분.
                    lbl_path.write_text((rec.get("annotations") or "").strip() + "\n",
                                        encoding="utf-8")
                    saved += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  [실패] {name}: {exc}")
                    failed += 1
                    continue

            manifest.append(
                {
                    "name": name,
                    "image": str(img_path.relative_to(DEST)),
                    "label": str(lbl_path.relative_to(DEST)),
                    "partner": rec.get("partner"),
                    "camera": rec.get("camera"),
                    "date": rec.get("date"),
                }
            )

        print(f"  진행 {saved + skipped}/{args.count}")

    manifest_path = DEST / f"{args.split}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n새로 받음 {saved} / 기존 {skipped} / 실패 {failed}")
    print(f"매니페스트: {manifest_path}")
    print("\n주의: 지상 감시탑 시점 데이터다. 드론 시점 성능의 근거가 아니다.")
    return 1 if failed and not saved else 0


if __name__ == "__main__":
    raise SystemExit(main())
