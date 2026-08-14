"""HPWREN FIgLib 화재 시퀀스를 내려받는다.

왜 이 데이터인가:
    각 시퀀스는 발화 **전** 40분과 **후** 40분을 60초 간격으로 담고 있다.
    발화 전 구간은 "카메라가 아무 일도 없는 산을 지켜본 40분"이며, 이것이
    오경보율의 **실제 시간 분모**가 된다. 시퀀스 하나당 0.67시간이므로
    소개서의 "오경보 0.5건/비행시간" 을 진짜 단위로 계산할 수 있다.

    합성 음성 샘플로는 이 숫자를 만들 수 없다. 실제 안개·구름그림자·
    노을·역광이 들어있는 진짜 40분이어야 의미가 있다.

한계:
    60초 간격이라 광류 기반 특징을 쓸 수 없다. 시퀀스 검증기의 **희소 모드**
    (면적 성장·중심 이동·지속성)만 검증 가능하다.

파일명 규약:
    <epoch>_<±초offset>.jpg   — 음수는 발화 전, 양수는 발화 후.

사용:
    uv run python scripts/fetch_figlib.py --sequences 12
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, unquote

import requests

BASE = "http://hpwren.ucsd.edu/FIgLib/HPWREN-FIgLib-Data"
DEST = Path(__file__).resolve().parents[1] / "data" / "figlib"

# href 가 따옴표 없이 쓰여 있다.
HREF = re.compile(r"href\s*=\s*['\"]?([^'\"\s>]+)")
# 파일명 끝의 부호 있는 오프셋(초).
OFFSET = re.compile(r"_([+-]\d+)\.jpg$", re.IGNORECASE)


def list_sequences() -> list[str]:
    resp = requests.get(f"{BASE}/", timeout=60)
    resp.raise_for_status()
    names = []
    for href in HREF.findall(resp.text):
        if href.endswith("/index.html") and not href.startswith("Tar"):
            names.append(href.split("/")[0])
    return names


def list_frames(sequence: str) -> list[tuple[str, int]]:
    """(파일명, 발화기준 오프셋 초) 목록.

    발화 **후** 프레임의 '+' 는 href 안에서 %2B 로 인코딩돼 있다.
    디코딩하지 않으면 양수 오프셋이 통째로 누락된다.
    """
    resp = requests.get(f"{BASE}/{sequence}/index.html", timeout=60)
    resp.raise_for_status()
    frames = []
    for href in HREF.findall(resp.text):
        name = unquote(href.split("/")[-1])
        match = OFFSET.search(name)
        if match:
            frames.append((name, int(match.group(1))))
    return sorted(set(frames), key=lambda t: t[1])


def download_one(args: tuple[str, str, Path]) -> bool:
    sequence, name, dest = args
    if dest.exists():
        return True
    try:
        # 로컬 파일명은 디코딩된 형태('+')를 쓰되, 요청 URL 은 다시 인코딩한다.
        url = f"{BASE}/{sequence}/{quote(name)}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    [실패] {sequence}/{name}: {exc}")
        return False


def rebuild_manifest() -> int:
    """로컬 디렉터리를 스캔해 매니페스트를 다시 만든다.

    다운로드가 중간에 끊겨도 받아둔 만큼은 쓸 수 있어야 한다. 완결된
    시퀀스(발화 전 프레임이 있는 것)만 넣는다.
    """
    manifest = []
    for seq_dir in sorted(p for p in DEST.iterdir() if p.is_dir()):
        frames = []
        for image in seq_dir.glob("*.jpg"):
            match = OFFSET.search(image.name)
            if match:
                frames.append({"file": image.name, "offset_s": int(match.group(1))})
        if not frames or not any(f["offset_s"] < 0 for f in frames):
            continue
        frames.sort(key=lambda f: f["offset_s"])
        manifest.append({"sequence": seq_dir.name, "frames": frames})

    path = DEST / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_neg = sum(1 for s in manifest for f in s["frames"] if f["offset_s"] < 0)
    print(f"시퀀스 {len(manifest)}개 · 발화 전 프레임 {total_neg}장")
    print(f"관측 시간 환산: {total_neg / 60.0:.2f} 시간 (60초 간격 기준)")
    print(f"매니페스트: {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences", type=int, default=12, help="받을 시퀀스 수")
    ap.add_argument("--skip", type=int, default=0, help="앞에서 건너뛸 시퀀스 수")
    ap.add_argument(
        "--negatives-only",
        action="store_true",
        help="발화 전 프레임만 받는다 (오경보율 측정 전용, 용량 절반)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--rebuild-only",
        action="store_true",
        help="내려받지 않고 로컬에 이미 있는 시퀀스만으로 매니페스트를 다시 만든다",
    )
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)

    if args.rebuild_only:
        return rebuild_manifest()
    print("시퀀스 목록 조회 중...")
    all_sequences = list_sequences()
    print(f"  전체 {len(all_sequences)}개")

    chosen = all_sequences[args.skip : args.skip + args.sequences]
    manifest = []

    for i, sequence in enumerate(chosen, 1):
        print(f"[{i}/{len(chosen)}] {sequence}")
        try:
            frames = list_frames(sequence)
        except Exception as exc:  # noqa: BLE001
            print(f"    목록 조회 실패: {exc}")
            continue

        if args.negatives_only:
            frames = [f for f in frames if f[1] < 0]
        if not frames:
            print("    프레임 없음 — 건너뜀")
            continue

        seq_dir = DEST / sequence
        jobs = [(sequence, name, seq_dir / name) for name, _ in frames]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(download_one, jobs))

        ok = sum(results)
        negatives = [f for f in frames if f[1] < 0]
        positives = [f for f in frames if f[1] >= 0]
        print(f"    {ok}/{len(jobs)}장 (발화 전 {len(negatives)} / 후 {len(positives)})")

        manifest.append(
            {
                "sequence": sequence,
                "frames": [
                    {"file": name, "offset_s": off}
                    for name, off in frames
                    if (seq_dir / name).exists()
                ],
            }
        )

    path = DEST / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_neg = sum(
        1 for s in manifest for f in s["frames"] if f["offset_s"] < 0
    )
    # 60초 간격 → 프레임 수가 곧 관측 시간(분)이다.
    print(f"\n시퀀스 {len(manifest)}개 · 발화 전 프레임 {total_neg}장")
    print(f"관측 시간 환산: {total_neg / 60.0:.2f} 시간 (60초 간격 기준)")
    print(f"매니페스트: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
