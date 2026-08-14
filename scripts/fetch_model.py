"""Pyronear 연기 탐지 ONNX 가중치를 받는다.

가중치는 Apache-2.0 이라 ultralytics(AGPL-3.0) 없이 쓸 수 있다.
추론은 onnxruntime + 자체 전/후처리로 하므로 공공 납품 시 소스공개
의무가 걸리지 않는다 (README 라이선스 주의 참조).

사용:
    uv run python scripts/fetch_model.py
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

import requests

REPO = "pyronear/yolo11s_sensitive-detector_v1.0.0"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
DEST_DIR = Path(__file__).resolve().parents[1] / "models" / "pyronear"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  받는 중: {url}")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    print(f"  저장: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 받는다")
    args = ap.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = DEST_DIR / "model.onnx"

    if onnx_path.exists() and not args.force:
        print(f"이미 있음: {onnx_path}")
        return 0

    tarball = DEST_DIR / "onnx_cpu.tar.gz"
    download(f"{BASE}/onnx_cpu.tar.gz", tarball)

    print("  압축 해제 중...")
    extract_dir = DEST_DIR / "_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with tarfile.open(tarball) as tf:
        # 경로 탈출 방지 — 신뢰하는 출처라도 아카이브는 검증하고 푼다.
        for member in tf.getmembers():
            target = (extract_dir / member.name).resolve()
            if not str(target).startswith(str(extract_dir.resolve())):
                raise RuntimeError(f"아카이브 경로가 대상 밖을 가리킨다: {member.name}")
        tf.extractall(extract_dir, filter="data")

    found = sorted(extract_dir.rglob("*.onnx"))
    if not found:
        print("  [실패] 아카이브 안에 .onnx 가 없다", file=sys.stderr)
        print(f"  내용: {[str(p.relative_to(extract_dir)) for p in extract_dir.rglob('*')]}")
        return 1

    shutil.copy2(found[0], onnx_path)
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    print(f"  ONNX: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  sha256: {digest}")

    # 부산물 정리.
    shutil.rmtree(extract_dir, ignore_errors=True)
    tarball.unlink(missing_ok=True)

    print("\n라이선스: 가중치 Apache-2.0 (Pyronear). 추론은 onnxruntime 로만 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
