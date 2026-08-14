"""단일 파일 데모 빌드를 아티팩트 배포용으로 변환한다.

Vite 는 완전한 HTML 문서(`<!doctype><html><head>...`)를 낸다. 반면 아티팩트는
본문만 받아 자체 스켈레톤으로 감싸므로, 문서 태그가 중복되면 안 된다.

이 스크립트는 head 의 <style>/<script> 와 body 내용만 꺼내 이어 붙인다.
favicon 링크는 버린다 — 아티팩트가 파라미터로 지정한다.

사용:
    uv run python scripts/make_artifact.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web" / "dist-demo" / "index.html"
DEST = ROOT / "out" / "dashboard-demo.html"

TITLE = "FIRSTLIGHT 관제 데모"


def extract(pattern: str, html: str) -> list[str]:
    return re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)


def main() -> int:
    if not SOURCE.exists():
        print(f"[실패] {SOURCE} 가 없다. 먼저 빌드한다:")
        print("  cd web && npm run build:demo")
        return 1

    html = SOURCE.read_text(encoding="utf-8")

    head_match = re.search(r"<head[^>]*>(.*?)</head>", html, re.DOTALL | re.I)
    body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.I)
    if not head_match or not body_match:
        print("[실패] head/body 를 찾지 못했다 — 빌드 산출물 형태가 바뀌었다.")
        return 1

    head, body = head_match.group(1), body_match.group(1)

    # head 에서 살릴 것: 스타일과 스크립트. 버릴 것: meta/title/favicon 링크.
    styles = extract(r"(<style[^>]*>.*?</style>)", head)
    head_scripts = extract(r"(<script[^>]*>.*?</script>)", head)

    parts = [
        f"<title>{TITLE}</title>",
        *styles,
        *head_scripts,
        body.strip(),
    ]
    output = "\n".join(parts)

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(output, encoding="utf-8")

    # 검증: 문서 태그가 남아 있으면 아티팩트 래퍼와 충돌한다.
    leftovers = [
        tag for tag in ("<!doctype", "<html", "<head", "<body")
        if tag in output.lower()
    ]
    size_mb = DEST.stat().st_size / 1_000_000
    print(f"저장: {DEST}  ({size_mb:.2f} MB)")
    print(f"  스타일 {len(styles)}개 · 스크립트 {len(head_scripts)}개")
    print(f"  문서 태그 잔존: {leftovers or '없음'}")
    print(f"  16MB 한도: {'OK' if size_mb < 16 else '초과'}")

    external = sorted(set(re.findall(r"https?://[a-zA-Z0-9.-]+", output)))
    print(f"  외부 호스트 참조: {external}")
    return 1 if leftovers else 0


if __name__ == "__main__":
    raise SystemExit(main())
