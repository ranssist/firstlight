"""탐지 스냅샷 — 작품설명서 「관제 대시보드 화면 설계」의 "연기 스냅샷".

    이벤트 카드 : 탐지 시각, 좌표, 신뢰도, **연기 스냅샷**, 시퀀스 GIF, 등급을
    통해 관제 요원이 1클릭으로 "실제 산불 / 오탐" 확인

좌표와 점수만으로는 오탐 여부를 판단할 수 없다. 요원이 실제로 보는 것은
"저게 연기처럼 생겼나"이고, 그러려면 **그 순간 그 자리의 그림**이 필요하다.
GLOW 큐의 1클릭 판정이 성립하려면 이 이미지가 있어야 한다.

무엇을 잘라내는가:
    탐지 박스만 자르면 맥락이 없어 안개인지 연기인지 알 수 없다. 박스 주변을
    넉넉히 포함해 능선·하늘과의 관계가 보이게 하고, 박스는 등급 색으로
    그려 어디를 봐야 하는지 표시한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# 박스 주변 여유 배수. 1.0 = 박스 크기만큼 사방으로 더 본다.
CONTEXT_MARGIN = 1.4
# 저장 크기. 카드 썸네일(작게)과 상세 확대(크게)를 모두 감당해야 한다.
OUTPUT_WIDTH = 384
# 박스가 너무 작으면 확대해도 뭉개지므로 최소 크롭 폭을 둔다.
MIN_CROP_PX = 160

TIER_BGR = {
    "FLARE": (70, 57, 230),      # #E63946
    "GLOW": (97, 162, 244),      # #F4A261
    "SPARK": (174, 153, 141),    # #8D99AE
}


def crop_detection(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    tier: str = "GLOW",
    draw_box: bool = True,
) -> np.ndarray | None:
    """탐지 주변을 맥락과 함께 잘라내고 박스를 그린다.

    Returns:
        BGR 이미지. 박스가 프레임 밖이거나 너무 작으면 None.
    """
    import cv2

    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)

    # 정사각 크롭으로 만든다 — 카드에서 크기가 들쭉날쭉하면 훑기 어렵다.
    side = max(box_w, box_h) * (1.0 + 2.0 * CONTEXT_MARGIN)
    side = max(side, MIN_CROP_PX)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    # 프레임 안으로 밀어 넣는다. 잘라내면 박스가 가장자리로 몰리므로 이동시킨다.
    side = min(side, float(min(width, height)))
    left = int(round(min(max(cx - side / 2, 0), width - side)))
    top = int(round(min(max(cy - side / 2, 0), height - side)))
    size = int(round(side))

    crop = frame_bgr[top : top + size, left : left + size]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None

    crop = crop.copy()
    if draw_box:
        colour = TIER_BGR.get(tier, TIER_BGR["GLOW"])
        thickness = max(1, size // 120)
        cv2.rectangle(
            crop,
            (int(round(x1 - left)), int(round(y1 - top))),
            (int(round(x2 - left)), int(round(y2 - top))),
            colour,
            thickness,
        )

    if size != OUTPUT_WIDTH:
        interp = cv2.INTER_AREA if size > OUTPUT_WIDTH else cv2.INTER_CUBIC
        crop = cv2.resize(crop, (OUTPUT_WIDTH, OUTPUT_WIDTH), interpolation=interp)
    return crop


def save_snapshot(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    dest_dir: Path,
    name: str,
    tier: str = "GLOW",
    quality: int = 82,
) -> str | None:
    """크롭을 JPEG 로 저장하고 파일명을 돌려준다.

    **이벤트마다 별도 파일이다.** 트랙 단위로 덮어쓰면 디스크는 아끼지만,
    과거 이벤트가 나중 프레임의 그림을 가리키게 된다 — 요원이 t=60s 이벤트를
    열었는데 t=2400s 의 다 큰 연기가 보이는 식이다. 이벤트는 특정 시점의
    기록이므로 그 시점의 그림이어야 한다.

    장기 운용에서는 보존 정책(오래된 스냅샷 삭제)으로 용량을 관리한다.
    이벤트 행 자체도 같은 속도로 쌓이므로 스냅샷만의 문제는 아니다.
    """
    import cv2

    crop = crop_detection(frame_bgr, bbox, tier=tier)
    if crop is None:
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}.jpg"
    path = dest_dir / filename
    ok = cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return filename if ok else None
