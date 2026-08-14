"""자기운동 보정 — 드론이 움직이는 동안 연기만의 움직임을 분리한다.

지상 감시탑 시스템에는 이 단계가 없다. 카메라가 고정이라 화면의 움직임이
곧 대상의 움직임이기 때문이다. 드론은 다르다. 기체가 초속 몇 m 로 날고
짐벌이 흔들리면 **정지한 바위도 화면에서 움직인다.** 보정 없이 광류를
재면 "상승"도 "팽창"도 전부 기체 운동에 묻힌다.

절차:
    1. 탐지 박스를 **제외한** 배경에서 ORB 특징점을 뽑는다. 연기 위의
       특징점을 쓰면 보정하려는 대상으로 보정하게 되어 자기 발등을 찍는다.
    2. RANSAC 으로 프레임 간 호모그래피를 추정한다.
    3. 이전 프레임을 현재 프레임에 정합시킨다.
    4. 정합된 두 프레임 사이의 **잔차** 광류를 박스 안에서만 계산한다.
       이 잔차가 곧 대상 고유의 운동이다.

잔차 광류에서 뽑는 세 값:
    divergence      — 팽창(연기)인가 평행이동(구름그림자)인가
    upward_ratio    — 위로 향하는 벡터 비율 (연기는 오른다)
    translation_mag — 잔차 평행이동 크기
"""

from __future__ import annotations

import numpy as np

from firstlight.detect.detector import Detection


def _background_mask(shape: tuple[int, int], detections: list[Detection],
                     margin: float = 0.25) -> np.ndarray:
    """탐지 박스(여유 포함)를 제외한 배경 마스크."""
    mask = np.full(shape, 255, dtype=np.uint8)
    h, w = shape
    for det in detections:
        pad_x = det.width * margin
        pad_y = det.height * margin
        x1 = max(0, int(det.x1 - pad_x))
        y1 = max(0, int(det.y1 - pad_y))
        x2 = min(w, int(np.ceil(det.x2 + pad_x)))
        y2 = min(h, int(np.ceil(det.y2 + pad_y)))
        mask[y1:y2, x1:x2] = 0
    return mask


class EgoMotionCompensator:
    """연속 프레임 간 카메라 운동을 추정한다.

    Args:
        max_features: ORB 특징점 상한.
        min_matches: 이보다 적으면 추정을 포기하고 None 을 돌려준다.
            무리하게 추정한 호모그래피는 보정 안 한 것보다 나쁘다.
        ransac_threshold: RANSAC 재투영 오차 허용 (px).
    """

    def __init__(
        self,
        max_features: int = 1500,
        min_matches: int = 12,
        ransac_threshold: float = 3.0,
    ) -> None:
        import cv2

        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def estimate(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        detections: list[Detection] | None = None,
    ) -> np.ndarray | None:
        """이전 → 현재 호모그래피. 실패하면 None."""
        import cv2

        mask = (
            _background_mask(prev_gray.shape[:2], detections)
            if detections
            else None
        )
        kp1, des1 = self._orb.detectAndCompute(prev_gray, mask)
        kp2, des2 = self._orb.detectAndCompute(curr_gray, mask)
        if des1 is None or des2 is None or len(kp1) < self.min_matches:
            return None
        if len(kp2) < self.min_matches:
            return None

        # Lowe 비율 검정으로 모호한 대응을 버린다.
        pairs = self._matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < self.min_matches:
            return None

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_threshold)
        if H is None or inliers is None or int(inliers.sum()) < self.min_matches:
            return None
        return H


def residual_flow_stats(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    bbox: tuple[float, float, float, float],
    homography: np.ndarray | None = None,
) -> dict:
    """정합 후 박스 안의 잔차 광류 통계.

    homography 가 None 이면 보정 없이 계산한다 (고정 카메라).
    """
    import cv2

    h, w = curr_gray.shape[:2]
    if homography is not None:
        aligned = cv2.warpPerspective(
            prev_gray, homography, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        aligned = prev_gray

    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(np.ceil(bbox[2])))
    y2 = min(h, int(np.ceil(bbox[3])))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return {"divergence": 0.0, "upward_ratio": 0.0, "translation_mag": 0.0,
                "valid": False}

    flow = cv2.calcOpticalFlowFarneback(
        aligned[y1:y2, x1:x2], curr_gray[y1:y2, x1:x2],
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    vx, vy = flow[..., 0], flow[..., 1]

    # 발산 = ∂vx/∂x + ∂vy/∂y. 양수면 팽창이다.
    dvx_dx = np.gradient(vx, axis=1)
    dvy_dy = np.gradient(vy, axis=0)
    divergence = float((dvx_dx + dvy_dy).mean())

    # 이미지 y 는 아래가 양수이므로 상승은 vy < 0.
    magnitude = np.hypot(vx, vy)
    moving = magnitude > max(0.2, float(np.median(magnitude)))
    upward_ratio = float((vy[moving] < 0).mean()) if moving.any() else 0.0

    return {
        "divergence": divergence,
        "upward_ratio": upward_ratio,
        "translation_mag": float(np.hypot(vx.mean(), vy.mean())),
        "valid": True,
    }
