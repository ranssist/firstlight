"""탐지기 베이스라인 — pyro-sdis 검증셋 PR 곡선과 AP50.

이 숫자의 성격을 오해하면 안 된다:
    pyro-sdis 는 **지상 감시탑** 영상이다. 여기서 잘 나온다고 드론 시점에서
    잘 나온다는 뜻이 아니다. 이 측정의 목적은 (a) 추론 경로가 옳게 구현됐는지
    (b) 시퀀스 검증기를 붙이기 전의 출발점이 어디인지를 못박는 것이다.

특히 **프레임 단위 오탐률**을 따로 뽑는다. 연기가 없는 이미지에서 탐지가
몇 번 나오는지가 M3 시퀀스 검증기가 줄여야 할 바로 그 숫자이기 때문이다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from firstlight.detect.detector import Detection, Detector


@dataclass
class PRPoint:
    confidence: float
    precision: float
    recall: float
    f1: float


@dataclass
class DetectBaselineResult:
    n_images: int
    n_images_with_smoke: int
    n_images_without_smoke: int
    n_ground_truth: int
    n_predictions: int
    ap50: float
    best: PRPoint
    curve: list[PRPoint] = field(default_factory=list)
    # 연기가 없는 이미지에서 나온 탐지 수 / 그런 이미지 수.
    # 시퀀스 검증기가 공격해야 할 숫자다.
    false_positives_per_negative_image: float = 0.0
    negative_image_alarm_rate: float = 0.0      # 한 번이라도 탐지가 뜬 음성 이미지 비율
    median_latency_ms: float = float("nan")
    device: str = "cpu"
    iou_threshold: float = 0.5


def load_yolo_labels(label_path: Path, width: int, height: int) -> list[tuple[float, ...]]:
    """YOLO 정규화 라벨 → 픽셀 (x1, y1, x2, y2)."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = (float(p) for p in parts[:5])
        boxes.append(
            (
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            )
        )
    return boxes


def _iou(pred: Detection, gt: tuple[float, ...]) -> float:
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(pred.x1, gx1), max(pred.y1, gy1)
    ix2, iy2 = min(pred.x2, gx2), min(pred.y2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pred.area + max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1) - inter
    return inter / union if union > 0 else 0.0


def _match(
    preds: list[Detection], gts: list[tuple[float, ...]], iou_threshold: float
) -> list[bool]:
    """신뢰도 내림차순 탐욕 매칭. 각 예측이 TP 인지."""
    taken = [False] * len(gts)
    flags = []
    for pred in sorted(preds, key=lambda d: d.confidence, reverse=True):
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gts):
            if taken[i]:
                continue
            value = _iou(pred, gt)
            if value > best_iou:
                best_iou, best_idx = value, i
        if best_iou >= iou_threshold and best_idx >= 0:
            taken[best_idx] = True
            flags.append(True)
        else:
            flags.append(False)
    return flags


def _average_precision(confidences: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """전점 보간 AP (PASCAL VOC 2010 이후 방식)."""
    if n_gt == 0 or confidences.size == 0:
        return 0.0
    order = np.argsort(-confidences)
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])

    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)

    # precision 을 우측에서 단조 감소하도록 포락선을 씌운다.
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def run_baseline(
    detector: Detector,
    manifest_path: Path,
    iou_threshold: float = 0.5,
    limit: int | None = None,
    device: str = "cpu",
    progress=None,
) -> DetectBaselineResult:
    """검증셋 전체를 돌려 PR 곡선을 만든다.

    탐지기의 conf 임계값은 낮게 두고 호출해야 한다. 곡선을 그리려면
    낮은 신뢰도 예측도 있어야 하기 때문이다.
    """
    import time

    import cv2

    root = manifest_path.parent
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if limit is not None:
        records = records[:limit]

    all_conf: list[float] = []
    all_tp: list[bool] = []
    n_gt_total = 0
    n_pred_total = 0
    n_with = n_without = 0
    fp_on_negatives = 0
    negatives_with_any_alarm = 0
    latencies: list[float] = []

    for i, rec in enumerate(records):
        image = cv2.imread(str(root / rec["image"]))
        if image is None:
            continue
        h, w = image.shape[:2]
        gts = load_yolo_labels(root / rec["label"], w, h)

        start = time.perf_counter()
        preds = detector.detect(image)
        latencies.append((time.perf_counter() - start) * 1000.0)

        n_gt_total += len(gts)
        n_pred_total += len(preds)
        if gts:
            n_with += 1
        else:
            n_without += 1
            fp_on_negatives += len(preds)
            if preds:
                negatives_with_any_alarm += 1

        flags = _match(preds, gts, iou_threshold)
        for pred, flag in zip(sorted(preds, key=lambda d: d.confidence, reverse=True), flags):
            all_conf.append(pred.confidence)
            all_tp.append(flag)

        if progress is not None:
            progress(i + 1, len(records))

    conf_arr = np.array(all_conf, dtype=float)
    tp_arr = np.array(all_tp, dtype=bool)
    ap50 = _average_precision(conf_arr, tp_arr, n_gt_total)

    curve = _build_curve(conf_arr, tp_arr, n_gt_total)
    best = max(curve, key=lambda p: p.f1) if curve else PRPoint(0.0, 0.0, 0.0, 0.0)

    return DetectBaselineResult(
        n_images=len(records),
        n_images_with_smoke=n_with,
        n_images_without_smoke=n_without,
        n_ground_truth=n_gt_total,
        n_predictions=n_pred_total,
        ap50=ap50,
        best=best,
        curve=curve,
        false_positives_per_negative_image=(
            fp_on_negatives / n_without if n_without else 0.0
        ),
        negative_image_alarm_rate=(
            negatives_with_any_alarm / n_without if n_without else 0.0
        ),
        median_latency_ms=float(np.median(latencies)) if latencies else float("nan"),
        device=device,
        iou_threshold=iou_threshold,
    )


def _build_curve(
    confidences: np.ndarray, is_tp: np.ndarray, n_gt: int,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
                                     0.50, 0.60, 0.70, 0.80, 0.90),
) -> list[PRPoint]:
    points = []
    for thr in thresholds:
        keep = confidences >= thr
        tp = int(is_tp[keep].sum())
        fp = int((~is_tp[keep]).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_gt if n_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        points.append(PRPoint(thr, precision, recall, f1))
    return points


def save_json(result: DetectBaselineResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
