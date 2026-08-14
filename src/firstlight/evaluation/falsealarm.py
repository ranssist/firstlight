"""오경보율 측정 — 소개서 §4① 의 근거.

측정 대상:
    FIgLib 시퀀스의 **발화 전 40분** 구간. 이 구간에는 산불이 없다는 것이
    데이터 자체로 보장된다. 그러면서 안개·구름그림자·노을·역광은 전부
    들어있다. 합성 음성 샘플로는 만들 수 없는 조건이다.

    60초 간격이므로 프레임 수가 곧 관측 분(分)이다. 시퀀스 16개면
    560프레임 = 9.33시간. 이것이 "시간당 오경보"의 진짜 분모다.

세 가지 정책을 나란히 잰다. 하나의 숫자만 내면 어디서 개선이 나왔는지
알 수 없기 때문이다:

    1. raw       — 탐지 하나가 곧 경보 (프레임 단위 경보)
    2. dedup     — 트랙 하나당 경보 1건 (중복 억제만, 검증 없음)
    3. verified  — 시퀀스 검증을 통과(FLARE)한 트랙만 경보

1→2 는 중복 억제의 효과, 2→3 이 **시퀀스 검증 자체의 효과**다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from firstlight.detect.detector import Detector
from firstlight.verify.criteria import evaluate_criteria
from firstlight.verify.features import VerifierMode, extract_features
from firstlight.verify.scorer import SequenceScorer, Tier
from firstlight.verify.tracker import Track, Tracker

# FIgLib 은 60초 간격이다.
FIGLIB_INTERVAL_S = 60.0


@dataclass
class SequenceOutcome:
    """시퀀스 하나의 처리 결과."""

    sequence: str
    n_negative_frames: int
    n_positive_frames: int
    raw_detections_negative: int
    tracks_negative: int
    flares_negative: int
    # --- T2 프레임 단위 오경보 (작품설명서 표 Ⅲ-2 는 %로 목표를 적었다) ---
    # 단일 프레임 판정: 탐지가 하나라도 있으면 그 프레임은 오경보.
    frames_alarmed_single: int = 0
    # 시퀀스 판정: FLARE 가 뜬 프레임만 오경보.
    frames_alarmed_sequence: int = 0
    # 발화 후 첫 FLARE 까지 걸린 시간 (초). 못 잡았으면 None.
    time_to_flare_s: float | None = None
    detected_after_ignition: bool = False
    # --- T4 End-to-End 지연 (프레임 수신 → 경보 확정) ---
    latencies_ms: list[float] = field(default_factory=list)


@dataclass
class FalseAlarmResult:
    n_sequences: int
    negative_hours: float
    positive_hours: float
    raw_per_hour: float
    dedup_per_hour: float
    verified_per_hour: float
    reduction_vs_raw: float          # 1 - verified/raw
    reduction_vs_dedup: float        # 1 - verified/dedup
    detection_rate: float            # 발화 후 FLARE 를 낸 시퀀스 비율
    median_time_to_flare_s: float | None
    scorer_fitted: bool
    conf_threshold: float
    tau_high: float

    # --- T2: 프레임 단위 오경보율 (작품설명서 표 Ⅲ-2 형식) ---
    n_negative_frames: int = 0
    fpr_single_frame: float = 0.0      # 단일 프레임 판정 오경보율 (목표 비교용 기준)
    fpr_sequence: float = 0.0          # 시퀀스 판정 오경보율 (목표 3% 이하)
    fpr_reduction: float = 0.0         # 1 - 시퀀스/단일

    # --- T4: End-to-End 지연 (목표 30초 이내) ---
    latency_median_ms: float = float("nan")
    latency_p95_ms: float = float("nan")
    latency_max_ms: float = float("nan")

    sequences: list[SequenceOutcome] = field(default_factory=list)


# --------------------------------------------------------------------------


def _patch_for(frame: np.ndarray, det, size: int = 48) -> np.ndarray | None:
    """탐지 박스에서 특징 계산용 축소 그레이스케일 패치를 뜬다."""
    import cv2

    h, w = frame.shape[:2]
    x1 = max(0, int(det.x1))
    y1 = max(0, int(det.y1))
    x2 = min(w, int(np.ceil(det.x2)))
    y2 = min(h, int(np.ceil(det.y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def process_sequence(
    frames: list[tuple[Path, int]],
    detector: Detector,
    scorer: SequenceScorer,
    tracker_factory=Tracker,
) -> tuple[SequenceOutcome, list[tuple[Track, int]]]:
    """시퀀스 하나를 시간순으로 흘려보낸다.

    실시간과 같은 조건을 만들기 위해 **온라인으로** 점수를 매긴다. 매 프레임
    시점에서 그때까지의 관측만으로 특징을 뽑는다. 전체를 다 본 뒤에 채점하면
    운용 성능을 과대평가하게 된다.

    Returns:
        (시퀀스 결과, [(트랙, 최초 관측 오프셋)]) — 뒤쪽은 학습 라벨링용.
    """
    import time

    import cv2

    tracker = tracker_factory()
    raw_negative = 0
    n_neg = n_pos = 0
    flared_tracks: set[int] = set()
    flares_negative = 0
    frames_alarmed_single = 0
    frames_alarmed_sequence = 0
    time_to_flare: float | None = None
    track_first_offset: dict[int, int] = {}
    latencies: list[float] = []

    for frame_index, (path, offset_s) in enumerate(frames):
        image = cv2.imread(str(path))
        if image is None:
            continue
        is_negative = offset_s < 0
        n_neg += int(is_negative)
        n_pos += int(not is_negative)

        # T4: 프레임을 손에 쥔 순간부터 경보 확정까지를 잰다.
        # 영상 디코딩(imread)은 뺀다 — 스트리밍에서는 별도 계층이고,
        # 파일 I/O 속도가 파이프라인 지연으로 잘못 계상되기 때문이다.
        started = time.perf_counter()

        detections = detector.detect(image)
        if is_negative:
            raw_negative += len(detections)
            # 단일 프레임 판정: 탐지가 하나라도 있으면 그 프레임은 오경보.
            if detections:
                frames_alarmed_single += 1

        patches = [_patch_for(image, d) for d in detections]
        active = tracker.update(
            detections, timestamp=float(offset_s), frame_index=frame_index, patches=patches
        )

        frame_flared = False
        for track in active:
            track_first_offset.setdefault(track.track_id, int(track.first.timestamp))
            feats = extract_features(track, FIGLIB_INTERVAL_S, mode=scorer.mode)
            criteria = evaluate_criteria(track, feats)
            # 이 평가에서는 지오레퍼런싱을 붙이지 않으므로 geo_ok=True 로 둔다.
            # 실제 운용에서는 좌표 실패가 FLARE 를 GLOW 로 강등시키므로
            # 여기 수치는 오경보 측면에서 **보수적**(더 나쁜 쪽)이다.
            is_flare = scorer.tier_from_criteria(criteria, geo_ok=True) is Tier.FLARE
            if is_flare:
                frame_flared = True
            if is_flare and track.track_id not in flared_tracks:
                flared_tracks.add(track.track_id)
                if is_negative:
                    flares_negative += 1
                elif time_to_flare is None:
                    time_to_flare = float(offset_s)

        latencies.append((time.perf_counter() - started) * 1000.0)
        if is_negative and frame_flared:
            frames_alarmed_sequence += 1

    all_tracks = tracker.close()
    negative_tracks = sum(
        1 for t in all_tracks if track_first_offset.get(t.track_id, 0) < 0
    )

    outcome = SequenceOutcome(
        sequence=frames[0][0].parent.name if frames else "?",
        n_negative_frames=n_neg,
        n_positive_frames=n_pos,
        raw_detections_negative=raw_negative,
        tracks_negative=negative_tracks,
        flares_negative=flares_negative,
        frames_alarmed_single=frames_alarmed_single,
        frames_alarmed_sequence=frames_alarmed_sequence,
        time_to_flare_s=time_to_flare,
        detected_after_ignition=time_to_flare is not None,
        latencies_ms=latencies,
    )
    labelled = [(t, track_first_offset.get(t.track_id, 0)) for t in all_tracks]
    return outcome, labelled


def load_manifest(manifest_path: Path) -> list[tuple[str, list[tuple[Path, int]]]]:
    """매니페스트 → [(시퀀스명, [(경로, 오프셋초)])], 시간 오름차순."""
    root = manifest_path.parent
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = []
    for entry in data:
        seq = entry["sequence"]
        frames = sorted(
            ((root / seq / f["file"], int(f["offset_s"])) for f in entry["frames"]),
            key=lambda t: t[1],
        )
        frames = [(p, o) for p, o in frames if p.exists()]
        if frames:
            out.append((seq, frames))
    return out


def run_false_alarm(
    detector: Detector,
    manifest_path: Path,
    scorer: SequenceScorer | None = None,
    limit: int | None = None,
    only: set[str] | None = None,
    progress=None,
) -> tuple[FalseAlarmResult, list[tuple[str, Track, int]]]:
    """전 시퀀스를 돌려 세 정책의 오경보율을 비교한다.

    Args:
        only: 지정하면 이 시퀀스들만 평가한다 (홀드아웃 평가용).
    """
    scorer = scorer or SequenceScorer(mode=VerifierMode.SPARSE)
    sequences = load_manifest(manifest_path)
    if limit is not None:
        sequences = sequences[:limit]
    if only is not None:
        sequences = [(s, f) for s, f in sequences if s in only]

    outcomes: list[SequenceOutcome] = []
    all_labelled: list[tuple[str, Track, int]] = []

    for i, (seq, frames) in enumerate(sequences):
        outcome, labelled = process_sequence(frames, detector, scorer)
        outcome.sequence = seq
        outcomes.append(outcome)
        all_labelled.extend((seq, t, off) for t, off in labelled)
        if progress is not None:
            progress(i + 1, len(sequences), seq)

    neg_frames = sum(o.n_negative_frames for o in outcomes)
    pos_frames = sum(o.n_positive_frames for o in outcomes)
    neg_hours = neg_frames * FIGLIB_INTERVAL_S / 3600.0
    pos_hours = pos_frames * FIGLIB_INTERVAL_S / 3600.0

    raw = sum(o.raw_detections_negative for o in outcomes)
    dedup = sum(o.tracks_negative for o in outcomes)
    verified = sum(o.flares_negative for o in outcomes)

    raw_ph = raw / neg_hours if neg_hours else 0.0
    dedup_ph = dedup / neg_hours if neg_hours else 0.0
    verified_ph = verified / neg_hours if neg_hours else 0.0

    times = [o.time_to_flare_s for o in outcomes if o.time_to_flare_s is not None]

    # T2: 프레임 단위 오경보율 (작품설명서 표 Ⅲ-2 는 %로 목표를 적었다).
    alarmed_single = sum(o.frames_alarmed_single for o in outcomes)
    alarmed_sequence = sum(o.frames_alarmed_sequence for o in outcomes)
    fpr_single = alarmed_single / neg_frames if neg_frames else 0.0
    fpr_sequence = alarmed_sequence / neg_frames if neg_frames else 0.0

    # T4: End-to-End 지연.
    all_latencies = np.concatenate(
        [np.asarray(o.latencies_ms) for o in outcomes if o.latencies_ms]
    ) if any(o.latencies_ms for o in outcomes) else np.array([])

    result = FalseAlarmResult(
        n_sequences=len(outcomes),
        negative_hours=neg_hours,
        positive_hours=pos_hours,
        raw_per_hour=raw_ph,
        dedup_per_hour=dedup_ph,
        verified_per_hour=verified_ph,
        reduction_vs_raw=1.0 - verified_ph / raw_ph if raw_ph else 0.0,
        reduction_vs_dedup=1.0 - verified_ph / dedup_ph if dedup_ph else 0.0,
        detection_rate=(
            sum(o.detected_after_ignition for o in outcomes) / len(outcomes)
            if outcomes
            else 0.0
        ),
        median_time_to_flare_s=float(np.median(times)) if times else None,
        scorer_fitted=scorer.is_fitted,
        conf_threshold=getattr(detector, "conf_threshold", float("nan")),
        tau_high=scorer.tau_high,
        n_negative_frames=neg_frames,
        fpr_single_frame=fpr_single,
        fpr_sequence=fpr_sequence,
        fpr_reduction=1.0 - fpr_sequence / fpr_single if fpr_single else 0.0,
        latency_median_ms=(
            float(np.median(all_latencies)) if all_latencies.size else float("nan")
        ),
        latency_p95_ms=(
            float(np.percentile(all_latencies, 95)) if all_latencies.size else float("nan")
        ),
        latency_max_ms=(
            float(all_latencies.max()) if all_latencies.size else float("nan")
        ),
        sequences=outcomes,
    )
    return result, all_labelled


# ------------------------------------------------------------------ 학습


def build_training_set(
    labelled: list[tuple[str, Track, int]],
    mode: VerifierMode = VerifierMode.SPARSE,
    min_observations: int = 2,
    sequences: set[str] | None = None,
) -> tuple[list, list[int]]:
    """트랙에 라벨을 붙여 학습셋을 만든다.

    라벨링 규칙 — **트랙이 아니라 구간에 라벨을 붙인다**:

        음성(0) — 트랙의 **발화 전 구간**. 그 시각에 산불이 없었다는 것은
                  데이터가 보장하므로, 그 구간에서 잡힌 것은 정의상 오탐이다.
                  트랙이 발화 후까지 이어지더라도 발화 전 구간은 그대로
                  음성으로 쓴다.
        양성(1) — **발화 이후에 처음 나타난** 트랙의 발화 후 구간. 없던
                  자리에 새로 생긴 것이므로 연기일 가능성이 높다.

    초안은 "발화를 걸치는 트랙"을 통째로 버렸는데, 그러면 **발화 전부터
    계속 잡히던 안개·구름이 전부 사라진다.** 실제로 그렇게 했더니 음성이
    3개밖에 남지 않아 계수 9개를 맞출 수 없었다 (계수 하나가 정확히 0 으로
    죽고 다른 하나는 부호가 뒤집혔다). 오래 지속되는 오탐이야말로 가장
    중요한 학습 대상인데 그걸 버리고 있었던 셈이다.

    양성 라벨에는 여전히 잡음이 있다 — FIgLib 본 아카이브에 박스 주석이
    없어 "발화 후 새로 생긴 탐지"를 연기의 대용으로 쓰기 때문이다.
    반면 **음성 라벨은 깨끗하다.** 오경보율 측정이 이 평가의 주된 결과이고
    학습이 부수적인 이유가 여기 있다.
    """
    features, labels = [], []
    for seq, track, first_offset in labelled:
        if sequences is not None and seq not in sequences:
            continue

        pre = [o for o in track.observations if o.timestamp < 0]
        post = [o for o in track.observations if o.timestamp >= 0]

        if len(pre) >= min_observations:
            features.append(
                extract_features(Track(track.track_id, pre), FIGLIB_INTERVAL_S, mode=mode)
            )
            labels.append(0)

        if first_offset >= 0 and len(post) >= min_observations:
            features.append(
                extract_features(Track(track.track_id, post), FIGLIB_INTERVAL_S, mode=mode)
            )
            labels.append(1)

    return features, labels


def save_json(result: FalseAlarmResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False),
                    encoding="utf-8")
