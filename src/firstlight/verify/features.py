"""시간축 특징 — 연기와 오탐원을 가르는 물리량.

각 특징이 어떤 오탐을 겨냥하는지:

    특징                    연기          안개        구름그림자      반사광
    ----------------------  ------------  ----------  -------------  --------
    area_growth_rate        증가          0           0              0
    centroid_rise_rate      상승          0           수평           0
    persistence             높음          높음        중간           산발
    intensity_flicker       있음          없음        중간           강함
    flow_divergence*        큼            0           0 (평행이동)   —
    flow_upward_ratio*      높음          —           낮음           —
    (* dense 모드에서만)

두 가지 모드:
    DENSE  — 프레임 간격 ≤ 2초. 광류를 쓸 수 있다.
    SPARSE — 간격이 길다 (FIgLib 은 60초). 기하·측광 특징만 쓴다.

모드를 나눈 이유는 타협이 아니라 필요다. 60초 간격에서 광류를 계산하면
숫자는 나오지만 의미가 없다 — 그 사이 장면이 통째로 바뀌기 때문이다.
없는 정보를 있는 척하느니 특징을 줄이고 그 사실을 명시하는 편이 낫다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from firstlight.verify.tracker import Track

# 이 간격 이하면 광류가 의미 있다고 본다.
#
# 작품설명서는 "후속 12프레임(약 0.5~1초)에 걸쳐 검증"이라고 적었다.
# 12프레임이 0.5~1초면 간격은 0.04~0.08초, 즉 12~25fps 다. 여유를 두어
# 0.2초(5fps)까지를 dense 로 본다 — 그 이상 벌어지면 프레임 간 장면이
# 너무 달라져 광류가 의미를 잃는다.
#
# (초안은 2.0초였는데, 그러면 12프레임이 24초가 되어 명세의 "약 0.5~1초"와
#  전혀 맞지 않았다.)
DENSE_MAX_INTERVAL_S = 0.2


class VerifierMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"

    @classmethod
    def from_interval(cls, median_interval_s: float) -> VerifierMode:
        return cls.DENSE if median_interval_s <= DENSE_MAX_INTERVAL_S else cls.SPARSE


@dataclass
class TrackFeatures:
    """트랙 하나의 특징 벡터.

    dense 전용 특징은 sparse 모드에서 0.0 이다. 스코어러가 모드별로 다른
    가중치를 쓰므로 문제되지 않는다.
    """

    # --- 공통 (기하·측광) ---
    persistence: float = 0.0            # 관측된 프레임 비율 [0,1]
    duration_s: float = 0.0             # 트랙 지속 시간
    n_observations: int = 0
    area_growth_rate: float = 0.0       # d(log 면적)/dt [1/s]
    # 면적이 **단조** 증가하는가 — 연속 구간 중 늘어난 비율 [0,1].
    # 작품설명서가 "면적은 시간에 따라 단조 증가"라고 못박았으므로,
    # 추세(성장률)만이 아니라 일관성도 따로 재야 한다.
    area_monotonicity: float = 0.0
    centroid_rise_rate: float = 0.0     # 상승 속도 / sqrt(면적) [1/s], 위가 양수
    centroid_drift_rate: float = 0.0    # 수평 이동 속도 / sqrt(면적) [1/s]
    translation_over_growth: float = 0.0  # 이동/성장 비. 구름그림자는 크다.
    aspect_growth_rate: float = 0.0     # d(log(h/w))/dt [1/s]
    intensity_flicker: float = 0.0      # 박스 내 평균밝기의 시간축 변동계수
    edge_softness: float = 0.0          # 경계 흐림 정도 [0,1]
    confidence_mean: float = 0.0
    confidence_slope: float = 0.0       # [1/s]

    # --- dense 전용 (광류) ---
    flow_divergence: float = 0.0        # 자기운동 보정 후 발산 (팽창)
    flow_upward_ratio: float = 0.0      # 위를 향하는 벡터 비율 [0,1]
    flow_translation_mag: float = 0.0   # 잔차 평행이동 크기

    mode: VerifierMode = VerifierMode.SPARSE

    def as_vector(self, names: list[str]) -> np.ndarray:
        data = asdict(self)
        return np.array([float(data[n]) for n in names], dtype=float)


# 모드별로 스코어러가 쓰는 특징 이름. 순서가 곧 가중치 순서다.
SPARSE_FEATURES = [
    "persistence",
    "area_growth_rate",
    "area_monotonicity",
    "centroid_rise_rate",
    "translation_over_growth",
    "aspect_growth_rate",
    "intensity_flicker",
    "edge_softness",
    "confidence_mean",
    "confidence_slope",
]

DENSE_FEATURES = SPARSE_FEATURES + [
    "flow_divergence",
    "flow_upward_ratio",
    "flow_translation_mag",
]


def feature_names(mode: VerifierMode) -> list[str]:
    return DENSE_FEATURES if mode is VerifierMode.DENSE else SPARSE_FEATURES


# --------------------------------------------------------------------------


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    """최소제곱 기울기. 표본이 부족하거나 시간이 안 변하면 0."""
    if t.size < 2:
        return 0.0
    t = t - t.mean()
    denom = float((t * t).sum())
    if denom < 1e-12:
        return 0.0
    return float((t * (y - y.mean())).sum() / denom)


def extract_features(
    track: Track,
    frame_interval_s: float,
    mode: VerifierMode | None = None,
) -> TrackFeatures:
    """트랙에서 특징을 뽑는다.

    Args:
        track: 대상 트랙.
        frame_interval_s: 프레임 간격 중앙값. persistence 분모와 모드 결정에 쓴다.
    """
    mode = mode or VerifierMode.from_interval(frame_interval_s)
    feats = TrackFeatures(mode=mode, n_observations=track.hits)

    if track.hits == 0:
        return feats

    times = track.times()
    areas = np.maximum(track.areas(), 1.0)
    centroids = track.centroids()
    confs = track.confidences()

    feats.duration_s = track.duration_s
    feats.confidence_mean = float(confs.mean())

    # persistence: 관측 구간에서 실제로 몇 번 잡혔는가.
    # 놓친 프레임이 많다는 것은 산발적 반사광일 가능성이 높다는 뜻이다.
    expected = max(1.0, track.duration_s / max(frame_interval_s, 1e-6) + 1.0)
    feats.persistence = float(min(1.0, track.hits / expected))

    if track.hits < 2:
        return feats

    # 로그를 취해 크기에 무관한 성장률로 만든다. 100px²→200px² 와
    # 10000→20000 이 같은 값이 되어야 한다.
    feats.area_growth_rate = _slope(times, np.log(areas))
    feats.confidence_slope = _slope(times, confs)

    # 단조성: 연속한 관측 사이에서 면적이 늘어난 비율.
    # 노을·반사광은 면적이 그대로거나 갑자기 사라지므로 0.5 아래로 떨어진다.
    steps = np.diff(areas)
    feats.area_monotonicity = float((steps > 0).mean()) if steps.size else 0.0

    # 이미지 y 는 아래가 양수이므로 상승은 y 감소다.
    scale = float(np.sqrt(areas).mean())
    rise_px_s = -_slope(times, centroids[:, 1])
    drift_px_s = abs(_slope(times, centroids[:, 0]))
    feats.centroid_rise_rate = rise_px_s / max(scale, 1.0)
    feats.centroid_drift_rate = drift_px_s / max(scale, 1.0)

    # 구름그림자를 겨냥한 핵심 비율: 커지지 않고 움직이기만 하면 크다.
    speed = float(np.hypot(rise_px_s, drift_px_s)) / max(scale, 1.0)
    feats.translation_over_growth = speed / (abs(feats.area_growth_rate) + 1e-3)

    heights = np.maximum(np.array([o.detection.height for o in track.observations]), 1.0)
    widths = np.maximum(np.array([o.detection.width for o in track.observations]), 1.0)
    feats.aspect_growth_rate = _slope(times, np.log(heights / widths))

    _appearance_features(track, feats)
    if mode is VerifierMode.DENSE:
        _flow_features(track, feats)

    return feats


def _appearance_features(track: Track, feats: TrackFeatures) -> None:
    """패치 기반 측광 특징. 패치가 없으면 건드리지 않는다."""
    patches = [o.patch for o in track.observations if o.patch is not None]
    if not patches:
        return

    means = np.array([float(p.mean()) for p in patches])
    if means.size >= 2 and means.mean() > 1e-6:
        # 변동계수 — 안개는 밝기가 거의 변하지 않는다.
        feats.intensity_flicker = float(means.std() / means.mean())

    # 경계 흐림: **가장 급한 전이가 얼마나 급한가**를 잰다.
    #
    # 계단 에지(건물·비닐하우스)는 전체 명암차를 픽셀 한두 개 만에 넘긴다.
    # 연기 경계는 같은 명암차를 수십 픽셀에 걸쳐 넘긴다. 따라서 최대
    # 그래디언트를 명암 범위로 나누면 전자는 ~1, 후자는 ~1/폭 이 된다.
    #
    # (초안은 "강한 그래디언트를 가진 픽셀의 비율"을 썼는데 부호가 반대로
    #  나왔다 — 균일한 완만 기울기는 모든 픽셀이 문턱을 넘고, 국소 계단은
    #  소수만 넘기 때문이다. 그건 흐림이 아니라 그래디언트 집중도다.)
    softness = []
    for patch in patches[-5:]:
        if patch.ndim != 2 or min(patch.shape) < 4:
            continue
        arr = patch.astype(np.float32)
        contrast = float(arr.max() - arr.min())
        if contrast < 1e-6:
            softness.append(1.0)              # 평평하면 에지가 없다
            continue
        gy, gx = np.gradient(arr)
        magnitude = np.hypot(gx, gy)
        # 최대 대신 99퍼센타일 — 잡음 픽셀 하나에 휘둘리지 않게.
        steepest = float(np.percentile(magnitude, 99)) / contrast
        softness.append(float(np.clip(1.0 - steepest, 0.0, 1.0)))
    if softness:
        feats.edge_softness = float(np.mean(softness))


def _flow_features(track: Track, feats: TrackFeatures) -> None:
    """자기운동 보정 후 잔차 광류 통계를 집계한다."""
    flows = [o.flow for o in track.observations if o.flow]
    if not flows:
        return
    feats.flow_divergence = float(np.mean([f.get("divergence", 0.0) for f in flows]))
    feats.flow_upward_ratio = float(np.mean([f.get("upward_ratio", 0.0) for f in flows]))
    feats.flow_translation_mag = float(
        np.mean([f.get("translation_mag", 0.0) for f in flows])
    )
