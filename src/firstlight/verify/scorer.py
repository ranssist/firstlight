"""검증 점수 → 3단계 등급.

**로지스틱 회귀를 쓴다.** 신경망이 아니다. 이유가 셋 있다:

1. 이 노트북에 NVIDIA GPU 가 없다. 로지스틱 회귀는 CPU 에서 1초 안에
   재학습된다 — 소개서 §4③ 의 "쓸수록 오탐이 줄어드는 구조"가 실제로
   돌아가는 유일한 현실적 선택지다.
2. 계수를 사람이 읽을 수 있다. 왜 GLOW 로 떨어졌는지 관제 요원에게
   설명할 수 있어야 하는 시스템이다.
3. 특징이 10여 개, 라벨이 수백 개 규모다. 이 데이터에서 큰 모델은
   과적합만 한다.

기본 가중치는 **사전값(prior)일 뿐 검증된 값이 아니다.** 물리적 방향
(연기는 커지고 오르고, 구름그림자는 옮겨가기만 한다)만 반영해 놓았고,
라벨이 모이면 `fit()` 이 통째로 갈아엎는다. `is_fitted` 로 지금 어느
쪽인지 항상 확인할 수 있게 했다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from firstlight.verify.features import TrackFeatures, VerifierMode, feature_names


class Tier(str, Enum):
    """작품설명서 표 3-1 「FIRSTLIGHT 경보 등급 체계」."""

    FLARE = "FLARE"      # 타오름 — 자동 경보 + 담당자 푸시 + 좌표 확정
    GLOW = "GLOW"        # 어른거림 — 대시보드 큐에 적재, 사람이 확인
    SPARK = "SPARK"      # 스침 — 알림 없이 로그만, 재학습 데이터로 환류

    @property
    def colour(self) -> str:
        return {"FLARE": "#E63946", "GLOW": "#F4A261", "SPARK": "#8D99AE"}[self.value]

    @property
    def label_ko(self) -> str:
        return {"FLARE": "타오름", "GLOW": "어른거림", "SPARK": "스침"}[self.value]


# 특징별 (중심, 척도). z = (x - 중심)/척도 가 O(1) 이 되도록 잡은 값이다.
# 성장률 계열이 1e-3 인 것은 60초 간격 데이터 기준으로, 40분에 걸쳐
# 면적이 10배 되면 log(10)/2400 ≈ 1e-3 /s 이기 때문이다.
DEFAULT_SCALES: dict[str, tuple[float, float]] = {
    "persistence": (0.5, 0.3),
    "area_growth_rate": (0.0, 1e-3),
    "area_monotonicity": (0.5, 0.25),
    "centroid_rise_rate": (0.0, 1e-3),
    "translation_over_growth": (0.0, 50.0),
    "aspect_growth_rate": (0.0, 1e-3),
    "intensity_flicker": (0.05, 0.05),
    "edge_softness": (0.5, 0.2),
    "confidence_mean": (0.4, 0.2),
    "confidence_slope": (0.0, 1e-3),
    "flow_divergence": (0.0, 0.5),
    "flow_upward_ratio": (0.5, 0.25),
    "flow_translation_mag": (0.0, 2.0),
}

# 물리적 방향만 담은 사전 가중치. 부호가 핵심이고 크기는 임의다.
DEFAULT_WEIGHTS: dict[str, float] = {
    "persistence": 1.2,               # 산발적이면 반사광
    "area_growth_rate": 1.0,          # 연기는 커진다
    "area_monotonicity": 0.8,         # 꾸준히 커진다 (노을은 그대로거나 사라진다)
    "centroid_rise_rate": 0.8,        # 연기는 오른다
    "translation_over_growth": -0.7,  # 커지지 않고 옮겨만 가면 구름그림자
    "aspect_growth_rate": 0.3,        # 기둥은 세로로 길어진다
    "intensity_flicker": 0.4,         # 안개는 정적이다
    "edge_softness": 0.3,             # 연기 경계는 흐리다
    "confidence_mean": 0.8,
    "confidence_slope": 0.3,
    "flow_divergence": 0.9,           # 팽창 (연기) vs 평행이동 (그림자)
    "flow_upward_ratio": 0.6,
    "flow_translation_mag": -0.4,
}
DEFAULT_INTERCEPT = -0.5

# FLARE 를 주기 위한 최소 관측 수.
#
# 왜 필요한가: 관측 2개짜리 트랙에서도 최소제곱 기울기는 계산된다. 그런데
# 두 점을 지나는 직선의 기울기는 **추정이 아니라 그냥 두 점의 차이**다.
# 잡음이 그대로 성장률로 들어가고, 척도가 1e-3 이라 곧바로 시그모이드를
# 포화시킨다. 실제로 대시보드에서 관측 2회짜리 발화 전 트랙이 0.96 점으로
# FLARE 를 받는 것이 관찰됐다 — 명백한 오경보다.
#
# 시간축 검증을 표방하면서 표본 2개로 판정하는 것은 앞뒤가 맞지 않는다.
# 소개서 §3 시나리오도 "후속 12프레임 시퀀스 검증"이라고 적고 있다.
# 이 문턱에 못 미치면 점수와 무관하게 GLOW 로 묶어 사람이 보게 한다.
DEFAULT_MIN_OBSERVATIONS = {
    VerifierMode.DENSE: 12,     # 소개서의 12프레임
    VerifierMode.SPARSE: 3,     # 60초 간격이면 3회 = 3분간의 지속
}


@dataclass
class ScoreBreakdown:
    """왜 이 점수가 나왔는지. 관제 화면과 디버깅에 쓴다."""

    score: float
    contributions: dict[str, float]

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.contributions.items(), key=lambda kv: -abs(kv[1]))[:n]


class SequenceScorer:
    """특징 → [0,1] 점수 → 등급."""

    def __init__(
        self,
        mode: VerifierMode = VerifierMode.SPARSE,
        weights: dict[str, float] | None = None,
        intercept: float = DEFAULT_INTERCEPT,
        scales: dict[str, tuple[float, float]] | None = None,
        tau_high: float = 0.70,
        tau_low: float = 0.35,
        is_fitted: bool = False,
        n_train: int = 0,
        min_observations_for_flare: int | None = None,
    ) -> None:
        self.mode = mode
        self.names = feature_names(mode)
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.intercept = intercept
        self.scales = dict(scales or DEFAULT_SCALES)
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.is_fitted = is_fitted
        self.n_train = n_train
        self.min_observations_for_flare = (
            min_observations_for_flare
            if min_observations_for_flare is not None
            else DEFAULT_MIN_OBSERVATIONS[mode]
        )

    # ------------------------------------------------------------- 점수

    def _standardise(self, feats: TrackFeatures) -> np.ndarray:
        raw = feats.as_vector(self.names)
        centre = np.array([self.scales[n][0] for n in self.names])
        scale = np.array([self.scales[n][1] for n in self.names])
        return (raw - centre) / np.where(np.abs(scale) < 1e-12, 1.0, scale)

    def score(self, feats: TrackFeatures) -> float:
        z = self._standardise(feats)
        w = np.array([self.weights.get(n, 0.0) for n in self.names])
        logit = float(np.dot(w, z) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))))

    def explain(self, feats: TrackFeatures) -> ScoreBreakdown:
        z = self._standardise(feats)
        contributions = {
            n: float(self.weights.get(n, 0.0) * z[i]) for i, n in enumerate(self.names)
        }
        return ScoreBreakdown(self.score(feats), contributions)

    # ------------------------------------------------------------- 등급

    def tier(
        self, score: float, geo_ok: bool = True, n_observations: int | None = None
    ) -> Tier:
        """**보조 경로** — 점수만으로 등급을 추정한다.

        운용 등급은 `tier_from_criteria()` 가 정한다. 이 함수는 시퀀스
        검증을 돌리지 않은 맥락(단일 프레임 비교 실험 등)에서만 쓴다.
        """
        if score >= self.tau_high:
            enough = (
                n_observations is None
                or n_observations >= self.min_observations_for_flare
            )
            return Tier.FLARE if (geo_ok and enough) else Tier.GLOW
        if score >= self.tau_low:
            return Tier.GLOW
        return Tier.SPARK

    # ------------------------------------------------------- 운용 등급 판정

    def tier_from_criteria(
        self, criteria, geo_ok: bool = True
    ) -> Tier:
        """작품설명서의 판정 규칙 — **등급은 만족한 조건 수로 정해진다.**

            두 조건을 동시에 만족하면 FLARE,
            하나만 만족하면 GLOW,
            둘 다 미달이면 SPARK.
                                    — 작품설명서 Ⅱ-1 「시퀀스 기반 연기 탐지」

        여기에 두 개의 강등 조건이 더 붙는다. 둘 다 명세에서 유도된 것이다:

        - **프레임 부족** — 명세가 "후속 12프레임 시퀀스 검증"이라고 못박았다.
          그만큼 못 봤으면 시퀀스 검증을 한 것이 아니므로 FLARE 를 줄 수 없다.
        - **좌표 미발행** — 표 3-1 의 FLARE 동작이 "자동 경보 · 담당자 푸시 ·
          **좌표 확정**"이다. 확정할 좌표가 없으면 FLARE 의 정의를 채우지 못한다.

        강등은 언제나 GLOW 까지다. 조건을 통과한 후보를 버리지 않고 사람에게
        넘긴다 — 안전 시스템은 놓치는 쪽이 더 위험하기 때문이다.
        """
        satisfied = criteria.n_satisfied

        if satisfied >= 2:
            if not criteria.enough_frames or not geo_ok:
                return Tier.GLOW
            return Tier.FLARE
        if satisfied == 1:
            return Tier.GLOW
        return Tier.SPARK

    # ------------------------------------------------------------- 학습

    def fit(self, features: list[TrackFeatures], labels: list[int]) -> SequenceScorer:
        """라벨로 가중치를 다시 맞춘다. CPU 에서 1초 이내.

        음성이 압도적으로 많으므로 class_weight='balanced' 를 쓴다.
        """
        from sklearn.linear_model import LogisticRegression

        if len(features) != len(labels):
            raise ValueError("특징과 라벨 개수가 다르다")
        if len(set(labels)) < 2:
            raise ValueError("양성과 음성이 모두 있어야 학습할 수 있다")

        X = np.vstack([self._standardise(f) for f in features])
        y = np.array(labels, dtype=int)

        model = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs"
        )
        model.fit(X, y)

        self.weights = {n: float(c) for n, c in zip(self.names, model.coef_[0])}
        self.intercept = float(model.intercept_[0])
        self.is_fitted = True
        self.n_train = len(labels)
        return self

    # ------------------------------------------------------------ 저장

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "weights": self.weights,
            "intercept": self.intercept,
            "scales": {k: list(v) for k, v in self.scales.items()},
            "tau_high": self.tau_high,
            "tau_low": self.tau_low,
            "is_fitted": self.is_fitted,
            "n_train": self.n_train,
            "min_observations_for_flare": self.min_observations_for_flare,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SequenceScorer:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mode=VerifierMode(raw["mode"]),
            weights=raw["weights"],
            intercept=raw["intercept"],
            scales={k: tuple(v) for k, v in raw["scales"].items()},
            tau_high=raw.get("tau_high", 0.70),
            tau_low=raw.get("tau_low", 0.35),
            is_fitted=raw.get("is_fitted", False),
            n_train=raw.get("n_train", 0),
            min_observations_for_flare=raw.get("min_observations_for_flare"),
        )

    def __repr__(self) -> str:
        state = f"학습됨(n={self.n_train})" if self.is_fitted else "사전값(미학습)"
        return (
            f"SequenceScorer(mode={self.mode.value}, {state}, "
            f"τ={self.tau_low}/{self.tau_high}, 최소관측={self.min_observations_for_flare})"
        )
