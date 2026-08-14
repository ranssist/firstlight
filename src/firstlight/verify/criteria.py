"""시퀀스 검증 2조건 — 작품설명서 Ⅱ-1 「시퀀스 기반 연기 탐지」의 판정 규칙.

명세 원문:

    단일 프레임 탐지 모델(YOLO 계열)이 연기 후보를 검출하면, 후속 n프레임
    (약 0.5~1초)에 걸쳐 다음 두 가지를 검증한다.

    광류(Optical Flow) : 연기는 상승/확산 운동을 보인다. 안개는 정지하거나
        수평 이동하고, 구름 그림자는 기하학적으로 평행 이동하므로, 프레임 간
        움직임 벡터의 방향/크기 분포를 분석하면 이 패턴 차이를 정량화할 수 있다.
    마스크 면적 변화율 : 연기 마스크의 면적은 시간에 따라 단조 증가해, 노을이나
        반사광은 면적이 변하지 않거나 갑자기 사라진다.

    **두 조건을 동시에 만족하면 FLARE, 하나만 만족하면 GLOW, 둘 다 미달이면
    SPARK로 분류한다.**

이 모듈이 그 규칙 자체다. 등급은 점수 임계값이 아니라 **조건 몇 개를
만족했는가**로 정해진다 — 관제 요원에게 "왜 GLOW냐"고 물었을 때
"둘 중 면적 조건만 통과했습니다"라고 답할 수 있어야 하기 때문이다.

로지스틱 스코어러(`scorer.py`)는 이 판정을 대체하지 않는다. 같은 등급
안에서 우선순위를 매기고 재학습 환류를 담당하는 보조 신호로 남는다.

희소 프레임(FIgLib 60초 간격 등)에서는 광류를 계산할 수 없다. 그때는
광류 조건을 **기하 대용값**(자기운동 보정 후 중심 상승 속도)으로 판정하고
`flow_from_proxy=True` 로 표시한다. 없는 정보를 있는 척하지 않되,
판정 자체를 포기하지도 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from firstlight.verify.features import TrackFeatures, VerifierMode
from firstlight.verify.tracker import Track


@dataclass(frozen=True)
class CriteriaConfig:
    """두 조건의 통과 문턱.

    기본값은 명세의 물리적 서술("상승/확산", "단조 증가")을 수치화한 것이며
    현장 데이터로 재조정할 대상이다.
    """

    # --- 광류 조건 ---
    # 마스크 내부 벡터 중 위를 향하는 비율. 0.5 = 방향이 무작위.
    min_flow_upward_ratio: float = 0.55
    # 발산(팽창). 평행이동만 하는 구름그림자는 0 근처다.
    min_flow_divergence: float = 0.02
    # 희소 모드 대용값: sqrt(면적)로 정규화한 상승 속도 [1/s].
    min_rise_rate_proxy: float = 1e-4

    # --- 면적 조건 ---
    # d(log 면적)/dt [1/s]. 양수면 커지고 있다는 뜻.
    min_area_growth_rate: float = 1e-4
    # 단조성: 연속 구간 중 면적이 증가한 비율. 명세의 "단조 증가".
    min_area_monotonicity: float = 0.55

    # --- 공통 ---
    # 명세 「후속 12프레임 시퀀스 검증」. 이보다 적으면 판정 자체를 보류한다.
    min_observations: int = 12
    # 희소 모드에서는 12프레임이 12분이라 현실적이지 않다.
    min_observations_sparse: int = 3

    def observations_required(self, mode: VerifierMode) -> int:
        return (
            self.min_observations
            if mode is VerifierMode.DENSE
            else self.min_observations_sparse
        )


@dataclass
class SequenceCriteria:
    """한 트랙에 대한 2조건 판정 결과."""

    flow_ok: bool
    area_ok: bool
    enough_frames: bool
    mode: VerifierMode
    # 광류를 직접 재지 못해 기하 대용값으로 판정했는가.
    flow_from_proxy: bool = False

    # 판정 근거 수치 (대시보드에 그대로 보여준다).
    flow_upward_ratio: float = 0.0
    flow_divergence: float = 0.0
    rise_rate: float = 0.0
    area_growth_rate: float = 0.0
    area_monotonicity: float = 0.0
    n_observations: int = 0

    @property
    def n_satisfied(self) -> int:
        """만족한 조건 수. 등급이 이 값에서 나온다."""
        return int(self.flow_ok) + int(self.area_ok)

    def reasons(self) -> list[str]:
        """왜 이 판정이 나왔는지 사람 말로."""
        out = []
        if not self.enough_frames:
            out.append(
                f"관측 {self.n_observations}회 — 시퀀스 검증에 필요한 프레임 부족"
            )
        if self.flow_ok:
            out.append(
                "상승·확산 운동 확인"
                + (" (기하 대용값)" if self.flow_from_proxy else "")
            )
        else:
            out.append(
                "상승·확산 운동 미확인"
                + (" (기하 대용값)" if self.flow_from_proxy else "")
            )
        out.append("면적 단조 증가 확인" if self.area_ok else "면적 증가 미확인")
        return out


def evaluate_criteria(
    track: Track,
    features: TrackFeatures,
    config: CriteriaConfig | None = None,
) -> SequenceCriteria:
    """트랙 하나에 명세의 두 조건을 적용한다."""
    config = config or CriteriaConfig()
    mode = features.mode
    n_obs = track.hits
    enough = n_obs >= config.observations_required(mode)

    # --- 조건 ①: 광류 (상승 + 확산) -------------------------------------
    # 명세는 "방향/크기 분포"를 보라고 한다. 상승 비율과 발산을 함께 본다 —
    # 구름 그림자는 평행이동이라 발산이 0 이고, 안개는 아예 움직이지 않는다.
    use_proxy = mode is not VerifierMode.DENSE
    if use_proxy:
        flow_ok = features.centroid_rise_rate >= config.min_rise_rate_proxy
    else:
        flow_ok = (
            features.flow_upward_ratio >= config.min_flow_upward_ratio
            and features.flow_divergence >= config.min_flow_divergence
        )

    # --- 조건 ②: 마스크 면적 변화율 (단조 증가) --------------------------
    # 성장률만 보면 한 번 크게 튄 프레임에 속는다. 명세가 "단조 증가"라고
    # 못박았으므로 추세와 일관성을 함께 요구한다.
    area_ok = (
        features.area_growth_rate >= config.min_area_growth_rate
        and features.area_monotonicity >= config.min_area_monotonicity
    )

    return SequenceCriteria(
        flow_ok=bool(flow_ok),
        area_ok=bool(area_ok),
        enough_frames=enough,
        mode=mode,
        flow_from_proxy=use_proxy,
        flow_upward_ratio=features.flow_upward_ratio,
        flow_divergence=features.flow_divergence,
        rise_rate=features.centroid_rise_rate,
        area_growth_rate=features.area_growth_rate,
        area_monotonicity=features.area_monotonicity,
        n_observations=n_obs,
    )
