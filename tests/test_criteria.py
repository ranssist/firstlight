"""시퀀스 검증 2조건 판정 — 작품설명서 Ⅱ-1 의 규칙 그대로인지.

명세 원문:
    두 조건을 동시에 만족하면 FLARE, 하나만 만족하면 GLOW,
    둘 다 미달이면 SPARK로 분류한다.

등급이 점수 임계값이 아니라 **만족한 조건 수**에서 나온다는 것이 핵심이다.
"""

import numpy as np
import pytest

from firstlight.detect.detector import Detection
from firstlight.verify.criteria import CriteriaConfig, evaluate_criteria
from firstlight.verify.features import VerifierMode, extract_features
from firstlight.verify.scorer import SequenceScorer, Tier
from firstlight.verify.tracker import Track, TrackObservation

DENSE_INTERVAL = 1 / 25      # 25fps — 명세의 "12프레임 ≈ 0.5초"
SPARSE_INTERVAL = 60.0       # FIgLib


def make_track(
    n: int = 12,
    *,
    growth: float = 0.0,
    rise_px: float = 0.0,
    drift_px: float = 0.0,
    interval: float = DENSE_INTERVAL,
    flow: dict | None = None,
    jitter_area: bool = False,
) -> Track:
    """지정한 거동의 합성 트랙."""
    rng = np.random.default_rng(0)
    track = Track(track_id=0)
    for i in range(n):
        scale = (1.0 + growth) ** i
        if jitter_area:
            # 커지긴 하는데 들쭉날쭉 — 단조성이 깨진다.
            scale *= 1.0 + 0.5 * (rng.random() - 0.5)
        size = 40.0 * scale
        cx = 500.0 + drift_px * i
        cy = 400.0 - rise_px * i
        det = Detection(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, 0.6)
        track.observations.append(
            TrackObservation(i * interval, i, det, None, flow)
        )
    return track


def criteria_for(track: Track, interval: float, config=None):
    feats = extract_features(track, interval)
    return evaluate_criteria(track, feats, config)


RISING_FLOW = {"divergence": 0.15, "upward_ratio": 0.8, "translation_mag": 0.3}
DRIFTING_FLOW = {"divergence": 0.0, "upward_ratio": 0.2, "translation_mag": 2.5}
STILL_FLOW = {"divergence": 0.0, "upward_ratio": 0.5, "translation_mag": 0.0}


# ------------------------------------------------------- 두 조건이 각각 동작


def test_rising_growing_smoke_satisfies_both():
    """연기: 상승·확산 + 면적 단조 증가 → 두 조건 모두."""
    c = criteria_for(make_track(growth=0.06, rise_px=3.0, flow=RISING_FLOW),
                     DENSE_INTERVAL)
    assert c.flow_ok and c.area_ok
    assert c.n_satisfied == 2


def test_static_fog_satisfies_neither():
    """안개: 움직이지 않고 커지지도 않는다 → 둘 다 미달."""
    c = criteria_for(make_track(growth=0.0, flow=STILL_FLOW), DENSE_INTERVAL)
    assert not c.flow_ok and not c.area_ok
    assert c.n_satisfied == 0


def test_cloud_shadow_satisfies_neither():
    """구름 그림자: 평행이동만 한다 — 발산 0, 면적 불변."""
    c = criteria_for(make_track(growth=0.0, drift_px=12.0, flow=DRIFTING_FLOW),
                     DENSE_INTERVAL)
    assert not c.flow_ok, "평행이동은 광류 조건을 통과하면 안 된다"
    assert not c.area_ok
    assert c.n_satisfied == 0


def test_growing_without_rising_satisfies_one():
    """면적은 늘지만 상승 운동이 없으면 조건 하나만."""
    c = criteria_for(make_track(growth=0.06, flow=STILL_FLOW), DENSE_INTERVAL)
    assert c.area_ok and not c.flow_ok
    assert c.n_satisfied == 1


def test_rising_without_growing_satisfies_one():
    """상승하지만 면적이 그대로면 조건 하나만."""
    c = criteria_for(make_track(growth=0.0, rise_px=3.0, flow=RISING_FLOW),
                     DENSE_INTERVAL)
    assert c.flow_ok and not c.area_ok
    assert c.n_satisfied == 1


def test_erratic_area_fails_monotonicity():
    """명세가 요구한 것은 '단조 증가'다 — 들쭉날쭉하면 통과하면 안 된다."""
    steady = criteria_for(make_track(growth=0.06, flow=RISING_FLOW), DENSE_INTERVAL)
    erratic = criteria_for(
        make_track(growth=0.06, flow=RISING_FLOW, jitter_area=True), DENSE_INTERVAL
    )
    assert steady.area_monotonicity > erratic.area_monotonicity


# --------------------------------------------------------------- 등급 매핑


@pytest.mark.parametrize(
    "flow_ok,area_ok,expected",
    [
        (True, True, Tier.FLARE),    # 두 조건 동시 만족
        (True, False, Tier.GLOW),    # 하나만
        (False, True, Tier.GLOW),    # 하나만
        (False, False, Tier.SPARK),  # 둘 다 미달
    ],
)
def test_tier_follows_number_of_satisfied_conditions(flow_ok, area_ok, expected):
    """명세의 판정표 그대로여야 한다."""
    from firstlight.verify.criteria import SequenceCriteria

    criteria = SequenceCriteria(
        flow_ok=flow_ok, area_ok=area_ok, enough_frames=True,
        mode=VerifierMode.DENSE, n_observations=12,
    )
    assert SequenceScorer().tier_from_criteria(criteria, geo_ok=True) is expected


def test_insufficient_frames_downgrades_flare_to_glow():
    """명세: "후속 12프레임 시퀀스 검증". 그만큼 못 봤으면 검증한 게 아니다."""
    from firstlight.verify.criteria import SequenceCriteria

    criteria = SequenceCriteria(
        flow_ok=True, area_ok=True, enough_frames=False,
        mode=VerifierMode.DENSE, n_observations=4,
    )
    assert SequenceScorer().tier_from_criteria(criteria, geo_ok=True) is Tier.GLOW


def test_missing_coordinate_downgrades_flare_to_glow():
    """표 3-1 의 FLARE 동작이 '좌표 확정'이다 — 확정할 좌표가 없으면 FLARE 가 아니다."""
    from firstlight.verify.criteria import SequenceCriteria

    criteria = SequenceCriteria(
        flow_ok=True, area_ok=True, enough_frames=True,
        mode=VerifierMode.DENSE, n_observations=12,
    )
    scorer = SequenceScorer()
    assert scorer.tier_from_criteria(criteria, geo_ok=True) is Tier.FLARE
    assert scorer.tier_from_criteria(criteria, geo_ok=False) is Tier.GLOW


def test_downgrade_never_goes_below_glow():
    """조건을 통과한 후보를 버리지 않는다 — 놓치는 쪽이 더 위험하다."""
    from firstlight.verify.criteria import SequenceCriteria

    criteria = SequenceCriteria(
        flow_ok=True, area_ok=True, enough_frames=False,
        mode=VerifierMode.DENSE, n_observations=1,
    )
    assert SequenceScorer().tier_from_criteria(criteria, geo_ok=False) is not Tier.SPARK


# ------------------------------------------------------------- 프레임 요건


def test_dense_mode_requires_twelve_frames():
    """명세가 못박은 12프레임."""
    config = CriteriaConfig()
    assert config.observations_required(VerifierMode.DENSE) == 12

    short = criteria_for(make_track(n=8, growth=0.06, flow=RISING_FLOW), DENSE_INTERVAL)
    full = criteria_for(make_track(n=12, growth=0.06, flow=RISING_FLOW), DENSE_INTERVAL)
    assert not short.enough_frames
    assert full.enough_frames


def test_sparse_mode_uses_geometric_proxy_and_says_so():
    """60초 간격에서는 광류를 못 쓴다 — 대용값을 쓰되 그 사실을 표시한다."""
    c = criteria_for(make_track(n=6, growth=0.3, rise_px=4.0,
                                interval=SPARSE_INTERVAL), SPARSE_INTERVAL)
    assert c.mode is VerifierMode.SPARSE
    assert c.flow_from_proxy is True
    assert c.flow_ok, "상승하는 트랙은 대용값으로도 광류 조건을 통과해야 한다"


def test_reasons_are_human_readable():
    c = criteria_for(make_track(growth=0.06, rise_px=3.0, flow=RISING_FLOW),
                     DENSE_INTERVAL)
    reasons = c.reasons()
    assert any("상승" in r for r in reasons)
    assert any("면적" in r for r in reasons)
