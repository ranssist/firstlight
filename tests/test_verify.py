"""시퀀스 검증 — 특징이 실제로 오탐원을 구분하는지.

합성 트랙을 만들어 검증한다. 실제 데이터 평가(`firstlight falsealarm`)와
별개로, **각 특징이 겨냥한 물리 현상을 실제로 재는지**를 여기서 못박는다.
특징 하나의 부호가 뒤집혀도 전체 오경보율은 그럴듯하게 나올 수 있으므로,
개별 특징을 따로 잡아둘 필요가 있다.
"""

import numpy as np
import pytest

from firstlight.detect.detector import Detection
from firstlight.verify.features import (
    SPARSE_FEATURES,
    VerifierMode,
    extract_features,
)
from firstlight.verify.scorer import SequenceScorer, Tier
from firstlight.verify.tracker import Track, TrackObservation, Tracker

INTERVAL = 60.0          # FIgLib 간격


def make_track(
    n: int = 10,
    *,
    x0: float = 500.0,
    y0: float = 400.0,
    size0: float = 40.0,
    growth: float = 0.0,      # 프레임당 크기 배율 증가
    rise_px: float = 0.0,     # 프레임당 상승 픽셀
    drift_px: float = 0.0,    # 프레임당 수평 이동 픽셀
    conf: float = 0.5,
    interval: float = INTERVAL,
    patch_maker=None,
) -> Track:
    """지정한 거동을 가진 합성 트랙."""
    track = Track(track_id=0)
    for i in range(n):
        size = size0 * (1.0 + growth) ** i
        cx = x0 + drift_px * i
        cy = y0 - rise_px * i
        det = Detection(
            cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, conf
        )
        patch = patch_maker(i) if patch_maker else None
        track.observations.append(TrackObservation(i * interval, i, det, patch))
    return track


# ---------------------------------------------------------------- 개별 특징


def test_growing_smoke_has_positive_area_growth():
    smoke = extract_features(make_track(growth=0.12), INTERVAL)
    assert smoke.area_growth_rate > 0


def test_static_fog_has_no_area_growth():
    fog = extract_features(make_track(growth=0.0), INTERVAL)
    assert fog.area_growth_rate == pytest.approx(0.0, abs=1e-9)


def test_rising_plume_has_positive_rise_rate():
    """이미지 y 는 아래가 양수 — 상승은 y 감소다. 부호를 여기서 못박는다."""
    rising = extract_features(make_track(rise_px=8.0), INTERVAL)
    sinking = extract_features(make_track(rise_px=-8.0), INTERVAL)
    assert rising.centroid_rise_rate > 0
    assert sinking.centroid_rise_rate < 0


def test_cloud_shadow_has_high_translation_over_growth():
    """구름그림자: 커지지 않고 옮겨가기만 한다 → 비율이 커야 한다."""
    shadow = extract_features(make_track(growth=0.0, drift_px=25.0), INTERVAL)
    smoke = extract_features(make_track(growth=0.12, drift_px=2.0), INTERVAL)
    assert shadow.translation_over_growth > smoke.translation_over_growth


def test_area_growth_is_scale_free():
    """로그를 쓰므로 작은 연기와 큰 연기의 같은 배율 성장은 같은 값이어야 한다."""
    small = extract_features(make_track(size0=20.0, growth=0.1), INTERVAL)
    large = extract_features(make_track(size0=400.0, growth=0.1), INTERVAL)
    assert small.area_growth_rate == pytest.approx(large.area_growth_rate, rel=1e-9)


def test_vertical_elongation_shows_in_aspect():
    track = Track(track_id=0)
    for i in range(8):
        w, h = 40.0, 40.0 + 12.0 * i          # 세로로만 길어진다
        track.observations.append(
            TrackObservation(i * INTERVAL, i,
                             Detection(500, 400, 500 + w, 400 + h, 0.5))
        )
    assert extract_features(track, INTERVAL).aspect_growth_rate > 0


def test_flicker_separates_static_from_varying():
    """안개는 밝기가 일정하고, 연기는 변한다."""
    rng = np.random.default_rng(0)
    static = make_track(patch_maker=lambda i: np.full((48, 48), 0.5, np.float32))
    varying = make_track(
        patch_maker=lambda i: np.full((48, 48), 0.3 + 0.25 * rng.random(), np.float32)
    )
    assert extract_features(static, INTERVAL).intensity_flicker == pytest.approx(0, abs=1e-6)
    assert extract_features(varying, INTERVAL).intensity_flicker > 0.05


def test_edge_softness_separates_sharp_from_diffuse():
    """건물·비닐하우스는 경계가 날카롭고 연기는 흐리다."""
    def sharp(i):
        p = np.zeros((48, 48), np.float32)
        p[:, 24:] = 1.0                        # 계단 모양 에지
        return p

    def diffuse(i):
        x = np.linspace(0, 1, 48, dtype=np.float32)
        return np.tile(x, (48, 1))             # 완만한 기울기

    soft = extract_features(make_track(patch_maker=diffuse), INTERVAL).edge_softness
    hard = extract_features(make_track(patch_maker=sharp), INTERVAL).edge_softness
    assert soft > hard


def test_persistence_penalises_sporadic_tracks():
    """산발적으로 잡히는 반사광은 persistence 가 낮아야 한다."""
    steady = make_track(n=10)
    sporadic = Track(track_id=1)
    for i in (0, 3, 7, 9):                     # 10프레임 중 4번만
        sporadic.observations.append(
            TrackObservation(i * INTERVAL, i, Detection(500, 400, 540, 440, 0.5))
        )
    assert extract_features(steady, INTERVAL).persistence > \
           extract_features(sporadic, INTERVAL).persistence


def test_single_observation_track_is_safe():
    """관측 1개짜리 트랙에서 기울기 계산이 터지면 안 된다."""
    feats = extract_features(make_track(n=1), INTERVAL)
    assert feats.n_observations == 1
    assert feats.area_growth_rate == 0.0


def test_empty_track_is_safe():
    assert extract_features(Track(track_id=0), INTERVAL).n_observations == 0


# ------------------------------------------------------------------- 모드


def test_mode_selected_from_interval():
    """작품설명서: "후속 12프레임(약 0.5~1초)" → 12~25fps 가 dense 다.

    간격 1초는 12프레임이 12초가 되어 명세의 "약 0.5~1초"와 맞지 않으므로
    sparse 로 떨어져야 한다.
    """
    assert VerifierMode.from_interval(0.04) is VerifierMode.DENSE     # 25fps
    assert VerifierMode.from_interval(0.08) is VerifierMode.DENSE     # 12.5fps
    assert VerifierMode.from_interval(1.0) is VerifierMode.SPARSE
    assert VerifierMode.from_interval(60.0) is VerifierMode.SPARSE    # FIgLib


def test_sparse_mode_zeroes_flow_features():
    """60초 간격에서 광류 특징은 0 이어야 한다 — 없는 정보를 만들면 안 된다."""
    feats = extract_features(make_track(growth=0.1), INTERVAL)
    assert feats.mode is VerifierMode.SPARSE
    assert feats.flow_divergence == 0.0
    assert feats.flow_upward_ratio == 0.0


# ----------------------------------------------------------------- 스코어러


def test_scorer_prefers_smoke_over_fog_and_shadow():
    """사전 가중치만으로도 물리적 방향은 맞아야 한다."""
    scorer = SequenceScorer(mode=VerifierMode.SPARSE)
    smoke = scorer.score(extract_features(make_track(growth=0.12, rise_px=6.0, conf=0.6), INTERVAL))
    fog = scorer.score(extract_features(make_track(growth=0.0, conf=0.6), INTERVAL))
    shadow = scorer.score(extract_features(make_track(growth=0.0, drift_px=25.0, conf=0.6), INTERVAL))

    assert smoke > fog, f"연기 {smoke:.3f} vs 안개 {fog:.3f}"
    assert smoke > shadow, f"연기 {smoke:.3f} vs 구름그림자 {shadow:.3f}"


def test_score_is_a_probability():
    scorer = SequenceScorer()
    for track in (make_track(growth=0.5), make_track(growth=-0.3), make_track(n=1)):
        assert 0.0 <= scorer.score(extract_features(track, INTERVAL)) <= 1.0


@pytest.mark.parametrize(
    "score,expected",
    [(0.95, Tier.FLARE), (0.70, Tier.FLARE), (0.69, Tier.GLOW),
     (0.35, Tier.GLOW), (0.34, Tier.SPARK), (0.0, Tier.SPARK)],
)
def test_tier_thresholds(score, expected):
    assert SequenceScorer().tier(score, geo_ok=True) is expected


def test_high_score_without_coordinate_is_not_flare():
    """좌표를 못 내면 자동 경보를 울리면 안 된다 — 갈 곳을 모르기 때문이다."""
    scorer = SequenceScorer()
    assert scorer.tier(0.99, geo_ok=True) is Tier.FLARE
    assert scorer.tier(0.99, geo_ok=False) is Tier.GLOW


def test_high_score_with_too_few_observations_is_not_flare():
    """표본 2개로 계산한 기울기는 추정이 아니다 — 자동 경보를 주면 안 된다.

    대시보드에서 관측 2회짜리 발화 **전** 트랙이 0.96 점으로 FLARE 를 받는
    것이 실제로 관찰돼서 넣은 게이트다.
    """
    scorer = SequenceScorer(mode=VerifierMode.SPARSE)
    assert scorer.min_observations_for_flare == 3
    assert scorer.tier(0.99, geo_ok=True, n_observations=2) is Tier.GLOW
    assert scorer.tier(0.99, geo_ok=True, n_observations=3) is Tier.FLARE


def test_dense_mode_needs_more_observations():
    """소개서의 '후속 12프레임 검증'이 dense 모드 기본값이다."""
    assert SequenceScorer(mode=VerifierMode.DENSE).min_observations_for_flare == 12


def test_low_score_stays_spark_regardless_of_observations():
    """관측이 많다고 낮은 점수가 올라가지는 않는다."""
    scorer = SequenceScorer()
    assert scorer.tier(0.10, geo_ok=True, n_observations=100) is Tier.SPARK


def test_min_observations_survives_roundtrip(tmp_path):
    scorer = SequenceScorer(mode=VerifierMode.SPARSE, min_observations_for_flare=7)
    path = tmp_path / "s.json"
    scorer.save(path)
    assert SequenceScorer.load(path).min_observations_for_flare == 7


def test_explain_contributions_sum_to_logit():
    scorer = SequenceScorer()
    feats = extract_features(make_track(growth=0.1, rise_px=5.0), INTERVAL)
    breakdown = scorer.explain(feats)
    logit = sum(breakdown.contributions.values()) + scorer.intercept
    assert breakdown.score == pytest.approx(1 / (1 + np.exp(-logit)), rel=1e-9)


def test_fit_changes_weights_and_marks_fitted():
    """라벨이 들어오면 가중치가 갈아엎어져야 한다 (환류 루프의 핵심)."""
    rng = np.random.default_rng(0)
    features, labels = [], []
    for _ in range(40):
        features.append(extract_features(
            make_track(growth=0.10 + 0.05 * rng.random(), rise_px=5.0), INTERVAL))
        labels.append(1)
        features.append(extract_features(
            make_track(growth=0.0, drift_px=20.0 * rng.random()), INTERVAL))
        labels.append(0)

    scorer = SequenceScorer(mode=VerifierMode.SPARSE)
    before = dict(scorer.weights)
    scorer.fit(features, labels)

    assert scorer.is_fitted and scorer.n_train == 80
    assert any(
        abs(scorer.weights[n] - before[n]) > 1e-6 for n in SPARSE_FEATURES
    ), "학습 후에도 가중치가 그대로다"

    # 학습 후에는 두 부류를 분리해야 한다.
    pos = np.mean([scorer.score(f) for f, y in zip(features, labels) if y == 1])
    neg = np.mean([scorer.score(f) for f, y in zip(features, labels) if y == 0])
    assert pos > neg + 0.2


def test_fit_rejects_single_class():
    scorer = SequenceScorer()
    feats = [extract_features(make_track(), INTERVAL)] * 5
    with pytest.raises(ValueError, match="양성과 음성"):
        scorer.fit(feats, [0, 0, 0, 0, 0])


def test_scorer_roundtrip(tmp_path):
    scorer = SequenceScorer(mode=VerifierMode.SPARSE, tau_high=0.8, tau_low=0.4)
    scorer.weights["persistence"] = 3.14
    path = tmp_path / "scorer.json"
    scorer.save(path)

    loaded = SequenceScorer.load(path)
    assert loaded.tau_high == 0.8 and loaded.tau_low == 0.4
    assert loaded.weights["persistence"] == 3.14
    feats = extract_features(make_track(growth=0.1), INTERVAL)
    assert loaded.score(feats) == pytest.approx(scorer.score(feats))


# ------------------------------------------------------------------ 트래커


def test_tracker_links_overlapping_detections():
    tracker = Tracker()
    for i in range(5):
        tracker.update([Detection(100 + i * 3, 100, 160 + i * 3, 160, 0.6)],
                       timestamp=i * 1.0, frame_index=i)
    tracks = tracker.close()
    assert len(tracks) == 1 and tracks[0].hits == 5


def test_tracker_separates_distant_detections():
    tracker = Tracker()
    for i in range(4):
        tracker.update(
            [Detection(100, 100, 160, 160, 0.6), Detection(900, 700, 960, 760, 0.6)],
            timestamp=i * 1.0, frame_index=i,
        )
    tracks = tracker.close()
    assert len(tracks) == 2
    assert all(t.hits == 4 for t in tracks)


def test_tracker_closes_after_max_misses():
    tracker = Tracker(max_misses=2)
    tracker.update([Detection(100, 100, 160, 160, 0.6)], 0.0, 0)
    for i in range(1, 5):
        tracker.update([], float(i), i)
    assert len(tracker.finished) == 1
    assert tracker.active == []


def test_tracker_survives_brief_gap():
    """한 프레임 놓쳤다고 트랙이 끊기면 persistence 가 무의미해진다."""
    tracker = Tracker(max_misses=2)
    box = Detection(100, 100, 160, 160, 0.6)
    tracker.update([box], 0.0, 0)
    tracker.update([], 1.0, 1)
    tracker.update([box], 2.0, 2)
    assert len(tracker.close()) == 1


def test_tracker_assigns_unique_ids():
    tracker = Tracker()
    tracker.update([Detection(100, 100, 160, 160, 0.6)], 0.0, 0)
    tracker.update([Detection(900, 700, 960, 760, 0.6)], 1.0, 1)
    ids = {t.track_id for t in tracker.all_tracks}
    assert len(ids) == len(tracker.all_tracks)


def test_track_window_trims_to_recent():
    track = make_track(n=10, interval=10.0)         # 0..90초
    recent = track.window(30.0)
    assert recent.hits == 4                          # 60,70,80,90
    assert recent.first.timestamp == 60.0
