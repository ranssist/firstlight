"""평가 하네스 자체가 옳은지 확인한다.

측정 도구가 조용히 망가지면 잘못된 숫자가 소개서로 그대로 흘러간다.
숫자 자체(CEP 몇 m)를 고정하지는 않는다 — 지형·설정에 따라 달라지는 값이라
그걸 못박으면 부서지기 쉬운 테스트가 된다. 대신 **성립해야 하는 관계**를
검증한다.
"""

import numpy as np
import pytest

from firstlight.evaluation.geo_accuracy import measure_roundtrip, run_report, sweep_cep
from firstlight.geo.camera import CameraIntrinsics
from firstlight.geo.dem import synthetic_dem
from firstlight.geo.uncertainty import NoiseModel

LAT0, LON0 = 36.4127, 128.7043
BBOX = (LON0 - 0.35, LAT0 - 0.35, LON0 + 0.35, LAT0 + 0.35)


@pytest.fixture(scope="module")
def dem():
    return synthetic_dem(BBOX, resolution_deg=1 / 1200, seed=7)


@pytest.fixture(scope="module")
def intrinsics():
    return CameraIntrinsics.from_fov(1920, 1080, hfov_deg=73.7)


def test_roundtrip_is_sub_metre(dem, intrinsics):
    """무노이즈 왕복은 1m 이내여야 한다. 여기가 새면 CEP 는 전부 무의미하다."""
    rt = measure_roundtrip(dem, intrinsics, LAT0, LON0, altitude_agl_m=300.0)
    assert rt.n_samples > 50, "표본이 너무 적다 — 대부분 거절되고 있다"
    assert rt.max_m < 1.0
    assert rt.pixel_residual_px < 0.01


def test_cep_grows_as_depression_falls(dem, intrinsics):
    """부각이 낮아질수록 CEP 가 단조 증가해야 한다.

    이 관계가 깨지면 오차전파가 기하를 반영하지 못하고 있다는 뜻이다.
    """
    rows = sweep_cep(
        dem, intrinsics, LAT0, LON0,
        noise=NoiseModel.consumer_gnss(),
        depressions=(10.0, 20.0, 45.0, 90.0),
        trials=64,
    )
    solved = [r for r in rows if r.n_solved]
    assert len(solved) == 4

    ceps = [r.cep50_m for r in sorted(solved, key=lambda r: r.depression_deg)]
    assert ceps == sorted(ceps, reverse=True), f"부각에 대해 단조 감소해야 한다: {ceps}"


def test_rtk_beats_consumer_gnss(dem, intrinsics):
    """측위 등급이 좋아지면 CEP 가 작아져야 한다."""
    kwargs = dict(depressions=(20.0, 45.0), trials=64)
    consumer = sweep_cep(dem, intrinsics, LAT0, LON0,
                         noise=NoiseModel.consumer_gnss(), **kwargs)
    rtk = sweep_cep(dem, intrinsics, LAT0, LON0, noise=NoiseModel.rtk(), **kwargs)

    for c, r in zip(consumer, rtk):
        assert r.cep50_m < c.cep50_m, f"부각 {c.depression_deg}도에서 RTK 가 더 나빠졌다"


def test_range_grows_as_depression_falls(dem, intrinsics):
    """부각이 낮으면 사거리가 길어야 한다 (기하 정합성)."""
    rows = sweep_cep(dem, intrinsics, LAT0, LON0, noise=NoiseModel.perfect(),
                     depressions=(15.0, 30.0, 60.0, 90.0), trials=8)
    ranges = [r.median_range_m for r in rows]
    assert ranges == sorted(ranges, reverse=True), f"단조 감소해야 한다: {ranges}"


def test_nadir_range_equals_altitude(dem, intrinsics):
    """수직 하방 사거리는 대지고도와 같아야 한다."""
    rows = sweep_cep(dem, intrinsics, LAT0, LON0, noise=NoiseModel.perfect(),
                     depressions=(90.0,), azimuths=(0.0,), trials=4)
    assert rows[0].median_range_m == pytest.approx(300.0, rel=0.02)


def test_perfect_noise_gives_near_zero_cep(dem, intrinsics):
    rows = sweep_cep(dem, intrinsics, LAT0, LON0, noise=NoiseModel.perfect(),
                     depressions=(30.0,), trials=16)
    assert rows[0].cep90_m == pytest.approx(0.0, abs=0.05)


def test_publish_rate_falls_at_grazing_angles(dem, intrinsics):
    """스침각에서는 발행률이 떨어져야 한다 — 게이트가 실제로 작동하는지."""
    rows = sweep_cep(dem, intrinsics, LAT0, LON0, noise=NoiseModel.consumer_gnss(),
                     depressions=(5.0, 45.0), trials=64, max_cep90_m=150.0)
    grazing, steep = rows[0], rows[1]
    assert grazing.publish_rate < steep.publish_rate
    assert steep.publish_rate == 1.0


def test_report_has_both_noise_grades(dem, intrinsics):
    report = run_report(dem, intrinsics, LAT0, LON0, trials=16)
    assert set(report.sweeps) == {"일반 GNSS", "RTK"}
    assert all(len(rows) > 0 for rows in report.sweeps.values())
    assert np.isfinite(report.roundtrip.max_m)
