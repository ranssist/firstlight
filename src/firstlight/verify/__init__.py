"""시퀀스 검증 — 소개서 §4① "오탐을 줄이는 것이 성능이다".

탐지기는 한 장만 보고 판단하므로 안개·구름그림자·노을·반사광을 연기와
구분하지 못한다. 실측(pyro-sdis 검증셋 200장): 연기가 없는 이미지의
**92.1%** 에서 탐지가 떴다. 이대로 프레임 단위 경보를 울리면 담당자는
하루 만에 알림을 꺼버린다.

이 패키지는 시간축을 본다. 연기는 **퍼지고, 올라가고, 계속 변한다.**
안개는 그대로 있고 구름그림자는 통째로 평행이동한다.
"""

from firstlight.verify.tracker import Track, TrackObservation, Tracker
from firstlight.verify.features import TrackFeatures, VerifierMode, extract_features
from firstlight.verify.criteria import (
    CriteriaConfig,
    SequenceCriteria,
    evaluate_criteria,
)
from firstlight.verify.scorer import SequenceScorer, Tier

__all__ = [
    "CriteriaConfig",
    "SequenceCriteria",
    "SequenceScorer",
    "Tier",
    "Track",
    "TrackFeatures",
    "TrackObservation",
    "Tracker",
    "VerifierMode",
    "evaluate_criteria",
    "extract_features",
]
