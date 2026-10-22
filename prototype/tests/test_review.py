import numpy as np
import pytest

from hrapometr_proto.audio import normalize_level
from hrapometr_proto.review import split_labels, validate_labels_doc


def test_normalize_level_brings_typical_loudness_to_target():
    signal = np.full(1000, 0.02, dtype=np.float32)
    result = normalize_level(signal, target_db=-6.0206, max_gain_db=60.0)
    assert np.max(np.abs(result)) == pytest.approx(0.5, abs=1e-3)


def test_single_click_does_not_set_the_gain():
    signal = np.full(1000, 0.01, dtype=np.float32)
    signal[500] = 0.9  # хлопок двери
    result = normalize_level(signal, target_db=-6.0206, max_gain_db=60.0)
    assert result[0] == pytest.approx(0.5, abs=1e-2)  # тихий храп поднят, а не задавлен хлопком
    assert result[500] == 1.0  # сам хлопок обрезан


def test_normalize_level_limits_gain_for_near_silence():
    signal = np.full(100, 1e-6, dtype=np.float32)
    result = normalize_level(signal, target_db=-3.0, max_gain_db=20.0)
    assert np.max(np.abs(result)) == pytest.approx(1e-5, rel=1e-3)


def test_normalize_level_of_digital_silence_stays_silent():
    assert not normalize_level(np.zeros(10, dtype=np.float32), -3.0, 30.0).any()


def test_validate_labels_accepts_matching_token_and_ids():
    key = {"token": "abc", "clips": {"c001": {}, "c002": {}}}
    validate_labels_doc(key, {"token": "abc", "labels": {"c001": "snore"}})


def test_validate_labels_rejects_other_selection():
    key = {"token": "abc", "clips": {"c001": {}}}
    with pytest.raises(ValueError, match="другой выборки"):
        validate_labels_doc(key, {"token": "zzz", "labels": {"c001": "snore"}})


def test_validate_labels_rejects_unknown_clip_ids():
    key = {"token": "abc", "clips": {"c001": {}}}
    with pytest.raises(ValueError, match="c099"):
        validate_labels_doc(key, {"token": "abc", "labels": {"c099": "snore"}})


KEY = {
    "clips": {
        "c001": {"group": "flagged"},
        "c002": {"group": "flagged"},
        "c003": {"group": "unflagged"},
        "c004": {"group": "unflagged"},
    }
}


def test_split_labels_maps_snore_to_true_and_others_to_false():
    flagged, unflagged, reasons = split_labels(
        KEY, {"c001": "snore", "c002": "breathing", "c003": "noise", "c004": "snore"}
    )
    assert flagged == [True, False]
    assert unflagged == [False, True]
    assert reasons == {"breathing": 1}


def test_split_labels_unsure_becomes_none_and_missing_is_skipped():
    flagged, unflagged, _ = split_labels(KEY, {"c001": "unsure", "c003": "noise"})
    assert flagged == [None]
    assert unflagged == [False]


def test_split_labels_rejects_unknown_label():
    with pytest.raises(ValueError):
        split_labels(KEY, {"c001": "banana"})
