import numpy as np
import pytest

from hrapometr_proto.audio import normalize_peak
from hrapometr_proto.review import split_labels


def test_normalize_peak_brings_loudest_sample_to_target():
    signal = np.array([0.01, -0.02, 0.005], dtype=np.float32)
    result = normalize_peak(signal, target_db=-6.0206, max_gain_db=60.0)
    assert np.max(np.abs(result)) == pytest.approx(0.5, abs=1e-3)


def test_normalize_peak_limits_gain_for_near_silence():
    signal = np.array([1e-6, -1e-6], dtype=np.float32)
    result = normalize_peak(signal, target_db=-3.0, max_gain_db=20.0)
    assert np.max(np.abs(result)) == pytest.approx(1e-5, rel=1e-3)


def test_normalize_peak_of_digital_silence_stays_silent():
    assert not normalize_peak(np.zeros(10, dtype=np.float32), -3.0, 30.0).any()


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
