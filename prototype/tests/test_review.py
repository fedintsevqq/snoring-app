import numpy as np
import pytest

from hrapometr_proto.audio import normalize_level
from hrapometr_proto.review import _segment_path, split_labels, sweep, validate_labels_doc


def test_segment_name_is_looked_up_in_run_dir_not_in_current_directory(tmp_path):
    run_dir = tmp_path / "night"
    run_dir.mkdir()
    assert _segment_path("seg_0000.ogg", run_dir) == run_dir / "seg_0000.ogg"
    # путь, записанный на Windows, на Mac превращается в имя файла рядом с записью
    assert _segment_path("runs\\night\\seg_0001.ogg", run_dir) == run_dir / "seg_0001.ogg"


def test_sweep_refuses_stale_frames(tmp_path):
    review = tmp_path / "review"
    review.mkdir()
    (tmp_path / "frames.csv").write_text("t_s,seg,rms_db,snoring,breathing,snort,speech\n")
    (review / "key.json").write_text(
        '{"token": "t", "frames_sha256": "not-the-real-hash", "clips": {"c001": {"group": "flagged", '
        '"start_s": 0, "end_s": 30}}, "flagged_total": 1, "unflagged_total": 1}'
    )
    labels = tmp_path / "labels.json"
    labels.write_text('{"token": "t", "labels": {"c001": "snore"}}')
    with pytest.raises(ValueError, match="frames.csv изменился"):
        sweep(tmp_path, labels)


def test_report_without_prepare_says_so(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text('{"token": "t", "labels": {}}')
    with pytest.raises(ValueError, match="prepare"):
        from hrapometr_proto.review import report

        report(tmp_path, labels)


def test_sweep_rejects_mistyped_labels(tmp_path):
    import hashlib

    review = tmp_path / "review"
    review.mkdir()
    frames = tmp_path / "frames.csv"
    frames.write_text("t_s,seg,rms_db,snoring,breathing,snort,speech\n")
    digest = hashlib.sha256(frames.read_bytes()).hexdigest()
    (review / "key.json").write_text(
        '{"token": "t", "frames_sha256": "%s", "clips": {"c001": {"group": "flagged", '
        '"start_s": 0, "end_s": 30}}, "flagged_total": 1, "unflagged_total": 1}' % digest
    )
    labels = tmp_path / "labels.json"
    labels.write_text('{"token": "t", "labels": {"c001": "Snore"}}')
    with pytest.raises(ValueError, match="Неизвестная метка"):
        sweep(tmp_path, labels)


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
