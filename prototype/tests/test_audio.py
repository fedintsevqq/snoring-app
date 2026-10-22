import numpy as np
import pytest

from hrapometr_proto.audio import apply_gain_db, frame_stream, rms_db


def collect(blocks, win, hop):
    return [frame.copy() for frame in frame_stream(iter(blocks), win=win, hop=hop)]


def test_frames_follow_hop_and_window():
    signal = np.arange(25, dtype=np.float32)
    frames = collect([signal], win=8, hop=4)
    assert [int(f[0]) for f in frames] == [0, 4, 8, 12, 16]
    assert all(len(f) == 8 for f in frames)


def test_block_boundaries_do_not_change_frames():
    signal = np.arange(100, dtype=np.float32)
    whole = collect([signal], win=8, hop=4)
    split = collect([signal[:7], signal[7:31], signal[31:32], signal[32:]], win=8, hop=4)
    assert len(whole) == len(split)
    for a, b in zip(whole, split):
        assert np.array_equal(a, b)


def test_signal_shorter_than_window_gives_no_frames():
    assert collect([np.zeros(5, dtype=np.float32)], win=8, hop=4) == []


def test_rms_db_of_full_scale_square_wave_is_zero():
    assert rms_db(np.ones(100, dtype=np.float32)) == pytest.approx(0.0, abs=1e-6)


def test_rms_db_of_silence_is_floor_not_minus_infinity():
    assert np.isfinite(rms_db(np.zeros(100, dtype=np.float32)))


def test_gain_scales_amplitude_and_clips():
    signal = np.array([0.1, -0.4, 0.9], dtype=np.float32)
    boosted = apply_gain_db(signal, 6.0206)
    assert boosted[0] == pytest.approx(0.2, abs=1e-3)
    assert boosted[2] == 1.0
    assert signal[0] == pytest.approx(0.1)  # исходный массив не тронут
