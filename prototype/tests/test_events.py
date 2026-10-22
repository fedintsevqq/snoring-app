import numpy as np

from hrapometr_proto.events import frames_to_events

HOP = 0.5
WIN = 1.0


def run(scores, **overrides):
    params = dict(hop_s=HOP, win_s=WIN, threshold=0.5, max_gap_s=1.0, min_duration_s=1.0)
    params.update(overrides)
    return frames_to_events(np.array(scores, dtype=np.float32), **params)


def test_empty_input_gives_no_events():
    assert run([]) == []


def test_scores_below_threshold_give_no_events():
    assert run([0.1, 0.2, 0.3]) == []


def test_single_frame_event_spans_one_window():
    events = run([0.0, 0.9, 0.0])
    assert len(events) == 1
    assert events[0].start_s == 0.5
    assert events[0].end_s == 1.5
    assert events[0].peak_score == np.float32(0.9)


def test_adjacent_frames_merge_into_one_event():
    events = run([0.9, 0.8, 0.7, 0.0])
    assert len(events) == 1
    assert events[0].start_s == 0.0
    assert events[0].end_s == 2.0


def test_gap_within_limit_merges():
    # кадры 1 и 4: [0.5, 1.5] и [2.0, 3.0], пауза 0.5 с при лимите 1.0 с
    events = run([0, 0.9, 0, 0, 0.9, 0])
    assert len(events) == 1
    assert (events[0].start_s, events[0].end_s) == (0.5, 3.0)


def test_gap_over_limit_splits_events():
    # кадры 1 и 6: [0.5, 1.5] и [3.0, 4.0], пауза 1.5 с при лимите 1.0 с
    events = run([0, 0.9, 0, 0, 0, 0, 0.9, 0])
    assert len(events) == 2


def test_gap_equal_to_limit_still_merges():
    # пауза ровно 1.0 с: кадры 1 и 5 -> [0.5, 1.5] и [2.5, 3.5]
    events = run([0, 0.9, 0, 0, 0, 0.9, 0])
    assert len(events) == 1


def test_short_event_is_dropped():
    assert run([0, 0.9, 0], min_duration_s=2.0) == []


def test_peak_score_is_maximum_inside_event():
    events = run([0.6, 0.95, 0.7])
    assert events[0].peak_score == np.float32(0.95)


def test_duration_property():
    events = run([0.9, 0.9])
    assert events[0].duration_s == 1.5
