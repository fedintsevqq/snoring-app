from hrapometr_proto.events import Event
from hrapometr_proto.windows import (
    BORDERLINE,
    FLAGGED,
    UNFLAGGED,
    Segment,
    classify_windows,
    make_windows,
    sample_windows,
)


def seg(index, start, duration):
    return Segment(index=index, path=f"seg_{index}.ogg", start_s=start, duration_s=duration)


def test_windows_never_cross_segment_boundaries():
    windows = make_windows([seg(0, 0.0, 90.0), seg(1, 90.0, 60.0)], win_s=30.0)
    assert [(w.seg_index, w.start_s, w.end_s) for w in windows] == [
        (0, 0.0, 30.0),
        (0, 30.0, 60.0),
        (0, 60.0, 90.0),
        (1, 90.0, 120.0),
        (1, 120.0, 150.0),
    ]


def test_long_partial_window_is_kept_short_one_dropped():
    kept = make_windows([seg(0, 0.0, 70.0)], win_s=30.0, min_partial_s=10.0)
    dropped = make_windows([seg(0, 0.0, 65.0)], win_s=30.0, min_partial_s=10.0)
    assert [w.end_s for w in kept] == [30.0, 60.0, 70.0]
    assert [w.end_s for w in dropped] == [30.0, 60.0]


def test_window_is_flagged_when_event_overlaps_it_enough():
    windows = make_windows([seg(0, 0.0, 90.0)], win_s=30.0)
    events = [Event(start_s=25.0, end_s=35.0, peak_score=0.9)]
    assert classify_windows(windows, events) == [FLAGGED, FLAGGED, UNFLAGGED]


def test_event_touching_boundary_does_not_flag_next_window():
    windows = make_windows([seg(0, 0.0, 60.0)], win_s=30.0)
    events = [Event(start_s=10.0, end_s=30.0, peak_score=0.9)]
    assert classify_windows(windows, events) == [FLAGGED, UNFLAGGED]


def test_short_tail_of_event_makes_window_borderline():
    # хвост эпизода 0.5 с в соседнем окне: не считаем ни помеченным, ни чистым
    windows = make_windows([seg(0, 0.0, 60.0)], win_s=30.0)
    events = [Event(start_s=10.0, end_s=30.5, peak_score=0.9)]
    assert classify_windows(windows, events, min_overlap_s=1.0) == [FLAGGED, BORDERLINE]


def test_sampling_is_deterministic_and_split_by_group():
    windows = make_windows([seg(0, 0.0, 3000.0)], win_s=30.0)  # 100 окон
    classes = [FLAGGED if i % 4 == 0 else UNFLAGGED for i in range(len(windows))]  # 25 помеченных
    first = sample_windows(windows, classes, per_group=10, seed=7)
    second = sample_windows(windows, classes, per_group=10, seed=7)
    assert first == second
    flagged, unflagged = first
    assert len(flagged) == 10 and len(unflagged) == 10
    assert all(classes[windows.index(w)] == FLAGGED for w in flagged)
    assert all(classes[windows.index(w)] == UNFLAGGED for w in unflagged)


def test_sampling_takes_everything_when_group_is_small():
    windows = make_windows([seg(0, 0.0, 120.0)], win_s=30.0)  # 4 окна
    classes = [FLAGGED, UNFLAGGED, UNFLAGGED, UNFLAGGED]
    flagged, unflagged = sample_windows(windows, classes, per_group=10, seed=1)
    assert len(flagged) == 1 and len(unflagged) == 3


def test_sampling_never_picks_borderline_windows():
    windows = make_windows([seg(0, 0.0, 120.0)], win_s=30.0)
    classes = [FLAGGED, BORDERLINE, UNFLAGGED, BORDERLINE]
    flagged, unflagged = sample_windows(windows, classes, per_group=10, seed=1)
    assert len(flagged) == 1 and len(unflagged) == 1
