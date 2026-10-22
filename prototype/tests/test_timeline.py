from hrapometr_proto.timeline import place_segments, segment_number


def test_segment_number_from_file_name():
    assert segment_number("seg_0007.ogg") == 7
    assert segment_number("night.ogg") is None
    assert segment_number("._seg_0001.ogg") is None


def test_without_runs_segments_go_back_to_back():
    starts = place_segments([0, 1, 2], [10.0, 20.0, 5.0], runs=[], nominal_s=3600)
    assert starts == [0.0, 10.0, 30.0]


def test_single_run_starts_at_its_offset():
    runs = [{"start_s": 0.0, "first_segment": 0}]
    assert place_segments([0, 1], [3600.0, 3600.0], runs, 3600) == [0.0, 3600.0]


def test_gap_between_runs_is_kept():
    # ffmpeg упал и перезапустился через 400 с после конца кусков первого запуска
    runs = [{"start_s": 0.0, "first_segment": 0}, {"start_s": 8000.0, "first_segment": 2}]
    starts = place_segments([0, 1, 2, 3], [3600.0, 3600.0, 3600.0, 100.0], runs, 3600)
    assert starts == [0.0, 3600.0, 8000.0, 11600.0]


def test_unreadable_segment_counts_as_nominal_length():
    runs = [{"start_s": 0.0, "first_segment": 0}]
    starts = place_segments([0, 1, 2], [None, 3600.0, 3600.0], runs, 3600)
    assert starts == [0.0, 3600.0, 7200.0]


def test_missing_file_in_the_middle_shifts_later_segments():
    # файла seg_0001 нет совсем: после него время сдвигается на номинальную длину
    runs = [{"start_s": 0.0, "first_segment": 0}]
    starts = place_segments([0, 2], [3600.0, 3600.0], runs, 3600)
    assert starts == [0.0, 7200.0]


def test_files_without_numbers_fall_back_to_cumulative_order():
    runs = [{"start_s": 5000.0, "first_segment": 0}]
    starts = place_segments([None, None], [30.0, 30.0], runs, 3600)
    assert starts == [0.0, 30.0]
