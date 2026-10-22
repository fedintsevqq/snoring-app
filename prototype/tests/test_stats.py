import pytest

from hrapometr_proto.stats import estimate_metrics, wilson_interval


def test_wilson_with_no_data_is_whole_range():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_known_value():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.490, abs=0.005)
    assert high == pytest.approx(0.943, abs=0.005)


def test_wilson_all_successes_upper_bound_is_one():
    _, high = wilson_interval(10, 10)
    assert high == pytest.approx(1.0, abs=1e-9)


def test_perfect_detector():
    result = estimate_metrics(
        flagged_total=100,
        unflagged_total=900,
        flagged_labels=[True] * 10,
        unflagged_labels=[False] * 10,
    )
    assert result.precision == 1.0
    assert result.miss_rate == 0.0
    assert result.recall == 1.0


def test_recall_combines_precision_and_miss_rate_by_group_size():
    # 8 из 10 помеченных окон — храп, 1 из 10 непомеченных — тоже храп
    result = estimate_metrics(
        flagged_total=100,
        unflagged_total=900,
        flagged_labels=[True] * 8 + [False] * 2,
        unflagged_labels=[True] + [False] * 9,
    )
    assert result.precision == pytest.approx(0.8)
    assert result.miss_rate == pytest.approx(0.1)
    assert result.recall == pytest.approx(80 / (80 + 90))


def test_unsure_labels_are_excluded_and_counted():
    result = estimate_metrics(
        flagged_total=10,
        unflagged_total=10,
        flagged_labels=[True, None, False, None],
        unflagged_labels=[False, None],
    )
    assert result.flagged_labeled == 2
    assert result.flagged_unsure == 2
    assert result.unflagged_labeled == 1
    assert result.unflagged_unsure == 1
    assert result.precision == 0.5


def test_recall_interval_is_conservative_and_contains_estimate():
    result = estimate_metrics(
        flagged_total=100,
        unflagged_total=900,
        flagged_labels=[True] * 8 + [False] * 2,
        unflagged_labels=[True] + [False] * 9,
    )
    low, high = result.recall_ci
    assert low <= result.recall <= high


def test_recall_interval_is_wider_than_naive_corners_of_two_95_percent_intervals():
    flagged = [True] * 8 + [False] * 2
    unflagged = [True] + [False] * 9
    result = estimate_metrics(
        flagged_total=100, unflagged_total=900, flagged_labels=flagged, unflagged_labels=unflagged
    )
    p_lo, p_hi = wilson_interval(8, 10)
    m_lo, m_hi = wilson_interval(1, 10)
    naive_low = p_lo * 100 / (p_lo * 100 + m_hi * 900)
    naive_high = p_hi * 100 / (p_hi * 100 + m_lo * 900)
    assert result.recall_ci[0] <= naive_low
    assert result.recall_ci[1] >= naive_high


def test_recall_is_none_when_no_snoring_anywhere():
    result = estimate_metrics(
        flagged_total=10,
        unflagged_total=10,
        flagged_labels=[False, False],
        unflagged_labels=[False, False],
    )
    assert result.precision == 0.0
    assert result.recall is None
    assert result.recall_ci is None


def test_threshold_too_high_nothing_flagged_recall_is_zero_without_flagged_labels():
    # детектор не пометил ни одного окна, а в непомеченных есть храп
    result = estimate_metrics(
        flagged_total=0,
        unflagged_total=100,
        flagged_labels=[],
        unflagged_labels=[True, False, False, False],
    )
    assert result.precision is None
    assert result.recall == 0.0
    assert result.recall_ci[0] == 0.0


def test_everything_flagged_recall_is_full_without_unflagged_labels():
    result = estimate_metrics(
        flagged_total=100,
        unflagged_total=0,
        flagged_labels=[True, True, False],
        unflagged_labels=[],
    )
    assert result.recall == 1.0
    assert result.recall_ci[1] == 1.0


def test_no_flagged_and_no_snoring_anywhere_is_still_undefined():
    result = estimate_metrics(
        flagged_total=0, unflagged_total=100, flagged_labels=[], unflagged_labels=[False] * 4
    )
    assert result.recall is None


def test_unsure_ranges_cover_both_extremes():
    result = estimate_metrics(
        flagged_total=100,
        unflagged_total=100,
        flagged_labels=[True, True, None, None],
        unflagged_labels=[False, False, None, None],
    )
    assert result.precision == 1.0
    assert result.precision_unsure_range == pytest.approx((0.5, 1.0))
    assert result.miss_rate == 0.0
    assert result.miss_rate_unsure_range == pytest.approx((0.0, 0.5))
    low, high = result.recall_unsure_range
    assert low < result.recall <= high


def test_no_unsure_means_no_unsure_ranges():
    result = estimate_metrics(
        flagged_total=10, unflagged_total=10, flagged_labels=[True], unflagged_labels=[False]
    )
    assert result.precision_unsure_range is None
    assert result.recall_unsure_range is None


def test_no_labeled_samples_gives_none():
    result = estimate_metrics(
        flagged_total=10, unflagged_total=10, flagged_labels=[None], unflagged_labels=[]
    )
    assert result.precision is None
    assert result.recall is None
