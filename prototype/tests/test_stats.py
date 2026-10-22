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


def test_no_labeled_samples_gives_none():
    result = estimate_metrics(
        flagged_total=10, unflagged_total=10, flagged_labels=[None], unflagged_labels=[]
    )
    assert result.precision is None
    assert result.recall is None
