import pytest

from hrapometr_proto.sweep import WindowSample, sweep_thresholds


def test_perfect_separation_gives_full_precision_and_recall():
    samples = [
        WindowSample("flagged", True, 0.9),
        WindowSample("flagged", True, 0.8),
        WindowSample("unflagged", False, 0.05),
        WindowSample("unflagged", False, 0.1),
    ]
    (row,) = sweep_thresholds(samples, {"flagged": 10, "unflagged": 90}, [0.5])
    assert row.precision == 1.0
    assert row.recall == 1.0
    assert row.flagged_windows == pytest.approx(10.0)


def test_groups_are_weighted_by_their_size_in_the_night():
    # помеченные: 2 окна на 10 в ночи (вес 5 каждое); непомеченные: 2 на 90 (вес 45)
    samples = [
        WindowSample("flagged", True, 0.9),
        WindowSample("flagged", False, 0.8),
        WindowSample("unflagged", True, 0.2),
        WindowSample("unflagged", False, 0.0),
    ]
    strict, loose = sweep_thresholds(samples, {"flagged": 10, "unflagged": 90}, [0.5, 0.1])
    # порог 0.5: TP=5, FP=5, FN=45
    assert strict.precision == pytest.approx(0.5)
    assert strict.recall == pytest.approx(0.1)
    # порог 0.1: сработало и окно 0.2 -> TP=50, FP=5, FN=0
    assert loose.precision == pytest.approx(50 / 55)
    assert loose.recall == pytest.approx(1.0)


def test_unsure_windows_are_ignored():
    samples = [
        WindowSample("flagged", True, 0.9),
        WindowSample("flagged", None, 0.9),
        WindowSample("unflagged", False, 0.0),
    ]
    (row,) = sweep_thresholds(samples, {"flagged": 4, "unflagged": 4}, [0.5])
    # вес помеченных 4 / 1 размеченное = 4
    assert row.flagged_windows == pytest.approx(4.0)


def test_no_predictions_gives_none_precision():
    samples = [WindowSample("flagged", True, 0.2), WindowSample("unflagged", False, 0.0)]
    (row,) = sweep_thresholds(samples, {"flagged": 1, "unflagged": 1}, [0.9])
    assert row.precision is None
    assert row.recall == 0.0


def test_empty_group_does_not_crash():
    (row,) = sweep_thresholds([WindowSample("flagged", True, 0.9)], {"flagged": 5, "unflagged": 5}, [0.5])
    assert row.precision == 1.0
