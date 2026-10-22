"""Оценка точности и полноты детектора по ручной разметке выборки окон."""

from __future__ import annotations

import math
from dataclasses import dataclass

_Z95 = 1.96


def wilson_interval(successes: int, total: int, z: float = _Z95) -> tuple[float, float]:
    """Доверительный интервал Уилсона для доли. Без данных — весь диапазон [0, 1]."""
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass(frozen=True)
class Metrics:
    flagged_labeled: int
    flagged_unsure: int
    unflagged_labeled: int
    unflagged_unsure: int
    precision: float | None
    precision_ci: tuple[float, float] | None
    miss_rate: float | None
    miss_rate_ci: tuple[float, float] | None
    recall: float | None
    recall_ci: tuple[float, float] | None


def _share(labels: list[bool | None]) -> tuple[int, int, int]:
    """Возвращает (число разметки, число «не уверен», число «храп»)."""
    decided = [label for label in labels if label is not None]
    return len(decided), len(labels) - len(decided), sum(decided)


def _recall(precision: float, miss_rate: float, flagged_total: int, unflagged_total: int) -> float:
    found = precision * flagged_total
    missed = miss_rate * unflagged_total
    return found / (found + missed) if found + missed > 0 else 0.0


def estimate_metrics(
    *,
    flagged_total: int,
    unflagged_total: int,
    flagged_labels: list[bool | None],
    unflagged_labels: list[bool | None],
) -> Metrics:
    """Точность и полнота в окнах по случайным выборкам из двух групп.

    precision — доля окон с настоящим храпом среди помеченных детектором.
    miss_rate — доля окон с храпом среди непомеченных.
    recall — храп в помеченных / (храп в помеченных + пропущенный храп), группы взвешены размером.
    Интервал полноты консервативный: берём худшее и лучшее сочетание границ двух долей.
    """
    f_n, f_unsure, f_hits = _share(flagged_labels)
    u_n, u_unsure, u_hits = _share(unflagged_labels)

    precision = f_hits / f_n if f_n else None
    miss_rate = u_hits / u_n if u_n else None
    precision_ci = wilson_interval(f_hits, f_n) if f_n else None
    miss_rate_ci = wilson_interval(u_hits, u_n) if u_n else None

    recall = recall_ci = None
    if precision is not None and miss_rate is not None:
        recall = _recall(precision, miss_rate, flagged_total, unflagged_total)
        recall_ci = (
            _recall(precision_ci[0], miss_rate_ci[1], flagged_total, unflagged_total),
            _recall(precision_ci[1], miss_rate_ci[0], flagged_total, unflagged_total),
        )

    return Metrics(
        flagged_labeled=f_n,
        flagged_unsure=f_unsure,
        unflagged_labeled=u_n,
        unflagged_unsure=u_unsure,
        precision=precision,
        precision_ci=precision_ci,
        miss_rate=miss_rate,
        miss_rate_ci=miss_rate_ci,
        recall=recall,
        recall_ci=recall_ci,
    )
