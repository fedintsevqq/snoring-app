"""Оценка точности и полноты детектора по ручной разметке выборки окон."""

from __future__ import annotations

import math
from dataclasses import dataclass

_Z95 = 1.96
# Интервал полноты строим из двух независимых долей. Чтобы вместе они покрывали около 95%,
# каждая берётся с покрытием около 97.5% (z = 2.24): 0.975 * 0.975 > 0.95. Интервал Уилсона
# приближённый, поэтому «около», а не «гарантированно».
_Z_JOINT95 = 2.24


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
    # границы, если все «не уверен» оказались бы не храпом или, наоборот, храпом; None, если «не уверен» нет
    precision_unsure_range: tuple[float, float] | None = None
    miss_rate_unsure_range: tuple[float, float] | None = None
    recall_unsure_range: tuple[float, float] | None = None


def _share(labels: list[bool | None]) -> tuple[int, int, int]:
    """Возвращает (число разметки, число «не уверен», число «храп»)."""
    decided = [label for label in labels if label is not None]
    return len(decided), len(labels) - len(decided), sum(decided)


def _recall(
    precision: float, miss_rate: float, flagged_total: int, unflagged_total: int
) -> float | None:
    """Доля найденного храпа. Если храпа нет нигде, полноту не определить (None)."""
    found = precision * flagged_total
    missed = miss_rate * unflagged_total
    return found / (found + missed) if found + missed > 0 else None


def _recall_range(
    precision_range: tuple[float, float],
    miss_range: tuple[float, float],
    flagged_total: int,
    unflagged_total: int,
) -> tuple[float, float] | None:
    low = _recall(precision_range[0], miss_range[1], flagged_total, unflagged_total)
    high = _recall(precision_range[1], miss_range[0], flagged_total, unflagged_total)
    if low is None or high is None:
        return None
    return (low, high)


def _unsure_range(hits: int, decided: int, unsure: int) -> tuple[float, float] | None:
    total = decided + unsure
    if total == 0:
        return None
    return (hits / total, (hits + unsure) / total)


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
    Интервал полноты считаем по границам двух долей (z = 2.24), он покрывает около 95%.
    «Не уверен» в оценку не входит, а крайние случаи по ним отдельно: *_unsure_range.
    """
    f_n, f_unsure, f_hits = _share(flagged_labels)
    u_n, u_unsure, u_hits = _share(unflagged_labels)

    precision = f_hits / f_n if f_n else None
    miss_rate = u_hits / u_n if u_n else None
    precision_ci = wilson_interval(f_hits, f_n) if f_n else None
    miss_rate_ci = wilson_interval(u_hits, u_n) if u_n else None

    # Пустая группа в ночи (детектор не пометил ни одного окна или пометил все) не требует разметки:
    # в ней найдено 0 или пропущено 0 по определению.
    p_for_recall = 0.0 if flagged_total == 0 else precision
    m_for_recall = 0.0 if unflagged_total == 0 else miss_rate

    recall = recall_ci = None
    if p_for_recall is not None and m_for_recall is not None:
        recall = _recall(p_for_recall, m_for_recall, flagged_total, unflagged_total)
        if recall is not None:
            recall_ci = _recall_range(
                (0.0, 0.0) if flagged_total == 0 else wilson_interval(f_hits, f_n, _Z_JOINT95),
                (0.0, 0.0) if unflagged_total == 0 else wilson_interval(u_hits, u_n, _Z_JOINT95),
                flagged_total,
                unflagged_total,
            )

    precision_unsure_range = miss_rate_unsure_range = recall_unsure_range = None
    if f_unsure + u_unsure > 0:
        p_range = (0.0, 0.0) if flagged_total == 0 else _unsure_range(f_hits, f_n, f_unsure)
        m_range = (0.0, 0.0) if unflagged_total == 0 else _unsure_range(u_hits, u_n, u_unsure)
        precision_unsure_range, miss_rate_unsure_range = p_range, m_range
        if p_range is not None and m_range is not None:
            recall_unsure_range = _recall_range(p_range, m_range, flagged_total, unflagged_total)

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
        precision_unsure_range=precision_unsure_range,
        miss_rate_unsure_range=miss_rate_unsure_range,
        recall_unsure_range=recall_unsure_range,
    )
