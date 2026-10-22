"""Подбор порога по уже размеченным окнам, без новой разметки.

Выборка окон взята из двух групп (помеченные и непомеченные при исходном пороге), поэтому каждое
размеченное окно «представляет» вес = размер группы в ночи / число размеченных окон группы.
Окно считаем найденным, если пик оценки Snoring внутри него >= порога.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowSample:
    group: str  # "flagged" или "unflagged" при исходном пороге
    is_snore: bool | None  # None — «не уверен», в расчёт не входит
    peak_score: float  # максимум оценки Snoring по кадрам окна


@dataclass(frozen=True)
class SweepRow:
    threshold: float
    precision: float | None
    recall: float | None
    flagged_windows: float  # сколько окон ночи сработало бы при этом пороге (оценка)


def sweep_thresholds(
    samples: list[WindowSample], group_totals: dict[str, int], thresholds: Iterable[float]
) -> list[SweepRow]:
    labeled = [s for s in samples if s.is_snore is not None]
    counts = {group: sum(1 for s in labeled if s.group == group) for group in group_totals}
    weights = {
        group: (group_totals[group] / counts[group] if counts[group] else 0.0)
        for group in group_totals
    }

    rows = []
    for threshold in thresholds:
        tp = fp = fn = 0.0
        for sample in labeled:
            weight = weights.get(sample.group, 0.0)
            predicted = sample.peak_score >= threshold
            if predicted and sample.is_snore:
                tp += weight
            elif predicted:
                fp += weight
            elif sample.is_snore:
                fn += weight
        rows.append(
            SweepRow(
                threshold=threshold,
                precision=tp / (tp + fp) if tp + fp > 0 else None,
                recall=tp / (tp + fn) if tp + fn > 0 else None,
                flagged_windows=tp + fp,
            )
        )
    return rows
