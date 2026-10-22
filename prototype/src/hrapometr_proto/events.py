"""Склейка покадровых оценок храпа в эпизоды."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# допуск на ошибки округления при сравнении пауз
_EPS = 1e-9


@dataclass(frozen=True)
class Event:
    start_s: float
    end_s: float
    peak_score: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def as_dict(self) -> dict:
        return {**asdict(self), "duration_s": self.duration_s}


def frames_to_events(
    scores: np.ndarray,
    *,
    hop_s: float,
    win_s: float,
    threshold: float,
    max_gap_s: float,
    min_duration_s: float,
) -> list[Event]:
    """Кадры с оценкой >= threshold склеиваем в эпизод, если пауза между ними <= max_gap_s.

    Кадр i покрывает отрезок [i * hop_s, i * hop_s + win_s].
    Эпизоды короче min_duration_s отбрасываем.
    """
    scores = np.asarray(scores)
    positive = np.flatnonzero(scores >= threshold)
    if positive.size == 0:
        return []

    starts = positive * hop_s
    ends = starts + win_s
    breaks = np.flatnonzero(starts[1:] - ends[:-1] > max_gap_s + _EPS) + 1
    groups = np.split(np.arange(positive.size), breaks)

    events = [
        Event(
            start_s=float(starts[group[0]]),
            end_s=float(ends[group[-1]]),
            peak_score=float(scores[positive[group]].max()),
        )
        for group in groups
    ]
    return [event for event in events if event.duration_s >= min_duration_s - _EPS]
