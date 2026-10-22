"""Окна по 30 секунд для ручной проверки: окно с эпизодом детектора — «помеченное»."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .events import Event


@dataclass(frozen=True)
class Segment:
    index: int
    path: str
    start_s: float  # начало сегмента на общей шкале ночи
    duration_s: float


@dataclass(frozen=True)
class Window:
    seg_index: int
    start_s: float  # на общей шкале ночи
    end_s: float


def make_windows(
    segments: list[Segment], win_s: float = 30.0, min_partial_s: float = 10.0
) -> list[Window]:
    """Режем каждый сегмент на окна, чтобы ни одно не пересекало границу файла.

    Неполное окно в конце сегмента оставляем, если оно не короче min_partial_s.
    """
    windows: list[Window] = []
    for segment in segments:
        offset = 0.0
        while offset < segment.duration_s - 1e-9:
            length = min(win_s, segment.duration_s - offset)
            if length >= min_partial_s - 1e-9:
                windows.append(
                    Window(
                        seg_index=segment.index,
                        start_s=segment.start_s + offset,
                        end_s=segment.start_s + offset + length,
                    )
                )
            offset += win_s
    return windows


FLAGGED = "flagged"
UNFLAGGED = "unflagged"
# хвост эпизода короче min_overlap_s на краю окна: детектор тут «почти» сработал, судить нельзя
BORDERLINE = "borderline"


def _overlap_s(window: Window, event: Event) -> float:
    return max(0.0, min(window.end_s, event.end_s) - max(window.start_s, event.start_s))


def classify_windows(
    windows: list[Window], events: list[Event], min_overlap_s: float = 1.0
) -> list[str]:
    """Для каждого окна: FLAGGED (эпизоды в нём дают >= min_overlap_s), UNFLAGGED (ни одного) или BORDERLINE."""
    result = []
    for window in windows:
        overlap = sum(_overlap_s(window, event) for event in events)
        if overlap >= min_overlap_s - 1e-9:
            result.append(FLAGGED)
        elif overlap > 0.0:
            result.append(BORDERLINE)
        else:
            result.append(UNFLAGGED)
    return result


def sample_windows(
    windows: list[Window], classes: list[str], *, per_group: int, seed: int
) -> tuple[list[Window], list[Window]]:
    """Случайно выбираем до per_group окон из помеченных и из непомеченных (спорные не берём). Повторяемо по seed."""
    flagged = [w for w, c in zip(windows, classes) if c == FLAGGED]
    unflagged = [w for w, c in zip(windows, classes) if c == UNFLAGGED]
    rng = random.Random(seed)
    pick_flagged = sorted(rng.sample(flagged, min(per_group, len(flagged))), key=lambda w: w.start_s)
    pick_unflagged = sorted(
        rng.sample(unflagged, min(per_group, len(unflagged))), key=lambda w: w.start_s
    )
    return pick_flagged, pick_unflagged
