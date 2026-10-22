"""Раскладка кусков ночи по общей шкале времени с учётом обрывов записи."""

from __future__ import annotations

import re

_NAME = re.compile(r"^seg_(\d+)\.")


def segment_number(file_name: str) -> int | None:
    """Номер куска из имени seg_0007.ogg; для чужих имён и служебных файлов macOS — None."""
    match = _NAME.match(file_name)
    return int(match.group(1)) if match else None


def place_segments(
    numbers: list[int | None],
    durations: list[float | None],
    runs: list[dict],
    nominal_s: float,
) -> list[float]:
    """Начало каждого куска, секунды от старта первого запуска записи.

    numbers — номера кусков из имён, durations — их длины (None: файл не прочитался, берём nominal_s),
    runs — запуски ffmpeg [{"start_s", "first_segment"}] по возрастанию first_segment.
    Внутри одного запуска куски идут встык, между запусками остаётся дыра, равная реальному перерыву.
    Если номеров или запусков нет, куски просто идут друг за другом с нуля.
    """

    def length(value: float | None) -> float:
        return nominal_s if value is None else value

    if not runs or any(number is None for number in numbers):
        starts, cursor = [], 0.0
        for duration in durations:
            starts.append(cursor)
            cursor += length(duration)
        return starts

    starts = []
    current_run = None
    cursor = 0.0
    previous_number = None
    for number, duration in zip(numbers, durations):
        run = next(
            (r for r in reversed(runs) if r["first_segment"] <= number), runs[0]
        )
        if run is not current_run:
            current_run, cursor, previous_number = run, run["start_s"], None
        if previous_number is not None:
            cursor += (number - previous_number - 1) * nominal_s  # пропавшие файлы
        starts.append(cursor)
        cursor += length(duration)
        previous_number = number
    return starts
