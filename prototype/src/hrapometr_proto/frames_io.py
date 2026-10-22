"""Таблица кадров ночи (frames.csv): запись, чтение, округление."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

FRAMES_HEADER = "t_s,seg,rms_db,snoring,breathing,snort,speech"
FRAMES_DECIMALS = (3, 0, 2, 4, 4, 4, 4)  # знаков после запятой по колонкам


def quantize_frames(frames: np.ndarray) -> np.ndarray:
    """Округляет колонки так же, как они пишутся в frames.csv, чтобы --reuse давал те же эпизоды."""
    result = frames.copy()
    for column, decimals in enumerate(FRAMES_DECIMALS):
        result[:, column] = np.round(result[:, column], decimals)
    return result


def write_frames(path: Path, frames: np.ndarray) -> None:
    fmt = [f"%.{decimals}f" for decimals in FRAMES_DECIMALS]
    np.savetxt(path, frames, delimiter=",", header=FRAMES_HEADER, comments="", fmt=fmt)


def load_frames(path: Path) -> np.ndarray:
    """Читает frames.csv. Пустая таблица (только заголовок) даёт массив (0, 7)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # «input contained no data» для пустой ночи
        frames = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return np.empty((0, 7)) if frames.size == 0 else frames
