"""Чтение аудио через ffmpeg и нарезка на кадры для YAMNet."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
YAMNET_WIN = 15600  # 0.975 с — вход модели
YAMNET_HOP = 7800  # половина окна
_FLOOR = 1e-9


def rms_db(frame: np.ndarray) -> float:
    """Громкость кадра в dBFS. Тишина даёт около -180, а не -inf."""
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    return 20.0 * np.log10(rms + _FLOOR)


def apply_gain_db(signal: np.ndarray, gain_db: float) -> np.ndarray:
    """Усиление в dB с обрезкой по полной шкале. Возвращает новый массив."""
    if gain_db == 0.0:
        return signal
    return np.clip(signal * np.float32(10.0 ** (gain_db / 20.0)), -1.0, 1.0)


def normalize_level(
    signal: np.ndarray, target_db: float, max_gain_db: float, percentile: float = 99.5
) -> np.ndarray:
    """Поднимает типичную громкость клипа до target_db (dBFS), но не больше чем на max_gain_db.

    Опорная громкость — 99.5-й перцентиль модуля, а не максимум: один хлопок двери
    не должен глушить тихий храп в том же окне. Одиночные выбросы после усиления обрезаются.
    Нужно для прослушивания тихих клипов человеком. Для модели не используется.
    """
    reference = float(np.percentile(np.abs(signal), percentile)) if signal.size else 0.0
    if reference == 0.0:
        return signal.copy()
    gain_db = min(target_db - 20.0 * np.log10(reference), max_gain_db)
    return apply_gain_db(signal, gain_db)


def iter_pcm_blocks(
    path: str | Path,
    *,
    block_s: float = 60.0,
    gain_db: float = 0.0,
    start_s: float | None = None,
    duration_s: float | None = None,
    ffmpeg: str = "ffmpeg",
) -> Iterator[np.ndarray]:
    """Декодирует любой аудиофайл в моно 16 кГц float32 и отдаёт блоками по block_s секунд.

    start_s и duration_s вырезают кусок файла (быстрая перемотка до чтения).
    """
    seek = []
    if start_s is not None:
        seek += ["-ss", f"{start_s:.3f}"]
    if duration_s is not None:
        seek += ["-t", f"{duration_s:.3f}"]
    cmd = [
        ffmpeg, "-nostdin", "-v", "error", *seek, "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]  # fmt: skip
    block_bytes = int(block_s * SAMPLE_RATE) * 4
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(block_bytes)
            if not raw:
                break
            usable = len(raw) // 4 * 4
            yield apply_gain_db(np.frombuffer(raw[:usable], dtype="<f4"), gain_db)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg не смог прочитать {path} (код {proc.returncode})")


def frame_stream(
    blocks: Iterable[np.ndarray], *, win: int = YAMNET_WIN, hop: int = YAMNET_HOP
) -> Iterator[np.ndarray]:
    """Из потока блоков делает кадры длиной win с шагом hop. Границы блоков на кадры не влияют."""
    buffer = np.empty(0, dtype=np.float32)
    for block in blocks:
        buffer = np.concatenate([buffer, block])
        if len(buffer) < win:
            continue
        count = (len(buffer) - win) // hop + 1
        for k in range(count):
            yield buffer[k * hop : k * hop + win]
        buffer = buffer[count * hop :]
