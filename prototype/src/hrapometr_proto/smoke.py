"""Проверка проводки детектора на коротких клипах ESC-50: храп против похожих и обычных звуков.

ESC-50 — лицензия CC BY-NC 3.0 (личное некоммерческое использование). Клипы лежат в prototype/data,
в репозиторий не попадают.

Запуск:
  python -m hrapometr_proto.smoke fetch   # скачать ~110 клипов (около 50 МБ)
  python -m hrapometr_proto.smoke eval    # посчитать оценки Snoring
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .audio import frame_stream, iter_pcm_blocks
from .yamnet import CLASS_SNORING, DEFAULT_MODEL, Yamnet

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "esc50"
META_DIR = DATA_DIR / "meta"
AUDIO_DIR = DATA_DIR / "audio"
_BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
_FILENAME = re.compile(r"^\d-\d+-[A-Z]-\d+\.wav$")

POSITIVE = "snoring"
NEGATIVES = [
    "breathing", "coughing", "sneezing", "laughing", "keyboard_typing", "vacuum_cleaner",
    "washing_machine", "clock_tick", "door_wood_knock", "footsteps", "crying_baby", "wind",
]  # fmt: skip
PER_NEGATIVE = 6
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - адрес из константы
        target.write_bytes(response.read())


def _select_clips(meta_path: Path) -> list[tuple[str, str]]:
    """Пары (категория, имя файла): все клипы храпа и по PER_NEGATIVE клипов остальных категорий."""
    by_category: dict[str, list[str]] = {}
    with meta_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filename = row["filename"]
            if _FILENAME.match(filename):
                by_category.setdefault(row["category"], []).append(filename)
    chosen = [(POSITIVE, name) for name in sorted(by_category.get(POSITIVE, []))]
    for category in NEGATIVES:
        chosen.extend((category, n) for n in sorted(by_category.get(category, []))[:PER_NEGATIVE])
    return chosen


def fetch() -> int:
    META_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = META_DIR / "esc50.csv"
    if not meta_path.exists():
        _download(f"{_BASE}/meta/esc50.csv", meta_path)
    clips = _select_clips(meta_path)
    for number, (_, filename) in enumerate(clips, start=1):
        target = AUDIO_DIR / filename
        if not target.exists():
            _download(f"{_BASE}/audio/{filename}", target)
        if number % 20 == 0:
            print(f"  {number}/{len(clips)}", flush=True)
    print(f"Готово: {len(clips)} клипов в {AUDIO_DIR}")
    return 0


def evaluate() -> int:
    meta_path = META_DIR / "esc50.csv"
    if not meta_path.exists():
        print("Сначала: python -m hrapometr_proto.smoke fetch")
        return 1
    model = Yamnet(DEFAULT_MODEL)
    peaks: dict[str, list[float]] = {}
    for category, filename in _select_clips(meta_path):
        path = AUDIO_DIR / filename
        if not path.exists():
            continue
        scores = [model.scores(f)[CLASS_SNORING] for f in frame_stream(iter_pcm_blocks(path))]
        peaks.setdefault(category, []).append(max(scores) if scores else 0.0)

    positives = np.array(peaks.get(POSITIVE, []))
    if positives.size == 0:
        print("Нет клипов храпа, запустите fetch.")
        return 1
    print(f"Храп: {positives.size} клипов, пик Snoring: медиана {np.median(positives):.2f}, "
          f"мин {positives.min():.2f}, макс {positives.max():.2f}\n")
    print("Порог | храп найден | ложные срабатывания по категориям (клипов из %d)" % PER_NEGATIVE)
    for threshold in THRESHOLDS:
        found = int((positives >= threshold).sum())
        false_alarms = {
            cat: int((np.array(vals) >= threshold).sum())
            for cat, vals in peaks.items()
            if cat != POSITIVE and (np.array(vals) >= threshold).any()
        }
        print(f"{threshold:>5} | {found}/{positives.size}".ljust(22) + f" | {false_alarms or 'нет'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["fetch", "eval"])
    args = parser.parse_args()
    return fetch() if args.command == "fetch" else evaluate()


if __name__ == "__main__":
    sys.exit(main())
