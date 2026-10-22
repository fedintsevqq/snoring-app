"""Ручная проверка детектора: готовим слепую выборку окон и считаем точность и полноту.

  python -m hrapometr_proto.review prepare runs/night-1     # клипы и index.html
  python -m hrapometr_proto.review report runs/night-1 ~/Downloads/labels.json
  python -m hrapometr_proto.review sweep runs/night-1 ~/Downloads/labels.json
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
import wave
from collections import Counter
from pathlib import Path, PureWindowsPath

import numpy as np

from .audio import SAMPLE_RATE, iter_pcm_blocks, normalize_level
from .events import Event
from .review_page import render_page
from .stats import Metrics, estimate_metrics
from .sweep import WindowSample, sweep_thresholds
from .windows import (
    BORDERLINE,
    FLAGGED,
    UNFLAGGED,
    Window,
    classify_windows,
    make_windows,
    sample_windows,
    segments_from_doc,
)

NON_SNORE_LABELS = {"breathing", "speech", "noise"}
VALID_LABELS = {"snore", "unsure"} | NON_SNORE_LABELS
CLIP_TARGET_DB = -6.0
CLIP_MAX_GAIN_DB = 30.0
SWEEP_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def validate_labels_doc(key: dict, doc: dict) -> None:
    """labels.json должен быть от этой же выборки: совпадает токен, нет чужих номеров клипов."""
    if doc.get("token") != key.get("token"):
        raise ValueError(
            "labels.json от другой выборки (токен не совпадает). Скорее всего, после него снова "
            "запускали prepare. Откройте актуальный index.html и разметьте заново."
        )
    unknown = sorted(set(doc.get("labels", {})) - set(key["clips"]))
    if unknown:
        raise ValueError(f"В labels.json есть клипы, которых нет в выборке: {', '.join(unknown)}")


def split_labels(
    key: dict, labels: dict[str, str]
) -> tuple[list[bool | None], list[bool | None], dict[str, int]]:
    """Разбивает ответы по группам: (помеченные, непомеченные, причины «не храп» среди помеченных).

    snore -> True, breathing/speech/noise -> False, unsure -> None. Клипы без ответа пропускаем.
    """
    flagged: list[bool | None] = []
    unflagged: list[bool | None] = []
    reasons: Counter[str] = Counter()
    for clip_id, info in key["clips"].items():
        label = labels.get(clip_id)
        if label is None:
            continue
        if label not in VALID_LABELS:
            raise ValueError(f"Неизвестная метка {label!r} у {clip_id}")
        verdict = None if label == "unsure" else label == "snore"
        if info["group"] == FLAGGED:
            flagged.append(verdict)
            if label in NON_SNORE_LABELS:
                reasons[label] += 1
        else:
            unflagged.append(verdict)
    return flagged, unflagged, dict(reasons)


def _segment_path(raw: str, run_dir: Path) -> Path:
    """Путь к куску ночи. Если папку перенесли (в том числе с Windows на Mac), ищем по имени рядом."""
    path = Path(raw)
    if path.exists():
        return path
    return run_dir / PureWindowsPath(raw).name


def _write_clip(segment_path: Path, local_start_s: float, length_s: float, target: Path) -> None:
    blocks = list(iter_pcm_blocks(segment_path, start_s=local_start_s, duration_s=length_s))
    signal = np.concatenate(blocks) if blocks else np.zeros(1, dtype=np.float32)
    loud = normalize_level(signal, CLIP_TARGET_DB, CLIP_MAX_GAIN_DB)
    pcm = (np.clip(loud, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def prepare(run_dir: Path, per_group: int, win_s: float, seed: int) -> None:
    segments = segments_from_doc(json.loads((run_dir / "segments.json").read_text("utf-8")))
    events_doc = json.loads((run_dir / "events.json").read_text("utf-8"))
    events = [Event(e["start_s"], e["end_s"], e["peak_score"]) for e in events_doc["events"]]

    windows = make_windows(segments, win_s=win_s)
    classes = classify_windows(windows, events)
    flagged_pick, unflagged_pick = sample_windows(windows, classes, per_group=per_group, seed=seed)

    picks: list[tuple[str, Window]] = [(FLAGGED, w) for w in flagged_pick] + [
        (UNFLAGGED, w) for w in unflagged_pick
    ]
    random.Random(seed + 1).shuffle(picks)  # порядок не выдаёт группу

    review_dir = run_dir / "review"
    clips_dir = review_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("c*.wav"):  # клипы прошлой выборки не должны путаться с новыми
        old.unlink()
    by_index = {s.index: s for s in segments}

    key_clips: dict[str, dict] = {}
    page_clips: list[dict] = []
    for number, (group, window) in enumerate(picks, start=1):
        clip_id = f"c{number:03d}"
        segment = by_index[window.seg_index]
        _write_clip(
            _segment_path(segment.path, run_dir),
            window.start_s - segment.start_s,
            window.end_s - window.start_s,
            clips_dir / f"{clip_id}.wav",
        )
        key_clips[clip_id] = {"group": group, "start_s": window.start_s, "end_s": window.end_s}
        page_clips.append({"id": clip_id, "file": f"clips/{clip_id}.wav"})

    token = secrets.token_hex(6)
    key = {
        "token": token,
        "window_s": win_s,
        "seed": seed,
        "flagged_total": classes.count(FLAGGED),
        "unflagged_total": classes.count(UNFLAGGED),
        "borderline_total": classes.count(BORDERLINE),
        "event_params": events_doc["params"],
        "clips": key_clips,
    }
    (review_dir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), "utf-8")
    (review_dir / "index.html").write_text(render_page(run_dir.name, token, page_clips), "utf-8")
    print(
        f"Готово: {len(page_clips)} окон по {win_s:g} с "
        f"(в ночи помеченных {key['flagged_total']}, непомеченных {key['unflagged_total']}, "
        f"спорных {key['borderline_total']} — их в выборку не берём)."
    )
    print(f"Откройте в браузере: {(review_dir / 'index.html').resolve()}")


def _percent(value: float | None) -> str:
    return "нет данных" if value is None else f"{value * 100:.0f}%"


def _interval(ci: tuple[float, float] | None) -> str:
    return "" if ci is None else f" (95%: {ci[0] * 100:.0f}–{ci[1] * 100:.0f}%)"


def _unsure(range_: tuple[float, float] | None) -> str:
    if range_ is None or abs(range_[1] - range_[0]) < 1e-9:
        return ""
    return f" Если «не уверен» посчитать крайними случаями: от {range_[0] * 100:.0f}% до {range_[1] * 100:.0f}%."


def format_report(run_name: str, key: dict, metrics: Metrics, reasons: dict[str, int]) -> str:
    params = key["event_params"]
    lines = [
        f"# Результат проверки детектора: {run_name}",
        "",
        f"Параметры: порог {params['threshold']}, склейка пауз до {params['max_gap_s']} с, "
        f"минимум {params['min_duration_s']} с. Окно {key['window_s']:g} с.",
        "",
        f"- **Точность** (в помеченных окнах правда храп): {_percent(metrics.precision)}"
        f"{_interval(metrics.precision_ci)}. Размечено {metrics.flagged_labeled}, «не уверен» {metrics.flagged_unsure}."
        f"{_unsure(metrics.precision_unsure_range)}",
        f"- **Доля храпа среди непомеченных окон**: {_percent(metrics.miss_rate)}"
        f"{_interval(metrics.miss_rate_ci)}. Размечено {metrics.unflagged_labeled}, «не уверен» {metrics.unflagged_unsure}."
        f"{_unsure(metrics.miss_rate_unsure_range)}",
        f"- **Полнота** (какую часть храпа нашли): {_percent(metrics.recall)}{_interval(metrics.recall_ci)}."
        f"{_unsure(metrics.recall_unsure_range)}",
        "",
        f"Окон в ночи: помеченных {key['flagged_total']}, непомеченных {key['unflagged_total']}, "
        f"спорных {key.get('borderline_total', 0)} (хвост эпизода меньше секунды, в оценку не входят).",
    ]
    if reasons:
        lines += [
            "",
            "Что детектор принял за храп: " + ", ".join(f"{k} — {v}" for k, v in reasons.items()) + ".",
        ]
    return "\n".join(lines) + "\n"


def _load_labels(run_dir: Path, labels_path: Path) -> tuple[dict, dict]:
    key = json.loads((run_dir / "review" / "key.json").read_text("utf-8"))
    doc = json.loads(labels_path.read_text("utf-8"))
    validate_labels_doc(key, doc)
    return key, doc["labels"]


def report(run_dir: Path, labels_path: Path) -> None:
    key, labels = _load_labels(run_dir, labels_path)
    flagged, unflagged, reasons = split_labels(key, labels)
    metrics = estimate_metrics(
        flagged_total=key["flagged_total"],
        unflagged_total=key["unflagged_total"],
        flagged_labels=flagged,
        unflagged_labels=unflagged,
    )
    text = format_report(run_dir.name, key, metrics, reasons)
    (run_dir / "review" / "report.md").write_text(text, "utf-8")
    print(text)


def sweep(run_dir: Path, labels_path: Path) -> None:
    """Точность и полнота при других порогах по тем же размеченным окнам."""
    key, labels = _load_labels(run_dir, labels_path)
    frames = np.loadtxt(run_dir / "frames.csv", delimiter=",", skiprows=1, ndmin=2)
    times, scores = frames[:, 0], frames[:, 3]

    samples = []
    for clip_id, info in key["clips"].items():
        label = labels.get(clip_id)
        if label is None:
            continue
        inside = (times >= info["start_s"]) & (times < info["end_s"])
        peak = float(scores[inside].max()) if inside.any() else 0.0
        verdict = None if label == "unsure" else label == "snore"
        samples.append(WindowSample(info["group"], verdict, peak))

    rows = sweep_thresholds(
        samples,
        {FLAGGED: key["flagged_total"], UNFLAGGED: key["unflagged_total"]},
        SWEEP_THRESHOLDS,
    )
    print("Порог | точность | полнота | окон с храпом в ночи (оценка)")
    for row in rows:
        print(
            f"{row.threshold:>5} | {_percent(row.precision):>8} | {_percent(row.recall):>7} | "
            f"{row.flagged_windows:.0f}"
        )
    print("\nОценка по пику Snoring в окне, без склейки и минимальной длины эпизода. Выборка небольшая, цифры ориентировочные.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="нарезать клипы и собрать index.html")
    prep.add_argument("run_dir", type=Path)
    prep.add_argument("--per-group", type=int, default=40, help="окон из каждой группы")
    prep.add_argument("--win", type=float, default=30.0, help="длина окна, секунд")
    prep.add_argument("--seed", type=int, default=1)

    for name, help_text in (
        ("report", "посчитать точность и полноту по labels.json"),
        ("sweep", "то же при других порогах, без новой разметки"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("run_dir", type=Path)
        command.add_argument("labels", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.run_dir, args.per_group, args.win, args.seed)
        elif args.command == "report":
            report(args.run_dir, args.labels)
        else:
            sweep(args.run_dir, args.labels)
    except ValueError as error:
        print(f"Ошибка: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
