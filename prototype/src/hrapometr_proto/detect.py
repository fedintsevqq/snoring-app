"""Детект храпа по записи ночи: YAMNet по кадрам -> frames.csv -> эпизоды в events.json.

Запуск:  python -m hrapometr_proto.detect runs/night-1
Порог и склейку можно менять без повторного прогона модели: добавьте --reuse.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, YAMNET_HOP, YAMNET_WIN, frame_stream, iter_pcm_blocks, rms_db
from .events import Event, frames_to_events
from .windows import Segment
from .yamnet import (
    CLASS_BREATHING,
    CLASS_SNORING,
    CLASS_SNORT,
    CLASS_SPEECH,
    DEFAULT_MODEL,
    Yamnet,
)

HOP_S = YAMNET_HOP / SAMPLE_RATE
WIN_S = YAMNET_WIN / SAMPLE_RATE
FRAMES_HEADER = "t_s,seg,rms_db,snoring,breathing,snort,speech"
AUDIO_SUFFIXES = {".ogg", ".opus", ".wav", ".m4a", ".mp3", ".flac"}


@dataclass(frozen=True)
class EventParams:
    threshold: float = 0.3
    max_gap_s: float = 6.0
    min_duration_s: float = 1.0


def find_audio(path: Path) -> list[Path]:
    """Один файл или папка с кусками ночи, по порядку имён."""
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"В {path} нет аудиофайлов")
    return files


def analyze_segment(
    path: Path, model: Yamnet, *, gain_db: float, seg_index: int, offset_s: float
) -> tuple[np.ndarray, float]:
    """Прогоняет один файл через модель. Возвращает таблицу кадров и длительность файла в секундах."""
    samples = 0

    def counted_blocks():
        nonlocal samples
        for block in iter_pcm_blocks(path, gain_db=gain_db):
            samples += len(block)
            yield block

    rows = []
    for k, frame in enumerate(frame_stream(counted_blocks())):
        scores = model.scores(frame)
        rows.append(
            (
                offset_s + k * HOP_S,
                seg_index,
                rms_db(frame),
                scores[CLASS_SNORING],
                scores[CLASS_BREATHING],
                scores[CLASS_SNORT],
                scores[CLASS_SPEECH],
            )
        )
        if k % 1200 == 1199:
            print(f"  {path.name}: {(k + 1) * HOP_S / 60:.0f} мин", flush=True)
    table = np.array(rows, dtype=np.float64).reshape(-1, 7)
    return table, samples / SAMPLE_RATE


def detect_events(
    frames: np.ndarray, segments: list[Segment], params: EventParams
) -> list[Event]:
    """Эпизоды по каждому куску отдельно (эпизод не пересекает границу файла), время общее."""
    events: list[Event] = []
    for segment in segments:
        scores = frames[frames[:, 1] == segment.index][:, 3]
        local = frames_to_events(
            scores,
            hop_s=HOP_S,
            win_s=WIN_S,
            threshold=params.threshold,
            max_gap_s=params.max_gap_s,
            min_duration_s=params.min_duration_s,
        )
        events.extend(
            Event(e.start_s + segment.start_s, e.end_s + segment.start_s, e.peak_score)
            for e in local
        )
    return events


def summarize(frames: np.ndarray, segments: list[Segment], events: list[Event]) -> dict:
    total_s = sum(s.duration_s for s in segments)
    snoring_s = sum(e.duration_s for e in events)
    snoring_scores = frames[:, 3] if len(frames) else np.zeros(1)
    return {
        "night_hours": round(total_s / 3600, 3),
        "events": len(events),
        "events_per_hour": round(len(events) / (total_s / 3600), 2) if total_s else 0.0,
        "snoring_minutes": round(snoring_s / 60, 1),
        "snoring_percent": round(100 * snoring_s / total_s, 2) if total_s else 0.0,
        "snoring_score_quantiles": {
            q: round(float(np.quantile(snoring_scores, p)), 4)
            for q, p in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))
        },
    }


def _write_frames(path: Path, frames: np.ndarray) -> None:
    np.savetxt(
        path, frames, delimiter=",", header=FRAMES_HEADER, comments="",
        fmt=["%.3f", "%d", "%.2f", "%.4f", "%.4f", "%.4f", "%.4f"],
    )  # fmt: skip


def _load_run(out_dir: Path) -> tuple[np.ndarray, list[Segment]]:
    frames = np.loadtxt(out_dir / "frames.csv", delimiter=",", skiprows=1, ndmin=2)
    raw = json.loads((out_dir / "segments.json").read_text("utf-8"))
    return frames, [Segment(**item) for item in raw]


def run(
    audio_input: Path, out_dir: Path, model_path: Path, gain_db: float, params: EventParams,
    reuse: bool,
) -> dict:  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    if reuse and (out_dir / "frames.csv").exists() and (out_dir / "segments.json").exists():
        frames, segments = _load_run(out_dir)
        print("Беру готовые frames.csv, модель не запускаю.")
    else:
        model = Yamnet(model_path)
        tables, segments, offset = [], [], 0.0
        started = time.monotonic()
        for index, path in enumerate(find_audio(audio_input)):
            print(f"Файл {index + 1}: {path.name}", flush=True)
            table, duration = analyze_segment(
                path, model, gain_db=gain_db, seg_index=index, offset_s=offset
            )
            tables.append(table)
            segments.append(Segment(index=index, path=str(path), start_s=offset, duration_s=duration))
            offset += duration
        frames = np.vstack(tables) if tables else np.empty((0, 7))
        _write_frames(out_dir / "frames.csv", frames)
        (out_dir / "segments.json").write_text(
            json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2), "utf-8"
        )
        print(f"Модель отработала за {time.monotonic() - started:.0f} с.")

    events = detect_events(frames, segments, params)
    summary = summarize(frames, segments, events)
    (out_dir / "events.json").write_text(
        json.dumps(
            {
                "params": asdict(params),
                "gain_db": gain_db,
                "summary": summary,
                "events": [e.as_dict() for e in events],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Детект храпа по записи ночи")
    parser.add_argument("input", type=Path, help="папка с кусками ночи или один аудиофайл")
    parser.add_argument("--out", type=Path, help="куда писать результат (по умолчанию рядом с записью)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gain-db", type=float, default=0.0, help="усилить звук перед моделью")
    parser.add_argument("--threshold", type=float, default=EventParams.threshold)
    parser.add_argument("--max-gap", type=float, default=EventParams.max_gap_s, help="склеивать эпизоды с паузой до N секунд")
    parser.add_argument("--min-duration", type=float, default=EventParams.min_duration_s)
    parser.add_argument("--reuse", action="store_true", help="не гонять модель заново, если есть frames.csv")
    args = parser.parse_args()

    out_dir = args.out or (args.input if args.input.is_dir() else Path("runs") / args.input.stem)
    params = EventParams(args.threshold, args.max_gap, args.min_duration)
    summary = run(args.input, out_dir, args.model, args.gain_db, params, args.reuse)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Результат: {out_dir / 'events.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
