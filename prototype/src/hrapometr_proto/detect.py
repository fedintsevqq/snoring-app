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
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio import (
    SAMPLE_RATE,
    YAMNET_HOP,
    YAMNET_WIN,
    AudioDecodeError,
    frame_stream,
    iter_pcm_blocks,
    rms_db,
)
from .events import Event, frames_to_events
from .frames_io import load_frames, quantize_frames, write_frames
from .timeline import place_segments, segment_number
from .windows import Segment, segments_from_doc
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
AUDIO_SUFFIXES = {".ogg", ".opus", ".wav", ".m4a", ".mp3", ".flac"}
MIN_AUDIO_BYTES = 1024  # файлы меньше считаем мусором (оборванная запись, пустышка)
GAP_WARNING_S = 120.0  # на столько запись может отличаться от времени на часах без предупреждения


@dataclass(frozen=True)
class EventParams:
    threshold: float = 0.3
    max_gap_s: float = 6.0
    # Один кадр длится 0.975 с, поэтому одиночный всплеск оценки эпизодом не считаем. Два кадра
    # склеиваются, если пауза между ними не больше max_gap_s, и тогда эпизод не короче 1.46 с.
    min_duration_s: float = 1.4


def find_audio(path: Path) -> list[Path]:
    """Один файл или папка с кусками ночи по порядку имён. Служебные и пустые файлы пропускаем."""
    if path.is_file():
        return [path]
    files = sorted(
        p
        for p in path.iterdir()
        if p.suffix.lower() in AUDIO_SUFFIXES
        and not p.name.startswith(".")
        and p.stat().st_size >= MIN_AUDIO_BYTES
    )
    if not files:
        raise FileNotFoundError(f"В {path} нет читаемых аудиофайлов")
    return files


def analyze_segment(path: Path, model: Yamnet, *, gain_db: float) -> tuple[np.ndarray, float]:
    """Прогоняет один файл через модель. Возвращает таблицу кадров (время от начала файла) и длину файла в секундах.

    Если ffmpeg споткнулся в конце файла, уже прочитанный звук оставляем (с предупреждением).
    Если не прочитано ничего, поднимаем AudioDecodeError.
    """
    samples = 0

    def counted_blocks():
        nonlocal samples
        for block in iter_pcm_blocks(path, gain_db=gain_db):
            samples += len(block)
            yield block

    rows = []
    try:
        for k, frame in enumerate(frame_stream(counted_blocks())):
            scores = model.scores(frame)
            rows.append(
                (
                    k * HOP_S,
                    0,
                    rms_db(frame),
                    scores[CLASS_SNORING],
                    scores[CLASS_BREATHING],
                    scores[CLASS_SNORT],
                    scores[CLASS_SPEECH],
                )
            )
            if k % 1200 == 1199:
                print(f"  {path.name}: {(k + 1) * HOP_S / 60:.0f} мин", flush=True)
    except AudioDecodeError as error:
        if not rows:
            raise
        print(f"  ВНИМАНИЕ: {path.name} прочитан не до конца ({error}). Беру то, что успели прочитать.")
    table = np.array(rows, dtype=np.float64).reshape(-1, 7)
    return table, samples / SAMPLE_RATE


def detect_events(frames: np.ndarray, segments: list[Segment], params: EventParams) -> list[Event]:
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
    audio_s = sum(s.duration_s for s in segments)
    snoring_s = sum(e.duration_s for e in events)
    snoring_scores = frames[:, 3] if len(frames) else np.zeros(1)
    return {
        "audio_hours": round(audio_s / 3600, 3),
        "events": len(events),
        "events_per_hour": round(len(events) / (audio_s / 3600), 2) if audio_s else 0.0,
        "snoring_minutes": round(snoring_s / 60, 1),
        "snoring_percent": round(100 * snoring_s / audio_s, 2) if audio_s else 0.0,
        "snoring_score_quantiles": {
            q: round(float(np.quantile(snoring_scores, p)), 4)
            for q, p in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))
        },
    }


def _load_run(out_dir: Path) -> tuple[np.ndarray, list[Segment], dict]:
    frames = load_frames(out_dir / "frames.csv")
    doc = json.loads((out_dir / "segments.json").read_text("utf-8"))
    if isinstance(doc, list):  # старый формат без gain_db
        doc = {"gain_db": 0.0, "segments": doc}
    return frames, segments_from_doc(doc), doc


def _read_session(audio_input: Path) -> dict | None:
    path = audio_input / "session.json" if audio_input.is_dir() else None
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        print("session.json не читается, время считаю по длине файлов.")
        return None


def _runs_from_session(session: dict | None) -> list[dict]:
    """Запуски ffmpeg со временем от старта первого запуска в секундах."""
    if not session or not session.get("runs"):
        return []
    first = datetime.fromisoformat(session["runs"][0]["started_at"])
    return [
        {
            "start_s": (datetime.fromisoformat(r["started_at"]) - first).total_seconds(),
            "first_segment": r["first_segment"],
        }
        for r in session["runs"]
    ]


def _stored_path(path: Path, out_dir: Path) -> str:
    """Рядом с результатом — одно имя файла (папку можно переносить), иначе полный путь."""
    resolved = path.resolve()
    return resolved.name if resolved.parent == out_dir.resolve() else str(resolved)


def _warn_about_gaps(session: dict | None, audio_s: float) -> None:
    if not session or not session.get("ended_at") or not session.get("started_at"):
        return
    wall_s = (
        datetime.fromisoformat(session["ended_at"]) - datetime.fromisoformat(session["started_at"])
    ).total_seconds()
    if abs(wall_s - audio_s) > GAP_WARNING_S:
        print(
            f"ВНИМАНИЕ: по часам запись шла {wall_s / 3600:.2f} ч, а звука {audio_s / 3600:.2f} ч. "
            "В записи есть дыры (сбой микрофона, сон Mac). Время эпизодов учитывает перерывы только "
            "между перезапусками записи."
        )


def _analyze_night(
    audio_input: Path, out_dir: Path, model_path: Path, gain_db: float
) -> tuple[np.ndarray, list[Segment], str | None, dict | None]:
    files = find_audio(audio_input)
    session = _read_session(audio_input)
    model = Yamnet(model_path)
    started = time.monotonic()

    tables: list[np.ndarray | None] = []
    durations: list[float | None] = []
    for number, path in enumerate(files, start=1):
        print(f"Файл {number} из {len(files)}: {path.name}", flush=True)
        try:
            table, duration = analyze_segment(path, model, gain_db=gain_db)
        except AudioDecodeError as error:
            print(f"  ПРОПУСКАЮ {path.name}: {error}")
            tables.append(None)
            durations.append(None)
            continue
        tables.append(table)
        durations.append(duration)
    if all(table is None for table in tables):
        raise ValueError("Ни один файл не удалось прочитать.")

    nominal_s = float(session["segment_s"]) if session and session.get("segment_s") else 3600.0
    unnumbered = [p.name for p in files if segment_number(p.name) is None]
    if session and session.get("runs") and unnumbered:
        print(
            f"ВНИМАНИЕ: файлы без номера seg_N: {', '.join(unnumbered)}. "
            "Перерывы записи при расчёте времени не учитываются."
        )
    starts = place_segments(
        [segment_number(p.name) for p in files], durations, _runs_from_session(session), nominal_s
    )

    segments: list[Segment] = []
    shifted: list[np.ndarray] = []
    for path, table, duration, start in zip(files, tables, durations, starts):
        if table is None or duration is None:
            continue
        index = len(segments)
        segments.append(Segment(index, _stored_path(path, out_dir), start, duration))
        shifted.append(table + np.array([start, index, 0, 0, 0, 0, 0]))
    frames = quantize_frames(np.vstack(shifted))
    print(f"Модель отработала за {time.monotonic() - started:.0f} с.")
    _warn_about_gaps(session, sum(s.duration_s for s in segments))
    return frames, segments, (session or {}).get("started_at"), session


def run(
    audio_input: Path,
    out_dir: Path,
    model_path: Path,
    gain_db: float | None,
    params: EventParams,
    reuse: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path, segments_path = out_dir / "frames.csv", out_dir / "segments.json"

    if reuse and frames_path.exists() and segments_path.exists():
        frames, segments, doc = _load_run(out_dir)
        stored_gain = float(doc.get("gain_db", 0.0))
        if gain_db is not None and gain_db != stored_gain:
            raise ValueError(
                f"frames.csv посчитан с усилением {stored_gain} dB, а вы просите {gain_db}. "
                "Уберите --reuse, чтобы прогнать модель заново."
            )
        gain_db, started_at = stored_gain, doc.get("night_started_at")
        print("Беру готовые frames.csv, модель не запускаю.")
    else:
        gain_db = 0.0 if gain_db is None else gain_db
        frames, segments, started_at, _ = _analyze_night(audio_input, out_dir, model_path, gain_db)
        write_frames(frames_path, frames)
        segments_path.write_text(
            json.dumps(
                {
                    "gain_db": gain_db,
                    "night_started_at": started_at,
                    "segments": [asdict(s) for s in segments],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "utf-8",
        )

    events = detect_events(frames, segments, params)
    summary = {**summarize(frames, segments, events), "night_started_at": started_at}
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
    parser.add_argument("--gain-db", type=float, default=None, help="усилить звук перед моделью (по умолчанию 0)")
    parser.add_argument("--threshold", type=float, default=EventParams.threshold)
    parser.add_argument("--max-gap", type=float, default=EventParams.max_gap_s, help="склеивать эпизоды с паузой до N секунд")
    parser.add_argument("--min-duration", type=float, default=EventParams.min_duration_s)
    parser.add_argument("--reuse", action="store_true", help="не гонять модель заново, если есть frames.csv")
    args = parser.parse_args()

    out_dir = args.out or (args.input if args.input.is_dir() else Path("runs") / args.input.stem)
    params = EventParams(args.threshold, args.max_gap, args.min_duration)
    try:
        summary = run(args.input, out_dir, args.model, args.gain_db, params, args.reuse)
    except (ValueError, FileNotFoundError) as error:
        print(f"Ошибка: {error}")
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Результат: {out_dir / 'events.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
