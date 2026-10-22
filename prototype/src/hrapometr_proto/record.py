"""Запись ночи на Mac: ffmpeg пишет моно Opus кусками по часу, caffeinate не даёт Mac уснуть.

Запуск:  python -m hrapometr_proto.record --out runs/night-1
Остановка: Ctrl+C (ffmpeg корректно закрывает последний файл).
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def build_ffmpeg_command(
    *, input_args: list[str], out_dir: str | Path, segment_s: float, bitrate: str
) -> list[str]:
    """Команда записи: вход -> моно 16 кГц -> Opus -> файлы seg_0000.ogg, seg_0001.ogg ..."""
    pattern = str(Path(out_dir) / "seg_%04d.ogg")
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        *input_args,
        "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", bitrate,
        "-f", "segment", "-segment_time", f"{segment_s:g}", "-reset_timestamps", "1",
        pattern,
    ]  # fmt: skip


def parse_on_ac_power(pmset_output: str) -> bool | None:
    """Разбор вывода `pmset -g batt`. True — от сети, False — от батареи, None — непонятно."""
    match = re.search(r"Now drawing from '([^']+)'", pmset_output)
    if not match:
        return None
    return match.group(1) == "AC Power"


def _check_power() -> None:
    try:
        output = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        print("Не смог проверить питание (pmset недоступен).")
        return
    if parse_on_ac_power(output) is False:
        print("ВНИМАНИЕ: Mac работает от батареи. Включите зарядку, иначе ночь может прерваться.")


def _list_devices() -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        check=False,
    )


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("seg_*.ogg")) / 1_000_000


def _write_session(out_dir: Path, data: dict) -> None:
    (out_dir / "session.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Запись ночи (Opus, моно, куски по часу)")
    parser.add_argument("--out", type=Path, help="папка для ночи, например runs/night-1")
    parser.add_argument("--device", default="0", help="номер аудиоустройства avfoundation (по умолчанию 0)")
    parser.add_argument("--list-devices", action="store_true", help="показать устройства и выйти")
    parser.add_argument("--segment-min", type=float, default=60.0, help="длина куска, минут")
    parser.add_argument("--bitrate", default="16k", help="битрейт Opus (по умолчанию 16k)")
    parser.add_argument(
        "--test-seconds", type=float, default=0.0,
        help="проверка без микрофона: пишет тестовый тон N секунд",
    )  # fmt: skip
    args = parser.parse_args()

    if args.list_devices:
        _list_devices()
        return 0
    if args.out is None:
        parser.error("нужен --out")

    if args.test_seconds > 0:
        input_args = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-t", f"{args.test_seconds:g}"]
    elif platform.system() == "Darwin":
        input_args = ["-f", "avfoundation", "-i", f":{args.device}"]
    else:
        print("Запись с микрофона поддержана только на Mac. Для проверки используйте --test-seconds.")
        return 2

    out_dir: Path = args.out
    if out_dir.exists() and any(out_dir.glob("seg_*.ogg")):
        print(f"В {out_dir} уже есть записи. Выберите другую папку.")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    segment_s = args.segment_min * 60.0
    cmd = build_ffmpeg_command(
        input_args=input_args, out_dir=out_dir, segment_s=segment_s, bitrate=args.bitrate
    )
    started = datetime.now().astimezone()
    session = {
        "started_at": started.isoformat(timespec="seconds"),
        "device": args.device,
        "bitrate": args.bitrate,
        "segment_s": segment_s,
        "host": platform.platform(),
        "ffmpeg_command": cmd,
    }
    _write_session(out_dir, session)

    if platform.system() == "Darwin":
        _check_power()
    print(f"Пишу в {out_dir}. Остановить: Ctrl+C.")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    keep_awake = None
    if platform.system() == "Darwin":
        # -i: не засыпать от простоя, -s: не засыпать системе при работе от сети, -w: до конца ffmpeg
        keep_awake = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(proc.pid)])

    try:
        last_report = time.monotonic()
        while proc.poll() is None:
            time.sleep(1)
            if time.monotonic() - last_report >= 60:
                last_report = time.monotonic()
                print(f"{datetime.now():%H:%M} записано {_dir_size_mb(out_dir):.1f} МБ", flush=True)
    except KeyboardInterrupt:
        print("Останавливаю запись...")
        try:
            if proc.stdin:
                proc.stdin.write(b"q")
                proc.stdin.flush()
        except OSError:
            pass
    finally:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=10)
        if keep_awake is not None:
            keep_awake.terminate()

    ended = datetime.now().astimezone()
    hours = (ended - started).total_seconds() / 3600
    size_mb = _dir_size_mb(out_dir)
    final = {
        **session,
        "ended_at": ended.isoformat(timespec="seconds"),
        "hours": round(hours, 3),
        "size_mb": round(size_mb, 2),
        "mb_per_hour": round(size_mb / hours, 2) if hours > 0 else None,
    }
    _write_session(out_dir, final)
    print(f"Готово: {hours:.2f} ч, {size_mb:.1f} МБ ({final['mb_per_hour']} МБ/ч).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
