"""Запись ночи на Mac: ffmpeg пишет моно Opus кусками по часу, caffeinate не даёт Mac уснуть.

Запуск:  python -m hrapometr_proto.record --out runs/night-1
Остановка: Ctrl+C (ffmpeg корректно закрывает последний файл).

Если ffmpeg упал или завис (отвалился микрофон), запись перезапускается в новые куски. Все запуски
и перезапуски записываются в session.json, а ошибки ffmpeg — в ffmpeg.log рядом с записью.
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

import numpy as np

from .audio import rms_db
from .timeline import segment_number

PREFLIGHT_S = 3  # столько секунд слушаем микрофон перед стартом
SILENCE_FLOOR_DB = -90.0  # тише — это цифровая тишина (нет доступа к микрофону, не то устройство)
STALL_S = 180  # столько секунд файлы не растут — ffmpeg завис
DEFAULT_MAX_RESTARTS = 10
DEFAULT_RESTART_DELAY_S = 5.0


def build_ffmpeg_command(
    *,
    input_args: list[str],
    out_dir: str | Path,
    segment_s: float,
    bitrate: str,
    start_number: int = 0,
) -> list[str]:
    """Команда записи: вход -> моно 16 кГц -> Opus -> файлы seg_0000.ogg, seg_0001.ogg ..."""
    pattern = str(Path(out_dir) / "seg_%04d.ogg")
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        *input_args,
        "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", bitrate,
        "-f", "segment", "-segment_time", f"{segment_s:g}", "-reset_timestamps", "1",
        "-segment_start_number", str(start_number),
        pattern,
    ]  # fmt: skip


def parse_on_ac_power(pmset_output: str) -> bool | None:
    """Разбор вывода `pmset -g batt`. True — от сети, False — от батареи, None — непонятно."""
    match = re.search(r"Now drawing from '([^']+)'", pmset_output)
    if not match:
        return None
    return match.group(1) == "AC Power"


def parse_audio_devices(output: str) -> dict[int, str]:
    """Аудиоустройства из вывода `ffmpeg -f avfoundation -list_devices true -i ""`."""
    devices: dict[int, str] = {}
    in_audio = False
    for line in output.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
        elif "AVFoundation video devices" in line:
            in_audio = False
        elif in_audio:
            match = re.search(r"\]\s+\[(\d+)\]\s+(.+?)\s*$", line)
            if match:
                devices[int(match.group(1))] = match.group(2)
    return devices


def next_segment_number(out_dir: Path) -> int:
    """Номер, с которого продолжать куски после перезапуска, чтобы не затереть записанное."""
    numbers = [segment_number(p.name) for p in out_dir.glob("seg_*.ogg")]
    known = [n for n in numbers if n is not None]
    return max(known) + 1 if known else 0


def input_level_ok(level_db: float) -> bool:
    """Микрофон отдаёт хоть какой-то звук (комнатный шум выше -90 dBFS), а не цифровую тишину."""
    return level_db > SILENCE_FLOOR_DB


def classify_exit(
    *, returncode: int | None, stalled: bool, interrupted: bool, finite_input: bool
) -> str:
    """«done» — запись закончена как задумано, «restart» — ffmpeg упал, завис или остановился сам."""
    if interrupted:
        return "done"
    if finite_input and returncode == 0 and not stalled:
        return "done"
    return "restart"


def _list_devices_output() -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    return result.stderr


def _measure_input_level(input_args: list[str]) -> float | None:
    """Пишет PREFLIGHT_S секунд с микрофона и возвращает громкость в dBFS; None — не удалось записать."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", *input_args,
        "-t", str(PREFLIGHT_S), "-ac", "1", "-ar", "16000", "-f", "f32le", "-",
    ]  # fmt: skip
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=90, check=False)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or len(result.stdout) < 4:
        return None
    return rms_db(np.frombuffer(result.stdout[: len(result.stdout) // 4 * 4], dtype="<f4"))


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


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.glob("seg_*.ogg"))


def _write_session(out_dir: Path, data: dict) -> None:
    (out_dir / "session.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def _tail(path: Path, lines: int = 5) -> str:
    try:
        return "\n".join(path.read_text("utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _watch(proc: subprocess.Popen, out_dir: Path) -> bool:
    """Ждёт конца ffmpeg. True — завис (файлы STALL_S секунд не растут), False — завершился сам."""
    now = time.monotonic()
    last_report = last_check = last_growth = now
    last_size = _dir_size_bytes(out_dir)
    while proc.poll() is None:
        time.sleep(1)
        now = time.monotonic()
        if now - last_check >= 10:
            last_check = now
            size = _dir_size_bytes(out_dir)
            if size != last_size:
                last_size, last_growth = size, now
        if now - last_growth >= STALL_S:
            return True
        if now - last_report >= 60:
            last_report = now
            print(f"{datetime.now():%H:%M} записано {last_size / 1_000_000:.1f} МБ", flush=True)
    return False


def _stop_process(proc: subprocess.Popen) -> int | None:
    """Мягко останавливает ffmpeg (команда q закрывает файл), при повторном Ctrl+C или зависании — жёстко."""
    if proc.poll() is None:
        try:
            if proc.stdin:
                proc.stdin.write(b"q")
                proc.stdin.flush()
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                proc.kill()
                proc.wait()
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Запись ночи (Opus, моно, куски по часу)")
    parser.add_argument("--out", type=Path, help="папка для ночи, например runs/night-1")
    parser.add_argument("--device", default="0", help="номер аудиоустройства avfoundation (по умолчанию 0)")
    parser.add_argument("--list-devices", action="store_true", help="показать устройства и выйти")
    parser.add_argument("--segment-min", type=float, default=60.0, help="длина куска, минут")
    parser.add_argument("--bitrate", default="16k", help="битрейт Opus (по умолчанию 16k)")
    parser.add_argument("--skip-check", action="store_true", help="не проверять микрофон перед стартом")
    parser.add_argument("--max-restarts", type=int, default=DEFAULT_MAX_RESTARTS)
    parser.add_argument("--restart-delay", type=float, default=DEFAULT_RESTART_DELAY_S)
    parser.add_argument(
        "--test-seconds", type=float, default=0.0,
        help="проверка без микрофона: пишет тестовый тон N секунд",
    )  # fmt: skip
    args = parser.parse_args()

    is_mac = platform.system() == "Darwin"
    if args.list_devices:
        print(_list_devices_output() if is_mac else "Список устройств доступен только на Mac.")
        return 0
    if args.out is None:
        parser.error("нужен --out")

    finite_input = args.test_seconds > 0
    if finite_input:
        input_args = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-t", f"{args.test_seconds:g}"]
    elif is_mac:
        input_args = ["-f", "avfoundation", "-i", f":{args.device}"]
    else:
        print("Запись с микрофона поддержана только на Mac. Для проверки используйте --test-seconds.")
        return 2

    out_dir: Path = args.out
    if out_dir.exists() and any(out_dir.glob("seg_*.ogg")):
        print(f"В {out_dir} уже есть записи. Выберите другую папку.")
        return 2

    device_name = None
    if is_mac and not finite_input:
        devices = parse_audio_devices(_list_devices_output())
        device_name = devices.get(int(args.device)) if args.device.isdigit() else None
        print(f"Микрофон: {device_name or 'не удалось определить имя'} (номер {args.device}).")
        if not args.skip_check:
            print(f"Проверяю микрофон ({PREFLIGHT_S} с)...")
            level = _measure_input_level(input_args)
            if level is None:
                print("Не удалось записать с микрофона. Проверьте номер (--list-devices) и доступ Терминала к микрофону.")
                return 3
            print(f"Уровень сигнала: {level:.0f} dBFS.")
            if not input_level_ok(level):
                print(
                    "Микрофон отдаёт тишину. Проверьте доступ Терминала к микрофону "
                    "(Системные настройки, Конфиденциальность, Микрофон) и номер устройства. "
                    "Если так и должно быть, запустите с --skip-check."
                )
                return 3

    out_dir.mkdir(parents=True, exist_ok=True)
    segment_s = args.segment_min * 60.0
    started = datetime.now().astimezone()
    session: dict = {
        "started_at": started.isoformat(timespec="seconds"),
        "device": args.device,
        "device_name": device_name,
        "bitrate": args.bitrate,
        "segment_s": segment_s,
        "runs": [],
        "restarts": 0,
    }
    _write_session(out_dir, session)

    if is_mac:
        _check_power()
    print(f"Пишу в {out_dir}. Остановить: Ctrl+C.")

    log_path = out_dir / "ffmpeg.log"
    interrupted = False
    gave_up = False
    restarts = 0
    with log_path.open("ab") as log:
        while True:
            start_number = next_segment_number(out_dir)
            session["runs"] = [
                *session["runs"],
                {"started_at": datetime.now().astimezone().isoformat(timespec="seconds"), "first_segment": start_number},
            ]
            _write_session(out_dir, session)
            cmd = build_ffmpeg_command(
                input_args=input_args, out_dir=out_dir, segment_s=segment_s,
                bitrate=args.bitrate, start_number=start_number,
            )  # fmt: skip
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
            keep_awake = None
            if is_mac:
                # -i: не засыпать от простоя, -s: не засыпать системе при работе от сети, -w: до конца ffmpeg
                keep_awake = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(proc.pid)])

            stalled = False
            try:
                stalled = _watch(proc, out_dir)
            except KeyboardInterrupt:
                interrupted = True
                print("Останавливаю запись...")
            returncode = _stop_process(proc)
            if keep_awake is not None:
                keep_awake.terminate()

            decision = classify_exit(
                returncode=None if stalled else returncode,
                stalled=stalled,
                interrupted=interrupted,
                finite_input=finite_input,
            )
            if decision == "done":
                break

            restarts += 1
            session["restarts"] = restarts
            reason = "завис (файлы перестали расти)" if stalled else f"остановился с кодом {returncode}"
            print(f"\n!!! {datetime.now():%H:%M} ffmpeg {reason}.")
            print(_tail(log_path))
            if restarts > args.max_restarts:
                gave_up = True
                break
            print(f"Перезапуск {restarts} из {args.max_restarts}.")
            try:
                time.sleep(args.restart_delay)
            except KeyboardInterrupt:
                interrupted = True
                break

    ended = datetime.now().astimezone()
    hours = (ended - started).total_seconds() / 3600
    size_mb = _dir_size_bytes(out_dir) / 1_000_000
    has_audio = size_mb > 0
    session = {
        **session,
        "ended_at": ended.isoformat(timespec="seconds"),
        "hours": round(hours, 3),
        "size_mb": round(size_mb, 2),
        "mb_per_hour": round(size_mb / hours, 2) if hours > 0 else None,
        "gave_up": gave_up,
    }
    _write_session(out_dir, session)

    if gave_up or not has_audio:
        print(
            f"\nЗАПИСЬ НЕ УДАЛАСЬ: ffmpeg не заработал даже после {args.max_restarts} перезапусков. "
            f"Ошибки: {log_path}"
        )
        return 1
    print(f"Готово: {hours:.2f} ч, {size_mb:.1f} МБ ({session['mb_per_hour']} МБ/ч), перезапусков: {restarts}.")
    if restarts:
        print(f"В записи есть перерывы ({restarts}). Подробности в {log_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
