from hrapometr_proto.record import (
    ProgressMonitor,
    _tail_since,
    _watch,
    build_ffmpeg_command,
    classify_exit,
    input_level_ok,
    next_segment_number,
    parse_audio_devices,
    parse_on_ac_power,
    parse_progress_line,
    restart_delay,
)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeProc:
    """ffmpeg, который жив, пока не вызвали finish()."""

    def __init__(self):
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def finish(self):
        self.alive = False


def test_command_reports_progress_to_stdout():
    cmd = build_ffmpeg_command(input_args=["-i", "x"], out_dir="x", segment_s=60, bitrate="16k")
    joined = " ".join(cmd)
    assert "-progress pipe:1" in joined
    assert "-stats_period 5" in joined
    assert cmd[-1].endswith("seg_%04d.ogg")  # выходной файл остаётся последним аргументом


def test_progress_line_parsing():
    assert parse_progress_line("out_time_us=5493500") == 5493500
    assert parse_progress_line("out_time_ms=10553500") == 10553500  # в ffmpeg это тоже микросекунды
    assert parse_progress_line("out_time_us=N/A") is None
    assert parse_progress_line("total_size=N/A") is None
    assert parse_progress_line("progress=continue") is None


def test_monitor_stalls_when_recording_time_stops_growing():
    clock = FakeClock()
    monitor = ProgressMonitor(stall_s=120, clock=clock)
    monitor.feed("out_time_us=5000000")
    clock.now += 100
    assert not monitor.stalled()
    monitor.feed("out_time_us=105000000")  # время записи продвинулось
    clock.now += 100
    assert not monitor.stalled()
    clock.now += 21
    assert monitor.stalled()
    assert monitor.recorded_s == 105.0


def test_monitor_ignores_repeated_or_backwards_time():
    clock = FakeClock()
    monitor = ProgressMonitor(stall_s=60, clock=clock)
    monitor.feed("out_time_us=9000000")
    clock.now += 50
    monitor.feed("out_time_us=9000000")  # то же значение — прогресса нет
    monitor.feed("out_time_us=1000000")
    clock.now += 11
    assert monitor.stalled()


def test_silent_quiet_room_is_not_a_stall_while_time_advances():
    # файл при этом может не расти минутами: на это не смотрим
    clock = FakeClock()
    monitor = ProgressMonitor(stall_s=120, clock=clock)
    for step in range(1, 100):
        clock.now += 5
        monitor.feed(f"out_time_us={step * 5_000_000}")
        assert not monitor.stalled()


def test_watch_returns_true_when_progress_stops():
    clock = FakeClock()
    monitor = ProgressMonitor(stall_s=30, clock=clock)
    assert _watch(FakeProc(), monitor, sleep=clock.sleep, clock=clock) is True


def test_watch_returns_false_when_ffmpeg_exits_on_its_own():
    clock = FakeClock()
    monitor = ProgressMonitor(stall_s=30, clock=clock)
    proc = FakeProc()

    def sleep_then_finish(seconds):
        clock.sleep(seconds)
        monitor.feed(f"out_time_us={int(clock.now * 1_000_000)}")
        if clock.now > 1005:
            proc.finish()

    assert _watch(proc, monitor, sleep=sleep_then_finish, clock=clock) is False


def test_restart_delay_grows_and_is_capped():
    assert [restart_delay(n, 5.0, 60.0) for n in (1, 2, 3, 4, 5, 6)] == [5, 10, 20, 40, 60, 60]


def test_tail_shows_only_current_run_errors(tmp_path):
    log = tmp_path / "ffmpeg.log"
    log.write_bytes("старая ошибка\n".encode())
    offset = log.stat().st_size
    with log.open("ab") as handle:
        handle.write("новая ошибка\nещё одна\n".encode())
    assert _tail_since(log, offset) == "новая ошибка\nещё одна"


def test_command_has_mono_opus_and_segmenting():
    cmd = build_ffmpeg_command(
        input_args=["-f", "avfoundation", "-i", ":0"],
        out_dir="runs/night",
        segment_s=3600,
        bitrate="16k",
    )
    joined = " ".join(cmd)
    assert cmd[0] == "ffmpeg"
    assert "-ac 1" in joined
    assert "-c:a libopus" in joined
    assert "-b:a 16k" in joined
    assert "-f segment" in joined
    assert "-segment_time 3600" in joined
    assert "-segment_start_number 0" in joined
    assert cmd[-1].replace("\\", "/") == "runs/night/seg_%04d.ogg"


def test_restart_continues_numbering():
    cmd = build_ffmpeg_command(
        input_args=["-i", "x"], out_dir="x", segment_s=60, bitrate="16k", start_number=5
    )
    assert "-segment_start_number 5" in " ".join(cmd)


def test_input_args_come_before_encoding_options():
    cmd = build_ffmpeg_command(
        input_args=["-f", "avfoundation", "-i", ":1"], out_dir="x", segment_s=60, bitrate="16k"
    )
    assert cmd.index("-i") < cmd.index("-c:a")


def test_next_segment_number_follows_existing_files(tmp_path):
    assert next_segment_number(tmp_path) == 0
    (tmp_path / "seg_0000.ogg").write_bytes(b"x")
    (tmp_path / "seg_0003.ogg").write_bytes(b"x")
    assert next_segment_number(tmp_path) == 4


def test_pmset_ac_power():
    output = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t95%; charging; 1:00 remaining"
    assert parse_on_ac_power(output) is True


def test_pmset_battery_power():
    output = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t95%; discharging"
    assert parse_on_ac_power(output) is False


def test_pmset_unknown_output():
    assert parse_on_ac_power("") is None
    assert parse_on_ac_power("what") is None


AVFOUNDATION_LIST = """\
[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] FaceTime HD Camera
[AVFoundation indev @ 0x1] [1] Capture screen 0
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone
[AVFoundation indev @ 0x1] [1] AirPods Pro
"""


def test_parse_audio_devices_takes_only_audio_section():
    assert parse_audio_devices(AVFOUNDATION_LIST) == {0: "MacBook Pro Microphone", 1: "AirPods Pro"}


def test_parse_audio_devices_of_garbage_is_empty():
    assert parse_audio_devices("nothing useful here") == {}


def test_digital_silence_fails_input_check_and_room_noise_passes():
    assert input_level_ok(-180.0) is False
    assert input_level_ok(-95.0) is False
    assert input_level_ok(-60.0) is True


def test_user_stop_is_never_a_failure():
    assert classify_exit(returncode=255, stalled=False, interrupted=True, finite_input=False) == "done"


def test_finite_test_input_finishing_cleanly_is_done():
    assert classify_exit(returncode=0, stalled=False, interrupted=False, finite_input=True) == "done"


def test_ffmpeg_error_means_restart():
    assert classify_exit(returncode=1, stalled=False, interrupted=False, finite_input=False) == "restart"


def test_live_recording_that_ends_by_itself_means_restart():
    assert classify_exit(returncode=0, stalled=False, interrupted=False, finite_input=False) == "restart"


def test_stalled_ffmpeg_means_restart():
    assert classify_exit(returncode=None, stalled=True, interrupted=False, finite_input=False) == "restart"
