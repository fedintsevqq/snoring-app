from hrapometr_proto.record import (
    build_ffmpeg_command,
    classify_exit,
    input_level_ok,
    next_segment_number,
    parse_audio_devices,
    parse_on_ac_power,
)


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
