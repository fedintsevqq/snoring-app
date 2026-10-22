from hrapometr_proto.record import build_ffmpeg_command, parse_on_ac_power


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
    assert cmd[-1].replace("\\", "/") == "runs/night/seg_%04d.ogg"


def test_input_args_come_before_encoding_options():
    cmd = build_ffmpeg_command(
        input_args=["-f", "avfoundation", "-i", ":1"], out_dir="x", segment_s=60, bitrate="16k"
    )
    assert cmd.index("-i") < cmd.index("-c:a")


def test_pmset_ac_power():
    output = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t95%; charging; 1:00 remaining"
    assert parse_on_ac_power(output) is True


def test_pmset_battery_power():
    output = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t95%; discharging"
    assert parse_on_ac_power(output) is False


def test_pmset_unknown_output():
    assert parse_on_ac_power("") is None
    assert parse_on_ac_power("what") is None
