import json
import shutil
import subprocess

import numpy as np
import pytest

from hrapometr_proto.detect import (
    HOP_S,
    WIN_S,
    EventParams,
    _load_run,
    _write_frames,
    analyze_segment,
    detect_events,
    find_audio,
    quantize_frames,
    summarize,
)
from hrapometr_proto.events import Event, frames_to_events
from hrapometr_proto.windows import Segment
from hrapometr_proto.yamnet import CLASS_SNORING

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")


def frame_rows(seg_index: int, scores: list[float], offset_s: float = 0.0) -> np.ndarray:
    rows = [
        (offset_s + i * HOP_S, seg_index, -40.0, score, 0.0, 0.0, 0.0)
        for i, score in enumerate(scores)
    ]
    return np.array(rows, dtype=np.float64)


def test_default_params_drop_single_frame_but_keep_two_adjacent_frames():
    # реальные длины кадра и шага, а не круглые числа
    p = EventParams()
    common = dict(
        hop_s=HOP_S, win_s=WIN_S, threshold=p.threshold, max_gap_s=p.max_gap_s,
        min_duration_s=p.min_duration_s,
    )  # fmt: skip
    assert frames_to_events(np.array([0, 0.9, 0]), **common) == []
    assert len(frames_to_events(np.array([0, 0.9, 0.9, 0]), **common)) == 1


def test_events_are_shifted_by_segment_start_and_never_cross_segments():
    frames = np.vstack(
        [
            frame_rows(0, [0, 0, 0.9, 0.9, 0, 0]),
            frame_rows(1, [0.9, 0.9, 0, 0, 0, 0], offset_s=100.0),
        ]
    )
    segments = [Segment(0, "a", 0.0, 10.0), Segment(1, "b", 100.0, 10.0)]
    events = detect_events(frames, segments, EventParams())
    assert len(events) == 2
    assert events[0].start_s == pytest.approx(2 * HOP_S)
    assert events[1].start_s == pytest.approx(100.0)


def test_summary_counts_snoring_share_of_night():
    segments = [Segment(0, "a", 0.0, 100.0)]
    events = [Event(10.0, 20.0, 0.9), Event(50.0, 60.0, 0.8)]
    summary = summarize(frame_rows(0, [0.0, 0.5, 1.0]), segments, events)
    assert summary["events"] == 2
    assert summary["snoring_minutes"] == pytest.approx(20 / 60, abs=0.05)
    assert summary["snoring_percent"] == pytest.approx(20.0)
    assert summary["snoring_score_quantiles"]["max"] == 1.0


def test_find_audio_skips_hidden_tiny_and_foreign_files(tmp_path):
    (tmp_path / "seg_0001.ogg").write_bytes(b"x" * 5000)
    (tmp_path / "seg_0000.ogg").write_bytes(b"x" * 5000)
    (tmp_path / "._seg_0000.ogg").write_bytes(b"x" * 5000)  # служебный файл macOS
    (tmp_path / "seg_0002.ogg").write_bytes(b"")  # пустой
    (tmp_path / "notes.txt").write_text("hi")
    assert [p.name for p in find_audio(tmp_path)] == ["seg_0000.ogg", "seg_0001.ogg"]


def test_find_audio_fails_loudly_when_nothing_usable(tmp_path):
    (tmp_path / "notes.txt").write_text("hi")
    with pytest.raises(FileNotFoundError):
        find_audio(tmp_path)


def test_quantized_frames_equal_what_reuse_reads_back(tmp_path):
    raw = np.array([[0.48751, 0, -40.123456, 0.299996, 0.00004, 0.5, 0.99999]])
    path = tmp_path / "frames.csv"
    _write_frames(path, quantize_frames(raw))
    reloaded = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    assert np.array_equal(reloaded, quantize_frames(raw))
    # граничная оценка 0.299996 при пороге 0.3 теперь одинакова и в памяти, и после --reuse
    assert quantize_frames(raw)[0, 3] >= 0.3


def test_load_run_with_header_only_frames_file(tmp_path):
    (tmp_path / "frames.csv").write_text("t_s,seg,rms_db,snoring,breathing,snort,speech\n")
    (tmp_path / "segments.json").write_text(json.dumps({"gain_db": 0.0, "segments": []}))
    frames, segments, doc = _load_run(tmp_path)
    assert frames.shape == (0, 7)
    assert segments == [] and doc["gain_db"] == 0.0


class FakeYamnet:
    def scores(self, frame):
        scores = np.zeros(521, dtype=np.float32)
        scores[CLASS_SNORING] = 0.9
        return scores


@needs_ffmpeg
def test_analyze_segment_counts_frames_and_duration(tmp_path):
    tone = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=16000",
         "-t", "5", str(tone)],
        check=True,
    )  # fmt: skip
    table, duration = analyze_segment(tone, FakeYamnet(), gain_db=0.0)
    assert duration == pytest.approx(5.0, abs=0.01)
    assert table.shape == (9, 7)  # (80000 - 15600) // 7800 + 1
    assert table[0, 0] == 0.0
    assert table[1, 0] == pytest.approx(HOP_S)
    assert np.all(table[:, 3] == np.float32(0.9))


def test_analyze_segment_raises_on_unreadable_file(tmp_path):
    bad = tmp_path / "bad.ogg"
    bad.write_bytes(b"not audio at all" * 100)
    if shutil.which("ffmpeg") is None:
        pytest.skip("нужен ffmpeg")
    with pytest.raises(RuntimeError):
        analyze_segment(bad, FakeYamnet(), gain_db=0.0)
