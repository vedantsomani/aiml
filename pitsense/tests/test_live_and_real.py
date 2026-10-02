import json

import pytest

from pitsense.events import load_archive_session
from pitsense.live import load_fastf1_recording, load_recording
from pitsense.state import replay

from .conftest import real_session_dir, synthetic_events


def test_live_jsonl_recording_replays(tmp_path):
    """A recording made by `pitsense record` goes through the same reducer."""
    path = tmp_path / "rec.jsonl"
    rows = synthetic_events()
    with open(path, "w", encoding="utf-8") as f:
        for t, topic, data in rows:
            f.write(json.dumps({"recv": 1_700_000_000 + t, "topic": topic, "kind": "delta", "data": data}) + "\n")
    s = replay(load_recording(path))
    assert len(s.laps) == 24 and len(s.pit_events) == 1


def test_fastf1_format_recording_replays(tmp_path):
    """Files saved by FastF1's recorder: python-literal lists, first lines are snapshots."""
    path = tmp_path / "ff1.txt"
    rows = synthetic_events()
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(["SessionInfo", json.dumps(rows[0][2]), ""]) + "\n")
        for t, topic, data in rows[1:]:
            secs = 1_700_000_000 + t
            from datetime import datetime, timezone

            ts = datetime.fromtimestamp(secs, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            f.write(str([topic, data, ts]) + "\n")
    s = replay(load_fastf1_recording(path))
    assert len(s.laps) == 24


@pytest.mark.integration
def test_real_race_reconstruction():
    d = real_session_dir(2026, "Hungarian")
    if d is None:
        pytest.skip("run `pitsense fetch --year 2026 --race hungary` first")
    s = replay(load_archive_session(d))
    assert s.total_laps == 70
    timed = [x for x in s.laps if x.lap > 1]
    assert sum(x.lap_time is not None for x in timed) == len(timed)
    assert len([p for p in s.pit_events if not p.under_red]) >= 30


def test_failed_connect_is_cleaned_up(tmp_path):
    """A connection attempt that raises must not leave the file open or the connection running."""
    from types import SimpleNamespace

    from pitsense.live import _cleanup

    stopped = []

    def stop():
        stopped.append(True)
        raise RuntimeError("already closed")  # cleanup must tolerate this

    client = SimpleNamespace(_connection=SimpleNamespace(stop=stop), _output_file=open(tmp_path / "x.jsonl", "a"))
    _cleanup(client)
    assert stopped and client._output_file.closed
    _cleanup(SimpleNamespace())  # failed before anything was opened


def test_recorder_stops_at_deadline_even_while_data_flows(tmp_path):
    """FastF1's client only stops when the feed goes quiet; ours also honours --minutes."""
    pytest.importorskip("fastf1")
    import threading
    import time

    from pitsense.live import _make_client

    client = _make_client(tmp_path / "x.jsonl", no_auth=True, timeout=0, deadline=time.time() + 1.5)
    client._exit = lambda: None  # no connection in this test
    stop = threading.Event()

    def feed():  # keep "receiving" messages, like heartbeats after a session
        while not stop.is_set():
            client._t_last_message = time.time()
            time.sleep(0.1)

    threading.Thread(target=feed, daemon=True).start()
    t0 = time.time()
    client._supervise()
    stop.set()
    assert 1.0 < time.time() - t0 < 4.0
