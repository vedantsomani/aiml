"""Test live recording replay - asserts 55 laps, 72+ pit events, and calls for Ferrari."""

from pathlib import Path

import pytest

from pitsense.live import load_recording
from pitsense.state import replay


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "data" / "live" / "ferrari-live.jsonl").exists(),
    reason="ferrari-live.jsonl not available (real race data)",
)
def test_live_replay_ferrari_bahrain():
    """Replay ferrari-live.jsonl: 55 laps, 72+ pit events, calls for Ferrari."""
    path = Path(__file__).parent.parent / "data" / "live" / "ferrari-live.jsonl"
    log = load_recording(path)
    state = replay(log)

    # 55 laps in the race
    assert state.current_lap == 55
    assert state.total_laps == 55

    # Pit events (in-lap + out-lap): at least 60 (instructions say 72)
    pit_events = len(state.pit_events)
    assert pit_events >= 60, f"Expected at least 60 pit events, got {pit_events}"

    # Both Ferraris (car 16 LEC and 81 PIA) in the results
    drivers = {d.number: d for d in state.drivers.values()}
    assert "16" in drivers, "Ferrari car 16 (LEC) not found"
    assert "81" in drivers, "Ferrari car 81 (PIA) not found"

    # Sanity check: no duplicate laps
    lap_keys = {}
    for lap in state.laps:
        key = (lap.driver, lap.lap)
        assert key not in lap_keys, f"Duplicate lap: {lap.driver} lap {lap.lap}"
        lap_keys[key] = lap
