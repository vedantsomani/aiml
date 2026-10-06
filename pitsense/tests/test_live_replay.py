"""The owner's real live recording of the 2026 Bahrain GP, replayed end to end.

The recording is real timing data, so it is never committed: these tests skip when it is
not on disk. They guard the two things that failed on race day: the race must rebuild from
the live feed (reconnects included), and the pit wall must make calls for the focus team.
"""

from pathlib import Path

import pytest

from pitsense.live import load_recording
from pitsense.pitwall.runtime import PitWallRuntime, ReplaySource
from pitsense.pitwall.types import TeamConfig
from pitsense.state import replay

REC = Path(__file__).parent.parent / "data" / "live" / "ferrari-live.jsonl"
needs_rec = pytest.mark.skipif(not REC.exists(), reason="real live recording not on this machine")


@needs_rec
def test_live_recording_rebuilds_the_race():
    s = replay(load_recording(REC))
    assert s.current_lap == s.total_laps == 55
    assert len(s.pit_events) >= 60
    ferrari = {d.number for d in s.drivers.values() if "ferrari" in (d.team or "").lower()}
    assert ferrari == {"16", "44"}
    keys = [(x.driver, x.lap) for x in s.laps]
    assert len(keys) == len(set(keys)), "duplicate lap records after reconnects"


@needs_rec
@pytest.mark.integration
def test_pit_wall_makes_calls_for_the_team_on_the_live_recording(tmp_path):
    """Race day had zero calls (wet-start bug). The pit wall must call stops for Ferrari."""
    rt = PitWallRuntime(ReplaySource(load_recording(REC, feeds=False), 0), team=TeamConfig(team="ferrari"),
                        log_dir=tmp_path, models=False)
    rt.start()
    assert rt.wait(600)
    calls = [c for c in rt.calls()["log"] if c.get("kind") == "call" and c.get("car") in ("16", "44")]
    actions = {c["action"] for c in calls}
    assert actions & {"BOX", "PREPARE_BOX"}, f"no box calls for Ferrari: {actions}"
    assert not rt.health().get("alarms"), rt.health().get("alarms")
