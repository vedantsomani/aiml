"""A small synthetic race in the live-timing message format.

Three cars, 8 laps, one pit stop (car 22 boxes at the end of lap 3, rejoins
behind car 33), a safety car during laps 5-7, a repeated lap time the feed
omits (car 33, lap 8), and a stint record flagged TyresNotChanged that is
corrected a lap later.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pitsense.events import Event, EventLog

CARS = ["11", "22", "33"]
TLA = {"11": "AAA", "22": "BBB", "33": "CCC"}
START = 100.0
LAP = 90.0


def _fmt(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:06.3f}"


def synthetic_events() -> list[tuple[float, str, dict]]:
    ev: list[tuple[float, str, dict]] = []
    add = lambda t, topic, data: ev.append((t, topic, data))  # noqa: E731

    add(0.0, "SessionInfo", {"Meeting": {"Name": "Test GP"}, "Name": "Race", "Type": "Race"})
    add(0.0, "SessionStatus", {"Status": "Inactive"})
    add(0.0, "TrackStatus", {"Status": "1", "Message": "AllClear"})
    add(0.5, "DriverList", {n: {"RacingNumber": n, "Tla": TLA[n], "TeamName": f"Team {n}"} for n in CARS})
    add(1.0, "LapCount", {"CurrentLap": 1, "TotalLaps": 8})
    add(1.0, "TimingAppData", {"Lines": {n: {"RacingNumber": n, "Stints": []} for n in CARS}})
    add(1.0, "TimingData", {"Lines": {
        n: {"Position": str(i + 1), "Line": i + 1, "GapToLeader": "", "IntervalToPositionAhead": {"Value": ""},
            "InPit": False, "PitOut": False, "NumberOfPitStops": 0, "NumberOfLaps": 0,
            "Retired": False, "Stopped": False, "LastLapTime": {"Value": ""}}
        for i, n in enumerate(CARS)}})
    add(10.0, "TimingAppData", {"Lines": {n: {"Stints": {"0": {"Compound": "MEDIUM", "New": "true",
                                                                "TyresNotChanged": "0", "TotalLaps": 0,
                                                                "StartLaps": 0}}} for n in CARS}})
    add(START, "SessionStatus", {"Status": "Started"})

    # base lap times; car 33 repeats 91.5 on laps 4 and 5 (the feed omits the repeat)
    base = {"11": 90.0, "22": 90.4, "33": 91.5}
    order_after_stop = False
    t_cross = {n: START for n in CARS}
    for lap in range(1, 9):
        for i, n in enumerate(CARS):
            lt = base[n] + (0.0 if n == "33" else 0.05 * lap)
            if n == "22" and lap == 3:
                lt += 8.0  # in-lap through the pit lane
            if n == "22" and lap == 4:
                lt += 14.0  # out-lap
            if lap >= 5 and lap <= 6:
                lt += 30.0  # safety car laps
            t_cross[n] += lt if lap > 1 else (95.0 + i)
            t = t_cross[n]
            upd: dict = {"NumberOfLaps": lap}
            if lap > 1 and not (n == "33" and lap == 8):  # lap 8 repeats lap 7: omitted
                upd["LastLapTime"] = {"Value": _fmt(lt)}
            if n == "11":
                upd["GapToLeader"] = f"LAP {lap + 1}"
                upd["IntervalToPositionAhead"] = {"Value": f"LAP {lap + 1}"}
            else:
                gap = t - t_cross["11"]
                upd["GapToLeader"] = f"+{gap:.3f}"
            add(t, "TimingData", {"Lines": {n: upd}})
            if n == "11":
                add(t + 0.01, "LapCount", {"CurrentLap": lap + 1})
        # intervals after everyone crossed
        t_last = max(t_cross.values()) + 0.5
        pos = ["11", "33", "22"] if order_after_stop else ["11", "22", "33"]
        upd = {}
        for k in range(1, 3):
            ahead, me = pos[k - 1], pos[k]
            upd[me] = {"IntervalToPositionAhead": {"Value": f"+{t_cross[me] - t_cross[ahead]:.3f}"}}
        add(t_last, "TimingData", {"Lines": upd})

        if lap == 2:
            # car 22 enters the pits during lap 3
            t_in = t_cross["22"] + 85.0
            add(t_in, "TimingData", {"Lines": {"22": {"InPit": True, "NumberOfPitStops": 1}}})
            # stint record first flagged as 'tyres not changed', corrected two laps later
            add(t_in + 10.0, "TimingAppData", {"Lines": {"22": {"Stints": {"1": {
                "Compound": "UNKNOWN", "New": "false", "TyresNotChanged": "1", "TotalLaps": 0, "StartLaps": 0}}}}})
            add(t_in + 16.0, "TimingData", {"Lines": {"22": {"InPit": False, "PitOut": True}}})
            add(t_in + 17.0, "TimingData", {"Lines": {"22": {"Position": "3", "Line": 3}, "33": {"Position": "2", "Line": 2}}})
            add(t_in + 40.0, "TimingData", {"Lines": {"22": {"PitOut": False}}})
            order_after_stop = True
        if lap == 4:
            add(t_cross["11"] + 40.0, "TrackStatus", {"Status": "4", "Message": "SCDeployed"})
            add(t_cross["11"] + 41.0, "TimingAppData", {"Lines": {"22": {"Stints": {"1": {
                "Compound": "HARD", "New": "true", "TyresNotChanged": "0"}}}}})
        if lap == 6:
            add(t_cross["11"] + 30.0, "TrackStatus", {"Status": "1", "Message": "AllClear"})
    add(max(t_cross.values()) + 5.0, "SessionStatus", {"Status": "Finished"})
    return ev


def make_log(events: list[tuple[float, str, dict]] | None = None) -> EventLog:
    from pitsense.config import TOPIC_PRIORITY

    rows = events if events is not None else synthetic_events()
    keyed = sorted(enumerate(rows), key=lambda kv: (kv[1][0], TOPIC_PRIORITY.get(kv[1][1], 99), kv[1][1], kv[0]))
    return EventLog([Event(t, topic, data, i) for i, (_, (t, topic, data)) in enumerate(keyed)], {"slug": "test"})


@pytest.fixture
def race_log() -> EventLog:
    return make_log()


def real_session_dir(year: int, name: str) -> Path | None:
    """Path to a downloaded race if present (integration tests skip otherwise)."""
    root = Path(os.environ.get("PITSENSE_DATA", "data")) / "raw" / str(year)
    if not root.exists():
        return None
    for p in sorted(root.glob(f"*{name}*/*_Race")):
        if (p / "TimingData.jsonStream").exists():
            return p
    return None
