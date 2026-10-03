"""Feed differences in older seasons (2018-2024), on small synthetic snippets."""

from __future__ import annotations

import json

from pitsense.config import NOMINATIONS
from pitsense.state import relative_compound, replay

from .conftest import make_log, synthetic_events

PIRELLI_2018 = ("HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD", "SUPERHARD")  # softest first


def _renamed(events: list, names: dict[str, str]) -> list:
    out = []
    for t, topic, data in events:
        if topic == "TimingAppData":
            text = json.dumps(data)
            for old, new in names.items():
                text = text.replace(f'"{old}"', f'"{new}"')
            data = json.loads(text)
        out.append((t, topic, data))
    return out


def test_2018_compound_names_follow_the_weekends_nomination(monkeypatch):
    """2018 says SUPERSOFT/SOFT where later seasons say MEDIUM/HARD for the same tyres."""
    monkeypatch.setitem(NOMINATIONS, (2018, "Test Grand Prix"), ("HYPERSOFT", "SUPERSOFT", "SOFT"))
    modern = replay(make_log())
    log = make_log(_renamed(synthetic_events(), {"MEDIUM": "SUPERSOFT", "HARD": "SOFT"}))
    log.meta = {"year": 2018, "meeting_name": "Test Grand Prix"}
    old = replay(log)
    tyres = lambda s: [(x.driver, x.lap, x.compound, x.tyre_age, x.stint) for x in s.laps]  # noqa: E731
    assert tyres(old) == tyres(modern)
    assert old.drivers["22"].compounds_used == ["MEDIUM", "HARD"]  # that "SOFT" was the hardest tyre

    # no nomination (any later season): names are already relative and pass through
    log.meta = {"year": 2019, "meeting_name": "Test Grand Prix"}
    assert replay(log).drivers["22"].compounds_used == ["SUPERSOFT", "SOFT"]


def test_sets_published_in_one_message_all_count_as_used():
    """A car's first sets can arrive together, after a stop: 2018 publishes the first stint
    records minutes after the start, and Miami 2025 resent them after an outage."""
    events = []
    for t, topic, data in synthetic_events():
        car = (data.get("Lines") or {}).get("22") if topic == "TimingAppData" else None
        if car and car.get("Stints"):  # drop car 22's stint records ...
            data = {"Lines": {k: v for k, v in data["Lines"].items() if k != "22"}}
            if not data["Lines"]:
                continue
        events.append((t, topic, data))
    t_lap4 = next(t for t, topic, data in events
                  if topic == "TimingData" and (data["Lines"].get("22") or {}).get("NumberOfLaps") == 4)
    both = {"0": {"Compound": "MEDIUM", "New": "true", "TyresNotChanged": "0", "TotalLaps": 3, "StartLaps": 0},
            "1": {"Compound": "HARD", "New": "true", "TyresNotChanged": "0", "TotalLaps": 1, "StartLaps": 0}}
    events.append((t_lap4 + 5.0, "TimingAppData", {"Lines": {"22": {"Stints": both}}}))  # ... and send both late

    s = replay(make_log(events))
    lap = {x.lap: x for x in s.laps if x.driver == "22"}
    assert lap[4].compound is None  # nothing published yet
    assert (lap[5].compound, lap[5].tyre_age) == ("HARD", 2)  # fitted at the lap-3 stop
    assert s.drivers["22"].compounds_used == ["MEDIUM", "HARD"]  # the starting set counts too


def test_lap_time_sent_just_before_its_lap_count():
    """Some 2018 races publish LastLapTime ~1 s before NumberOfLaps. That value is the new
    lap's time, not a late value for the previous lap (or for lap 1, which has none)."""
    events = []
    for t, topic, data in synthetic_events():
        line = (data.get("Lines") or {}).get("11") if topic == "TimingData" else None
        if line and "NumberOfLaps" in line and "LastLapTime" in line:
            line = {k: v for k, v in line.items() if k != "LastLapTime"}
            events.append((t - 1.0, "TimingData", {"Lines": {"11": {"LastLapTime": data["Lines"]["11"]["LastLapTime"]}}}))
            data = {"Lines": {**data["Lines"], "11": line}}
        events.append((t, topic, data))
    times = lambda log: {x.lap: x.lap_time for x in replay(log).laps if x.driver == "11"}  # noqa: E731
    expected = times(make_log())
    assert expected[1] is None and len(set(expected.values())) == 8  # a different time every lap
    assert times(make_log(events)) == expected


def test_2018_nomination_table():
    entries = {race: c for race, c in NOMINATIONS.items() if race[0] == 2018}
    assert len(entries) == 21
    for race, compounds in entries.items():
        ranks = [PIRELLI_2018.index(c) for c in compounds]
        assert ranks == sorted(set(ranks)), race  # three distinct compounds, softest first
    melbourne = {"year": 2018, "meeting_name": "Australian Grand Prix"}
    assert [relative_compound(c, melbourne) for c in ("ULTRASOFT", "SUPERSOFT", "SOFT")] == ["SOFT", "MEDIUM", "HARD"]
    assert relative_compound("INTERMEDIATE", melbourne) == "INTERMEDIATE"
    assert relative_compound("SOFT", {}) == "SOFT"
