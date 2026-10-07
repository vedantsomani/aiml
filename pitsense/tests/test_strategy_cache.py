"""The strategy engineer's plan cache: what makes a focus car's plan stale.

A plan is cached per car and served until its key changes. The key once held the lap, track status, rule facts
and penalty, so a tyre change the feed confirms late (1-8 laps after the stop), the stop count and rain starting or
stopping were not seen until the car's next lap. The planner itself is replaced by a counting stand-in: only
the cache decisions are under test.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, PitWall, TeamConfig
from pitsense.pitwall.engineers.rules import RulesEngineer
from pitsense.pitwall.engineers.strategy import analysis
from pitsense.pitwall.engineers.strategy.engineer import get_analysis
from pitsense.pitwall.engineers.weather import WeatherEngineer
from pitsense.state import DriverState, RaceState

from .conftest import make_log, synthetic_events


class Planner:
    """Stands in for ``analysis.analyse``: a new object per car per call, and a log of what was asked."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def __call__(self, state, view, ctx, memory, cars, *, light=False):
        self.calls.append((list(cars), light))
        return {n: SimpleNamespace(car=n, run=len(self.calls)) for n in cars}


class World:
    """Two focus cars on lap 10 in a dry race, with the values an engineer view would hold for them."""

    def __init__(self, monkeypatch, cars=("16", "44")):
        self.drivers = {n: DriverState(number=n, tla=f"T{n}", laps=10, compound="MEDIUM", pit_stops=0, stint=1) for n in cars}
        self.state = SimpleNamespace(drivers=self.drivers, track_status="1")
        self.rules = {"sc_phase": "none", "pit_lane_open": True, "race_dry": True}
        self.weather = {"rainfall": 0.0, "rain_now": 0.0, "wet_running": False, "crossover": "none", "rain_prob_10min": 0.05,
                        "minutes_since_rain": -1.0, "rain_minutes": 0.0, "track_temp": 40.0, "inters_vs_slicks_s": None}
        self.car_rules = {n: {"must_stop": False, "penalty_s_pending": 0.0} for n in cars}
        self.ctx = Context(team=TeamConfig(cars=tuple(cars)))
        self.planner = Planner()
        monkeypatch.setattr(analysis, "analyse", self.planner)

    def view(self):
        w = self
        return SimpleNamespace(race=lambda name: dict({"rules": w.rules, "weather": w.weather}[name]),
                               car=lambda name, n: dict(w.car_rules[n]))

    def get(self):
        return get_analysis(self.ctx, None, self.state, self.view())


def test_nothing_changed_serves_the_cached_objects(monkeypatch):
    w = World(monkeypatch)
    first = w.get()
    assert w.get() == first and all(w.get()[n] is first[n] for n in first)
    assert len(w.planner.calls) == 1 and w.planner.calls[0][0] == ["16", "44"]


@pytest.mark.parametrize("attr,value", [
    ("compound", "HARD"),  # the feed confirms the new set a lap after the stop
    ("stint", 2),  # a stop to the same compound: the new stint record is all that changes
    ("pit_stops", 1),  # the stop count arrives before the stint record
])
def test_tyre_change_on_the_same_lap_refreshes_that_car_only(monkeypatch, attr, value):
    w = World(monkeypatch)
    first = w.get()
    setattr(w.drivers["16"], attr, value)
    again = w.get()
    assert again["16"] is not first["16"] and again["44"] is first["44"]
    assert w.planner.calls[-1][0] == ["16"]
    assert w.get()["16"] is again["16"]  # and it is cached again


@pytest.mark.parametrize("where,key,value", [
    ("weather", "rainfall", 1.0),  # rain starts
    ("weather", "rain_now", 1.0),
    ("weather", "wet_running", True),  # a car is on wet tyres
    ("weather", "crossover", "to_inters"),
    ("rules", "race_dry", False),  # wet tyres have been used: the race is no longer a dry one
])
def test_weather_flag_refreshes_every_plan_and_back(monkeypatch, where, key, value):
    w = World(monkeypatch)
    first = w.get()
    table = getattr(w, where)
    old = table[key]
    table[key] = value
    wet = w.get()
    assert all(wet[n] is not first[n] for n in first)  # every focus car, same lap
    assert w.get()["16"] is wet["16"]
    table[key] = old  # rain stops
    dry = w.get()
    assert all(dry[n] is not wet[n] for n in first)


def test_nowcast_values_do_not_defeat_the_cache(monkeypatch):
    w = World(monkeypatch)
    first = w.get()
    for k in range(20):  # these move on every message while it rains
        w.weather.update(rain_prob_10min=0.05 + k / 100, minutes_since_rain=float(k), rain_minutes=k / 3, track_temp=40.0 - k / 10,
                         inters_vs_slicks_s=k / 7)
        assert w.get()["16"] is first["16"]
    assert len(w.planner.calls) == 1


@pytest.mark.parametrize("change", [
    lambda w: setattr(w.drivers["16"], "laps", 11),
    lambda w: setattr(w.state, "track_status", "4"),
    lambda w: w.rules.update(sc_phase="sc"),
    lambda w: w.rules.update(pit_lane_open=False),
    lambda w: w.car_rules["16"].update(must_stop=True),
    lambda w: w.car_rules["16"].update(penalty_s_pending=5.0),
], ids=["lap", "track status", "sc phase", "pit lane", "must stop", "penalty"])
def test_the_earlier_triggers_still_refresh(monkeypatch, change):
    w = World(monkeypatch)
    first = w.get()
    change(w)
    assert w.get()["16"] is not first["16"]


def _same_compound_stop(events):
    """The synthetic race, with car 22's stop (laps 3-4) put on the compound it already had: the rules engineer's
    must-stop flag then does not change when the feed confirms the stop, so only the tyre values can show it."""
    out = copy.deepcopy(events)
    for _, topic, data in out:
        stint = ((data.get("Lines") or {}).get("22") or {}).get("Stints") if topic == "TimingAppData" else None
        if isinstance(stint, dict) and stint.get("1", {}).get("Compound") == "HARD":
            stint["1"]["Compound"] = "MEDIUM"
    return out


def test_replayed_race_refreshes_on_a_late_tyre_confirmation_and_on_rain(monkeypatch):
    planner = Planner()
    monkeypatch.setattr(analysis, "analyse", planner)
    rain = [(2.0, "WeatherData", {"Rainfall": "0", "TrackTemp": "40", "AirTemp": "25", "Humidity": "50"}),
            (320.0, "WeatherData", {"Rainfall": "1"}), (345.0, "WeatherData", {"Rainfall": "0"})]  # during lap 3
    log = make_log(_same_compound_stop(synthetic_events()) + rain)
    wall = PitWall(Context(prior=PitLossPrior()), engineers=[RulesEngineer, WeatherEngineer])
    state = RaceState(log.meta)
    last: dict[str, object] = {}
    refreshes: dict[str, list[tuple]] = {}  # car -> (t, laps, stint, stops, rain flag) each time its plan was recomputed
    for e in log.events:
        state.apply(e)
        wall.observe(state)
        for n, a in get_analysis(wall.ctx, wall.memory, state, wall.view(state)).items():
            if last.get(n) is not a:
                last[n] = a
                d = state.drivers[n]
                refreshes.setdefault(n, []).append((state.t, d.laps, d.stint, d.pit_stops, state.weather.get("Rainfall")))

    def steps(car):  # consecutive refreshes
        return list(zip(refreshes[car], refreshes[car][1:]))

    # car 22: the stint record arrives after the stop, with the lap count unchanged (same compound, so only stint / tyre values moved)
    assert any(a[1] == b[1] and (a[2], b[2]) == (1, 2) for a, b in steps("22")), refreshes["22"]
    assert any(a[1] == b[1] and (a[3], b[3]) == (0, 1) for a, b in steps("22")), refreshes["22"]  # and the stop count, before that
    for car in refreshes:  # rain on and off, mid-lap, for every car
        assert any(a[1] == b[1] and (a[4], b[4]) == (0.0, 1.0) for a, b in steps(car)), (car, refreshes[car])
        assert any(a[1] == b[1] and (a[4], b[4]) == (1.0, 0.0) for a, b in steps(car)), (car, refreshes[car])
        # a key that moved with every message would refresh on each of the 60-odd events
        assert len(refreshes[car]) <= 20, len(refreshes[car])


def test_risk_preference_changes_which_plan_ranks_first():
    import numpy as np

    from pitsense.pitwall.engineers.strategy.analysis import risk_score

    steady = np.full(100, 5.0)  # always P5
    gamble = np.array([3.0] * 75 + [8.0] * 25)  # usually P3, a quarter of the time P8
    best = {r: min(("steady", steady), ("gamble", gamble), key=lambda p: risk_score(p[1], r))[0]
            for r in ("expected", "protect", "aggressive")}
    assert best == {"expected": "gamble", "protect": "steady", "aggressive": "gamble"}
    assert risk_score(gamble, "unknown") == risk_score(gamble) == gamble.mean()


def test_teammates_are_not_double_stacked_under_green_unless_waiting_costs():
    from types import SimpleNamespace as NS

    from pitsense.pitwall.engineers.head import double_stack
    from pitsense.pitwall.types import Call

    drivers = {"16": NS(team="Ferrari", position=3, tla="LEC"), "44": NS(team="Ferrari", position=5, tla="HAM"),
               "1": NS(team="Red Bull", position=4, tla="VER")}
    calls = [Call(1.0, "16", "BOX", "HARD"), Call(1.0, "44", "BOX", "HARD"), Call(1.0, "1", "BOX", "HARD")]

    def plans(cost):
        return {"44": NS(ranked=[NS(first_offset=0, util=5.0, stops=((20, "HARD"),)),
                                 NS(first_offset=1, util=5.0 + cost, stops=((21, "MEDIUM"),))])}

    out = {c.car: c for c in double_stack(calls, drivers, "1", plans(0.2))}
    assert out["16"].action == "BOX" and out["1"].action == "BOX"  # the car ahead, and another team, untouched
    assert out["44"].action == "PREPARE_BOX" and out["44"].compound == "MEDIUM" and out["44"].reasons[0].code == "double_stack"
    kept = {c.car: c for c in double_stack(calls, drivers, "1", plans(0.8))}["44"]
    assert kept.action == "BOX" and "waiting a lap would cost 0.8 places" in kept.reasons[0].text
    sc = {c.car: c for c in double_stack(calls, drivers, "4", plans(0.2))}["44"]
    assert sc.action == "BOX" and "double stack behind LEC" in sc.reasons[0].text
