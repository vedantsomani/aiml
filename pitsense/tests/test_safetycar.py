"""Safety-car engineer: message classes, yellow bookkeeping, as-of values, training samples, strategy hook (synthetic)."""

from __future__ import annotations

import numpy as np

from pitsense.pitloss import PitLossPrior
from pitsense.pitwall.engineer import Context
from pitsense.pitwall.engineers.safetycar import (FEATURES, SafetyCarEngineer, Tracker, _classify, fit_logit,
                                                  race_samples, status_starts)
from pitsense.pitwall.memory import RaceMemory
from pitsense.pitwall.wall import PitWall
from pitsense.state import RaceState

from .conftest import make_log


def test_message_classes():
    assert _classify("Flag", "DOUBLE YELLOW", "DOUBLE YELLOW IN TRACK SECTOR 7") == ("dy", 7)
    assert _classify("Flag", "YELLOW", "YELLOW IN TRACK SECTOR 3") == ("y", 3)
    assert _classify("Flag", "CLEAR", "CLEAR IN TRACK SECTOR 3") == ("clear", 3)
    assert _classify("Other", "", "CAR 14 (ALO) STOPPED AT TURN 5")[0] == "stopped"
    assert _classify("Other", "", "CAR 1 (VER) OFF TRACK AND CONTINUED AT TURN 4")[0] == "off"
    assert _classify("Other", "", "RECOVERY VEHICLE ON TRACK AT TURN 9")[0] == "vehicle"
    assert _classify("Other", "", "FIA STEWARDS: TURN 1 INCIDENT INVOLVING CARS 4 AND 81 NOTED")[0] == "incident"
    assert _classify("Other", "", "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 4")[0] == "other"


def test_yellow_sectors_open_and_clear():
    tr = Tracker()
    tr.add_rc(100.0, "Flag", "YELLOW", "YELLOW IN TRACK SECTOR 3", 5)
    tr.add_rc(110.0, "Flag", "DOUBLE YELLOW", "DOUBLE YELLOW IN TRACK SECTOR 3", 5)
    f = tr.features(120.0, "2", 100.0, 5, 50)
    assert f["dy_active"] > 0 and f["y_active"] == 0 and f["yellow_now"] == 1.0 and abs(f["yellow_age"] - 20 / 120) < 1e-9
    tr.add_rc(130.0, "Flag", "CLEAR", "CLEAR IN TRACK SECTOR 3", 5)
    g = tr.features(140.0, "1", 130.0, 5, 50)
    assert g["dy_active"] == 0 and g["yellow_now"] == 0.0
    assert tr.features(110.0 + 1000, "1", 0.0, 5, 50)["yellow_old"] == 0.0  # nothing left active
    assert all(0.0 <= v <= 1.0 for v in f.values())
    assert set(f) == set(FEATURES)


def test_fit_logit_recovers_a_signal_and_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.random((4000, 3))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(-4 + 4 * X[:, 0])))).astype(float)
    w = fit_logit(X, y)
    assert w[1] > 1.0 and abs(w[2]) < 1.0
    assert np.array_equal(w, fit_logit(X, y))


def _run(events=None, ctx=None):
    log = make_log(events)
    wall = PitWall(ctx or Context(prior=PitLossPrior()), RaceMemory(), [SafetyCarEngineer])
    st = RaceState(log.meta)
    vals = []
    for e in log.events:
        st.apply(e)
        wall.observe(st)
        vals.append((st.t, st.track_status, wall.race_values(st)))
    return wall, st, vals


def test_values_scalar_bounded_zero_while_neutralised():
    wall, st, vals = _run()
    for t, status, v in vals:
        assert 0.0 <= v["safetycar__sc_prob_2laps"] <= 1.0 and 0.0 <= v["safetycar__vsc_prob_2laps"] <= 1.0
        assert isinstance(v["safetycar__sc_reason"], str)
        if status == "4":
            assert v["safetycar__sc_prob_2laps"] == 0.0 and v["safetycar__sc_reason"] == "safety car out"
    assert any(s == "4" for _, s, _ in vals)


def test_yellow_message_raises_probability_and_alerts():
    ev = [(150.0, "RaceControlMessages", {"Messages": [{"Category": "Flag", "Flag": "DOUBLE YELLOW", "Scope": "Sector",
                                                        "Sector": 7, "Message": "DOUBLE YELLOW IN TRACK SECTOR 7"}]}),
          (150.5, "TrackStatus", {"Status": "2", "Message": "Yellow"})]
    from .conftest import synthetic_events

    wall, st, vals = _run(synthetic_events() + ev)
    at = [v for t, s, v in vals if s == "2"]
    before = [v for t, s, v in vals if t < 150.0 and s == "1"]
    assert at and at[0]["safetycar__sc_prob_2laps"] > before[-1]["safetycar__sc_prob_2laps"]
    assert at[0]["safetycar__dy_sectors"] == 1


def test_race_samples_and_status_starts():
    wall, st, _ = _run()
    starts = status_starts(list(st.status_log))
    assert len(starts["sc"]) == 1 and starts["vsc"] == []
    out = race_samples(st)
    assert out["sc_laps"] and all(len(r) == len(FEATURES) + 2 for r in out["samples"])
    assert not any(v != v for r in out["samples"] for v in r)  # no NaN


def test_strategy_hook_falls_back_without_the_engineer():
    from pitsense.pitwall.engineers.strategy.analysis import near_rates

    class V:
        def race(self, name):
            return wall.view(st).race(name)

    wall_without = PitWall(Context(prior=PitLossPrior()), RaceMemory(), [])
    wall, st, _ = _run()
    assert near_rates(wall.view(st))[0] is not None
    st2 = RaceState({})
    assert near_rates(wall_without.view(st2)) == (None, None)
