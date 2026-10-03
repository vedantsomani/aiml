"""Rival strategist: prior hazard, summaries, undercut maths, and the engineer on the synthetic feed."""

import math
from datetime import datetime, timezone

import pandas as pd
import pytest

from pitsense.asof import RaceSummary
from pitsense.bench.rivals import _Race, rivals_labels
from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, PitWall
from pitsense.pitwall.engineers.pitstop import PitStopEngineer
from pitsense.pitwall.engineers.rivals import RivalsEngineer
from pitsense.pitwall.engineers.tyre import TyreEngineer
from pitsense.state import RaceState, replay

from .conftest import make_log

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _history(stints, teams=None, circuit=7):
    extra = {"rivals": {"stints": stints, "teams": teams or {}, "circuit": circuit}}
    return RaceSummary("r1", 2025, circuit, T0, T0, [], [], [], 50, 0, 0, extra)


def _engineer(history=(), circuit=7):
    ctx = Context(prior=PitLossPrior(), past_races=tuple(history), meta={"circuit_key": circuit})
    return RivalsEngineer(ctx, None)


def test_default_hazard_rises_with_age_and_horizon():
    e = _engineer()
    assert e.prior_hazard("MEDIUM", 5, 3) < e.prior_hazard("MEDIUM", 25, 3)
    assert e.prior_hazard("MEDIUM", 20, 1) < e.prior_hazard("MEDIUM", 20, 5)
    assert e.prior_hazard("SOFT", 15, 3) > e.prior_hazard("HARD", 15, 3)


def test_hazard_learns_the_circuit_stint_length():
    stints = {"MEDIUM": [[20, 1]] * 30}  # every medium stint at this circuit ends after 20 laps
    e = _engineer([_history(stints)])
    assert e.prior_hazard("MEDIUM", 18, 3) > 0.8  # stop due
    assert e.prior_hazard("MEDIUM", 5, 3) < 0.2
    other = _engineer([_history(stints)], circuit=99)  # another circuit only sees the pooled stints
    assert other.prior_hazard("MEDIUM", 18, 3) == pytest.approx(e.prior_hazard("MEDIUM", 18, 3), abs=0.2)
    assert e._typical["MEDIUM"] == 20.0


def test_team_cover_rate_shrinks_to_the_field():
    e = _engineer([_history({}, {"Fast": [40, 30], "Slow": [40, 0], "New": [0, 0]})])
    assert e.team_cover_rate("Fast") > e.team_cover_rate("New") > e.team_cover_rate("Slow")
    assert 0 < e.team_cover_rate("Slow") < e.team_cover_rate("Fast") < 1


def test_summarize_race_reads_stints_and_covers(race_log):
    final = replay(race_log)
    out = RivalsEngineer.summarize_race(final, {"circuit_key": 3})
    assert out["circuit"] == 3
    done = [s for rows in out["stints"].values() for s in rows if s[1] == 1]
    assert done, "car 22's stop should give one finished stint"
    assert all(math.isfinite(n) and n >= 1 for rows in out["stints"].values() for n, _ in rows)


def _wall(race_log, history=()):
    ctx = Context(prior=PitLossPrior(), past_races=tuple(history), meta={"circuit_key": 7})
    return PitWall(ctx, None, [TyreEngineer, PitStopEngineer, RivalsEngineer])


def test_engineer_fills_contract_keys_as_scalars_and_is_deterministic(race_log):
    outs = []
    for _ in range(2):
        state, wall = RaceState(race_log.meta), _wall(race_log)
        seen = []
        for e in race_log.events:
            state.apply(e)
            wall.observe(state)
            if state.new_laps:
                seen.append({k: v for k, v in wall.car_values(state, "22").items() if k.startswith("rivals__")})
        outs.append(seen)
    assert outs[0] == outs[1]
    last = outs[0][-1]
    for key in ("ahead", "behind", "gap_ahead", "gap_behind", "pit_prob_1", "pit_prob_3", "undercut_threat",
                "undercut_chance"):
        assert f"rivals__{key}" in last
    for seen in outs[0]:
        for p in ("rivals__pit_prob_1", "rivals__pit_prob_3"):
            assert seen[p] is None or 0.0 <= seen[p] <= 1.0
    assert not wall.alerts(state) or all(a.engineer == "rivals" for a in wall.alerts(state))


def test_pit_prob_is_zero_when_the_race_is_over(race_log):
    state, wall = RaceState(race_log.meta), _wall(race_log)
    for e in race_log.events:
        state.apply(e)
        wall.observe(state)
        if state.new_laps and state.drivers["11"].laps >= 8:
            assert wall.car_values(state, "11")["rivals__pit_prob_3"] == 0.0
            return
    pytest.fail("never reached the last lap")


def test_margin_follows_tyre_gain_and_gap():
    e = _engineer()

    class V:  # tyre values for two cars
        def car(self, who, n):
            return {"1": {"fresh_soft_s": 89.0, "fresh_medium_s": 89.5, "fresh_hard_s": 90.0},
                    "2": {"pace_s": 92.0, "deg_s_per_lap": 0.2}}[n]

    near, far = e._margin(V(), "1", "2", 1.0), e._margin(V(), "1", "2", 4.0)
    assert near - far == pytest.approx(3.0)
    assert e._margin(V(), "1", "2", None) is None


class _Final:
    def __init__(self, stops, pos):
        from types import SimpleNamespace as N

        self.pit_events = [N(driver=d, in_lap=lap, under_red=False) for d, lap in stops]
        self.laps = [N(driver=d, lap=lap, position=p) for (d, lap), p in pos.items()]


def test_undercut_label_needs_chaser_first_and_ahead_after_both_stop():
    pos = {("B", 12): 1, ("A", 12): 2}  # chaser B is ahead of A after the stops (lap 10 + 3 -> 13 missing, 12 used)
    race = _Race(_Final([("B", 8), ("A", 9)], {("B", 12): 1, ("A", 12): 2}))
    assert race.over("B", "A", 6) == (1.0, 1.0)
    assert race.over("A", "B", 6) == (0.0, 0.0)  # A stops after B: not first
    assert _Race(_Final([("B", 8), ("A", 9)], {("B", 12): 2, ("A", 12): 1})).over("B", "A", 6) == (1.0, 0.0)
    assert race.over("B", "A", 1) == (0.0, 0.0)  # more than 5 laps away
    rows = [{"kind": "lap_end", "driver": "A", "lap": 6, "rivals__behind": "B", "rivals__ahead": None}]
    rivals_labels(rows, _Final([("B", 8), ("A", 9)], {("B", 12): 1, ("A", 12): 2}))
    assert rows[0]["y_uc"] == 1.0 and math.isnan(rows[0]["y_uc_chance"])
