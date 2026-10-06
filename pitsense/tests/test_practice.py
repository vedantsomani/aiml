"""Practice long runs and the race-start tyre / pit-crew priors (synthetic data only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from pitsense import practice
from pitsense.asof import RaceSummary
from pitsense.pitwall.engineer import Context
from pitsense.pitwall.engineers.pitstop import PitStopEngineer
from pitsense.pitwall.engineers.tyre.engineer import TyreEngineer
from pitsense.pitwall.engineers.tyre.model import Priors
from pitsense.state import LapRecord


def lap(driver, n, t, comp="MEDIUM", age=None, stint=1, **kw):
    d = dict(driver=driver, lap=n, t_start=0.0, t_end=0.0, lap_time=t, lap_time_t=0.0, position=1, gap_to_leader=None,
             laps_down=0, interval=None, compound=comp, tyre_age=n if age is None else age, stint=stint,
             is_in_lap=False, is_out_lap=False, track_status="1")
    d.update(kw)
    return LapRecord(**d)


def run(driver, start, n, slope, comp="MEDIUM", base=90.0, stint=1, rng=None):
    rng = rng or np.random.default_rng(0)
    return [lap(driver, start + i, base + slope * i + float(rng.normal(0, 0.05)), comp, i + 1, stint) for i in range(n)]


def test_long_runs_keep_five_consecutive_clean_laps_only():
    laps = run("1", 2, 8, 0.05) + run("2", 2, 4, 0.05)  # driver 2's run is too short
    laps[3] = lap("1", 5, 90.4, is_in_lap=True)  # an in-lap splits driver 1's run: 2,3,4 | 6..9
    assert practice.long_runs(laps) == []
    laps = run("1", 2, 8, 0.05)
    out = practice.long_runs(laps)
    assert len(out) == 1 and len(out[0]) == 8


def test_long_runs_drop_slow_laps_and_unknown_compounds():
    laps = run("1", 2, 8, 0.0) + [lap("2", i, 90.0, "INTERMEDIATE") for i in range(2, 9)]
    laps.append(lap("1", 10, 140.0))  # cool-down lap
    out = practice.long_runs(laps)
    assert [len(r) for r in out] == [8]


def test_fit_runs_recovers_the_slope_and_an_offset():
    rng = np.random.default_rng(1)
    runs, groups = [], []
    for d in range(10):
        runs.append(run(str(d), 2, 10, 0.08, "MEDIUM", 90 + d * 0.3, rng=rng)); groups.append(("S1", str(d)))
        runs.append(run(str(d), 14, 10, 0.08, "SOFT", 89.6 + d * 0.3, rng=rng)); groups.append(("S1", str(d)))
    e = practice.fit_runs(runs, groups)
    assert e.net[1] == pytest.approx(0.08, abs=0.01) and e.net_sd[1] >= practice.NET_FLOOR
    assert e.net[2] is None  # no hard laps: unknown, not zero
    assert e.off[0] == pytest.approx(-0.4, abs=0.15)


def _meta(tmp_path, race_start="2026-03-15T15:00:00", sessions=(("Practice 1", "2026-03-13T12:30:00"),
                                                                ("Practice 2", "2026-03-13T16:00:00"),
                                                                ("Practice 3", "2026-03-16T11:30:00"))):
    meeting = tmp_path / "2026" / "2026-03-15_Test_Grand_Prix"
    for i, (name, start) in enumerate(sessions):
        d = meeting / f"{start[:10]}_{name.replace(' ', '_')}"
        d.mkdir(parents=True)
        (d / "TimingData.jsonStream").write_text("")
        (d / "session.json").write_text(json.dumps({"start_local": start, "gmt_offset": "00:00:00"}))
    return {"path": "2026/2026-03-15_Test_Grand_Prix/2026-03-15_Race/", "start_local": race_start, "gmt_offset": "00:00:00"}


def test_only_sessions_that_started_before_the_race_are_read(tmp_path, monkeypatch):
    monkeypatch.setattr(practice, "raw_dir", lambda: tmp_path)
    meta = _meta(tmp_path)
    found = [d.name for d in practice.earlier_practice(meta)]
    assert found == ["2026-03-13_Practice_1", "2026-03-13_Practice_2"]  # Practice 3 is after this race's start
    assert practice.practice_estimate({"path": "2026/none/x/"}) is None
    assert practice.earlier_practice({}) == []


def test_missing_practice_changes_nothing():
    pri, src = practice.blend(Priors(), None, None)
    assert pri.net == Priors().net and pri.off == Priors().off


def test_blend_moves_toward_history_and_practice_by_precision():
    prac = practice.Estimate((0.05, None, None), (0.01, None, None), (None, None, None), (None, None, None), (50, 0, 0), 2, "practice")
    pri, src = practice.blend(Priors(), None, prac)
    d = Priors().net[0]
    assert d < pri.net[0] < 0.05 and "practice" in src
    assert pri.net_sds[0] < pri.net_sds[1] and pri.net[1] == Priors().net[1]
    hist = {"net": [0.03, None, None], "off": [-0.2, 0.0, 0.1], "n_races": 2}
    both, _ = practice.blend(Priors(), hist, prac)
    assert both.net_sds[0] < pri.net_sds[0] and both.off[0] != Priors().off[0]


def test_circuit_history_uses_only_the_circuit_and_recent_weighs_more():
    def s(i, circuit, net):
        return RaceSummary(f"r{i}", 2025, circuit, datetime(2025, 1, i + 1, tzinfo=timezone.utc),
                           datetime(2025, 1, i + 1, 2, tzinfo=timezone.utc), [], [], [], 50, 0, 0,
                           {"tyre": {"net": [net, None, None], "off": [None, 0.0, None]}})
    h = practice.circuit_history([s(1, 7, 0.10), s(2, 8, 0.50), s(3, 7, 0.00)], 7)
    assert h["n_races"] == 2 and 0.0 < h["net"][0] < 0.05  # 0.00 (latest) outweighs 0.10
    assert practice.circuit_history([], 7)["n_races"] == 0


def test_tyre_engineer_runs_without_practice_or_history():
    eng = TyreEngineer(Context.for_race(None, {"circuit_key": 1}), SimpleNamespace(index=None))
    assert eng.pri.net == Priors().net


def stops(team_times):
    return {"stationary": {t: list(v) for t, v in team_times.items()}, "stops": [], "costop": {}}


def pit_engineer(history):
    past = tuple(RaceSummary(f"r{i}", 2025, 1, datetime(2025, 1, i + 1, tzinfo=timezone.utc),
                             datetime(2025, 1, i + 1, 2, tzinfo=timezone.utc), [], [], [], 50, 0, 0, {"pitstop": h})
                 for i, h in enumerate(history))
    ctx = Context(past_races=past, meta={"circuit_key": 1})
    eng = PitStopEngineer(ctx, SimpleNamespace(index=None))
    eng._learn()
    return eng


def test_crew_prior_is_shrunk_and_follows_team_renames():
    fast = [2.2, 2.3, 2.1, 2.4]
    field = [2.6, 2.7, 2.5, 2.8]
    eng = pit_engineer([stops({"Kick Sauber": fast, "X": field, "Y": field}) for _ in range(3)])
    team, f = eng._stationary("Audi")  # Kick Sauber's crew under its new name
    assert team < f and f - team < 0.5  # faster, but not the raw gap (shrunk)
    assert eng._stationary("Never Seen")[0] == pytest.approx(f)
    raw_gap = np.median(field) - np.median(fast)
    assert f - team < raw_gap


def test_crew_prior_ignores_a_slow_outlier_stop():
    eng = pit_engineer([stops({"A": [2.4, 2.4, 2.5, 7.9], "B": [2.6, 2.7, 2.6, 2.7]}) for _ in range(4)])
    team, f = eng._stationary("A")
    assert team < f  # a median, not a mean: the 7.9 s stop doesn't make A slow
