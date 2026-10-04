from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from pitsense.asof import LeakageError
from pitsense.bench.dataset import decision_rows
from pitsense.bench.evaluate import RACE_SPAN
from pitsense.bench.features import FeatureBuilder, feature_columns
from pitsense.bench.models import BaseRate, NoChange
from pitsense.modelstore import TrainedBundle, load_bundle, save_bundle, train_bundle
from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, PitWall
from pitsense.state import RaceState

T0 = datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc)


def _bundle(train_end=T0 - timedelta(days=1)):
    b = TrainedBundle(train_end, T0, ("r0",), "v0.1", None)
    b.models = {"pit_within_1": BaseRate().fit(pd.DataFrame({"y": [0, 0, 0, 1]}), "y"),
                "pit_within_3": BaseRate().fit(pd.DataFrame({"y": [0, 1]}), "y"),
                "position_after_stop": NoChange()}
    return b


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def test_live_row_equals_benchmark_row(race_log):
    """Replaying a race: the engineer's decision rows equal the benchmark's in every shared column."""
    prior = PitLossPrior()
    ctx = Context(prior=prior, models=_bundle())
    wall = PitWall(ctx)  # every registered engineer, models included
    fb = FeatureBuilder(prior, race_log.meta)
    me = wall.engineer("models")
    state = RaceState(race_log.meta)
    events, n_pits, checked = race_log.events, 0, {"lap_end": 0, "pit_entry": 0}
    queued_laps, queued_pits = [], []
    for k, e in enumerate(events):
        state.apply(e)
        fb.observe(state)
        wall.observe(state)
        queued_laps += state.new_laps
        queued_pits += range(n_pits, len(state.pit_events))
        n_pits = len(state.pit_events)
        if not ((queued_laps or queued_pits) and (k + 1 == len(events) or events[k + 1].t > e.t)):
            continue
        view = wall.view(state)
        for rec in queued_laps:
            bench = fb.row(state, rec)
            live, _ = me.decision_rows(state, rec.driver, view)
            assert (bench is None) == (live is None)
            if bench is not None:
                assert set(feature_columns()) <= live.keys()
                assert all(_same(bench[c], live[c]) for c in bench if c in live), [c for c in bench if not _same(bench[c], live[c])]
                assert {"position", "n_running", "cars_within_pitloss"} <= live.keys()
                checked["lap_end"] += 1
        for i in queued_pits:
            pe = state.pit_events[i]
            prev = fb.index.by_driver.get(pe.driver, {}).get(pe.in_lap - 1)
            bench = fb.row(state, prev, driver=pe.driver, kind="pit_entry", in_lap=pe.in_lap)
            _, live = me.decision_rows(state, pe.driver, view)
            if bench is not None and live is not None:
                assert all(_same(bench[c], live[c]) for c in bench if c in live)
                checked["pit_entry"] += 1
        queued_laps, queued_pits = [], []
    assert checked["lap_end"] > 10 and checked["pit_entry"] >= 1


def test_engineer_fills_keys_only_with_a_bundle(race_log):
    for models, expect in ((None, False), (_bundle(), True)):
        wall = PitWall(Context(prior=PitLossPrior(), models=models))
        state = RaceState(race_log.meta)
        for e in race_log.events[: len(race_log) // 2]:
            state.apply(e)
            wall.observe(state)
        vals = [wall.view(state).car("models", n) for n in state.drivers]
        from pitsense.pitwall.engineers.models import KEYS

        assert all(set(v) == set(KEYS) and {"pit_prob_1", "pit_prob_3", "rejoin_pred", "p_stop_le_2", "laps_to_stop_med"} <= set(v) for v in vals)
        assert any(v["pit_prob_3"] is not None for v in vals) == expect
        if expect:
            assert {v["pit_prob_1"] for v in vals if v["pit_prob_1"] is not None} == {0.25}
        wall.snapshot(state).to_dict()  # strict JSON


def test_guard_refuses_a_bundle_trained_after_the_race_started(tmp_path):
    late = _bundle(train_end=T0 + timedelta(hours=1))
    with pytest.raises(LeakageError):
        Context.for_race(PitLossPrior(), race_start_utc=T0, models=late)
    path = save_bundle(late, tmp_path / "b.pkl")
    with pytest.raises(LeakageError):
        load_bundle(path, T0)
    assert load_bundle(path).train_end_utc == late.train_end_utc  # loading without a race is allowed
    ok = save_bundle(_bundle(), tmp_path / "ok.pkl")
    assert load_bundle(ok, T0).models.keys() == _bundle().models.keys()


def _fake_bench(n_races=6):
    rng = np.random.default_rng(0)
    parts = []
    for r in range(n_races):
        n = 120
        df = pd.DataFrame({c: rng.normal(size=n) for c in feature_columns()})
        df["position"] = rng.integers(1, 20, n).astype(float)
        df["cars_within_pitloss"] = rng.integers(0, 4, n)
        df["n_running"] = 20
        df["kind"] = ["lap_end"] * 100 + ["pit_entry"] * 20
        df["y_retire_3"] = 0
        df["y_pit_1"] = rng.integers(0, 2, n)
        df["y_pit_3"] = rng.integers(0, 2, n)
        df["y_pos_after_stop"] = df["position"] + rng.integers(0, 3, n)
        df["race_id"] = f"r{r}"
        df["start_utc"] = pd.Timestamp(T0 + timedelta(days=7 * r))
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def test_bundle_window_and_determinism():
    from pitsense.bench.tasks import core_tasks
    from pitsense.bench.models import GBMHazard

    df = _fake_bench()
    tasks = [t for t in core_tasks() if t.name == "pit_within_1"]
    cutoff = T0 + timedelta(days=7 * 4)  # races 0..3 ended; race 4 starts at the cutoff
    a = train_bundle(df, cutoff, tasks=tasks, choices={"pit_within_1": "gbm_hazard"}, holdout=1)
    b = train_bundle(df, cutoff, tasks=tasks, choices={"pit_within_1": "gbm_hazard"}, holdout=1)
    assert a.trained_on == ("r0", "r1", "r2", "r3")
    assert a.train_end_utc == T0 + timedelta(days=21) + RACE_SPAN < cutoff
    test = tasks[0].select(df[df.race_id == "r4"])
    assert np.array_equal(a.predict("pit_within_1", test), b.predict("pit_within_1", test))
    assert a.metrics["pit_within_1"]["holdout_races"] == ["r3"]
    # equals what the benchmark's fit_predict gives for that race
    from pitsense.bench.evaluate import fit_predict

    train = tasks[0].select(df[df.start_utc + RACE_SPAN < pd.Timestamp(cutoff)])
    assert np.array_equal(fit_predict(GBMHazard, train, test, "y_pit_1"), a.predict("pit_within_1", test))
