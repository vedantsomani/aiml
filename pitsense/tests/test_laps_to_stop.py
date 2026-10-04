from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pitsense.bench import laps_to_stop as L
from pitsense.bench.dataset import decision_rows
from pitsense.bench.labels import add_labels
from pitsense.modelstore import TrainedBundle
from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, PitWall
from pitsense.state import RaceState

T0 = datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc)


def _frame(race_log) -> pd.DataFrame:
    rows, final = decision_rows(race_log, PitLossPrior())
    df = pd.DataFrame(add_labels(rows, final))
    df["race_id"], df["start_utc"] = "r0", pd.Timestamp(T0 - timedelta(days=1))
    return df[df.kind == "lap_end"].reset_index(drop=True)


def test_labels_are_consistent(race_log):
    df = _frame(race_log)
    ev = df[df.y_stop_event == 1]
    assert len(ev) and (ev.y_laps_to_stop >= 1).all()
    assert (ev.y_stop_obs == ev.y_laps_to_stop).all()
    nev = df[df.y_stop_event == 0]
    assert nev.y_laps_to_stop.isna().all() and (nev.y_stop_obs >= 0).all()
    # same next stop as the existing label
    assert ((ev.lap + ev.y_laps_to_stop) == ev.y_next_in_lap).all()


def test_cdf_and_summary():
    hz = np.full((2, L.H), 0.2)
    cdf = L.cdf_from_hazard(hz, np.array([99.0, 2.0]))
    assert (np.diff(cdf, axis=1) >= -1e-12).all()
    assert np.allclose(cdf[0, :3], 1 - 0.8 ** np.arange(1, 4))
    assert np.allclose(cdf[1, 2:], cdf[1, 1])  # no hazard beyond the laps left in the race
    s = L.summarize(cdf)
    assert s["laps_to_stop_med"][0] == 4 and s["laps_to_stop_med"][1] == L.CAP
    assert 1 < s["laps_to_stop_exp"][0] < s["laps_to_stop_exp"][1] <= L.CAP
    assert abs(s["p_stop_le_1"][0] - 0.2) < 1e-9 and set(s) >= {f"p_stop_le_{k}" for k in L.KS}


def test_concordance_handles_never_stopping_and_retirement():
    obs, ev, ret = np.array([2, 5, 9, 4.0]), np.array([1, 1, 1, 0]), np.array([0, 0, 0, 1])
    assert L.concordance(obs, ev, np.array([2.0, 5, 9, 1]), ret) < 1.0  # the retiree (seen 4 laps) is later than the stop at 2 only
    assert L.concordance(obs, ev, np.array([2.0, 5, 9, 4.5]), ret) == 1.0
    assert L.concordance(obs, ev, np.array([9.0, 5, 2, 4.5]), ret) == 0.0


def test_survival_gbm_fits_and_is_deterministic(race_log):
    df = _frame(race_log)
    p = dict(L.SurvivalGBM.params, max_iter=5, min_samples_leaf=5)
    cls = type("M", (L.SurvivalGBM,), {"params": p, "thin": 1})
    a, b = cls().fit(df).predict_cdf(df), cls().fit(df).predict_cdf(df)
    assert a.shape == (len(df), L.H) and np.array_equal(a, b)
    assert (np.diff(a, axis=1) >= -1e-12).all() and (a >= 0).all() and (a <= 1).all()
    sc = L.survival_scores(df, a)
    assert 0 <= sc["ibs"] <= 1 and 0 <= sc["window_acc"] <= 1


def test_live_values_match_the_benchmark_predictions(race_log):
    df = _frame(race_log)
    p = dict(L.SurvivalGBM.params, max_iter=5, min_samples_leaf=5)
    model = type("M", (L.SurvivalGBM,), {"params": p, "thin": 1})().fit(df)
    b = TrainedBundle(T0 - timedelta(days=1), T0, ("r0",), "v0.1", None, models={"laps_to_stop": model})
    wall = PitWall(Context(prior=PitLossPrior(), models=b))
    state, seen = RaceState(race_log.meta), 0
    for e in race_log.events[: len(race_log) // 2]:
        state.apply(e)
        wall.observe(state)
    view = wall.view(state)
    for n, d in state.drivers.items():
        vals = view.car("models", n)
        if not d.running or vals.get("p_stop_le_1") is None:
            continue
        lap_end, _ = wall.engineer("models").decision_rows(state, n, view)
        want = L.summarize(model.predict_cdf(pd.DataFrame([lap_end])))
        assert all(abs(vals[k] - float(want[k][0])) < 1e-12 for k in want)
        assert vals["p_stop_le_1"] <= vals["p_stop_le_2"] <= vals["p_stop_le_3"] <= vals["p_stop_le_5"] <= vals["p_stop_le_8"]
        seen += 1
    assert seen > 0
