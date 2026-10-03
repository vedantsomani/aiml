"""Pit-stop benchmark pieces: rejoin distribution, models, the pit_loss label (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitsense.bench.dataset import decision_rows
from pitsense.bench.labels import add_labels
from pitsense.bench.pitstop import PitstopGBM, PitstopRule, V01Prior, loss_rows
from pitsense.pitloss import PitLossPrior, expected_rejoin


def test_rejoin_counts_passers_and_respects_cars_that_stop():
    gaps = [3.0, 6.0, 9.0, 30.0]
    r = expected_rejoin(10, gaps, loss=7.0, sd=0.5, p_stop=[0.0] * 4)
    assert r.position == 12 and r.within == 2
    r = expected_rejoin(10, gaps, loss=7.0, sd=0.5, p_stop=[1.0, 0.0, 0.0, 0.0])
    assert r.position == 11  # the first car stops too and stays behind
    assert r.within_stopping == 1.0


def test_rejoin_summary_is_the_mode_of_the_passers_distribution():
    # three cars, each 60% to get by: expected 1.8 passers (mean rounds to 2), mode is 2, median 2
    gaps, p = [1.0, 2.0, 3.0], [0.4] * 3
    assert expected_rejoin(5, gaps, 100.0, 1.0, p, stat="mode").position == 7
    # one car at 30%: the mode is "nobody gets by" although the mean is 0.3
    assert expected_rejoin(5, [1.0], 100.0, 1.0, [0.7], stat="mode").position == 5
    assert expected_rejoin(5, [1.0] * 4, 100.0, 1.0, [0.5] * 4, stat="mean").position == 7


def _frame():
    return pd.DataFrame({
        "position": [5.0, 8.0], "n_running": [20, 20], "pitstop__rejoin_if_box_now": [9.0, 12.0],
        "pitstop__pass_through": [False, True], "pit_loss_now": [21.0, 22.0],
    })


def test_rule_keeps_places_while_the_field_drives_through_the_pit_lane():
    assert list(PitstopRule().fit(_frame(), "y").predict(_frame())) == [9.0, 8.0]
    assert list(V01Prior().fit(_frame(), "y").predict(_frame())) == [21.0, 22.0]


def test_gbm_learns_a_constant_correction_deterministically():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"position": rng.integers(1, 20, n).astype(float), "n_running": 20})
    df["pitstop__rejoin_if_box_now"] = df["position"] + 3
    df["pitstop__pass_through"] = False
    from pitsense.bench.pitstop import PS
    for k in PS:
        if f"pitstop__{k}" not in df:
            df[f"pitstop__{k}"] = rng.normal(size=n)
    df["y"] = np.minimum(df["position"] + 4, 20)  # reality is one place worse than the engineer says
    a = PitstopGBM().fit(df, "y").predict(df)
    b = PitstopGBM().fit(df, "y").predict(df)
    assert (a == b).all()
    assert np.mean(a == df["y"]) > 0.8


def test_pit_loss_label_is_the_measured_stop(race_log):
    rows, final = decision_rows(race_log, PitLossPrior(), with_hash=False)
    rows = add_labels(rows, final)
    df = pd.DataFrame(rows)
    pits = loss_rows(df)
    assert len(pits) == 1
    assert pits.iloc[0]["y_pit_loss"] == pytest.approx(22.0, abs=0.5)
    assert pits.iloc[0]["y_pit_cond"] == "green"
    assert df.loc[df.kind == "lap_end", "y_pit_loss"].isna().all()
