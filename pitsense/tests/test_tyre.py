"""Tyre & pace engineer: model recovery on synthetic laps, and the engineer on the synthetic feed."""

import numpy as np
import pytest

from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, PitWall
from pitsense.pitwall.engineers.tyre import TyreEngineer
from pitsense.pitwall.engineers.tyre.model import CI, Priors, expected, fit_car, fit_field
from pitsense.state import RaceState

FUEL, DEG, OFF = 0.06, (0.09, 0.05, 0.03), (-0.5, 0.0, 0.4)


def synthetic_field(seed=0, n_cars=16, n_laps=50, noise=0.2):
    """Laps generated from the field model: base + off + deg*age - fuel*lap + noise.

    Each car stops once, between laps 12 and 35, switching compound; tyre age resets.
    Rows: (car, lap, age, compound index, time, stint).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cars):
        base = 90.0 + 0.1 * i
        stop = 12 + (i * 7) % 24
        c1, c2 = ("SOFT", "HARD") if i % 3 == 0 else ("MEDIUM", "HARD") if i % 3 == 1 else ("HARD", "MEDIUM")
        for lap in range(2, n_laps + 1):
            comp, age = (c1, lap) if lap <= stop else (c2, lap - stop)
            if lap in (stop, stop + 1):
                continue  # in-lap and out-lap
            j = CI[comp]
            t = base + OFF[j] + DEG[j] * age - FUEL * lap + rng.normal(0, noise)
            rows.append((str(i), lap, age, j, t, 1 if lap <= stop else 2))
    return rows


def _field(rows, pri=Priors()):
    a = np.array([r[1:5] for r in rows], dtype=float)
    return fit_field([r[0] for r in rows], a[:, 0], a[:, 1], a[:, 2].astype(int), a[:, 3], np.ones(len(rows)),
                     np.array([r[5] for r in rows]), pri)


def test_field_fit_separates_fuel_from_wear():
    ff = _field(synthetic_field())
    assert ff.fuel == pytest.approx(FUEL, abs=0.01)
    for j in range(3):
        assert ff.net[j] == pytest.approx(DEG[j] - FUEL, abs=0.01)  # within-stint slope
        assert ff.deg[j] == pytest.approx(DEG[j], abs=0.015)  # wear with fuel removed
    assert ff.off[0] == pytest.approx(OFF[0], abs=0.15) and ff.off[2] == pytest.approx(OFF[2], abs=0.15)


def test_field_fit_is_the_prior_without_data():
    pri = Priors()
    ff = fit_field(["1"], np.array([5.0]), np.array([4.0]), np.array([1]), np.array([90.0]), np.ones(1),
                   np.array([1]), pri)
    assert (ff.fuel, ff.net, ff.off, ff.deg) == (pri.fuel, pri.net, pri.off, pri.deg)


def test_car_fit_tracks_pace_and_wear():
    field = synthetic_field(noise=0.1)
    ff = _field(field)
    rows = [r for r in field if r[0] == "1"]  # MEDIUM then HARD (stop after lap 19)
    a = np.array([r[1:5] for r in rows], dtype=float)
    stint = np.array([r[5] for r in rows])
    pri = Priors()
    fit = fit_car(a[:, 0], a[:, 1], a[:, 2].astype(int), a[:, 3], np.ones(len(rows)), stint, 2, ff, pri)
    lap, age = 51, int(a[-1, 1]) + 1
    truth = 90.1 + OFF[2] + DEG[2] * age - FUEL * lap
    assert expected(fit, ff, pri, CI["HARD"], fit.net, age, lap) == pytest.approx(truth, abs=0.2)
    assert fit.net + ff.fuel == pytest.approx(DEG[2], abs=0.02)
    assert fit.n_stint == int((stint == 2).sum()) and fit.last_lap == 50
    # a new set of softs fitted now: first flying lap two laps after the out-lap
    fresh = expected(fit, ff, pri, CI["SOFT"], ff.net[CI["SOFT"]], 2, lap + 2)
    assert fresh == pytest.approx(90.1 + OFF[0] + DEG[0] * 2 - FUEL * (lap + 2), abs=0.25)


def _replay(log, until=None):
    wall = PitWall(Context(prior=PitLossPrior()), engineers=[TyreEngineer])
    state = RaceState(log.meta)
    for e in log.events:
        if until is not None and e.t > until:
            break
        state.apply(e)
        wall.observe(state)
    return state, wall


def test_engineer_on_the_synthetic_feed(race_log):
    state, wall = _replay(race_log)
    v = wall.view(state)
    race = v.race("tyre")
    assert race["fuel_s_per_lap"] == Priors().fuel  # 3 cars, 8 laps: too little to move the prior
    car11 = v.car("tyre", "11")
    # car 11: clean laps 2-4 and 8 at 90.0 + 0.05 * lap on new mediums
    assert car11["stint_clean_laps"] == 4 and car11["last_clean_s"] == pytest.approx(90.4)
    assert car11["pace_s"] == pytest.approx(90.0 + 0.05 * 9, abs=0.5)
    assert all(car11[k] is not None for k in ("pace_sd_s", "pace_trend_s", "deg_s_per_lap", "fresh_soft_s",
                                               "fresh_medium_s", "fresh_hard_s"))
    assert car11["fresh_soft_s"] < car11["fresh_medium_s"] < car11["fresh_hard_s"]  # prior offsets
