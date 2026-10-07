"""Analysis panels (tyre deg, track evolution, two drivers) and the driving coach, on synthetic laps."""

from types import SimpleNamespace as NS

import numpy as np

from pitsense import coach, insights


def lap(driver, n, t, *, stint=1, age=None, status="1", interval=2.0, pos=2, inlap=False, outlap=False, comp="MEDIUM"):
    return NS(driver=driver, lap=n, lap_time=t, t_end=100.0 * n + t, t_start=100.0 * n, position=pos, interval=interval,
              compound=comp, tyre_age=n if age is None else age, stint=stint, is_in_lap=inlap, is_out_lap=outlap,
              track_status=status, lap_time_inferred=False)


def state(laps):
    return NS(laps=laps, drivers={"1": NS(tla="AAA"), "2": NS(tla="BBB")})


def test_deg_fits_clean_laps_only_with_fuel_correction():
    laps = [lap("1", n, 90.0 + 0.1 * n - insights.FUEL_S_PER_LAP * n) for n in range(2, 12)]
    laps += [lap("1", 12, 80.0, status="1,4"), lap("1", 13, 80.0, interval=0.5), lap("1", 1, 80.0), lap("1", 14, 80.0, inlap=True)]
    d = insights.tyre_deg(state(laps))[0]
    assert d["n"] == 10 and abs(d["deg_s_per_lap"] - 0.1) < 1e-6 and d["band"][0] <= 0 <= d["band"][1]


def test_track_evolution_trend():
    laps = [lap(c, n, 95.0 - 0.02 * n - insights.FUEL_S_PER_LAP * n) for n in range(2, 12) for c in ("1", "2", "3")]
    assert abs(insights.track_evolution(state(laps))["evo_s_per_lap"] + 0.02) < 1e-6


def test_compare_two_drivers():
    laps = [lap("1", n, 90.0) for n in range(2, 8)] + [lap("2", n, 90.5) for n in range(2, 8)]
    c = insights.compare(state(laps), "1", "2")
    assert c["gap_s"][-1] == -0.5 and c["stints"][0]["delta_s"] == -0.5 and c["tla"] == {"1": "AAA", "2": "BBB"}


def synthetic_lap(driver, brake_early=0.0):
    """A 3 km lap with two corners (apex at 1000 m and 2200 m); ``brake_early`` m earlier braking into the first."""
    d = np.arange(0.0, 3000.0, 2.0)
    v = np.full_like(d, 300.0)
    for apex, vmin in ((1000.0, 100.0), (2200.0, 140.0)):
        v = np.minimum(v, vmin + np.abs(d - apex) * 0.6)
    if brake_early:
        v = np.where((d > 1000 - 400 - brake_early) & (d < 1000), np.minimum(v, 100 + np.abs(d - 1000) * 0.45), v)
    brake = np.zeros_like(d)
    for apex, look in ((1000.0, 330.0 + brake_early), (2200.0, 270.0)):
        brake[(d > apex - look) & (d < apex)] = 1
    thr = np.where(brake > 0, 0.0, 100.0)
    t = np.concatenate([[0.0], np.cumsum(np.diff(d) / (v[1:] / 3.6))])
    return coach.Lap(driver, 5, d, t, v, thr, brake)


def test_coach_finds_the_corner_where_time_goes():
    r = coach.compare(synthetic_lap("1"), synthetic_lap("2", brake_early=30.0))
    assert len(r["corners"]) == 2 and r["total_s"] > 0
    worst = max(r["corners"], key=lambda c: c["time_lost_s"])
    assert worst["turn"] == 1 and worst["brake_later_m"] == -30 and worst["time_lost_s"] > 0.05
    assert r["tips"][0].startswith("Turn 1: brake 30 m earlier")
