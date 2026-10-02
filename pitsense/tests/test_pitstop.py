"""Pit-stop engineer: loss measurement, priors, rejoin estimate (synthetic data only)."""

from __future__ import annotations

import pytest

from pitsense.pitloss import (
    LapIndex,
    lane_condition,
    lap_number_loss,
    measure_stop,
    measure_stops_timed,
)
from pitsense.state import LapRecord, PitEvent, replay


# --------------------------------------------------------------------------- a field of cars under a VSC
def vsc_field(n_cars=10, gap=5.0, lap=90.0, slow=1.5, t_vsc=400.0, laps=8, stopper=2, stop_lap=5, delay=12.0):
    """Cars ``gap`` s apart at the same pace; at ``t_vsc`` everyone slows by ``slow`` at once.

    The stopper loses ``delay`` s around the line ending its in-lap (half on each lap).
    Returns (lap records in completion order, the stop, the status log).
    """

    def cross(i: int, k: int) -> float:
        t0 = i * gap
        green_laps = (t_vsc - t0) / lap
        return t0 + k * lap if k <= green_laps else t_vsc + (k - green_laps) * lap * slow

    recs = []
    for i in range(n_cars):
        for k in range(1, laps + 1):
            ts, te = cross(i, k - 1), cross(i, k)
            if i == stopper and k == stop_lap:
                te += delay / 2
            elif i == stopper and k > stop_lap:
                ts += delay / 2 if k == stop_lap + 1 else delay
                te += delay
            status = "1" if te <= t_vsc else "6" if ts >= t_vsc else "1,6"
            recs.append(LapRecord(str(i), k, ts, te, round(te - ts, 3), te, i + 1, None, 0, None, "MEDIUM", k, 1,
                                  i == stopper and k == stop_lap, i == stopper and k == stop_lap + 1, status))
    recs.sort(key=lambda r: r.t_end)
    t_line = cross(stopper, stop_lap)
    stop = PitEvent(str(stopper), stop_lap, t_line - 5.0, stop_lap + 1, t_line + 15.0, status_at_entry="6")
    return recs, stop, [(t_vsc, "6")]


def test_lane_condition():
    log = [(100.0, "4"), (200.0, "1"), (300.0, "6"), (350.0, "7"), (380.0, "1"), (500.0, "2"), (520.0, "1"), (600.0, "5")]
    assert lane_condition(log, 50, 80) == "green"
    assert lane_condition(log, 120, 150) == "sc"
    assert lane_condition(log, 190, 210) == "mixed"  # SC ended in the pit lane
    assert lane_condition(log, 310, 360) == "vsc"  # VSC then VSC ending
    assert lane_condition(log, 490, 510) == "green"  # local yellow
    assert lane_condition(log, 590, 610) is None  # red flag: not a stop in the usual sense


def test_time_alignment_removes_the_vsc_bias():
    """A VSC starts during the in-lap. Cars behind on the road get more of their window
    slowed, cars ahead less; same-lap-number medians inherit the field's skew, the
    time-aligned measure averages it out."""
    recs, stop, log = vsc_field()
    idx = LapIndex(recs)
    m = measure_stop(idx, stop, log)
    assert m is not None and m.condition == "vsc" and m.method == "time"
    assert m.loss == pytest.approx(12.0, abs=1.0)
    assert m.n_refs >= 4
    biased = lap_number_loss(idx, stop)
    assert abs(biased - 12.0) > 5.0  # the v0.1 comparison is off by several seconds


def test_green_stop_uses_the_lap_number_method():
    recs, stop, _ = vsc_field(t_vsc=10_000.0)  # no VSC at all
    m = measure_stop(LapIndex(recs), stop, [])
    assert m.condition == "green" and m.method == "laps"
    assert m.loss == pytest.approx(12.0, abs=0.01)


def test_measurement_only_uses_laps_published_by_its_cutoff():
    recs, stop, log = vsc_field()
    idx = LapIndex(recs)
    m = measure_stop(idx, stop, log)
    late = [r for r in recs if r.t_end > m.available_at]
    early = LapIndex([r for r in recs if r.t_end <= m.available_at])
    assert late  # the field kept running after the cutoff
    assert measure_stop(early, stop, log) == m


def test_synthetic_race_stop(race_log):
    """Three cars: too few for a field median, so v0.1 can't measure it; by time we can."""
    s = replay(race_log)
    (m,) = measure_stops_timed(LapIndex(s.laps), s.pit_events, s.status_log)
    assert (m.driver, m.in_lap, m.condition, m.method) == ("22", 3, "green", "time")
    assert m.loss == pytest.approx(22.0, abs=0.5)  # +8 s in-lap, +14 s out-lap
    assert m.typical
