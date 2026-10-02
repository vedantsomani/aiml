"""Tyre benchmark labels: next-lap / 5-lap / fresh-tyre labels and the cliff definition."""

import math

import pytest

from pitsense.bench.dataset import decision_rows
from pitsense.bench.labels import add_labels
from pitsense.bench.tyre import tyre_labels, tyre_tasks
from pitsense.pitloss import PitLossPrior
from pitsense.state import LapRecord, PitEvent, RaceState


def test_labels_on_the_synthetic_feed(race_log):
    rows, final = decision_rows(race_log, PitLossPrior())
    add_labels(rows, final)
    by = {(r["kind"], r["driver"], r["lap"]): r for r in rows}
    # car 11: lap 3 is clean (90.0 + 0.05 * 3); laps 5-6 are under the safety car
    assert by[("lap_end", "11", 2)]["y_next_lap"] == pytest.approx(90.15)
    assert math.isnan(by[("lap_end", "11", 4)]["y_next_lap"])
    assert all(math.isnan(r["y_lap_5"]) for r in rows if r["kind"] == "lap_end")  # SC within every 5 laps
    pit = next(r for r in rows if r["kind"] == "pit_entry")
    assert pit["driver"] == "22" and pit["y_fresh_compound"] == "HARD"
    assert math.isnan(pit["y_fresh_lap"])  # its first flying lap is under the safety car
    assert {"next_lap_time", "lap_time_5", "tyre_cliff_3", "fresh_tyre_pace"} == {t.name for t in tyre_tasks()}


def _race(times: dict[str, list[float]], intervals: dict[str, float] | None = None, stops=()):
    """A finished race from per-car lap times (lap 1 first); all green, mediums."""
    st = RaceState()
    for pos, (drv, ts) in enumerate(times.items(), 1):
        for lap, t in enumerate(ts, 1):
            stopped_after = [s for s in stops if s[0] == drv and s[1] < lap]
            age = lap - stopped_after[-1][1] if stopped_after else lap
            st.laps.append(LapRecord(
                driver=drv, lap=lap, t_start=None, t_end=float(lap * 100 + pos), lap_time=t, lap_time_t=None,
                position=pos, gap_to_leader=None, laps_down=0, interval=(intervals or {}).get(drv, 3.0),
                compound="MEDIUM", tyre_age=age, stint=1 + len(stopped_after),
                is_in_lap=any(s == (drv, lap) for s in stops),
                is_out_lap=any(s[0] == drv and s[1] == lap - 1 for s in stops), track_status="1"))
    for drv, in_lap in stops:
        st.pit_events.append(PitEvent(drv, in_lap, 0.0, in_lap + 1, 0.0))
    return st


def _rows(drv, laps):
    return [{"kind": "lap_end", "driver": drv, "lap": L} for L in laps]


def test_cliff_label():
    base = [95.0] + [90.0] * 19
    cliff = [95.0] + [90.0] * 11 + [91.5] * 8  # falls off from lap 13
    mistake = [95.0] + [90.0] * 11 + [91.5] + [90.0] * 7  # one slow lap (13)
    held = list(cliff)
    times = {"c": cliff, "m": mistake, "h": held, **{f"f{i}": list(base) for i in range(6)}}
    final = _race(times, intervals={"h": 0.6})
    rows = _rows("c", range(6, 18)) + _rows("m", range(6, 18)) + _rows("h", range(6, 18))
    tyre_labels(rows, final)
    y = {(r["driver"], r["lap"]): r["y_cliff_3"] for r in rows}
    # 1 when lap 13 is within the next 3 laps, and at L = 13 (its reference is still 90 s)
    assert [y[("c", L)] for L in range(7, 15)] == [0, 0, 0, 1, 1, 1, 1, 0]
    assert all(y[("m", L)] == 0 for L in range(7, 15))  # a single slow lap isn't a cliff
    assert all(y[("h", L)] == 0 for L in range(7, 15))  # held up in traffic: not the tyres
    assert math.isnan(y[("c", 17)])  # can't see three more laps at the end of the race


def test_cliff_label_is_censored_by_a_stop():
    times = {f"f{i}": [95.0] + [90.0] * 19 for i in range(7)}
    final = _race(times, stops=[("f0", 10)])
    rows = _rows("f0", [7, 8, 9])
    tyre_labels(rows, final)
    assert all(math.isnan(r["y_cliff_3"]) for r in rows)  # stops before laps L+1..L+4 are seen
    assert rows[0]["y_lap_5"] != rows[0]["y_lap_5"] and rows[0]["y_next_lap"] == 90.0
