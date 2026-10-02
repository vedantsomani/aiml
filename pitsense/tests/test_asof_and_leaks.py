from datetime import datetime, timedelta, timezone

import pytest

from pitsense.asof import AsOfTable, HistoryStore, LeakageError, RaceSummary, assert_trained_before
from pitsense.bench import leakcheck
from pitsense.bench.dataset import decision_rows
from pitsense.bench.features import FeatureBuilder
from pitsense.bench.labels import add_labels
from pitsense.pitloss import PitLossPrior


def test_asof_table_hides_future_rows():
    t = AsOfTable()
    for i in range(5):
        t.add(float(i), i)
    assert t.asof(2.5) == [0, 1, 2]
    t.horizon = 3.0
    with pytest.raises(LeakageError):
        t.asof(3.5)


def _summary(name: str, end: datetime) -> RaceSummary:
    return RaceSummary(name, end.year, 1, end - timedelta(hours=2), end, [20.0], [], [], 50, 0, 0)


def test_history_cutoff_and_training_guard():
    d0 = datetime(2026, 7, 1, tzinfo=timezone.utc)
    h = HistoryStore([_summary("a", d0), _summary("b", d0 + timedelta(days=7))])
    assert [s.race_id for s in h.asof(d0 + timedelta(days=3))] == ["a"]
    assert_trained_before([d0], d0 + timedelta(days=1))
    with pytest.raises(LeakageError):
        assert_trained_before([d0 + timedelta(days=2)], d0 + timedelta(days=1))


def test_decision_rows_and_labels(race_log):
    rows, final = decision_rows(race_log, PitLossPrior())
    rows = add_labels(rows, final)
    lap_end = {(r["driver"], r["lap"]): r for r in rows if r["kind"] == "lap_end"}
    # decision at the end of lap 2: car 22 stops at the end of lap 3
    assert lap_end[("22", 2)]["y_pit_1"] == 1
    assert lap_end[("22", 1)]["y_pit_1"] == 0 and lap_end[("22", 1)]["y_pit_2"] == 1
    assert lap_end[("11", 2)]["y_pit_5"] == 0
    entry = [r for r in rows if r["kind"] == "pit_entry"]
    assert len(entry) == 1 and entry[0]["driver"] == "22"
    assert entry[0]["y_pos_after_stop"] == 3.0


def test_retire_label_ignores_lapped_finishers(race_log):
    rows, final = decision_rows(race_log, PitLossPrior())
    # car 33 takes the flag a lap down: its last lap record is lap 7, but it is still running
    final.laps = [x for x in final.laps if not (x.driver == "33" and x.lap == 8)]
    lap6 = next(r for r in add_labels(rows, final) if r["kind"] == "lap_end" and (r["driver"], r["lap"]) == ("33", 6))
    assert lap6["y_retire_3"] == 0
    # the same laps from a car that was out at the flag are a retirement within 3 laps
    final.drivers["33"].retired = True
    add_labels(rows, final)
    assert lap6["y_retire_3"] == 1


def test_leakcheck_passes_on_clean_features(race_log):
    for mode in ("truncate", "scramble"):
        for cut in leakcheck.random_cuts(race_log, 3, seed=1):
            rep = leakcheck.check(race_log, cut, PitLossPrior(), mode=mode)
            assert rep.rows_checked > 0
            assert rep.ok, rep.mismatches[:5]


class LeakyBuilder(FeatureBuilder):
    """Planted leak of a classic kind: the row keeps a live reference to a list that
    keeps growing after the decision time, so by the time anyone reads the row it
    contains future laps."""

    def row(self, state, rec, **kw):
        out = super().row(state, rec, **kw)
        if out is not None:
            out["laps_so_far"] = state.laps  # bug: alias instead of a copy
        return out


def test_leakcheck_catches_a_planted_leak(race_log):
    cut = leakcheck.random_cuts(race_log, 1, seed=2)[0]
    rep = leakcheck.check(race_log, cut, PitLossPrior(), mode="truncate", builder=LeakyBuilder)
    assert not rep.ok
    assert {m[2] for m in rep.mismatches} == {"laps_so_far"}
