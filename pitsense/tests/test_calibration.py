"""Calibrated call confidence: scoring rules, isotonic fit, and the as-of rule (only earlier races)."""

from datetime import datetime, timedelta, timezone

from pitsense import calibration as cal


def test_calls_are_scored_by_the_benchmark_rules():
    assert cal.right("BOX", 10, [11]) and not cal.right("BOX", 10, [13])
    assert cal.right("PREPARE_BOX", 10, [13]) and not cal.right("PREPARE_BOX", 10, [9])
    assert cal.right("STAY_OUT", 10, [13]) and not cal.right("STAY_OUT", 10, [12])
    assert cal.right("NO_CALL", 10, []) is None


def test_fit_maps_an_overconfident_score_down_and_stays_monotone():
    rows = [{"action": "BOX", "raw": 0.9, "right": i % 3 == 0} for i in range(60)]
    rows += [{"action": "BOX", "raw": 0.5, "right": False} for _ in range(30)]
    fits = cal.fit(rows)
    hi, lo = cal.apply(fits, "BOX", 0.9), cal.apply(fits, "BOX", 0.5)
    assert 0.25 < hi < 0.45 and lo <= hi
    assert cal.apply(fits, "STAY_OUT", 0.8) == 0.8  # too few calls of that action: the raw score stands


def test_a_race_uses_only_fits_trained_before_it_started(tmp_path):
    t0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
    early, late = {"BOX": {"x": [0, 1], "y": [0.1, 0.2], "n": 50, "rate": 0.15}}, {"BOX": {"x": [0, 1], "y": [0.7, 0.8], "n": 90, "rate": 0.75}}
    cal.save(early, t0, ["a"], tmp_path)
    cal.save(late, t0 + timedelta(days=30), ["a", "b"], tmp_path)
    assert cal.for_race(t0 - timedelta(days=1), tmp_path) is None
    assert cal.for_race(t0 + timedelta(days=10), tmp_path) == early
    assert cal.for_race(t0 + timedelta(days=60), tmp_path) == late
