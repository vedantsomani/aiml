"""Validation helpers on synthetic snippets (no FastF1 download)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pitsense.archive import SessionRef
from pitsense.validate import fastf1_compound, ours_laps

from .conftest import synthetic_events


def _write_session(root: Path, events: list) -> None:
    """Events -> one archive-style ``Topic.jsonStream`` file per topic."""
    root.mkdir(parents=True)
    by_topic = defaultdict(list)
    for t, topic, data in events:
        by_topic[topic].append((t, data))
    for topic, rows in by_topic.items():
        lines = []
        for t, data in sorted(rows, key=lambda r: r[0]):
            h, rest = divmod(t, 3600)
            m, s = divmod(rest, 60)
            lines.append(f"{int(h):02d}:{int(m):02d}:{s:06.3f}{json.dumps(data)}")
        (root / f"{topic}.jsonStream").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tyre_age_is_also_checked_against_the_feeds_own_counter(tmp_path, monkeypatch):
    monkeypatch.setenv("PITSENSE_DATA", str(tmp_path))
    events = synthetic_events()
    crossings = [(t, data["Lines"]["11"]["NumberOfLaps"]) for t, topic, data in events
                 if topic == "TimingData" and "NumberOfLaps" in (data["Lines"].get("11") or {})]
    for t, lap in crossings:  # the counter follows each line crossing; lap 5's is planted wrong
        events.append((t + 1.0, "TimingAppData", {"Lines": {"11": {"Stints": {"0": {"TotalLaps": lap + (lap == 5)}}}}}))
    ref = SessionRef(1999, 1, "Test Grand Prix", "Nowhere", "Erewhon", 1, "Test", 1, 1, "Race", "Race",
                     "1999-07-01T15:00:00", "00:00:00", "1999/test/race/")
    _write_session(ref.local_dir, events)

    car = ours_laps(ref).query("driver == '11'").set_index("lap")
    assert car.feed_age.tolist() == [1, 2, 3, 4, 6, 6, 7, 8]
    assert car.index[car.tyre_age != car.feed_age].tolist() == [5]


def test_pooled_validation_skips_races_without_a_check(tmp_path, monkeypatch, capsys):
    """Spa 2021 has no comparable lap times; that must not turn the pooled number into NaN."""
    from types import SimpleNamespace

    from pitsense import cli
    from pitsense.validate import ValidationReport

    monkeypatch.setenv("PITSENSE_DATA", str(tmp_path))

    def report(slug, time, n):
        return ValidationReport(slug, n, n, n, time, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, n)

    canned = {"a": report("a", 1.0, 100), "b": report("b", float("nan"), 60), "c": "FastF1 failed"}
    monkeypatch.setattr(cli, "_races", lambda years, query: [SimpleNamespace(slug=s) for s in canned])
    monkeypatch.setattr(cli, "_validate_one", lambda args: (args[0].slug, canned[args[0].slug]))
    cli.cmd_validate(SimpleNamespace(year=[1999], race=None, jobs=1))
    out = capsys.readouterr().out
    assert "Pooled over 2 sessions, 160 laps: lap time 100.000%" in out
    assert "c" in out.splitlines()[-1] and "not compared" in out


def test_fastf1_missing_compounds_are_missing():
    melbourne = {"year": 2018, "meeting_name": "Australian Grand Prix"}
    missing = ("nan", "None", "", "UNKNOWN", None, float("nan"))
    assert [fastf1_compound(c, melbourne) for c in missing] == [None] * len(missing)
    assert fastf1_compound("SOFT", melbourne) == "HARD"  # Melbourne 2018's hardest nomination
    assert fastf1_compound("SOFT", {"year": 2025}) == "SOFT"
