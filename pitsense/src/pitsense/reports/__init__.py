"""Pre-race briefing and post-race report: ``pitsense briefing`` and ``pitsense report`` (docs/reports.md)."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _json(obj) -> str:
    from ..pitwall.types import jsonable

    return json.dumps(jsonable(obj), indent=1, sort_keys=True, default=str)


def _slug(*parts) -> str:
    return re.sub(r"[^a-z0-9]+", "-", "-".join(str(p).lower() for p in parts)).strip("-")


def _load(a, *, feeds: bool):
    from .. import archive
    from ..events import load_archive_session

    ref = archive.find_session(a.year, a.race, "Sprint" if a.sprint else "Race")
    if not (ref.local_dir / "TimingData.jsonStream").exists():
        archive.download_session(ref)
    files = {f.stem for f in ref.local_dir.glob("*.jsonStream")}
    skip = {"Position.z"} | (set() if getattr(a, "telemetry", False) else {"CarData.z"})
    topics = tuple(sorted(files - skip)) if feeds else None
    return ref, load_archive_session(ref.local_dir, topics)


def cmd_briefing(a) -> None:
    from . import briefing, prerace

    ref, log = _load(a, feeds=False)
    data = prerace.prerace(log, a.team, sims=a.sims)
    out = Path(a.out) if a.out else Path("reports") / f"{_slug(a.year, a.race, a.team)}-briefing.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(briefing.render(data), encoding="utf-8")
    jp = out.with_suffix(".json")
    jp.write_text(_json(data), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1024:.0f} KB), {jp} ({jp.stat().st_size / 1024:.0f} KB)")


def cmd_report(a) -> None:
    from ..pitwall.runtime import load_call_log
    from . import postrace

    ref, log = _load(a, feeds=True)
    calls = load_call_log(Path(a.log)) if a.log else None
    rep = postrace.build_report(log, a.team, calls=calls, k=a.k, ref=ref, sims=a.sims)
    out = Path(a.out) if a.out else Path("reports") / f"{_slug(a.year, a.race, a.team)}-report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(postrace.render(rep), encoding="utf-8")
    jp = out.with_suffix(".json")
    jp.write_text(_json(rep), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1024:.0f} KB), {jp} ({jp.stat().st_size / 1024:.0f} KB)")


def add_commands(sub) -> None:
    def common(s):
        s.add_argument("--year", type=int, default=2026)
        s.add_argument("--race", required=True, help="e.g. 'azerbaijan'")
        s.add_argument("--team", required=True, help="team name as in the feed ('ferrari') or car numbers '16,44'")
        s.add_argument("--sprint", action="store_true")
        s.add_argument("--sims", type=int, default=288, help="simulated races for the plans")
        s.add_argument("--out", help="HTML file; the JSON summary is written next to it")

    s = sub.add_parser("briefing", help="pre-race briefing page (as of lights out): grid, tyres, circuit, weather, Plan A/B/C, rivals")
    common(s)
    s.set_defaults(fn=cmd_briefing)

    s = sub.add_parser("report", help="post-race report: grade the pit wall's calls against the race")
    common(s)
    s.add_argument("--log", help="a pit-wall calls log (calls-*.jsonl); without it the race is replayed through the runtime")
    s.add_argument("--k", type=int, default=2, help="a box call is right if the stop is within +-k laps")
    s.add_argument("--telemetry", action="store_true", help="also load CarData (slower; lets the telemetry mechanic alert)")
    s.set_defaults(fn=cmd_report)
