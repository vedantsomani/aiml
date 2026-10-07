"""Championship standings and the projection "if the race finished in the current order".

Points are data-driven by era (``POINTS``): race points for the top ten, the fastest-lap point (2019-2024, top-ten
finishers only) and sprint points. A season's results come from each finished race's own timing feed, replayed to
its end (``pitsense standings --year Y`` caches them in ``data/bench/results_<year>.json``). As every cross-race
input here, a race only sees results of sessions that ended before it started (``before``).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

RACE_TOP10 = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
# (first season, race points, fastest-lap point for a top-ten finisher, sprint points); the last row that applies wins
POINTS = (
    (2010, RACE_TOP10, False, ()),
    (2019, RACE_TOP10, True, ()),
    (2021, RACE_TOP10, True, (3, 2, 1)),
    (2022, RACE_TOP10, True, (8, 7, 6, 5, 4, 3, 2, 1)),
    (2025, RACE_TOP10, False, (8, 7, 6, 5, 4, 3, 2, 1)),
)
CLASSIFIED_SHARE = 0.9  # a car is classified when it completed 90 % of the winner's laps


def rules(year: int) -> tuple[tuple[int, ...], bool, tuple[int, ...]]:
    row = POINTS[0]
    for r in POINTS:
        if year >= r[0]:
            row = r
    return row[1], row[2], row[3]


def points_for(pos: int | None, year: int, sprint: bool = False, fastest: bool = False) -> int:
    race, fl, spr = rules(year)
    table = spr if sprint else race
    p = table[pos - 1] if pos and 1 <= pos <= len(table) else 0
    return p + (1 if fl and fastest and not sprint and pos and pos <= 10 else 0)


def classification(state) -> list[dict]:
    """Final order of a finished session: classified cars by position, then the rest (no points)."""
    cars = [d for d in state.drivers.values() if d.position]
    top = max((d.laps for d in cars), default=0)
    fastest = min(((d.best_lap_time, d.number) for d in cars if d.best_lap_time), default=(None, None))[1]
    out = []
    for d in sorted(cars, key=lambda d: d.position):
        out.append({"car": d.number, "tla": d.tla, "team": d.team, "pos": d.position, "laps": d.laps,
                    "classified": d.laps >= CLASSIFIED_SHARE * top, "fastest": d.number == fastest})
    return out


def season_points(results: list[dict], year: int) -> dict[str, dict]:
    """{tla: {"points", "team"}} over sessions (``build`` output rows)."""
    tot: dict[str, dict] = {}
    for r in results:
        sprint = r["session"] != "Race"
        for c in r["results"]:
            row = tot.setdefault(c["tla"], {"points": 0, "team": c["team"]})
            row["team"] = c["team"] or row["team"]
            if c["classified"]:
                row["points"] += points_for(c["pos"], year, sprint, c["fastest"])
    return tot


def before(results: list[dict], start_utc: datetime) -> list[dict]:
    return [r for r in results if datetime.fromisoformat(r["end_utc"]) < start_utc]


def project(standings: dict[str, dict], running_order: list[tuple[str, str]], year: int) -> list[dict]:
    """Standings now and if the race finished in ``running_order`` [(tla, team), ...], with the change in place."""
    now = sorted(standings, key=lambda t: -standings[t]["points"])
    rank_now = {t: i + 1 for i, t in enumerate(now)}
    gain = {tla: points_for(i + 1, year) for i, (tla, _) in enumerate(running_order)}
    proj = {t: standings.get(t, {}).get("points", 0) + gain.get(t, 0) for t in set(standings) | set(gain)}
    team = {t: standings.get(t, {}).get("team") for t in proj} | {t: tm for t, tm in running_order if tm}
    order = sorted(proj, key=lambda t: (-proj[t], rank_now.get(t, 99)))
    return [{"tla": t, "team": team.get(t), "points": standings.get(t, {}).get("points", 0), "race": gain.get(t, 0),
             "projected": proj[t], "rank_now": rank_now.get(t), "rank": i + 1,
             "delta": (rank_now[t] - (i + 1)) if t in rank_now else None} for i, t in enumerate(order)]


# ----------------------------------------------------------------------------- the cache
def path(year: int) -> Path:
    from .config import bench_dir

    return bench_dir() / f"results_{year}.json"


def build(year: int) -> list[dict]:
    """Replay every downloaded race and sprint of ``year`` to its end and keep the classification."""
    from . import archive
    from .events import load_archive_session
    from .state import replay

    out = []
    for ref in archive.races(year, include_sprints=True):
        if not (ref.local_dir / "TimingData.jsonStream").exists():
            continue
        final = replay(load_archive_session(ref.local_dir))
        st, fin = final.started_t, final.finished_t  # the end: when the chequered flag was shown (fallback +3 h)
        end = ref.start_utc + (timedelta(seconds=fin - st) if fin is not None and st is not None else timedelta(hours=3))
        out.append({"slug": ref.slug, "session": ref.session_name, "start_utc": ref.start_utc.isoformat(),
                    "end_utc": end.isoformat(), "results": classification(final)})
    out.sort(key=lambda r: r["start_utc"])
    p = path(year)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def load(year: int) -> list[dict] | None:
    p = path(year)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cmd_standings(a) -> None:
    rows = build(a.year)
    tot = season_points(rows, a.year)
    print(f"{len(rows)} sessions -> {path(a.year)}")
    for i, (t, v) in enumerate(sorted(tot.items(), key=lambda kv: -kv[1]["points"])[:20], 1):
        print(f"  {i:2d} {t:<4} {v['points']:4d}  {v['team']}")


def add_commands(sub) -> None:
    s = sub.add_parser("standings", help="cache a season's race and sprint results for the championship panel")
    s.add_argument("--year", type=int, default=2026)
    s.set_defaults(fn=cmd_standings)
