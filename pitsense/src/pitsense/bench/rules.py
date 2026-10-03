"""Measurements for the rules engineer (``python -m pitsense rules-report``).

1. parser coverage over every race-control message of the chosen seasons, by type;
2. ``rules__must_stop`` against what finishers actually did;
3. lead time of "safety car in this lap" / "VSC ending" over the track-status change.

This module reads finished races on purpose (it scores the engineer); nothing
in ``pitwall/`` imports it. The core-leaderboard comparison is the ordinary
``bench build`` / ``bench run``.
"""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

from ..events import load_archive_session
from ..pitwall.engineers.rules.parser import parse
from ..state import RaceState


def _races(years):
    from ..archive import downloaded_sessions

    return [r for r in downloaded_sessions() if r.session_name == "Race" and r.year in years]


def _final_state(ref) -> RaceState:
    log = load_archive_session(ref.local_dir)
    st = RaceState(log.meta)
    for e in log.events:
        st.apply(e)
    return st


def measure(years=(2025, 2026)) -> dict:
    cov: Counter = Counter()
    unknown: Counter = Counter()
    sc_leads, vsc_leads, sc_deploy_lag = [], [], []
    rows = []
    for ref in _races(years):
        st = _final_state(ref)
        for m in st.rc:
            ev = parse(m.message, m.category, m.flag)
            cov[ev.kind] += 1
            if ev.kind == "unknown":
                unknown[m.message[:80]] += 1
        # track-status changes (t, code), as a lookup of "when did the status leave X after t"
        log = st.status_log

        def leaves(t0, codes):
            for t, code in log:
                if t > t0 and code not in codes:
                    return t
            return None

        for m in st.rc:
            ev = parse(m.message, m.category, m.flag)
            if ev.kind == "sc_in_this_lap":
                t1 = leaves(m.t, ("4",))
                if t1 is not None:
                    sc_leads.append((ref.slug, m.t, t1 - m.t, m.lap))
            elif ev.kind == "vsc_ending":
                t1 = leaves(m.t, ("6", "7"))
                if t1 is not None:
                    vsc_leads.append((ref.slug, m.t, t1 - m.t, m.lap))
            elif ev.kind == "sc_deployed":
                near = [t for t, c in log if c == "4"]
                if near:  # the status change closest in time to the message
                    sc_deploy_lag.append(m.t - min(near, key=lambda t: abs(t - m.t)))
    return {"coverage": cov, "unknown": unknown, "sc_leads": sc_leads, "vsc_leads": vsc_leads,
            "sc_deploy_lag": sc_deploy_lag}


def must_stop_report(df: pd.DataFrame) -> pd.DataFrame:
    """must_stop against the finishers' behaviour, from the benchmark's lap-end rows."""
    d = df[(df["kind"] == "lap_end") & (df["y_retire_3"] == 0)].copy()
    d["stops_later"] = d["y_next_in_lap"].notna()
    d["ms"] = d["rules__must_stop"].astype(float)
    last = d.sort_values("lap").groupby(["race_id", "driver"]).tail(1) 
    out = []
    for name, sub in (("must_stop=1", d[d.ms == 1]), ("must_stop=0, dry", d[(d.ms == 0) & (d.rules__race_dry == 1)])):
        out.append({"rows": name, "n": len(sub), "share_that_stop_later": round(sub.stops_later.mean(), 4) if len(sub) else None})
    if len(last):
        last = last[last.lap >= last.total_laps - 3]  # ran to the end (not an early retirement)
        dry = last[last.rules__race_dry == 1]
        out.append({"rows": "last rows of finishers (final 3 laps), dry race: must_stop still 1", "n": len(dry),
                    "share_that_stop_later": round(float((dry.ms == 1).mean()), 4) if len(dry) else None})
    for race_key in ("monaco", "qatar"):
        sub = d[(d.year == 2025) & d.race_id.astype(str).str.contains(race_key)]
        if len(sub):
            out.append({"rows": f"{race_key} 2025, must_stop=1 (regulated)", "n": int((sub.ms == 1).sum()),
                        "share_that_stop_later": round(sub[sub.ms == 1].stops_later.mean(), 4)})
            out.append({"rows": f"{race_key} 2025, base must_stop=1 (two-compound rule only)", "n": int((sub.must_stop == 1).sum()),
                        "share_that_stop_later": round(sub[sub.must_stop == 1].stops_later.mean(), 4)})
    return pd.DataFrame(out)


def _stats(x):
    return {"n": len(x), "median_s": round(statistics.median(x), 1), "min_s": round(min(x), 1),
            "p10_s": round(sorted(x)[len(x) // 10], 1), "max_s": round(max(x), 1)} if x else {"n": 0}


def report(years=(2025, 2026)) -> str:
    m = measure(years)
    lines = ["# Rules engineer measurements", "", "## 1. Parser coverage", "", "| type | messages |", "|---|---|"]
    total = sum(m["coverage"].values())
    for k, v in m["coverage"].most_common():
        lines.append(f"| {k} | {v} |")
    lines += ["", f"Total {total}; unknown {m['coverage'].get('unknown', 0)} "
              f"({100 * m['coverage'].get('unknown', 0) / max(total, 1):.2f} %).", ""]
    for msg, n in m["unknown"].most_common(10):
        lines.append(f"- unknown x{n}: {msg}")
    lines += ["", "## 3. Lead time over the track-status change (message to status leaving SC / VSC)", "",
              "| message | n | median s | p10 s | min s | max s |", "|---|---|---|---|---|---|"]
    for name, rows in (("SAFETY CAR IN THIS LAP", m["sc_leads"]), ("VSC ENDING", m["vsc_leads"])):
        s = _stats([r[2] for r in rows])
        lines.append(f"| {name} | {s['n']} | {s.get('median_s')} | {s.get('p10_s')} | {s.get('min_s')} | {s.get('max_s')} |")
    s = _stats(m["sc_deploy_lag"])
    lines += ["", f"SAFETY CAR DEPLOYED message vs status 4: n={s['n']}, median lag {s.get('median_s')} s "
              f"(positive = message after the status), min {s.get('min_s')}, max {s.get('max_s')}.", ""]
    return "\n".join(lines)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return chr(10).join(rows)


def cmd_rules_report(a) -> None:
    from ..config import bench_dir

    years = tuple(a.year)
    text = report(years)
    try:
        from .dataset import load_bench

        text += "\n## 2. must_stop against finishers\n\n" + _md(must_stop_report(load_bench())) + "\n"
    except Exception as exc:  # benchmark not built yet
        text += f"\n(2. must_stop: benchmark rows unavailable: {exc})\n"
    print(text)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rules.md").write_text(text, encoding="utf-8")


def add_commands(sub) -> None:
    s = sub.add_parser("rules-report", help="rules engineer: parser coverage, must_stop, SC lead time")
    s.add_argument("--year", type=int, nargs="+", default=[2025, 2026])
    s.add_argument("--out", help="folder for rules.md")
    s.set_defaults(fn=cmd_rules_report)
