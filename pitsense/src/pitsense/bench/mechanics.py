"""Measurements for the mechanic engineers (``python -m pitsense mechanics-eval``).

Events (per car and race, read from the finished race: this module scores the engineers, nothing in
``pitwall/`` imports it):

* ``retire``    the car ends not running with 2..total-3 laps done. Reference time ``t0`` is the
                last time the car moved above 25 km/h (it stopped), ``t_end`` the time the timing
                feed said it was out.
* ``slowdown``  a clean lap (green throughout, no pit, not lap 1, not within 2 s of the car ahead)
                at least 5 s slower than the median of the car's five previous clean laps.
                ``t0`` is the start of that lap, ``t_end`` its end (when the time is published).
* ``rc_stop``   a race-control line saying a car stopped or is slow (``t0`` = ``t_end`` = message).

Detectors: each telemetry check on its own, telemetry (any check), radio, the chief mechanic, and
the baseline: "a clean lap at least X s slower than the car's previous five clean laps".

For every event a detector counts as a hit if it raised an alert in [t0 - 3 laps, t_end] (an alert
already active at the window start also counts). ``recall_pre`` needs the alert strictly before
``t0``; ``recall_flag`` accepts any time up to ``t_end``. Lead time is ``t_end`` minus the time the
alert was raised, i.e. how much earlier than the timing feed the pit wall knew. A false alarm is an
alert episode (one car, one run of ``active``) that overlaps no event of that car within [t0 - 3
laps, t_end + 1 lap], or that starts after the car was already out. Rate = per car-race.

The slowdown event and the baseline use the same lap-time quantity, so the baseline's
``recall_flag`` on slowdowns is 1.0 by construction; read ``recall_pre`` there.

Protocol: thresholds chosen on 2025 for a fixed false-alarm budget; 2026 is scored once with them.
"""

from __future__ import annotations

import math
import pickle
import statistics
import time
from pathlib import Path

import numpy as np

SAMPLE_S = 5.0
PRE_LAPS = 3
SLOWDOWN_S = 5.0
FA_BUDGET = 0.25  # false alarms per car-race allowed when picking a threshold


# --------------------------------------------------------------------------- recording
def record_race(session_dir: str, meta_name: str = "") -> dict:
    """Replay one race with the three mechanic engineers and keep their values every SAMPLE_S."""
    from ..events import load_archive_session
    from ..pitloss import PitLossPrior
    from ..pitwall import Context, PitWall
    from ..pitwall.engineers.mechanic_chief import MechanicChief
    from ..pitwall.engineers.mechanic_radio import MechanicRadio
    from ..pitwall.engineers.mechanic_telemetry import CHECKS, MechanicTelemetry
    from ..state import RaceState

    log = load_archive_session(Path(session_dir), feeds=True)
    st = RaceState(log.meta)
    wall = PitWall(Context(prior=PitLossPrior(), meta=log.meta), engineers=[MechanicTelemetry, MechanicRadio, MechanicChief])
    nxt = 0.0
    cols: dict[str, dict[str, list]] = {}
    last_move: dict[str, float] = {}
    notrun: dict[str, tuple[float, int]] = {}
    alerts_log: list[tuple[float, str, str, str]] = []
    seen_alert: set[tuple[str, str, float]] = set()
    t_wall = time.time()
    for e in log.events:
        st.apply(e)
        wall.observe(st)
        for n, d in st.drivers.items():
            if not d.running and n not in notrun:
                notrun[n] = (st.t, d.laps)
        if st.t < nxt:
            continue
        nxt = st.t + SAMPLE_S
        view = wall.view(st)
        for n, x in st.feeds.telemetry.latest_telemetry().items():
            if x["speed"] > 25:
                last_move[n] = st.t
        for n in st.drivers:
            tel = view.car("mechanic_telemetry", n)
            rad = view.car("mechanic_radio", n)
            chf = view.car("mechanic_chief", n)
            c = cols.setdefault(n, {k: [] for k in ("t", *CHECKS, "tel_risk", "tel_active", "radio_risk", "radio_since", "radio_issue",
                                                     "chief_risk", "chief_since", "chief_out", "chief_issue", "chief_why", "laps")})
            c["t"].append(st.t)
            for k in CHECKS:
                c[k].append(np.nan if tel[k] is None else tel[k])
            c["tel_risk"].append(np.nan if tel["mech_risk"] is None else tel["mech_risk"])
            c["tel_active"].append(bool(tel["mech_active"]))
            c["radio_risk"].append(rad["radio_risk"])
            c["radio_since"].append(np.nan if rad["radio_since"] is None else rad["radio_since"])
            c["radio_issue"].append(rad["radio_issue"])
            c["chief_risk"].append(chf["risk"])
            c["chief_since"].append(np.nan if chf["since"] is None else chf["since"])
            c["chief_out"].append(bool(chf["out"]))
            c["chief_issue"].append(chf["issue"])
            c["chief_why"].append(chf["why"])
            c["laps"].append(st.drivers[n].laps)
    laps = [(l.driver, l.lap, l.t_start, l.t_end, l.lap_time, l.is_in_lap, l.is_out_lap, l.track_status, l.interval)
            for l in st.laps]
    radio = [(m.car, m.t, m.known_at, m.text) for m in st.feeds.radio.messages()]
    return {
        "slug": Path(session_dir).parent.name, "total": st.total_laps, "cols": cols, "laps": laps, "notrun": notrun,
        "last_move": last_move, "rc": [(m.t, m.lap, m.driver, m.message) for m in st.rc], "radio": radio,
        "tla": {n: d.tla for n, d in st.drivers.items()}, "end_t": st.t, "seconds": time.time() - t_wall,
    }


def _record_job(args):
    path, out = args
    out = Path(out)
    if out.exists():
        return pickle.loads(out.read_bytes())
    rec = record_race(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(rec))
    return rec


def load_records(years, jobs: int = 3, refresh: bool = False, folder: Path | None = None) -> list[dict]:
    from multiprocessing import Pool

    from ..config import data_dir
    from ..archive import downloaded_sessions

    folder = folder or (data_dir() / "scratch" / "mechanics")
    refs = [r for r in downloaded_sessions() if r.session_name == "Race" and r.year in years]
    jobs_ = []
    for r in refs:
        out = folder / f"{r.year}_{r.local_dir.parent.name}.pkl"
        if refresh and out.exists():
            out.unlink()
        jobs_.append((str(r.local_dir), str(out)))
    if jobs <= 1:
        return [_record_job(j) for j in jobs_]
    with Pool(min(jobs, 3)) as p:
        return list(p.imap(_record_job, jobs_))


# --------------------------------------------------------------------------- events
def clean_laps(rec: dict, car: str, traffic: bool = True) -> list[tuple]:
    """(lap, t_start, t_end, lap_time) of the car's clean laps, in order."""
    out = []
    for d, lap, t0, t1, lt, i_in, i_out, status, interval in rec["laps"]:
        if d != car or lt is None or lap <= 1 or i_in or i_out or status != "1" or t0 is None:
            continue
        if traffic and interval is not None and interval < 2.0:
            continue
        out.append((lap, t0, t1, lt))
    return out


def lap_drops(rec: dict, car: str, traffic: bool = True) -> list[tuple[int, float, float, float]]:
    """(lap, t_start, t_end, drop in s against the median of the previous five clean laps)."""
    laps = clean_laps(rec, car, traffic)
    out = []
    for i in range(5, len(laps)):
        med = statistics.median(x[3] for x in laps[i - 5:i])
        out.append((laps[i][0], laps[i][1], laps[i][2], laps[i][3] - med))
    return out


def typical_lap(rec: dict) -> float:
    xs = [x[3] for c in {l[0] for l in rec["laps"]} for x in clean_laps(rec, c, False)]
    return statistics.median(xs) if xs else 90.0


def events_of(rec: dict) -> list[dict]:
    ev = []
    total = rec["total"] or 0
    for car, (t_flag, laps) in rec["notrun"].items():
        if 2 <= laps <= total - 3:
            t0 = min(rec["last_move"].get(car, t_flag), t_flag)
            ev.append({"car": car, "kind": "retire", "t0": t0, "t_end": t_flag, "lap": laps})
    cars = {l[0] for l in rec["laps"]}
    for car in sorted(cars):
        for lap, t0, t1, drop in lap_drops(rec, car):
            if drop >= SLOWDOWN_S:
                ev.append({"car": car, "kind": "slowdown", "t0": t0, "t_end": t1, "lap": lap, "drop": drop})
    from ..pitwall.engineers.mechanic_chief import rc_car_problem

    for t, lap, drv, msg in rec["rc"]:
        c = rc_car_problem(msg)
        if c:
            ev.append({"car": c, "kind": "rc_stop", "t0": t, "t_end": t, "lap": lap})
    return ev


# --------------------------------------------------------------------------- detectors -> episodes
# an episode is (raised_at, last_seen, extra); one car may have several
def ep_from_flags(t: np.ndarray, on: np.ndarray) -> list[tuple[float, float]]:
    eps, start, last = [], None, None
    for ti, o in zip(t, on):
        if o:
            if start is None:
                start = ti
            last = ti
        elif start is not None:
            eps.append((float(start), float(last)))
            start = None
    if start is not None:
        eps.append((float(start), float(last)))
    return eps


def det_tel(rec: dict, checks, on: float, hold: float = None, off_ratio: float = 0.6, clear: float = 30.0):
    """Telemetry episodes by re-running the level logic on the recorded scores."""
    from ..pitwall.engineers.mechanic_telemetry import THRESH, Level

    out = {}
    for car, c in rec["cols"].items():
        t = np.array(c["t"])
        flags = np.zeros(len(t), bool)
        for ck in checks:
            h = THRESH[ck][2] if hold is None else hold
            lv = Level(on, on * off_ratio, h, clear)
            for i, (ti, s) in enumerate(zip(t, c[ck])):
                lv.update(float(ti), None if (s is None or (isinstance(s, float) and math.isnan(s))) else float(s))
                if lv.active:
                    flags[i] = True
        out[car] = ep_from_flags(t, flags)
    return out


def det_radio(rec: dict, thr: float):
    out = {}
    for car, c in rec["cols"].items():
        t = np.array(c["t"])
        risk = np.array(c["radio_risk"], float)
        known = np.array(c["radio_since"], float)
        out[car] = ep_from_flags(t, (risk >= thr) & (t >= np.nan_to_num(known, nan=1e18)))
    return out


def det_chief(rec: dict, thr: float):
    out = {}
    for car, c in rec["cols"].items():
        t = np.array(c["t"])
        risk = np.array(c["chief_risk"], float)
        o = np.array(c["chief_out"], bool)
        out[car] = ep_from_flags(t, (risk >= thr) & ~o)
    return out


def det_lapdrop(rec: dict, x: float):
    """Baseline: a clean lap >= x s slower than the previous five clean laps. Raised when the lap is published."""
    L = rec["L"]
    out = {}
    for car in rec["cols"]:
        eps = []
        for lap, t0, t1, drop in lap_drops(rec, car):
            if drop >= x:
                eps.append((float(t1), float(t1 + L)))
        out[car] = eps
    return out


# --------------------------------------------------------------------------- scoring
def prepare(records: list[dict]) -> list[dict]:
    for rec in records:
        rec["L"] = typical_lap(rec)
        rec["events"] = events_of(rec)
        rec["n_cars"] = sum(1 for c in rec["cols"] if any(l >= 1 for l in rec["cols"][c]["laps"]))
    return records


def score(records: list[dict], make, kinds=("retire", "slowdown", "rc_stop")) -> dict:
    """Score one detector. ``make(rec) -> {car: [(raised, last_seen)]}``."""
    stats = {k: {"n": 0, "pre": 0, "flag": 0, "lead_s": [], "lead_laps": []} for k in kinds}
    n_fa = 0
    n_ep = 0
    n_cars = 0
    for rec in records:
        eps = make(rec)
        L = rec["L"]
        n_cars += rec["n_cars"]
        by_car: dict[str, list] = {}
        for e in rec["events"]:
            by_car.setdefault(e["car"], []).append(e)
        for e in rec["events"]:
            if e["kind"] not in kinds:
                continue
            s = stats[e["kind"]]
            s["n"] += 1
            lo = e["t0"] - PRE_LAPS * L
            hits = [r for r, end in eps.get(e["car"], []) if r <= e["t_end"] and end >= lo]
            if hits:
                s["flag"] += 1
                first = min(hits)
                s["lead_s"].append(e["t_end"] - first)
                s["lead_laps"].append((e["t_end"] - first) / L)
                if first < e["t0"]:
                    s["pre"] += 1
        for car, lst in eps.items():
            out_t = rec["notrun"].get(car, (None,))[0]
            for r, end in lst:
                n_ep += 1
                if out_t is not None and r > out_t:
                    continue  # after the car was out: not counted either way
                if not any(r <= e["t_end"] + L and end >= e["t0"] - PRE_LAPS * L for e in by_car.get(car, [])):
                    n_fa += 1
    tot_n = sum(s["n"] for s in stats.values())
    tot_pre = sum(s["pre"] for s in stats.values())
    tot_flag = sum(s["flag"] for s in stats.values())
    leads = [x for s in stats.values() for x in s["lead_s"]]
    leads_l = [x for s in stats.values() for x in s["lead_laps"]]
    return {"stats": stats, "n": tot_n, "pre": tot_pre, "flag": tot_flag,
            "recall_pre": tot_pre / tot_n if tot_n else float("nan"), "recall_flag": tot_flag / tot_n if tot_n else float("nan"),
            "lead_s": statistics.median(leads) if leads else float("nan"),
            "lead_laps": statistics.median(leads_l) if leads_l else float("nan"),
            "fa": n_fa / max(n_cars, 1), "n_fa": n_fa, "episodes": n_ep, "car_races": n_cars}


GRIDS = {
    "tel_power": ("power_loss", [0.5, 0.6, 0.7, 0.8, 0.9]),
    "tel_gearbox": ("gearbox", [0.5, 0.6, 0.7, 0.8, 0.9]),
    "tel_brake": ("brake_issue", [0.5, 0.6, 0.7, 0.8, 0.9]),
    "tel_slow": ("slow_car", [0.5, 0.6, 0.7, 0.8, 0.9]),
}
ALL_CHECKS = ("power_loss", "gearbox", "brake_issue", "slow_car")


def detector_makers(params: dict):
    """name -> make(rec); params holds the chosen value of each detector's threshold."""
    m = {}
    for name, (chk, _) in GRIDS.items():
        m[name] = (lambda rec, chk=chk, v=params.get(name, 0.7): det_tel(rec, (chk,), v))
    m["tel_any"] = lambda rec: det_tel(rec, ALL_CHECKS, params.get("tel_any", 0.7))
    m["radio"] = lambda rec: det_radio(rec, params.get("radio", 0.5))
    m["chief"] = lambda rec: det_chief(rec, params.get("chief", 0.6))
    m["lapdrop_baseline"] = lambda rec: det_lapdrop(rec, params.get("lapdrop_baseline", 3.0))
    return m


GRID_ALL = {
    **{k: v[1] for k, v in GRIDS.items()},
    "tel_any": [0.5, 0.6, 0.7, 0.8, 0.9],
    "radio": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "chief": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "lapdrop_baseline": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
}


def tune(records: list[dict], budget: float = FA_BUDGET) -> tuple[dict, dict]:
    """Per detector: the threshold with the best recall_pre (then recall_flag) within the false-alarm budget."""
    params, table = {}, {}
    for name, grid in GRID_ALL.items():
        best = None
        rows = []
        for v in grid:
            mk = detector_makers({name: v})[name]
            r = score(records, mk)
            rows.append((v, r))
            key = (r["fa"] <= budget, r["recall_pre"] if r["fa"] <= budget else -r["fa"], r["recall_flag"])
            if best is None or key > best[0]:
                best = (key, v, r)
        params[name] = best[1]
        table[name] = rows
    return params, table


def fmt_table(rows: list[tuple[str, dict]]) -> str:
    out = ["| detector | events | recall before event | recall by timing flag | median lead (s) | median lead (laps) | false alarms / car-race |",
           "|---|---|---|---|---|---|---|"]
    for name, r in rows:
        out.append(f"| {name} | {r['n']} | {r['recall_pre']:.2f} ({r['pre']}) | {r['recall_flag']:.2f} ({r['flag']}) | "
                   f"{r['lead_s']:.0f} | {r['lead_laps']:.1f} | {r['fa']:.2f} ({r['n_fa']}) |")
    return "\n".join(out)


def fmt_kinds(rows: list[tuple[str, dict]]) -> str:
    out = ["| detector | kind | events | recall before | recall by flag | median lead (s) |", "|---|---|---|---|---|---|"]
    for name, r in rows:
        for k, s in r["stats"].items():
            if s["n"]:
                ld = statistics.median(s["lead_s"]) if s["lead_s"] else float("nan")
                out.append(f"| {name} | {k} | {s['n']} | {s['pre'] / s['n']:.2f} | {s['flag'] / s['n']:.2f} | {ld:.0f} |")
    return "\n".join(out)


def examples(records: list[dict], name: str, params: dict, limit: int = 12) -> list[str]:
    mk = detector_makers(params)[name]
    out = []
    for rec in records:
        eps = mk(rec)
        L = rec["L"]
        for e in rec["events"]:
            lo = e["t0"] - PRE_LAPS * L
            hits = [r for r, end in eps.get(e["car"], []) if r <= e["t_end"] and end >= lo]
            if hits and e["kind"] in ("retire", "slowdown"):
                r = min(hits)
                tla = rec["tla"].get(e["car"], e["car"])
                out.append((e["t_end"] - r, f"{rec['slug'][:10]} car {e['car']} {tla} {e['kind']} lap {e['lap']}: raised {(e['t_end'] - r) / L:+.1f} laps "
                            f"({e['t_end'] - r:.0f} s) before {'the timing flag' if e['kind'] == 'retire' else 'the lap was published'}"))
    out.sort(key=lambda x: -x[0])
    return [x[1] for x in out[:limit]]


def chief_alert_examples(records: list[dict], limit: int = 8) -> list[str]:
    out = []
    for rec in records:
        L = rec["L"]
        for e in rec["events"]:
            if e["kind"] != "retire":
                continue
            c = rec["cols"].get(e["car"])
            if not c:
                continue
            for i, t in enumerate(c["t"]):
                if c["chief_risk"][i] >= 0.6 and not c["chief_out"][i] and e["t0"] - PRE_LAPS * L <= t <= e["t_end"]:
                    out.append(f"{rec['slug'][:10]} car {e['car']} out lap {e['lap']}: {c['chief_why'][i]} "
                               f"[{c['chief_issue'][i] or 'problem'}, risk {c['chief_risk'][i]:.2f}], {(e['t_end'] - t) / L:.1f} laps before the timing flag")
                    break
    return out[:limit]


def report(years_tune=(2025,), years_final=(2026,), jobs: int = 3, refresh: bool = False, budget: float = FA_BUDGET) -> str:
    t0 = time.time()
    rec_t = prepare(load_records(years_tune, jobs, refresh))
    params, table = tune(rec_t, budget)
    lines = [f"# Mechanics evaluation (thresholds on {years_tune}, scored once on {years_final})", "",
             f"False-alarm budget when choosing a threshold: {budget} per car-race.", "", "## Chosen thresholds", "",
             "| detector | threshold |", "|---|---|"] + [f"| {k} | {v} |" for k, v in params.items()]
    mk = detector_makers(params)
    names = list(mk)
    lines += ["", f"## {years_tune} (tuning year)", ""]
    tune_rows = [(n, score(rec_t, mk[n])) for n in names]
    lines += [fmt_table(tune_rows), "", fmt_kinds(tune_rows)]
    rec_f = []
    if years_final:
        rec_f = prepare(load_records(years_final, jobs, refresh))
        final_rows = [(n, score(rec_f, mk[n])) for n in names]
        lines += ["", f"## {years_final} (scored once)", "", fmt_table(final_rows), "", fmt_kinds(final_rows)]
    lines += ["", "## Event counts", ""]
    for yrs, recs in ((years_tune, rec_t), (years_final, rec_f)):
        if not recs:
            continue
        cnt = {}
        for r in recs:
            for e in r["events"]:
                cnt[e["kind"]] = cnt.get(e["kind"], 0) + 1
        lines.append(f"- {yrs}: {cnt}, car-races {sum(r['n_cars'] for r in recs)}")
    ex = rec_f or rec_t
    lines += ["", f"## Examples ({years_final or years_tune}, earliest first)", ""] + [f"- {x}" for x in examples(ex, "tel_any", params)]
    lines += [""] + [f"- chief: {x}" for x in chief_alert_examples(ex)]
    lines += ["", f"(elapsed {time.time() - t0:.0f} s)"]
    return "\n".join(lines)


def cmd_mechanics_eval(a) -> None:
    from ..config import data_dir

    text = report(tuple(a.tune_year), tuple(a.year) if a.year else (), a.jobs, a.refresh, a.budget)
    print(text)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "mechanics.md").write_text(text, encoding="utf-8")


def add_commands(sub) -> None:
    s = sub.add_parser("mechanics-eval", help="mechanic engineers: recall, lead time, false alarms vs a lap-time baseline")
    s.add_argument("--tune-year", type=int, nargs="+", default=[2025])
    s.add_argument("--year", type=int, nargs="*", default=[2026], help="scored once, with the thresholds chosen on --tune-year (empty: tune only)")
    s.add_argument("--jobs", type=int, default=3)
    s.add_argument("--refresh", action="store_true", help="re-record races (default: reuse data/scratch/mechanics)")
    s.add_argument("--budget", type=float, default=FA_BUDGET)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_mechanics_eval)
