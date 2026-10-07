"""Scoring the incidents engineer (``python -m pitsense incidents-eval``).

Labels are read from the finished race (nothing in ``pitwall/`` imports this module).

An *unscheduled event* for a car is one of:

* ``retire``   the car ends not running with 2..total-3 laps done (event lap = the lap it died on)
* a pit stop (event lap = its in-lap) that is not a normal tyre stop, i.e. not under SC / VSC / red flag, not to or
  from wet tyres, and one of
  - ``early``   the stint is at most 0.6 x the race's median stint on that compound and 6+ laps shorter than it
  - ``collapse``  the stint is under 0.9 x the median and the car lost 2+ s against its field on one of its two
                  laps before the in-lap (the engineer's own ``pace_drop``: partly circular, so ``early`` is reported alone too)
  - ``incident``  the stint is under 0.9 x the median and race control named the car in an incident (collision,
                  off track) in the three laps before the stop

Rows are one per car per completed lap (lap in-laps excluded). A row at lap L is positive for ``damage`` when an
event falls on lap L+1 or L+2, and for ``stop3`` when it falls on L+1..L+3.

Protocol: logistic weights and the alert threshold are fitted on 2025 races only (the threshold on
leave-one-race-out predictions); 2026 is scored once.
"""

from __future__ import annotations

import pickle
import statistics
import time
from pathlib import Path

import numpy as np

STOP_STATUS = set("4567")  # SC, red, VSC deployed / ending
WET = {"INTERMEDIATE", "WET"}
EARLY_RATIO, EARLY_GAP, SOFT_RATIO, COLLAPSE_S = 0.6, 6, 0.9, 2.0
RC_LAPS = 3


# --------------------------------------------------------------------------- recording
def record_race(session_dir: str) -> dict:
    from ..events import load_archive_session
    from ..pitloss import PitLossPrior
    from ..pitwall import Context, PitWall
    from ..pitwall.engineers.incidents import IncidentsEngineer
    from ..pitwall.engineers.mechanic_radio import MechanicRadio
    from ..pitwall.engineers.mechanic_telemetry import MechanicTelemetry
    from ..state import RaceState

    t_wall = time.time()
    log = load_archive_session(Path(session_dir), feeds=True)
    st = RaceState(log.meta)
    wall = PitWall(Context(prior=PitLossPrior(), meta=log.meta), engineers=[MechanicTelemetry, MechanicRadio, IncidentsEngineer])
    seen = 0
    rows: list[dict] = []
    notrun: dict[str, tuple[float, int]] = {}
    for e in log.events:
        st.apply(e)
        wall.observe(st)
        for n, d in st.drivers.items():
            if not d.running and n not in notrun:
                notrun[n] = (st.t, d.laps)
        if len(st.laps) > seen:
            new = st.laps[seen:]
            seen = len(st.laps)
            view = wall.view(st)
            for l in new:
                if l.is_in_lap or l.driver not in st.drivers:
                    continue
                v = view.car("incidents", l.driver)
                rows.append({"car": l.driver, "lap": l.lap, "t": st.t, **v})
    laps = [(l.driver, l.lap, l.t_start, l.t_end, l.lap_time, l.is_in_lap, l.is_out_lap, l.track_status, l.compound, l.stint) for l in st.laps]
    pits = [(p.driver, p.in_lap, p.in_t, p.out_lap, p.status_at_entry) for p in st.pit_events]
    return {
        "slug": Path(session_dir).parent.name, "total": st.total_laps, "rows": rows, "laps": laps, "pits": pits,
        "notrun": notrun, "rc": [(m.t, m.lap, m.driver, m.message) for m in st.rc],
        "tla": {n: d.tla for n, d in st.drivers.items()}, "seconds": time.time() - t_wall,
    }


def _job(args):
    path, out = args
    out = Path(out)
    if out.exists():
        return pickle.loads(out.read_bytes())
    rec = record_race(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(rec))
    return rec


def load_records(years, jobs: int = 2, refresh: bool = False) -> list[dict]:
    from multiprocessing import Pool

    from ..archive import downloaded_sessions
    from ..config import data_dir

    folder = data_dir() / "scratch" / "incidents"
    work = []
    for r in downloaded_sessions():
        if r.session_name != "Race" or r.year not in years:
            continue
        out = folder / f"{r.year}_{r.local_dir.parent.name}.pkl"
        if refresh and out.exists():
            out.unlink()
        work.append((str(r.local_dir), str(out)))
    if jobs <= 1:
        return [_job(w) for w in work]
    with Pool(min(jobs, 2)) as p:
        return list(p.imap(_job, work))


# --------------------------------------------------------------------------- labels
def events_of(rec: dict) -> list[dict]:
    """Unscheduled events of one race: {car, lap, kind}. ``kind`` is retire / early / collapse / incident."""
    from ..pitwall.engineers.incidents import classify_rc

    total = rec["total"] or 0
    by = {(c, l): (st, comp, stint) for c, l, _, _, _, _, _, st, comp, stint in rec["laps"]}
    lap_end = {(c, l): te for c, l, _, te, *_ in rec["laps"]}
    row_at = {(r["car"], r["lap"]): r for r in rec["rows"]}
    stops: dict[str, list[tuple[int, str]]] = {}
    for car, in_lap, in_t, out_lap, status in rec["pits"]:
        stops.setdefault(car, []).append((in_lap, status))
    # stint lengths by compound, from completed stints (previous stop to this one)
    lens: dict[str, list[int]] = {}
    items = []
    for car, ss in stops.items():
        prev = 0
        for in_lap, status in sorted(ss):
            comp = (by.get((car, in_lap)) or (None, None, None))[1]
            items.append((car, in_lap, in_lap - prev, comp, status))
            if comp:
                lens.setdefault(comp, []).append(in_lap - prev)
            prev = in_lap
    allv = [x for v in lens.values() for x in v] or [1]
    norm = {c: statistics.median(v) if len(v) >= 3 else statistics.median(allv) for c, v in lens.items()}
    rcs = []
    for t, lap, drv, msg in rec["rc"]:
        kind, cars = classify_rc(msg)
        if kind in ("collision", "incident"):
            rcs += [(t, c) for c in cars]
    out = []
    for car, in_lap, length, comp, status in items:
        if in_lap < 1 or status == "5" or comp is None:
            continue
        st_in = (by.get((car, in_lap)) or ("1",))[0]
        if set(st_in.replace(",", "")) & STOP_STATUS or set(str(status)) & STOP_STATUS or comp in WET:
            continue
        nxt = next((c for (cc, l), (_, c, _) in by.items() if cc == car and l == in_lap + 1), None)
        if nxt in WET:
            continue
        nm = norm.get(comp, statistics.median(allv))
        kind = None
        if length <= EARLY_RATIO * nm and length <= nm - EARLY_GAP:
            kind = "early"
        elif length < SOFT_RATIO * nm:
            drops = [row_at[(car, l)]["pace_drop"] for l in (in_lap - 1, in_lap - 2) if (car, l) in row_at]
            t1 = lap_end.get((car, in_lap))
            t0 = lap_end.get((car, max(in_lap - RC_LAPS, 1)), 0.0)
            if drops and max(drops) >= COLLAPSE_S:
                kind = "collapse"
            elif t1 is not None and any(c == car and t0 - 30 <= t <= t1 for t, c in rcs):
                kind = "incident"
        if kind:
            out.append({"car": car, "lap": in_lap, "kind": kind, "comp": comp, "length": length})
    for car, (t, laps) in rec["notrun"].items():
        if 2 <= laps <= total - 3 and not any(e["car"] == car and abs(e["lap"] - (laps + 1)) <= 1 for e in out):
            out.append({"car": car, "lap": laps + 1, "kind": "retire", "comp": None, "length": None})
    return sorted(out, key=lambda e: (e["lap"], e["car"]))


def build_table(records: list[dict]) -> dict:
    """Rows as arrays: X (features), y_damage, y_stop3, and which race / car / lap each is."""
    from ..pitwall.engineers.incidents import FEATURES, clip_features

    X, yd, y3, meta = [], [], [], []
    evs = []
    for ri, rec in enumerate(records):
        ev = events_of(rec)
        evs.append(ev)
        lap_of: dict[str, set[int]] = {}
        for e in ev:
            lap_of.setdefault(e["car"], set()).add(e["lap"])
        for r in rec["rows"]:
            ls = lap_of.get(r["car"], ())
            L = r["lap"]
            x = clip_features(r)
            X.append([x[k] for k in FEATURES])
            yd.append(int(any(L + 1 <= l <= L + 2 for l in ls)))
            y3.append(int(any(L + 1 <= l <= L + 3 for l in ls)))
            meta.append((ri, r["car"], L))
    return {"X": np.array(X, float), "yd": np.array(yd), "y3": np.array(y3), "meta": meta, "events": evs}


# --------------------------------------------------------------------------- fitting
def fit(X, y, C: float = 0.5):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=C, max_iter=2000)
    m.fit(X, y)
    return float(m.intercept_[0]), m.coef_[0]


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_all(tab: dict, C: float) -> dict:
    from ..pitwall.engineers.incidents import FEATURES

    out = {}
    for key, y in (("damage", tab["yd"]), ("stop3", tab["y3"])):
        b, w = fit(tab["X"], y, C)
        out[key] = (b, {n: float(v) for n, v in zip(FEATURES, w)})
    return out


def loro_predict(tab: dict, C: float):
    """Leave-one-race-out probabilities for (damage, stop3)."""
    races = np.array([m[0] for m in tab["meta"]])
    pd, p3 = np.zeros(len(races)), np.zeros(len(races))
    for r in sorted(set(races)):
        te, tr = races == r, races != r
        for y, p in ((tab["yd"], pd), (tab["y3"], p3)):
            b, w = fit(tab["X"][tr], y[tr], C)
            p[te] = _sig(b + tab["X"][te] @ w)
    return pd, p3


def predict(tab: dict, params: dict):
    from ..pitwall.engineers.incidents import FEATURES

    out = []
    for key in ("damage", "stop3"):
        b, w = params[key]
        out.append(_sig(b + tab["X"] @ np.array([w[n] for n in FEATURES])))
    return out


# --------------------------------------------------------------------------- alert scoring
def alert_runs(tab: dict, p: np.ndarray, thr: float) -> dict[tuple[int, str], list[tuple[int, int]]]:
    """(race, car) -> list of (first lap, last lap) runs of consecutive rows at or above ``thr``."""
    by: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for (ri, car, lap), x in zip(tab["meta"], p):
        by.setdefault((ri, car), []).append((lap, x))
    runs = {}
    for k, seq in by.items():
        lst: list[tuple[int, int]] = []
        for lap, x in sorted(seq):
            if x < thr:
                continue
            if lst and lap - lst[-1][1] <= 1:
                lst[-1] = (lst[-1][0], lap)
            else:
                lst.append((lap, lap))
        if lst:
            runs[k] = lst
    return runs


def score(tab: dict, p: np.ndarray, thr: float, kinds=None) -> dict:
    """Episode precision, event recall and laps of warning at one threshold."""
    runs = alert_runs(tab, p, thr)
    n_alerts = n_true = 0
    evs_by = {}
    for ri, ev in enumerate(tab["events"]):
        for e in ev:
            evs_by.setdefault((ri, e["car"]), []).append(e)
    for k, lst in runs.items():
        for a, b in lst:
            n_alerts += 1
            if any(a + 1 <= e["lap"] <= b + 3 for e in evs_by.get(k, ())):
                n_true += 1
    n_ev = n_hit = 0
    warn = []
    for k, es in evs_by.items():
        for e in es:
            if kinds and e["kind"] not in kinds:
                continue
            n_ev += 1
            hit = [a for a, b in runs.get(k, ()) if a <= e["lap"] - 1 and b >= e["lap"] - 3]
            if hit:
                n_hit += 1
                warn.append(e["lap"] - max(min(hit), e["lap"] - 6))
    cars = len({(ri, c) for ri, c, _ in tab["meta"]})
    return {"alerts": n_alerts, "precision": n_true / n_alerts if n_alerts else float("nan"), "events": n_ev,
            "recall": n_hit / n_ev if n_ev else float("nan"), "warn_median": statistics.median(warn) if warn else float("nan"),
            "warn_mean": statistics.mean(warn) if warn else float("nan"), "false_per_car_race": (n_alerts - n_true) / max(cars, 1), "hits": n_hit}


MIN_ALERTS = 8


def choose_threshold(tab: dict, p: np.ndarray) -> float:
    """The threshold with the best F1 (episode precision, event recall) on the tuning races, with at least MIN_ALERTS alerts."""
    best, thr = -1.0, 0.1
    for t in np.arange(0.02, 0.6, 0.01):
        s = score(tab, p, float(t))
        if s["alerts"] < MIN_ALERTS:
            continue
        pr, rc = s["precision"], s["recall"]
        f = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0
        if f > best:
            best, thr = f, float(t)
    return round(thr, 2)


def choose_C(tab: dict, grid=(0.3, 1.0, 3.0, 10.0, 30.0)) -> float:
    """C with the best leave-one-race-out average precision for damage_prob on the tuning races."""
    from sklearn.metrics import average_precision_score

    best, bc = -1.0, grid[0]
    for C in grid:
        pd, _ = loro_predict(tab, C)
        ap = average_precision_score(tab["yd"], pd)
        if ap > best:
            best, bc = ap, C
    return bc


def fmt(s: dict) -> str:
    return (f"alerts {s['alerts']}, precision {s['precision']:.2f}, events {s['events']}, recall {s['recall']:.2f}, "
            f"warning median {s['warn_median']:.1f} laps (mean {s['warn_mean']:.1f}), false alerts/car-race {s['false_per_car_race']:.3f}")


def report(tune=(2025,), final=(2026,), jobs: int = 2, refresh: bool = False, C: float | None = None, quiet_table: bool = False) -> str:
    from ..pitwall.engineers.incidents import FEATURES

    rt = load_records(tune, jobs, refresh)
    tab = build_table(rt)
    C = C or choose_C(tab)
    pd, p3 = loro_predict(tab, C)
    thr = choose_threshold(tab, pd)
    params = fit_all(tab, C)
    L = [f"# Incidents evaluation (fit on {tune}, scored once on {final})", "",
         f"C={C}; alert = damage_prob >= {thr} (best F1, 8+ alerts, on leave-one-race-out 2025 predictions).", "",
         "## Weights (fitted on 2025)", "", "| feature | damage | stop3 |", "|---|---|---|",
         f"| bias | {params['damage'][0]:.3f} | {params['stop3'][0]:.3f} |"]
    L += [f"| {n} | {params['damage'][1][n]:.3f} | {params['stop3'][1][n]:.3f} |" for n in FEATURES]
    kinds = {}
    for ev in tab["events"]:
        for e in ev:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    L += ["", f"## 2025 (leave-one-race-out), events {kinds}", "", fmt(score(tab, pd, thr)), ""]
    for k in ("early", "retire", "collapse", "incident"):
        L.append(f"- {k}: " + fmt(score(tab, pd, thr, {k})).split(", precision")[1].split(", false")[0])
    for name, col in (("pace_drop >= 3 s", 0), ("pace_drop >= 2 s", 0)):
        thr_s = float(name.split(">= ")[1].split()[0]) / 4.0
        s = score(tab, tab["X"][:, FEATURES.index("pace_drop")], thr_s)
        L.append(f"- baseline {name}: " + fmt(s))
    final_tab = None
    if final:
        rf = load_records(final, jobs, refresh)
        final_tab = build_table(rf)
        qd, q3 = predict(final_tab, params)
        kinds = {}
        for ev in final_tab["events"]:
            for e in ev:
                kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        L += ["", f"## 2026 (scored once), events {kinds}", "", fmt(score(final_tab, qd, thr)), ""]
        for k in ("early", "retire", "collapse", "incident"):
            L.append(f"- {k}: " + fmt(score(final_tab, qd, thr, {k})).split(", precision")[1].split(", false")[0])
        for name in ("pace_drop >= 3 s", "pace_drop >= 2 s"):
            thr_s = float(name.split(">= ")[1].split()[0]) / 4.0
            L.append(f"- baseline {name}: " + fmt(score(final_tab, final_tab["X"][:, FEATURES.index("pace_drop")], thr_s)))
        L += ["", "## Unscheduled stops in 2026", ""] + stops_table(rf, final_tab, qd, thr)
    L.append(f"\nWEIGHTS_TEXT thr={thr}\nBIAS = {{'damage': {params['damage'][0]:.3f}, 'stop3': {params['stop3'][0]:.3f}}}")
    for k in ("damage", "stop3"):
        L.append(f"{k}: " + ", ".join(f'"{n}": {params[k][1][n]:.3f}' for n in FEATURES))
    return "\n".join(L)


def stops_table(records: list[dict], tab: dict, p: np.ndarray, thr: float) -> list[str]:
    runs = alert_runs(tab, p, thr)
    pk = {m: x for m, x in zip(tab["meta"], p)}
    rows = ["| race | car | lap | kind | compound / stint | max damage_prob in 3 laps before | warned | laps of warning |", "|---|---|---|---|---|---|---|---|"]
    for ri, (rec, ev) in enumerate(zip(records, tab["events"])):
        for e in ev:
            hit = [a for a, b in runs.get((ri, e["car"]), ()) if a <= e["lap"] - 1 and b >= e["lap"] - 3]
            mx = max([pk.get((ri, e["car"], l), 0.0) for l in range(e["lap"] - 3, e["lap"])] or [0.0])
            warn = f"{e['lap'] - max(min(hit), e['lap'] - 6)}" if hit else "-"
            tla = rec["tla"].get(e["car"], e["car"])
            ci = f"{e['comp']} / {e['length']} laps" if e["comp"] else "retired"
            rows.append(f"| {rec['slug'][11:].replace('_Grand_Prix', '')} | {tla} | {e['lap']} | {e['kind']} | {ci} | {mx:.2f} | {'yes' if hit else 'no'} | {warn} |")
    return rows


def cmd_incidents_eval(a) -> None:
    text = report(tuple(a.tune_year), tuple(a.year) if a.year else (), a.jobs, a.refresh, a.C)
    print(text)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "incidents.md").write_text(text, encoding="utf-8")


def add_commands(sub) -> None:
    s = sub.add_parser("incidents-eval", help="incidents engineer: precision / recall / laps of warning for unscheduled stops")
    s.add_argument("--tune-year", type=int, nargs="+", default=[2025])
    s.add_argument("--year", type=int, nargs="*", default=[2026], help="scored once (empty: fit only)")
    s.add_argument("--jobs", type=int, default=2)
    s.add_argument("--refresh", action="store_true")
    s.add_argument("--C", type=float, default=None, help="default: chosen on 2025 by leave-one-race-out average precision")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_incidents_eval)
