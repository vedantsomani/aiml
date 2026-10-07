"""Calibrated call confidence: the chance a call turns out right, from the head of strategy's raw score.

The head's confidence comes from the simulator's now-vs-later difference and its standard error: a statement
about the simulation, not about the race. Here it is mapped, per action, to the observed rate at which such
calls were right (``bench/callscore.py``'s rules: BOX right when the car stops in L .. L+2, PREPARE_BOX in
L .. L+3, STAY_OUT when it does not stop in L .. L+2, from the call's lap L). The map is isotonic: a higher raw
score never means a lower calibrated one.

As every cross-race model here, a fit uses only races that ended before the race being called: each fit is
stored with ``train_end_utc`` and a race uses the latest fit that ended before it started (``for_race``).
Refit with ``pitsense calibrate`` after the head or the simulator changes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .bench.callscore import STAY_K, WINDOWS

ACTIONS = ("BOX", "PREPARE_BOX", "STAY_OUT")
MIN_CALLS = 40  # an action with fewer calls to learn from keeps its raw score
FILE = "call_calibration.json"
BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)  # reliability table


def path(folder: Path | None = None) -> Path:
    from .modelstore import models_dir

    return (folder or models_dir()) / FILE


def right(action: str, car_lap: int, stops: list[int]) -> bool | None:
    """Was the call right, by the benchmark's call scoring? None for actions that are not scored here."""
    if action in WINDOWS:
        lo, hi = WINDOWS[action]
        return any(car_lap + lo <= s <= car_lap + hi for s in stops)
    if action == "STAY_OUT":
        return not any(car_lap <= s <= car_lap + STAY_K for s in stops)
    return None


def rows_from(results: list[dict]) -> list[dict]:
    """(race, end, action, raw, right) for every scored call of ``bench.strategy.collect`` results."""
    out = []
    for r in results:
        for c in r["calls"]:
            ok = right(c["action"], int(c.get("car_lap") or 0), r["stops"].get(c["car"], []))
            raw = c.get("confidence_raw", c.get("confidence"))
            if ok is not None and raw is not None:
                out.append({"race": r["slug"], "end_utc": r.get("end_utc"), "action": c["action"], "raw": float(raw), "right": ok})
    return out


def fit(rows: list[dict]) -> dict:
    """{action: {"x": [...], "y": [...], "n": N, "rate": base rate}} from scored calls (isotonic, clipped to 2-98 %)."""
    from sklearn.isotonic import IsotonicRegression

    out = {}
    for a in ACTIONS:
        r = [x for x in rows if x["action"] == a]
        if len(r) < MIN_CALLS:
            continue
        x = np.array([v["raw"] for v in r])
        y = np.array([float(v["right"]) for v in r])
        iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip").fit(x, y)
        out[a] = {"x": [round(float(v), 4) for v in iso.X_thresholds_], "y": [round(float(v), 4) for v in iso.y_thresholds_],
                  "n": len(r), "rate": round(float(y.mean()), 4)}
    return out


def apply(fits: dict | None, action: str, raw: float | None) -> float | None:
    """The calibrated confidence of a call (its raw score when there is no fit for that action)."""
    f = (fits or {}).get(action)
    if f is None or raw is None:
        return raw
    return round(float(np.interp(raw, f["x"], f["y"])), 2)


def save(fits: dict, train_end_utc: datetime, races: list[str], folder: Path | None = None) -> Path:
    """Add a fit to the store (one entry per training cutoff; a refit with the same cutoff replaces it)."""
    p = path(folder)
    store = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    store = [e for e in store if e["train_end_utc"] != train_end_utc.isoformat()]
    store.append({"train_end_utc": train_end_utc.isoformat(), "races": races, "fits": fits})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(store, key=lambda e: e["train_end_utc"]), indent=1), encoding="utf-8")
    return p


_CACHE: dict = {}


def for_race(race_start_utc: datetime | None, folder: Path | None = None) -> dict | None:
    """The latest fit trained only on races that ended before ``race_start_utc``; None without one (raw scores)."""
    if race_start_utc is None:
        return None
    p = path(folder)
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return None
    if _CACHE.get("key") != key:
        _CACHE.update(key=key, store=json.loads(p.read_text(encoding="utf-8")))
    best = None
    for e in _CACHE["store"]:
        if datetime.fromisoformat(e["train_end_utc"]) < race_start_utc:
            best = e
    return best["fits"] if best else None


def reliability(rows: list[dict], key: str) -> list[dict]:
    """Predicted vs observed by bin of ``key`` (raw or cal)."""
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        b = [r for r in rows if lo <= r[key] < hi]
        if b:
            out.append({"bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(b), "predicted": round(float(np.mean([r[key] for r in b])), 3),
                        "observed": round(float(np.mean([r["right"] for r in b])), 3)})
    return out


def scores(rows: list[dict], key: str) -> dict:
    """Brier score and expected calibration error (bins weighted by size) of ``key``."""
    p = np.array([r[key] for r in rows])
    y = np.array([float(r["right"]) for r in rows])
    ece = sum(b["n"] * abs(b["predicted"] - b["observed"]) for b in reliability(rows, key)) / max(len(rows), 1)
    return {"n": len(rows), "brier": round(float(np.mean((p - y) ** 2)), 4), "ece": round(float(ece), 4)}


# ----------------------------------------------------------------------------- pitsense calibrate
def build(years: list[int], test_year: int, *, jobs: int = 6, refresh: bool = False, folder: Path | None = None) -> dict:
    """Collect the head's calls on every race of ``years`` (bench.strategy, cached), score the calibration over an
    expanding window on ``test_year`` (each race calibrated by a fit on races that ended before it started) and
    store one fit per training cutoff."""
    from .bench import history as H
    from .bench import strategy as B
    from .config import bench_dir

    hist = {s.race_id: s for s in H.load(bench_dir() / "history.json").asof(datetime.max.replace(tzinfo=timezone.utc))}
    results = [r for y in years for r in B.collect(y, jobs=jobs, refresh=refresh, models=True, tag="_cal")]
    results = [dict(r, end_utc=hist[r["slug"]].end_utc, start_utc=hist[r["slug"]].start_utc) for r in results if r["slug"] in hist]
    results.sort(key=lambda r: r["start_utc"])
    rows = rows_from(results)
    test = []
    for r in results:
        if r["year"] != test_year:
            continue
        fits = fit([x for x in rows if x["end_utc"] < r["start_utc"]])
        test += [dict(x, cal=apply(fits, x["action"], x["raw"])) for x in rows if x["race"] == r["slug"]]
    n_saved = 0
    for i, r in enumerate(results):  # one fit per cutoff: after each race, on every race up to it
        fits = fit([x for x in rows if x["end_utc"] <= r["end_utc"]])
        if fits:
            save(fits, r["end_utc"], [x["slug"] for x in results[: i + 1]], folder)
            n_saved += 1
    by_action = {a: {"raw": scores([x for x in test if x["action"] == a], "raw"),
                     "cal": scores([x for x in test if x["action"] == a], "cal")} for a in ACTIONS if any(x["action"] == a for x in test)}
    return {"races": len(results), "calls_scored": len(rows), "test_year": test_year, "test_calls": len(test),
            "raw": scores(test, "raw") if test else None, "cal": scores(test, "cal") if test else None, "by_action": by_action,
            "reliability_raw": reliability(test, "raw"), "reliability_cal": reliability(test, "cal"),
            "fits_saved": n_saved, "file": str(path(folder))}


def cmd_calibrate(a) -> None:
    import os

    os.environ.setdefault("PITSENSE_VOICE", "off")
    res = build(a.year, a.test_year, jobs=a.jobs, refresh=a.refresh)
    if a.json:
        print(json.dumps(res, indent=1, default=str))
        return
    print(f"{res['races']} races, {res['calls_scored']} scored calls; {res['fits_saved']} fits -> {res['file']}")
    print(f"{res['test_year']} held out race by race ({res['test_calls']} calls): raw {res['raw']}  calibrated {res['cal']}")
    for a_, v in res["by_action"].items():
        print(f"  {a_:<12} raw brier {v['raw']['brier']} ece {v['raw']['ece']}  ->  calibrated brier {v['cal']['brier']} ece {v['cal']['ece']}  (n {v['raw']['n']})")
    print("reliability (bin, n, predicted, observed): raw | calibrated")
    for r, c in zip(res["reliability_raw"], res["reliability_cal"] + [None] * 9):
        print(f"  {r['bin']}  {r['n']:4d}  {r['predicted']:.2f}  {r['observed']:.2f}" + (f"   |  {c['bin']}  {c['n']:4d}  {c['predicted']:.2f}  {c['observed']:.2f}" if c else ""))


def add_commands(sub) -> None:
    s = sub.add_parser("calibrate", help="fit calibrated call confidence (as-of) and score it on a held-out season")
    s.add_argument("--year", type=int, nargs="+", default=[2025, 2026], help="races to collect calls from")
    s.add_argument("--test-year", type=int, default=2026)
    s.add_argument("--jobs", type=int, default=6)
    s.add_argument("--refresh", action="store_true", help="re-run the head on every race (after it or the simulator changed)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_calibrate)
