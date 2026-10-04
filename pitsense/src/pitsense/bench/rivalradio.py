"""Measurements for the rival-radio engineer (``python -m pitsense rivalradio-eval``). Owner: rivalradio.

Radio exists only for 2025-26 and only ~30 clips per race (none at all for 4 of the 15 radio-era 2026
races, whose mp3s were never published), so every number here covers the radio races only.

* **Event evaluation.** Each transcript becomes known 15 s after its message; at that moment the
  engineer's ``rr_box_intent`` for that car is read (a *signal* when >= theta). A signal is a hit when the
  car's next green-flag stop comes at the end of the lap in progress or of the next one (``in_lap <= laps
  completed + 2``). Precision is per signal, recall per stop (a stop is caught when a hit signal precedes
  it), lead time is stop time minus the moment the transcript was known. The base rate is the chance
  that a random car at a random lap end stops within 2 laps. ``extend`` signals are scored the same way
  with "no stop within 2 laps".
* **Ablation.** Rows of the radio races are rebuilt with the radio feed on (``bench/v0.1-rr``), and the
  ``pit_within_1/3`` tasks are scored with two TASK_MODELS entries that differ only in the rr columns:
  ``rr_gbm_with`` / ``rr_gbm_without`` (the core ``gbm_hazard`` settings). The core models are untouched.
* **Protocol.** theta is chosen on 2025 (F1 of precision and recall); 2026 is scored once with it. The
  ablation is also scored on 2025 (expanding window, >= 8 earlier radio races) and 2026 (>= 10).

Labels read the finished race; nothing in ``pitwall/`` imports this module.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import feature_columns

THETAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
WITHIN = 2
RR_VERSION = "v0.1-rr"


def _has_radio(ref) -> bool:
    d = Path(ref.local_dir) / "TeamRadio"
    return d.exists() and any(d.glob("*.mp3"))


def radio_refs(years):
    from ..archive import downloaded_sessions

    return [r for r in downloaded_sessions() if r.session_name == "Race" and r.year in years]


# ----------------------------------------------------------------------------- recording
def record_race(session_dir: str) -> dict:
    """Replay one race (radio on) and read ``rivalradio`` at the moment each transcript becomes known."""
    from ..events import load_archive_session
    from ..pitloss import PitLossPrior
    from ..pitwall import Context, PitWall
    from ..pitwall.engineers.rivalradio import RivalRadio
    from ..state import RaceState

    log = load_archive_session(Path(session_dir), feeds=True)
    st = RaceState(log.meta)
    wall = PitWall(Context(prior=PitLossPrior(), meta=log.meta), engineers=[RivalRadio])
    pending: list[tuple[float, str, str, float]] = []  # known_at, car, path, message t
    seen: set[str] = set()
    sigs: list[dict] = []
    n_clips: dict[str, int] = {}
    for e in log.events:
        st.apply(e)
        wall.observe(st)
        if e.topic == "TeamRadio":
            for m in st.feeds.radio.messages():
                if m.path not in seen:
                    seen.add(m.path)
                    pending.append((m.known_at, m.car, m.path, m.t))
            pending.sort()
        while pending and pending[0][0] <= st.t:
            known, car, path, mt = pending.pop(0)
            if car not in st.drivers:
                continue
            v = wall.view(st).car("rivalradio", car)
            n_clips[car] = n_clips.get(car, 0) + 1
            sigs.append({"car": car, "t": st.t, "known": known, "msg_t": mt, "laps_done": st.drivers[car].laps,
                         "quote": v["rr_last_quote"], "intent": v["rr_last_intent"], "last_t": v["rr_last_t"],
                         **{k: v[k] for k in ("rr_box_intent", "rr_extend", "rr_tyres_gone", "rr_planchange",
                                              "rr_push", "rr_save", "rr_cover", "rr_weather", "rr_problem")}})
    laps = [(l.driver, l.lap, l.t_end, l.track_status) for l in st.laps]
    pits = [(p.driver, p.in_lap, p.in_t, p.under_red) for p in st.pit_events]
    return {"slug": Path(session_dir).parent.name, "signals": sigs, "pits": pits, "laps": laps,
            "total": st.total_laps, "n_cars": len(st.drivers), "clips": len(seen), "cars_with_clips": len(n_clips)}


def _record_job(args):
    path, out = args
    out = Path(out)
    if out.exists():
        return pickle.loads(out.read_bytes())
    rec = record_race(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(rec))
    return rec


def load_records(years, jobs: int = 3, refresh: bool = False) -> list[dict]:
    from multiprocessing import Pool

    from ..config import data_dir

    folder = data_dir() / "scratch" / "rivalradio"
    todo = []
    for r in radio_refs(years):
        out = folder / f"{r.year}_{Path(r.local_dir).parent.name}.pkl"
        if refresh and out.exists():
            out.unlink()
        todo.append((str(r.local_dir), str(out), r.year))
    jobs_ = [(a, b) for a, b, _ in todo]
    if jobs <= 1:
        recs = [_record_job(j) for j in jobs_]
    else:
        with Pool(min(jobs, 3)) as p:
            recs = list(p.imap(_record_job, jobs_))
    for rec, (_, _, y) in zip(recs, todo):
        rec["year"] = y
    return recs


# ----------------------------------------------------------------------------- event scoring
def _race_events(rec: dict, theta: float, field: str = "rr_box_intent", want_stop: bool = True) -> dict:
    stops = sorted((c, il, it) for c, il, it, red in rec["pits"] if not red)
    by_car: dict[str, list[tuple[int, float]]] = {}
    for c, il, it in stops:
        by_car.setdefault(c, []).append((il, it))
    hits, signals, leads, caught = 0, 0, [], set()
    for s in rec["signals"]:
        if s[field] < theta:
            continue
        signals += 1
        nxt = next(((il, it) for il, it in by_car.get(s["car"], []) if it > s["t"]), None)
        stop_soon = nxt is not None and nxt[0] <= s["laps_done"] + WITHIN
        ok = stop_soon if want_stop else not stop_soon
        hits += ok
        if ok and want_stop:
            leads.append(nxt[1] - s["known"])
            caught.add((s["car"], nxt[0]))
    return {"signals": signals, "hits": hits, "leads": leads, "caught": len(caught), "stops": len(stops)}


def base_rate(recs: list[dict]) -> float:
    num = den = 0
    for rec in recs:
        by_car: dict[str, list[int]] = {}
        for c, il, _, red in rec["pits"]:
            if not red:
                by_car.setdefault(c, []).append(il)
        for c, lap, _, _ in rec["laps"]:
            if lap >= rec["total"] - 1:
                continue
            den += 1
            num += any(lap < il <= lap + WITHIN for il in by_car.get(c, ()))
    return num / max(den, 1)


def score_events(recs: list[dict], theta: float) -> dict:
    tot = {"signals": 0, "hits": 0, "caught": 0, "stops": 0}
    leads: list[float] = []
    for rec in recs:
        r = _race_events(rec, theta)
        for k in tot:
            tot[k] += r[k]
        leads += r["leads"]
    p = tot["hits"] / tot["signals"] if tot["signals"] else float("nan")
    rc = tot["caught"] / tot["stops"] if tot["stops"] else float("nan")
    f1 = 2 * p * rc / (p + rc) if tot["signals"] and p + rc > 0 else 0.0
    return {"theta": theta, **tot, "precision": p, "recall": rc, "f1": f1,
            "lead_med_s": float(np.median(leads)) if leads else float("nan"),
            "lead_p25_s": float(np.percentile(leads, 25)) if leads else float("nan"),
            "lead_p75_s": float(np.percentile(leads, 75)) if leads else float("nan")}


def score_extend(recs: list[dict], theta: float = 0.6) -> dict:
    sig = hit = 0
    for rec in recs:
        r = _race_events(rec, theta, "rr_extend", want_stop=False)
        sig += r["signals"]
        hit += r["hits"]
    return {"signals": sig, "no_stop_within_2": hit / sig if sig else float("nan")}


def coverage(recs: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in recs:
        stops = [p for p in rec["pits"] if not p[3]]
        sig = [s for s in rec["signals"] if s["rr_box_intent"] >= 0.5]
        rows.append({"race": rec["slug"][:22], "year": rec["year"], "clips": rec["clips"],
                     "cars_with_clip": rec["cars_with_clips"], "cars": rec["n_cars"], "stops": len(stops),
                     "box_signals": len(sig),
                     "stops_with_any_clip_2laps": sum(
                         any(s["car"] == c and s["t"] < it and s["laps_done"] + WITHIN >= il for s in rec["signals"])
                         for c, il, it, _ in stops)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- ablation models
RR_COLS = tuple(f"rivalradio__{k}" for k in ("rr_box_intent", "rr_extend", "rr_push", "rr_save", "rr_tyres_gone",
                                              "rr_planchange", "rr_cover", "rr_weather", "rr_problem", "rr_n", "rr_age_s"))


class _RRGBM:
    """The core ``gbm_hazard`` settings; the two entries differ only in the rival-radio columns."""

    name = ""
    with_rr = True

    def fit(self, df, target):
        self.cols = [c for c in feature_columns() if self.with_rr or c not in RR_COLS]
        self.model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=200,
            l2_regularization=10.0, random_state=0).fit(_x(df, self.cols), df[target].astype(int))
        return self

    def predict(self, df):
        return self.model.predict_proba(_x(df, self.cols))[:, 1]


def _x(df: pd.DataFrame, cols) -> pd.DataFrame:
    return pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df else np.nan for c in cols}, index=df.index).astype(float)


class RRGBMWith(_RRGBM):
    name = "rr_gbm_with"


class RRGBMWithout(_RRGBM):
    name = "rr_gbm_without"
    with_rr = False


def build_rr_rows(ref, history, out_dir: Path) -> Path:
    from ..events import load_archive_session
    from ..pitwall.engineer import Context
    from .dataset import decision_rows
    from .history import prior_for
    from .labels import add_labels

    log = load_archive_session(ref.local_dir, feeds=True)
    prior = prior_for(history, ref.circuit_key, ref.start_utc)
    ctx = Context.for_race(prior, log.meta, history=history, race_start_utc=ref.start_utc)
    rows, final = decision_rows(log, prior, ctx=ctx)
    df = pd.DataFrame(add_labels(rows, final))
    df.insert(0, "race_id", ref.slug)
    df.insert(1, "year", ref.year)
    df.insert(2, "round", ref.round_index)
    df.insert(3, "circuit_key", ref.circuit_key)
    df.insert(4, "start_utc", ref.start_utc.isoformat())
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ref.slug}.parquet"
    df.to_parquet(path, index=False)
    return path


def _rows_job(args):
    ref, out = args
    from ..config import bench_dir
    from . import history as H

    return str(build_rr_rows(ref, H.load(bench_dir() / "history.json"), Path(out)))


def load_rr_bench(years, jobs: int = 3, refresh: bool = False) -> pd.DataFrame:
    from multiprocessing import Pool

    from ..config import bench_dir

    out = bench_dir() / RR_VERSION
    todo = []
    for r in radio_refs(years):
        if _has_radio(r) and (refresh or not (out / f"{r.slug}.parquet").exists()):
            todo.append((r, str(out)))
    if todo:
        if jobs <= 1:
            list(map(_rows_job, todo))
        else:
            with Pool(min(jobs, 3)) as p:
                list(p.imap(_rows_job, todo))
    files = [out / f"{r.slug}.parquet" for r in radio_refs(years) if _has_radio(r)]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    return df.sort_values(["start_utc", "t", "driver"]).reset_index(drop=True)


def ablation(df: pd.DataFrame, test_year: int, min_train: int, horizons=(1, 3)) -> list[dict]:
    from .evaluate import evaluate_task
    from .tasks import Task, lap_end_rows
    from .models import GBMHazard

    out = []
    for h in horizons:
        task = Task(f"pit_within_{h}", "binary", f"y_pit_{h}", lap_end_rows, (GBMHazard, RRGBMWithout, RRGBMWith))
        res = evaluate_task(task, df, test_year=test_year, min_train_races=min_train)
        b = res.leaderboard.set_index("model")
        pr = res.per_race.set_index("race_id")
        d = pr["rr_gbm_without"] - pr["rr_gbm_with"]  # positive: radio helped
        for m in ("rr_gbm_without", "rr_gbm_with"):
            out.append({"task": task.name, "model": m, "log_loss": b.loc[m, "log_loss"], "brier": b.loc[m, "brier"],
                        "auc": b.loc[m].get("auc", float("nan"))})
        out.append({"task": task.name, "model": "gain (without - with)", "log_loss": float(d.mean()),
                    "brier": float("nan"), "auc": float("nan"), "races_better": int((d > 0).sum()), "races": len(d)})
    return out


# ----------------------------------------------------------------------------- command
def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def cmd_eval(a) -> None:
    tune, test = list(a.tune_year), list(a.year)
    recs = load_records(tune + test, jobs=a.jobs, refresh=a.refresh)
    rt = [r for r in recs if r["year"] in tune]
    rs = [r for r in recs if r["year"] in test]
    cov = coverage(recs)
    print(f"coverage ({len(recs)} races with radio; clips per race median {cov.clips.median():.0f}, "
          f"races with no radio at all are not in the list)")
    print(cov.groupby("year").agg(races=("clips", "size"), clips=("clips", "sum"), stops=("stops", "sum"),
                                  stops_with_clip=("stops_with_any_clip_2laps", "sum"), box_signals=("box_signals", "sum")).to_string())
    print(f"base rate (random car-lap stops within {WITHIN} laps): tune {base_rate(rt):.3f}" + (f", test {base_rate(rs):.3f}" if rs else ""))
    grid = pd.DataFrame([score_events(rt, th) for th in THETAS])
    print(f"\ntune {tune}:\n" + grid.round(3).to_string(index=False))
    best = float(grid.sort_values(["f1", "theta"], ascending=[False, False]).iloc[0].theta)
    print(f"chosen theta = {best}")
    if rs:
        print(f"\ntest {test} (scored once):\n" + pd.DataFrame([score_events(rs, best)]).round(3).to_string(index=False))
        print("extend signals (>=0.6) test:", score_extend(rs), " tune:", score_extend(rt))
    if a.no_ablation:
        return
    df = load_rr_bench(tune + test, jobs=a.jobs, refresh=a.refresh)
    print(f"\nablation rows: {len(df)} over {df.race_id.nunique()} radio races; rows with a known clip for the car: "
          f"{int((df['rivalradio__rr_n'] > 0).sum())}")
    for y, mt in ((tune[0], 8), (test[0] if test else None, 10)):
        if y is None:
            continue
        print(f"\npit_within ablation, test year {y}:")
        print(pd.DataFrame(ablation(df, y, mt)).round(4).to_string(index=False))


def cmd_leak(a) -> None:
    from .. import archive
    from ..events import load_archive_session
    from . import leakcheck
    from ..cli import _prior_for

    ref = archive.find_session(a.year, a.race)
    log = load_archive_session(ref.local_dir, feeds=True)
    prior = _prior_for(ref)
    failed = False
    for cut in leakcheck.random_cuts(log, a.cuts, seed=a.seed):
        for mode in ("truncate", "scramble"):
            rep = leakcheck.check(log, cut, prior, mode=mode, seed=a.seed)
            failed |= not rep.ok
            print(f"{'PASS' if rep.ok else 'FAIL'}  {mode:<8} cut t={cut:8.1f}s rows {rep.rows_checked:5d} mismatches {len(rep.mismatches)}")
            for m in rep.mismatches[:5]:
                print("   ", m)
    sys.exit(1 if failed else 0)


def add_commands(sub) -> None:
    s = sub.add_parser("rivalradio-eval", help="rival radio: box-intent precision/recall/lead and the pit_within ablation")
    s.add_argument("--tune-year", type=int, nargs="+", default=[2025])
    s.add_argument("--year", type=int, nargs="*", default=[2026])
    s.add_argument("--jobs", type=int, default=3)
    s.add_argument("--refresh", action="store_true")
    s.add_argument("--no-ablation", action="store_true")
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("rivalradio-leak", help="corrupted-future test with the radio feed loaded")
    s.add_argument("--year", type=int, default=2026)
    s.add_argument("--race", default="hungary")
    s.add_argument("--cuts", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_leak)
