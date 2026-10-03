"""Benchmark for the strategy engineer and the head of strategy (reads finished races on purpose).

Three tasks, each with its own ``evaluate`` (they replay races and run the simulator):

* ``strategy_finish``: finishing-position forecast at 25 / 50 / 75 % distance for the cars that finish
  (RPS, MAE of the expected position, calibration of top-3 / top-10) against the current position and a
  pace extrapolation.
* ``strategy_nextstop``: laps until the next stop, against the rivals hazard.
* ``strategy_calls``: the head's BOX / PREPARE_BOX calls for the race's top-5 finishers against their real
  stops (precision, recall, BOX with no stop after it), scored with the runtime's ``shadow_score``.

One replay per race serves all three; results are cached in ``data/scratch`` per year and setting.
"""

from __future__ import annotations

import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..archive import downloaded_sessions
from ..config import bench_dir, data_dir
from ..events import load_archive_session
from ..pitwall.engineer import Context
from ..pitwall.engineers.strategy import analysis
from ..pitwall.engineers.strategy.run import simulate_field
from ..pitwall.engineers.strategy.sim import NOSTOP, Draws
from ..pitwall.runtime import shadow_score
from ..pitwall.types import TeamConfig
from ..pitwall.wall import PitWall
from ..state import RaceState
from .evaluate import EvalResult
from .metrics import CAL_BINS, calibration_table
from .tasks import Task

FRACS = (0.25, 0.5, 0.75)
NPOS = 24


def _rps(pmf: np.ndarray, y: int) -> float:
    """Ranked probability score over finishing positions (lower is better)."""
    cdf = np.cumsum(pmf)
    obs = (np.arange(1, len(pmf) + 1) >= y).astype(float)
    return float(np.sum((cdf - obs) ** 2) / (len(pmf) - 1))


def _onehot(pos: float, n: int = NPOS) -> np.ndarray:
    p = np.zeros(n)
    p[int(np.clip(round(pos), 1, n)) - 1] = 1.0
    return p


def _rivals_next(rv: dict, A: int, stint_age: float) -> float:
    p1, p3, p5 = rv.get("pit_prob_1"), rv.get("pit_prob_3"), rv.get("pit_prob_5")
    if None in (p1, p3, p5):
        return A + 8.0
    if p1 >= 0.5:
        return A + 1.0
    if p3 >= 0.5:
        return A + 1 + 2 * (0.5 - p1) / max(p3 - p1, 1e-6)
    if p5 >= 0.5:
        return A + 3 + 2 * (0.5 - p3) / max(p5 - p3, 1e-6)
    typ = rv.get("typical_stint_laps") or 25.0
    return A + max(6.0, typ - stint_age)


def run_race(args) -> dict:
    """Replay one race: forecasts at the anchors, head calls for the top-5 finishers."""
    slug, year, stride, S, calls_on = args
    ref = next(r for r in downloaded_sessions() if r.slug == slug)
    from . import history as H

    hist = H.load(bench_dir() / "history.json")
    prior = H.prior_for(hist, ref.circuit_key, ref.start_utc)
    log = load_archive_session(ref.local_dir)
    final = RaceState(log.meta)
    for e in log.events:
        final.apply(e)
    total = final.total_laps or 0
    classified = [d for d in final.running_order() if d.running and d.laps >= 0.9 * total]
    y_pos = {d.number: d.position for d in classified}
    top5 = tuple(d.number for d in classified[:5])
    ctx = Context.for_race(prior, ref.to_dict(), history=hist, race_start_utc=ref.start_utc,
                           team=TeamConfig(cars=top5 if calls_on else ()))
    wall = PitWall(ctx)
    state = RaceState(log.meta)
    stops: dict[str, list[int]] = {}
    for pe in final.pit_events:
        if not pe.under_red:
            stops.setdefault(pe.driver, []).append(pe.in_lap)
    anchors = [int(round(f * total)) for f in FRACS]
    done: set[int] = set()
    fc_rows, ns_rows, calls, last_key = [], [], [], {}
    t_calls, n_calls = 0.0, 0
    for e in log.events:
        state.apply(e)
        wall.observe(state)
        for A in anchors:
            if A in done or not total:
                continue
            run_cars = [d for d in state.drivers.values() if d.running and d.position is not None]
            if run_cars and min(d.laps for d in run_cars) >= A:
                done.add(A)
                _forecast(wall, ctx, state, A, S, y_pos, stops, total, fc_rows, ns_rows, args)
        if calls_on:
            for lap in state.new_laps:
                if lap.driver in top5 and lap.lap >= 3 and lap.lap % stride == 0 and lap.lap < total - 1:
                    t0 = time.perf_counter()
                    out = wall.calls(state)
                    t_calls += time.perf_counter() - t0
                    n_calls += 1
                    for c in out:
                        if c.car != lap.driver:
                            continue
                        key = (c.action, c.compound)
                        if last_key.get(c.car) != key and not (c.action == "NO_CALL" and c.car not in last_key):
                            last_key[c.car] = key
                            calls.append({"kind": "call", "t": c.t, "lap": state.current_lap, "car_lap": lap.lap + 1,
                                          "car": c.car, "action": c.action, "compound": c.compound,
                                          "confidence": c.confidence})
    res = {"slug": slug, "year": year, "top5": top5, "fc": fc_rows, "ns": ns_rows, "calls": calls,
           "stops": {k: stops.get(k, []) for k in top5}, "calls_s": t_calls / max(n_calls, 1)}
    if calls_on:
        res["shadow"] = shadow_score(calls, final, 2, set(top5))
        out_dir = data_dir() / "scratch" / "calls"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"calls-{slug}.jsonl").write_text("\n".join(json.dumps(c) for c in calls), encoding="utf-8")
    return res


def _forecast(wall, ctx, state, A, S, y_pos, stops, total, fc_rows, ns_rows, args) -> None:
    view = wall.view(state)
    pri = analysis.priors_for(ctx)
    F, names, info = analysis.build_field(state, view, wall.memory, ctx, pri, A)
    if F.R <= 0 or F.C < 3:
        return
    D = Draws(F, S, analysis.seed_for(ctx, "", 0))
    run = simulate_field(F, D)
    now_order = {n: i + 1 for i, n in enumerate(sorted(names, key=lambda n: F.x0[names.index(n)]))}
    pace_fin = F.x0 + F.R * F.pace0
    pace_order = {n: int(np.sum(pace_fin < pace_fin[i])) + 1 for i, n in enumerate(names)}
    for i, n in enumerate(names):
        if n not in y_pos:
            continue
        pmf = np.bincount(run.pos[:, i].astype(int).clip(1, NPOS), minlength=NPOS + 1)[1:] / D.S
        y = int(y_pos[n])
        fc_rows.append({"anchor": A, "frac": A / total, "car": n, "y": y, "exp": float((pmf * np.arange(1, NPOS + 1)).sum()),
                        "p3": float(pmf[:3].sum()), "p10": float(pmf[:10].sum()), "rps_sim": _rps(pmf, y),
                        "rps_now": _rps(_onehot(now_order[n]), y), "rps_pace": _rps(_onehot(pace_order[n]), y),
                        "now": now_order[n], "pace": pace_order[n]})
        nxt = [l for l in stops.get(n, []) if l > A]
        if nxt:
            s0 = run.stop[:, i, 0]
            ok = s0 < NOSTOP
            pred = float(np.median(s0[ok])) if ok.any() else float("nan")
            rv = view.car("rivals", n)
            base = _rivals_next(rv, A, F.stint_age[i])
            ns_rows.append({"anchor": A, "car": n, "y": nxt[0], "sim": pred, "rivals": base, "p_stop": float(ok.mean())})


# --------------------------------------------------------------------------- scoring
def _races(year: int):
    return [r for r in downloaded_sessions() if r.year == year and r.session_name == "Race"]


def collect(year: int, *, jobs: int = 3, stride: int = 2, S: int = 192, calls_on: bool = True, refresh: bool = False) -> list[dict]:
    cache = data_dir() / "scratch" / f"strategy_{year}_{stride}_{S}_{int(calls_on)}.pkl"
    if cache.exists() and not refresh:
        return pickle.loads(cache.read_bytes())
    jobs_ = [(r.slug, year, stride, S, calls_on) for r in _races(year)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        out = list(ex.map(run_race, jobs_))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def _frame(results: list[dict], key: str) -> pd.DataFrame:
    rows = [dict(r, race_id=res["slug"]) for res in results for r in res[key]]
    return pd.DataFrame(rows)


def _finish(df, test_year, min_train_races, **kw) -> EvalResult:
    res = collect(test_year, **kw)
    d = _frame(res, "fc")
    board = []
    for name, col in (("sim", "rps_sim"), ("current_position", "rps_now"), ("pace_extrapolation", "rps_pace")):
        exp = {"sim": d.exp, "current_position": d.now, "pace_extrapolation": d.pace}[name]
        row = {"model": name, "n": len(d), "rps": d[col].mean(), "mae": (exp - d.y).abs().mean()}
        for f in FRACS:
            m = np.isclose(d.frac, f, atol=0.02)
            row[f"mae@{int(f * 100)}"] = (exp[m] - d.y[m]).abs().mean()
        board.append(row)
    board = pd.DataFrame(board).sort_values("rps").reset_index(drop=True)
    per = d.groupby("race_id").apply(lambda g: pd.Series({"sim": g.rps_sim.mean(), "current_position": g.rps_now.mean(),
                                                          "pace_extrapolation": g.rps_pace.mean()}), include_groups=False).reset_index()
    cal = {"sim": calibration_table((d.y <= 10).astype(float).to_numpy(), d.p10.to_numpy())}  # P(top 10)
    return EvalResult("strategy_finish", board, per, d, cal)


def _nextstop(df, test_year, min_train_races, **kw) -> EvalResult:
    res = collect(test_year, **kw)
    d = _frame(res, "ns").dropna(subset=["sim"])
    board = pd.DataFrame([{"model": n, "n": len(d), "mae_laps": (d[c] - d.y).abs().mean(),
                           "within_2": float(((d[c] - d.y).abs() <= 2).mean()), "bias": (d[c] - d.y).mean()}
                          for n, c in (("sim", "sim"), ("rivals_hazard", "rivals"))]).sort_values("mae_laps").reset_index(drop=True)
    per = d.groupby("race_id").apply(lambda g: pd.Series({"sim": (g.sim - g.y).abs().mean(), "rivals_hazard": (g.rivals - g.y).abs().mean()}),
                                     include_groups=False).reset_index()
    return EvalResult("strategy_nextstop", board, per, d, {})


def _calls(df, test_year, min_train_races, **kw) -> EvalResult:
    res = collect(test_year, **kw)
    rows = []
    for r in res:
        sh = r.get("shadow") or {}
        stops = r["stops"]
        box = [c for c in r["calls"] if c["action"] == "BOX"]
        no_stop = [not any(0 <= s - c["car_lap"] <= 2 for s in stops.get(c["car"], [])) for c in box]
        rows.append({"race_id": r["slug"], "box_calls": sh.get("box_calls", 0), "precision": sh.get("box_precision"),
                     "recall": sh.get("stop_recall"), "real_stops": sh.get("real_stops", 0),
                     "box_no_stop": float(np.mean(no_stop)) if no_stop else None, "n_box": len(box),
                     "stay_out_acc": sh.get("stay_out_accuracy"), "s_per_call": r["calls_s"]})
    per = pd.DataFrame(rows)

    def wavg(col, w):
        m = per[col].notna()
        return float((per[col][m] * per[w][m]).sum() / max(per[w][m].sum(), 1))

    board = pd.DataFrame([{"model": "head", "races": len(per), "box_calls": int(per.box_calls.sum()), "real_stops": int(per.real_stops.sum()),
                           "precision": wavg("precision", "box_calls"), "recall": wavg("recall", "real_stops"),
                           "box_no_stop": wavg("box_no_stop", "n_box"), "stay_out_acc": float(per.stay_out_acc.mean()),
                           "s_per_call": float(per.s_per_call.mean())}])
    return EvalResult("strategy_calls", board, per, per, {})


def strategy_tasks(**_) -> list[Task]:
    def mk(name, fn):
        return Task(name, "regression", "y", lambda d: d, (), evaluate=lambda df, *, test_year, min_train_races, _f=fn: _f(df, test_year, min_train_races))

    return [mk("strategy_finish", _finish), mk("strategy_nextstop", _nextstop), mk("strategy_calls", _calls)]


# --------------------------------------------------------------------------- leak test
def leakcheck(year: int, race: str, cuts: int = 3, seed: int = 0, cars: tuple = ()) -> list[dict]:
    """Plans and calls at time t must be identical built from the full log and from ``log.until(t)``.

    The full-log pit wall is fed the whole log but asked at t (every event up to t applied, none after);
    the other one is fed ``log.until(t)`` only. Both are asked once, at t.
    """
    from .. import archive
    from . import history as H
    from .leakcheck import random_cuts

    ref = archive.find_session(year, race)
    log = load_archive_session(ref.local_dir)
    hist = H.load(bench_dir() / "history.json")
    prior = H.prior_for(hist, ref.circuit_key, ref.start_utc)
    out = []
    for cut in random_cuts(log, cuts, seed):
        sides = []
        for source in (log, log.until(cut)):
            top = tuple(cars) or ("1", "16") 
            ctx = Context.for_race(prior, ref.to_dict(), history=hist, race_start_utc=ref.start_utc, team=TeamConfig(cars=top))
            wall, state = PitWall(ctx), RaceState(source.meta)
            for e in source.events:
                if e.t > cut:
                    break
                state.apply(e)
                wall.observe(state)
            sides.append([(c.car, c.action, c.compound, c.confidence, c.plan_a, c.plan_b, c.reasons) for c in wall.calls(state)])
        out.append({"cut": cut, "calls": len(sides[0]), "same": sides[0] == sides[1]})
    return out


def _cmd_leak(a) -> None:
    import sys

    res = leakcheck(a.year, a.race, a.cuts, a.seed, tuple(a.cars or ()))
    for r in res:
        print(("PASS" if r["same"] and r["calls"] else "FAIL"), f"cut t={r['cut']:8.1f}s  calls compared {r['calls']}")
    sys.exit(0 if all(r["same"] and r["calls"] for r in res) else 1)


def add_commands(sub) -> None:
    s = sub.add_parser("strategy-leakcheck", help="plans and calls at t: full log vs log.until(t)")
    s.add_argument("--year", type=int, default=2026)
    s.add_argument("--race", required=True)
    s.add_argument("--cuts", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--cars", nargs="+")
    s.set_defaults(fn=_cmd_leak)
