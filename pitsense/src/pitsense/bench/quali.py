"""Qualifying benchmark: the knockout cut time and who gets knocked out, at every moment.

Moments: every 20 s of a Q1/Q2 (SQ1/SQ2) part while its clock runs. Rows are replayed from the
archive streams with the same code the pit wall runs (``pitsense.quali``), so nothing later than the
moment is read; the labels (final cut time, final rank) are taken when the part ends.

Tasks
    cut time   MAE (s) of the predicted final cut time. Baselines: ``now`` (the cut time now, or the
               slowest time set while fewer cars than the cut have one) and ``now+mean`` (plus the
               mean 2025 improvement).
    knockout   log loss of P(car ends the part beyond the cut line). Baselines: ``prior`` (share
               of cars knocked out) and ``rank`` (0.8 / 0.2 by being in the drop zone now, the 2025 best).

Protocol: tune on 2025 (leave-one-session-out), fit on all of 2025, score 2026 once.
    pitsense quali-bench --out reports
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import quali as Q
from ..config import bench_dir
from ..events import load_archive_session
from ..state import RaceState
from ..weekend import quali_sessions

STEP_S = 20.0
CLIP = 1e-3


def collect_events(log, name: str, year: int, sprint: bool) -> tuple[list, list]:
    """(cut rows, knockout rows) of one session's event log; labels are filled when each part ends."""
    state, tr = RaceState(log.meta), Q.QualiTracker()
    cut_rows: list = []
    ko_rows: list = []
    mine_c: list = []
    mine_k: list = []
    cur_part, nxt = None, 0.0

    def finish() -> None:
        pic = Q.read(state, tr)
        if pic is not None and pic.cut_time is not None and pic.cut_pos is not None:
            for r in mine_c:
                r["y_cut"] = pic.cut_time
            for r in mine_k:
                c = pic.cars.get(r["car"])
                if c is not None:
                    r["y_ko"] = int(c.in_zone)
            cut_rows.extend(mine_c)
            ko_rows.extend(r for r in mine_k if "y_ko" in r)
        mine_c.clear()
        mine_k.clear()

    for e in log.events:
        if e.topic == "TimingData" and isinstance(e.data, dict) and "SessionPart" in e.data:
            new = int(e.data["SessionPart"])
            if cur_part is not None and new != cur_part:
                finish()  # the state still holds the part that just ended
            cur_part = new
        state.apply(e)
        tr.observe(state)
        if state.t < nxt or not state.started:
            continue
        pic = Q.read(state, tr)
        if pic is None or not pic.running or pic.cut_pos is None or not pic.part_len or not pic.time_left:
            continue
        nxt = state.t + STEP_S
        tr.sample_cut(pic.part, state.t, pic.cut_time)
        a, kind = Q.anchor(pic)
        if a is None:
            continue
        base = {"session": name, "year": year, "sprint": sprint, "t": round(state.t, 1), "part": pic.part,
                "anchor": a, "kind": kind, "tl": pic.time_left}
        mine_c.append({**base, **Q.cut_features(pic, tr, state.t)})
        for n, c in pic.cars.items():
            mine_k.append({**base, "car": n, "best": c.best, "rank": c.rank, "in_pit": c.in_pit,
                           "laps": c.laps_in_part, "cut_pos": pic.cut_pos, "n_elig": len(pic.order),
                           "in_zone": c.in_zone})
    return cut_rows, ko_rows


def collect(ref) -> tuple[list, list]:
    d = ref.local_dir
    topics = tuple(f.stem for f in d.glob("*.jsonStream") if f.stem not in ("CarData.z", "Position.z", "TeamRadio"))
    return collect_events(load_archive_session(d, topics), ref.slug, ref.year, ref.session_name == "Sprint Qualifying")


def collect_year(year: int, refresh: bool = False) -> tuple[list, list]:
    cache = bench_dir() / f"quali_rows_{year}.json"
    if cache.exists() and not refresh:
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["cut"], d["ko"]
    cut, ko = [], []
    for ref in quali_sessions(year):
        if not (ref.local_dir / "TimingData.jsonStream").exists():
            continue
        c, k = collect(ref)
        cut += c
        ko += k
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"cut": cut, "ko": ko}), encoding="utf-8")
    return cut, ko


# --------------------------------------------------------------------------- cut-time models
def _key(r: dict, by_sprint: bool) -> str:
    return f"{r['part']}{'s' if (by_sprint and r['sprint']) else ''}"


def _cell(r: dict) -> str:
    return f"{Q.time_bin(r['tl'])}{'k' if r['kind'] else ''}"


def fit_table(rows: list, by_sprint: bool, stat: str = "mean", min_n: int = 8) -> dict:
    """Mean (or median) of final cut - anchor per (part[, sprint]) x (time-left bin, anchor kind).

    Cells with fewer than ``min_n`` rows are left out (the prediction then adds 0).
    """
    cells: dict = defaultdict(list)
    for r in rows:
        cells[(_key(r, by_sprint), _cell(r))].append(r["y_cut"] - r["anchor"])
    f = np.mean if stat == "mean" else np.median
    out: dict = defaultdict(dict)
    for (k, c), v in sorted(cells.items()):
        if len(v) >= min_n:
            out[k][c] = round(float(f(v)), 4)
    return dict(out)


def predict_rows(rows: list, table: dict | None, by_sprint: bool) -> np.ndarray:
    out = []
    for r in rows:
        delta = 0.0
        if table is not None:
            tab = table.get(_key(r, by_sprint)) or table.get(str(r["part"])) or {}
            delta = tab.get(_cell(r), 0.0)
        out.append(r["anchor"] + delta)
    return np.array(out)


def mae(rows: list, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array([r["y_cut"] for r in rows]) - pred)))


def loso_cut(rows: list, by_sprint: bool, stat: str) -> float:
    err = []
    for s in sorted({r["session"] for r in rows}):
        tr = [r for r in rows if r["session"] != s]
        te = [r for r in rows if r["session"] == s]
        err.append(np.abs(np.array([r["y_cut"] for r in te]) - predict_rows(te, fit_table(tr, by_sprint, stat), by_sprint)))
    return float(np.mean(np.concatenate(err)))


# --------------------------------------------------------------------------- knockout models
def ko_matrix(rows: list, preds: np.ndarray) -> np.ndarray:
    return np.array([Q.ko_vec(r["best"], r["rank"], r["in_pit"], r["laps"], r["tl"], r["cut_pos"], r["n_elig"], p)
                     for r, p in zip(rows, preds)])


def logloss(y, p) -> float:
    p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logit(X, y, c: float):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=c, max_iter=2000).fit(X, y)


def _baselines(rows: list, prior: float, eps: float) -> dict:
    return {"prior": np.full(len(rows), prior),
            "rank": np.array([1 - eps if r["in_zone"] else eps for r in rows])}


TIME_BUCKETS = (("<1 min", 0, 60), ("1-4 min", 60, 240), (">4 min", 240, 1e9))


def run(out: str | None = "reports", write_model: bool = True, refresh: bool = False) -> dict:
    c25, k25 = collect_year(2025, refresh)
    # ---- tune on 2025 only
    grid = {(bs, st): loso_cut(c25, bs, st) for bs in (False, True) for st in ("mean", "median")}
    best_cfg = min(grid, key=grid.get)
    table = fit_table(c25, *best_cfg)
    X25 = ko_matrix(k25, predict_rows(k25, table, best_cfg[0]))
    y25 = np.array([r["y_ko"] for r in k25])
    groups = np.array([r["session"] for r in k25])
    sessions = sorted(set(groups))
    folds = [sessions[i::5] for i in range(5)]
    cv = {}
    for C in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        tot = 0.0
        for f in folds:
            te = np.isin(groups, f)
            m = fit_logit(X25[~te], y25[~te], C)
            tot += logloss(y25[te], m.predict_proba(X25[te])[:, 1]) * te.sum()
        cv[C] = tot / len(y25)
    bestC = min(cv, key=cv.get)
    ko_model = fit_logit(X25, y25, bestC)
    prior = float(y25.mean())
    eps = min((0.02, 0.05, 0.1, 0.15, 0.2), key=lambda e: logloss(y25, _baselines(k25, prior, e)["rank"]))
    mean_delta = float(np.mean([r["y_cut"] - r["anchor"] for r in c25]))
    model = {"evolution": table,
             "ko": {"intercept": float(ko_model.intercept_[0]), "coef": [float(x) for x in ko_model.coef_[0]]},
             "config": {"by_sprint": best_cfg[0], "stat": best_cfg[1], "C": bestC, "rank_eps": eps, "prior": prior,
                        "mean_delta": mean_delta, "fit": "2025 sessions", "step_s": STEP_S}}
    if write_model:
        Q.MODEL_FILE.write_text(json.dumps(model, indent=1, sort_keys=True), encoding="utf-8")
        Q._MODEL_CACHE.clear()
    # ---- score 2026, once
    c26, k26 = collect_year(2026, refresh)
    res = {"tuning_2025": {"cut_loso_mae": {f"by_sprint={a},{b}": round(v, 4) for (a, b), v in grid.items()},
                           "ko_cv_logloss": {str(k): round(v, 4) for k, v in cv.items()},
                           "chosen": model["config"], "rows": {"cut": len(c25), "ko": len(k25)}},
           "rows_2026": {"cut": len(c26), "ko": len(k26), "sessions": len({r["session"] for r in c26})}}
    now = predict_rows(c26, None, False)
    cut = {"now": now, "now+mean": now + mean_delta, "evolution table": predict_rows(c26, table, best_cfg[0])}
    res["cut_mae_s"] = {k: round(mae(c26, v), 4) for k, v in cut.items()}
    full = [i for i, r in enumerate(c26) if r["kind"] == 0]
    res["cut_mae_s_cut_known"] = {k: round(mae([c26[i] for i in full], v[full]), 4) for k, v in cut.items()} | {"n": len(full)}
    res["cut_mae_by_time_left"] = {}
    for name, lo, hi in TIME_BUCKETS:
        idx = [i for i, r in enumerate(c26) if lo <= r["tl"] < hi]
        res["cut_mae_by_time_left"][name] = {k: round(mae([c26[i] for i in idx], v[idx]), 4) for k, v in cut.items()} | {"n": len(idx)}
    y26 = np.array([r["y_ko"] for r in k26])
    ko = _baselines(k26, prior, eps)
    ko["logit"] = ko_model.predict_proba(ko_matrix(k26, predict_rows(k26, table, best_cfg[0])))[:, 1]
    res["ko_logloss"] = {k: round(logloss(y26, v), 4) for k, v in ko.items()}
    res["ko_logloss_by_time_left"] = {}
    for name, lo, hi in TIME_BUCKETS:
        idx = [i for i, r in enumerate(k26) if lo <= r["tl"] < hi]
        res["ko_logloss_by_time_left"][name] = {k: round(logloss(y26[idx], v[idx]), 4) for k, v in ko.items()} | {"n": len(idx)}
    if out:
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "quali_bench.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def cmd_quali_bench(a) -> None:
    print(json.dumps(run(a.out, write_model=not a.no_write, refresh=a.refresh), indent=1))


def add_commands(sub) -> None:
    s = sub.add_parser("quali-bench", help="qualifying benchmark: cut time MAE and knockout log loss (tune 2025, score 2026)")
    s.add_argument("--out", default="reports")
    s.add_argument("--refresh", action="store_true", help="rebuild the cached rows")
    s.add_argument("--no-write", action="store_true", help="do not write src/pitsense/quali_model.json")
    s.set_defaults(fn=cmd_quali_bench)
