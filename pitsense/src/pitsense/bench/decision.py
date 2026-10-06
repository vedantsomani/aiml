"""Decision value of the head's calls: a counterfactual **simulator estimate**, not an observed outcome.

Matching what the team did (``callscore``) measures prediction. Whether a call was *worth* making needs a
counterfactual, and the only one available is the strategy simulator. At every decision moment (a focus car
completes an even lap, as in the strategy benchmark) the simulator is run on the as-of state and these options
are scored on the **same fresh futures** (288, seeded from the race with the tag ``dv``, independent of the
futures the plans were chosen on, so the chosen plan gets no winner's-curse bonus):

* ``now``: the best plan (by the simulator's utility) whose first stop is this lap;
* ``soon``: the best plan stopping in 1-2 laps; ``later``: the best stopping in 3+ laps or never;
* ``stay``: the best plan not stopping this lap (``soon`` or ``later``);
* ``sc``: the BOX_IF_SC policy: ``stay``, but in a future with an SC / VSC starting within 5 laps the first stop
  moves to that lap (tyre and second stop as the strategy engineer's Plan B builds them);
* ``team``: what the team really did from here (its real remaining in-laps and tyres, red-flag stops excluded),
  as a fixed plan. Known only after the race: it is the action being evaluated, never an input to the futures.

Our action maps to an option (BOX -> now, PREPARE_BOX -> soon, STAY_OUT -> stay, BOX_IF_SC -> sc) and its
alternative (BOX <-> stay, PREPARE_BOX -> now, BOX_IF_SC -> stay). Per option: expected finishing position,
expected points, P(lose >= 2 places) against the position at the decision. Regret = our option's expected
position minus the best option's (and best points minus ours). Every simulated plan is open-loop (it does not
react to the simulated race, the team's real choices did), so ``team`` is a slightly pessimistic stand-in.

How far to trust the simulator: ``calibration`` scores the ``team`` option's distribution against the position
the car really finished (the one plan that was really driven): RPS against the current position, bias, PIT
histogram, 80 % interval coverage, points bias.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..config import bench_dir, data_dir
from ..pitwall.engineers.strategy import analysis
from ..pitwall.engineers.strategy.run import evaluate_plans, simulate_field
from ..pitwall.engineers.strategy.sim import K_STOPS, NOSTOP, Draws
from . import callscore

OPTIONS = ("now", "soon", "later", "stay", "sc", "team")
OURS = {"BOX": "now", "PREPARE_BOX": "soon", "STAY_OUT": "stay", "BOX_IF_SC": "sc"}
ALT = {"BOX": "stay", "PREPARE_BOX": "now", "STAY_OUT": "now", "BOX_IF_SC": "stay"}
BEST_OF = ("now", "soon", "later", "sc", "team")
SC_WINDOW = 5
STRIDE = 2
N_FOCUS = 4  # grid top 4: at most 4 focus cars keeps the simulator in full mode (as for a team's two cars)
VERSION = 1


# --------------------------------------------------------------------------- options at one moment
def option_plans(a) -> dict:
    """Option name -> stops ((in-lap, compound index), ...) from a CarAnalysis' ranked plans (best first)."""
    def first(pred):
        r = next((r for r in a.ranked if pred(r.first_offset)), None)
        return None if r is None else tuple((int(l), analysis.CIDX[c]) for l, c in r.stops)

    return {"now": first(lambda o: o == 0), "soon": first(lambda o: o is not None and 1 <= o <= 2),
            "later": first(lambda o: o is None or o >= 3), "stay": first(lambda o: o is None or o >= 1)}


def team_plan(stops: list, A: int, total: int) -> tuple | None:
    """The team's real stops after lap A as a plan; None if one of them fitted a wet-weather tyre."""
    out = []
    for lap, cj in stops:
        if lap < A + 1 or lap > total - 1:
            continue
        if cj is None:
            return None
        out.append((int(lap), int(cj)))
    return tuple(out[:K_STOPS])


def sc_policy(base: tuple, status: np.ndarray, A: int, total: int, life: np.ndarray, comp: int) -> tuple[np.ndarray, np.ndarray]:
    """[S,K] stop laps and compounds for "stay on ``base``; if an SC/VSC starts within SC_WINDOW laps, box that lap"."""
    S = status.shape[0]
    stop = np.full((S, K_STOPS), NOSTOP, dtype=np.int64)
    cmp_ = np.zeros((S, K_STOPS), dtype=np.int64)
    for k, (l, cj) in enumerate(base[:K_STOPS]):
        stop[:, k], cmp_[:, k] = l, cj
    w = min(SC_WINDOW, status.shape[1])
    on = status[:, :w] > 0
    for s in np.nonzero(on.any(1))[0]:
        l0 = A + 1 + int(on[s].argmax())
        if l0 > total - 3 or (base and base[0][0] <= l0):
            continue  # the plan already stops by then
        stop[s], cmp_[s] = NOSTOP, 0
        stop[s, 0], cmp_[s, 0] = l0, comp
        rem = total - l0
        gap = max(8, int(0.9 * life[comp]))
        second = min(l0 + gap, total - 3)
        if rem > 1.1 * life[comp] and second > l0 + 5:
            stop[s, 1], cmp_[s, 1] = second, (2 if rem - gap > life[1] else 1)
    return stop, cmp_


def _plan_arrays(plan: tuple, S: int):
    st, cp = analysis._arrays([plan], S)
    return st[0], cp[0]


def option_values(pos: np.ndarray, cur: int) -> dict:
    """Expected position, expected points, P(lose >= 2 places vs the position now) from simulated finishes."""
    return {"exp_pos": float(pos.mean()), "exp_pts": float(analysis._points(pos).mean()),
            "p_lose2": float(np.mean(pos >= cur + 2)), "pos": pos.astype(np.uint8).tobytes()}


def evaluate_moment(F, D2, run2, c: int, a, team: tuple | None) -> dict:
    """Every option of car index ``c`` on the fresh futures (D2, run2)."""
    plans = option_plans(a)
    plans["team"] = team
    S = D2.S
    stops, comps, names = [], [], []
    for n in ("now", "soon", "later", "stay", "team"):
        if plans.get(n) is not None:
            s_, c_ = _plan_arrays(plans[n], S)
            stops.append(s_), comps.append(c_), names.append(n)
    base = plans["stay"]
    if base is not None and F.R >= 6:
        comp = analysis.CIDX.get(a.sc_best_comp or "", base[0][1] if base else 1)
        s_, c_ = sc_policy(base, D2.status, F.A, F.total, F.life, comp)
        stops.append(s_), comps.append(c_), names.append("sc")
    if not names:
        return {}
    pos, _, _ = evaluate_plans(F, D2, run2, c, np.stack(stops), np.stack(comps))
    cur = 1 + int(np.sum(F.x0 < F.x0[c]))
    out = {n: dict(option_values(pos[i], cur), stops=plans.get(n) if n != "sc" else plans["stay"]) for i, n in enumerate(names)}
    out["_cur"] = cur
    return out


# --------------------------------------------------------------------------- the hook into the analysis
_HOOK: dict = {}


def _hooked(orig):
    def analyse_focus(F, info, run, D, number, ctx, S1, keep):
        a = orig(F, info, run, D, number, ctx, S1, keep)
        h = _HOOK
        if h and a.ok and a.ranked and F.A % STRIDE == 0 and number in h["truth"]:
            key = id(F)
            if h.get("field_key") != key:  # fresh futures, shared by the focus cars on this field
                D2 = Draws(F, analysis.SETTINGS["S"], analysis.seed_for(ctx, "", 0, "dv"))
                h["field"] = (D2, simulate_field(F, D2))
                h["field_key"], h["field_ref"] = key, F
            D2, run2 = h["field"]
            team = team_plan(h["truth"][number], F.A, F.total)
            h["caps"][(number, F.A)] = evaluate_moment(F, D2, run2, F.cars.index(number), a, team)
        return a

    return analyse_focus


def _install():
    if not getattr(analysis.analyse_focus, "_dv", False):
        f = _hooked(analysis.analyse_focus)
        f._dv = True
        analysis.analyse_focus = f


# --------------------------------------------------------------------------- one race
def _truth(final) -> dict:
    """car -> [(in-lap, dry compound index or None for a wet tyre)] from the finished race."""
    laps = {(x.driver, x.lap): x for x in final.laps}
    last = {}
    for x in final.laps:
        last[x.driver] = max(last.get(x.driver, 0), x.lap)
    ins: dict[str, list[int]] = {}
    for p in final.pit_events:
        if not p.under_red:
            ins.setdefault(p.driver, []).append(int(p.in_lap))
    out: dict[str, list] = {}
    for car, ls in ins.items():
        ls = sorted(ls)
        for i, l in enumerate(ls):
            # the stint's last lap: by then a late-confirmed tyre change is in the record
            end = (ls[i + 1] if i + 1 < len(ls) else last.get(car, l + 1))
            o = next((laps[(car, k)] for k in range(end, l, -1) if (car, k) in laps), None)
            out.setdefault(car, []).append((l, analysis.CIDX.get((o.compound if o else "") or "")))
    return out


def focus_from_grid(final, n: int = N_FOCUS) -> tuple:
    """The top ``n`` of the starting grid (a pre-race fact: no selection on the result)."""
    g = sorted((d for d in final.drivers.values() if d.grid), key=lambda d: d.grid)
    return tuple(d.number for d in g[:n])


def run_race(args) -> dict:
    """Replay one race through the pit wall (models bundle trained before the year) for the grid top 4."""
    slug, year = args[:2]
    from ..archive import downloaded_sessions
    from ..events import load_archive_session
    from ..pitwall.engineer import Context
    from ..pitwall.types import TeamConfig
    from ..pitwall.wall import PitWall
    from ..state import RaceState
    from . import history as H
    from .strategy import bundle_for_year

    ref = next(r for r in downloaded_sessions() if r.slug == slug)
    hist = H.load(bench_dir() / "history.json")
    prior = H.prior_for(hist, ref.circuit_key, ref.start_utc)
    log = load_archive_session(ref.local_dir)
    final = RaceState(log.meta)
    for e in log.events:
        final.apply(e)
    total = final.total_laps or 0
    focus = focus_from_grid(final)
    truth = _truth(final)
    finish = {d.number: d.position for d in final.running_order() if d.running and d.laps >= 0.9 * total}
    ctx = Context.for_race(prior, ref.to_dict(), history=hist, race_start_utc=ref.start_utc,
                           team=TeamConfig(cars=focus), models=bundle_for_year(year))
    _install()
    _HOOK.clear()
    _HOOK.update(truth={n: truth.get(n, []) for n in focus}, caps={})
    wall, state = PitWall(ctx), RaceState(log.meta)
    moments, calls, last_key = [], [], {}
    for e in log.events:
        state.apply(e)
        wall.observe(state)
        for lap in state.new_laps:
            if lap.driver not in focus or lap.lap < 3 or lap.lap % STRIDE or lap.lap >= total - 1:
                continue
            for c in wall.calls(state):
                if c.car != lap.driver:
                    continue
                key = (c.action, c.compound)
                if last_key.get(c.car) != key and not (c.action == "NO_CALL" and c.car not in last_key):
                    last_key[c.car] = key
                    calls.append({"kind": "call", "t": c.t, "lap": state.current_lap, "car_lap": lap.lap + 1, "car": c.car,
                                  "action": c.action, "compound": c.compound, "confidence": c.confidence})
                if c.action in OURS:
                    nxt = [l for l, _ in truth.get(c.car, []) if l >= lap.lap + 1]
                    moments.append({"race": slug, "car": c.car, "A": lap.lap, "car_lap": lap.lap + 1, "total": total,
                                    "action": c.action, "compound": c.compound, "track": state.track_status,
                                    "team_next": nxt[0] if nxt else None, "finish": finish.get(c.car),
                                    "dv": _HOOK["caps"].get((c.car, lap.lap))})
    _HOOK.clear()
    return {"slug": slug, "year": year, "circuit": ref.circuit_key, "focus": focus, "calls": calls, "moments": moments,
            "facts": callscore.race_facts(final)}


def settings_hash() -> str:
    from ..pitwall.engineers.strategy.sim import PARAMS

    blob = json.dumps({"S": analysis.SETTINGS, "P": PARAMS, "v": VERSION}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _cached_run(args) -> dict:
    slug, year, h, refresh = args
    path = data_dir() / "scratch" / "decision" / f"{slug}_{h}.pkl"
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes())
    res = run_race((slug, year))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(res))
    return res


def collect(year: int = 2026, *, jobs: int = 3, refresh: bool = False, races: list[str] | None = None) -> list[dict]:
    """One replay per race, cached per race and settings in ``data/scratch/decision`` (re-scoring is free)."""
    from ..archive import downloaded_sessions
    from .strategy import bundle_for_year

    refs = [r for r in downloaded_sessions() if r.year == year and r.session_name == "Race"]
    if races:
        refs = [r for r in refs if any(k in r.slug for k in races)]
    h = settings_hash()
    todo = [r for r in refs if refresh or not (data_dir() / "scratch" / "decision" / f"{r.slug}_{h}.pkl").exists()]
    if todo:
        bundle_for_year(year)  # train once here, not in every worker
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(_cached_run, [(r.slug, year, h, refresh) for r in refs]))


# --------------------------------------------------------------------------- scoring
def _pos(o: dict) -> np.ndarray:
    return np.frombuffer(o["pos"], dtype=np.uint8).astype(float)


def moment_metrics(m: dict) -> dict | None:
    """Regret and comparisons for one decision moment (None when the simulator had no dry plan)."""
    dv = m.get("dv") or {}
    ours, alt = dv.get(OURS[m["action"]]), dv.get(ALT[m["action"]])
    if ours is None:
        return None
    opts = {n: dv[n] for n in BEST_OF if n in dv}
    best_pos = min(o["exp_pos"] for o in opts.values())
    best_pts = max(o["exp_pts"] for o in opts.values())
    r = {"race": m["race"], "action": m["action"], "phase": callscore.phase_of(m["car_lap"], m["total"]),
         "exp_pos": ours["exp_pos"], "exp_pts": ours["exp_pts"], "p_lose2": ours["p_lose2"],
         "regret_pos": ours["exp_pos"] - best_pos, "regret_pts": best_pts - ours["exp_pts"],
         "best_is_ours": ours["exp_pos"] <= best_pos + 1e-9}
    if alt is not None:
        r.update(alt_pos=alt["exp_pos"], alt_pts=alt["exp_pts"], alt_p_lose2=alt["p_lose2"],
                 vs_alt_pos=ours["exp_pos"] - alt["exp_pos"], vs_alt_pts=ours["exp_pts"] - alt["exp_pts"])
    t = dv.get("team")
    if t is not None:
        po, pt = _pos(ours), _pos(t)
        r.update(team_pos=t["exp_pos"], team_pts=t["exp_pts"], team_p_lose2=t["p_lose2"],
                 vs_team_pos=ours["exp_pos"] - t["exp_pos"], vs_team_pts=ours["exp_pts"] - t["exp_pts"],
                 p_ours_ahead=float(np.mean(po < pt)), p_team_ahead=float(np.mean(pt < po)),
                 team_box_now=m.get("team_next") == m["car_lap"])
    return r


def _rps(pmf: np.ndarray, y: int) -> float:
    cdf = np.cumsum(pmf)
    obs = (np.arange(1, len(pmf) + 1) >= y).astype(float)
    return float(np.sum((cdf - obs) ** 2) / (len(pmf) - 1))


def calibration(moments: list[dict]) -> dict:
    """The simulator's forecast for the plan really driven (``team``) against the real finishing position."""
    rows = []
    for m in moments:
        t = (m.get("dv") or {}).get("team")
        if t is None or m.get("finish") is None:
            continue
        pos, y, cur = _pos(t), int(m["finish"]), int(m["dv"]["_cur"])
        n = 24
        pmf = np.bincount(pos.astype(int).clip(1, n), minlength=n + 1)[1:] / len(pos)
        now = np.zeros(n)
        now[min(cur, n) - 1] = 1.0
        lo, hi = np.quantile(pos, 0.1), np.quantile(pos, 0.9)
        rows.append({"race": m["race"], "rps": _rps(pmf, y), "rps_now": _rps(now, y), "err": float(pos.mean()) - y,
                     "abs_err": abs(float(pos.mean()) - y), "abs_err_now": abs(cur - y),
                     "pit": float(np.mean(pos < y) + 0.5 * np.mean(pos == y)), "in80": float(lo <= y <= hi),
                     "pts_err": t["exp_pts"] - float(analysis._points(np.array([y]))[0]), "p_lose2": t["p_lose2"],
                     "lost2": float(y >= cur + 2), "phase": callscore.phase_of(m["car_lap"], m["total"])})
    if not rows:
        return {"n": 0}
    by = _by_race(rows)

    def bm(k):
        return _ci(callscore.boot_mean([np.array([r[k] for r in rs]) for rs in by.values()]))

    pit = np.array([r["pit"] for r in rows])
    hist = np.histogram(pit, bins=10, range=(0, 1))[0]
    pl = np.array([r["p_lose2"] for r in rows])
    ls = np.array([r["lost2"] for r in rows])
    bins = [(0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 1.01)]
    rel = [{"bin": f"{a:.2f}-{b:.2f}", "n": int(((pl >= a) & (pl < b)).sum()),
            "predicted": float(pl[(pl >= a) & (pl < b)].mean()) if ((pl >= a) & (pl < b)).any() else None,
            "observed": float(ls[(pl >= a) & (pl < b)].mean()) if ((pl >= a) & (pl < b)).any() else None} for a, b in bins]
    return {"n": len(rows), "races": len(by), "rps": bm("rps"), "rps_current_position": bm("rps_now"),
            "bias_pos": bm("err"), "mae_pos": bm("abs_err"), "mae_current_position": bm("abs_err_now"),
            "coverage80": bm("in80"), "points_bias": bm("pts_err"), "pit_histogram": [int(h) for h in hist],
            "p_lose2_reliability": rel}


def _by_race(rows):
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["race"], []).append(r)
    return by


def _ci(t):
    pt, lo, hi = t
    return {"mean": pt, "ci90": [lo, hi]}


def summarise(results: list[dict]) -> dict:
    moments = [m for res in results for m in res["moments"]]
    rows = [r for r in (moment_metrics(m) for m in moments) if r is not None]
    out = {"moments": len(moments), "with_estimate": len(rows), "races": len({m["race"] for m in moments}),
           "by_action": {}, "calibration": calibration(moments)}

    def block(sel):
        by = _by_race(sel)
        if not sel:
            return {"n": 0}
        b = {"n": len(sel)}
        for k in ("exp_pos", "exp_pts", "p_lose2", "regret_pos", "regret_pts", "vs_alt_pos", "vs_alt_pts", "alt_p_lose2",
                  "vs_team_pos", "vs_team_pts", "team_p_lose2", "p_ours_ahead", "p_team_ahead"):
            vals = [np.array([r[k] for r in rs if k in r]) for rs in by.values()]
            vals = [v for v in vals if len(v)]
            if vals:
                b[k] = _ci(callscore.boot_mean(vals))
        b["best_is_ours"] = float(np.mean([r["best_is_ours"] for r in sel]))
        b["n_team"] = sum(1 for r in sel if "vs_team_pos" in r)
        return b

    out["all"] = block(rows)
    for a in OURS:
        out["by_action"][a] = block([r for r in rows if r["action"] == a])
    out["by_phase"] = {p: block([r for r in rows if r["phase"] == p]) for p in ("early", "mid", "late")}
    return out


def fair_scores(results: list[dict]) -> dict:
    per = []
    for res in results:
        sc = callscore.score_race(res["calls"], res["facts"], set(res["focus"]))
        per.append({"race": res["slug"], "facts": res["facts"], **sc})
    return {"summary": callscore.summarise(per), "splits": callscore.splits(per)}


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="python -m pitsense.bench.decision")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--races", nargs="*")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args(argv)
    res = collect(a.year, jobs=a.jobs, refresh=a.refresh, races=a.races)
    out = {"decision": summarise(res), "calls": fair_scores(res)}
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
