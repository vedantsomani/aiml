"""Everything the pit wall knows before the lights go out, for one team: the facts behind the briefing.

``prerace(log, team)`` cuts the log at the session start (``log.until(start)``) and reads only that: the grid and
starting tyres, tyre sets left, circuit history from ``ctx.past_races`` (races that ended before this one started),
the weather so far, and the strategy simulator's Plan A/B/C for the team's cars. Giving it the full log or the
already-cut log gives the identical result (``tests/test_reports.py``).

The strategy simulator is run from the grid (lap 0) with the same lap-by-lap model the live engineer uses
(``pitwall/engineers/strategy``): the field's pace comes from the weekend's qualifying times (or a grid-order
assumption when none are on disk), tyre wear from past stints at this circuit, stop timing from the circuit's
history. Plans are ranked on 288 seeded futures, so the same input gives the same plans.
"""

from __future__ import annotations

import os
import statistics as stats
from datetime import datetime
from pathlib import Path

import numpy as np

from ..events import EventLog
from ..pitwall.types import Call, Plan, PlanStop, Reason, Snapshot, TeamConfig
from ..state import RaceState

RACE_PACE_K = 1.06  # median race lap / best qualifying lap, over 2026 races (1.04 Azerbaijan to 1.10 Hungary)
DEG_K = 2.4  # tyre wear s/lap = DEG_K / typical stint laps
FUEL_S = 0.035  # s per lap from fuel burn
GRID_GAP_S = 0.3  # time gap between grid slots at the start
OFFSETS = (-0.6, 0.0, 0.5)  # soft, medium, hard pace vs medium (s)
SIMS = 288
DRY = ("SOFT", "MEDIUM", "HARD")


# ----------------------------------------------------------------------------- loading
def session_start(log: EventLog) -> float | None:
    """Session-clock time of the lights out (first ``SessionStatus: Started``), or None."""
    for e in log.events:
        if e.topic == "SessionStatus" and isinstance(e.data, dict) and e.data.get("Status") == "Started":
            return e.t
    return None


def team_config(text: str | None) -> TeamConfig:
    from ..pitwall.runtime import _team_config

    return _team_config(text)


def ref_from_meta(meta: dict):
    from ..archive import SessionRef

    try:
        return SessionRef.from_dict({k: meta[k] for k in SessionRef.__dataclass_fields__})
    except Exception:
        return None


def load_history():
    from ..bench import history as H
    from ..config import bench_dir

    path = bench_dir() / "history.json"
    return H.load(path) if path.exists() else None


def build_context(meta: dict, team: TeamConfig, history=None):
    """The pit wall's pre-race knowledge (as the runtime builds it, without a model bundle)."""
    from ..bench import history as H
    from ..pitloss import PitLossPrior
    from ..pitwall.engineer import Context

    ref = ref_from_meta(meta)
    start = ref.start_utc if ref else None
    prior = PitLossPrior()
    if history is not None and ref is not None:
        prior = H.prior_for(history, ref.circuit_key, start)
    return Context.for_race(prior, dict(meta), history=history if start else None, race_start_utc=start, team=team)


def weekend_pace(ref) -> tuple[dict[str, float], str]:
    """Best qualifying lap per car from the weekend sessions that started before the race."""
    if ref is None:
        return {}, ""
    from .. import weekend
    from ..events import load_archive_session
    from ..state import replay

    try:
        sessions = [s for s in weekend.before(ref) if s.session_name in ("Qualifying", "Sprint Qualifying")]
    except Exception:
        return {}, ""
    best: dict[str, float] = {}
    name = ""
    for s in sorted(sessions, key=lambda s: s.start_utc):
        if not (s.local_dir / "TimingData.jsonStream").exists():
            continue
        st = replay(load_archive_session(s.local_dir, ("DriverList", "TimingData", "SessionStatus", "LapCount")))
        got = {n: d.best_lap_time for n, d in st.drivers.items() if d.best_lap_time}
        if got and s.session_name == "Qualifying" or not best:
            best, name = got, s.session_name
    return best, name


# ----------------------------------------------------------------------------- circuit facts
def _q(xs: list[int]) -> tuple:
    xs = sorted(xs)
    n = len(xs)
    at = lambda p: xs[min(n - 1, int(p * (n - 1) + 0.5))]  # noqa: E731
    return (xs[0], at(0.25), at(0.5), at(0.75), xs[-1])


def circuit_facts(ctx, total: int) -> dict:
    from ..pitwall.engineers.strategy import analysis
    from ..pitwall.engineers.strategy.priors import Priors

    ck = ctx.meta.get("circuit_key")
    same = [s for s in ctx.past_races if s.circuit_key == ck and ck is not None]
    pri = analysis.priors_for(ctx)
    pool = Priors(ctx.past_races, None)
    out: dict = {"circuit": ctx.meta.get("circuit") or ctx.meta.get("meeting_name"), "past_races_here": len(same),
                 "past_races_all": len(ctx.past_races), "years_here": sorted({s.year for s in same}), "laps": total}
    pl = ctx.prior
    out["pit_loss"] = {"green": round(pl.green, 1), "sc": round(pl.sc, 1), "vsc": round(pl.vsc, 1), "source": pl.source}
    n = max(total, 1)
    out["safety_car"] = {
        "rate_per_lap": round(pri.sc_rate, 4), "vsc_rate_per_lap": round(pri.vsc_rate, 4),
        "p_sc_in_race": round(1 - (1 - pri.sc_rate) ** n, 3), "p_vsc_in_race": round(1 - (1 - pri.vsc_rate) ** n, 3),
        "expected_sc": round(pri.sc_rate * n, 2), "races_with_sc": sum(1 for s in same if s.sc_laps > 0),
        "races_with_vsc": sum(1 for s in same if s.vsc_laps > 0), "mean_sc_laps": round(pri.sc_len, 1),
    }
    p_here, p_all = pri.pass_p[3], pool.pass_p[3]
    ratio = p_here / p_all if p_all else 1.0
    out["overtaking"] = {"pass_chance_per_lap": round(p_here, 3), "all_circuits": round(p_all, 3), "ratio": round(ratio, 2),
                         "label": "hard" if ratio < 0.75 else "easy" if ratio > 1.3 else "average",
                         "note": "chance to pass per lap when 0.3-0.6 s a lap quicker and within 1 s"}
    stints = {}
    for c in DRY:
        lens = pri.stints.get(c) or []
        if lens:
            stints[c] = {"n": len(lens), "q": list(_q(lens)), "typical": round(pri.life[c], 1)}
    circ_n = {c: sum(1 for s in same for ln, _ in ((s.extra or {}).get("strategy", {}).get("stints", {}).get(c, []))) for c in DRY}
    out["stints"] = stints
    out["stint_scope"] = {c: ("circuit" if circ_n[c] >= 6 else "all circuits") for c in DRY}
    rain = [bool((s.extra or {}).get("weather", {}).get("rained")) for s in same if (s.extra or {}).get("weather")]
    out["rain_history"] = {"races": len(rain), "rained": sum(rain)}
    return out


# ----------------------------------------------------------------------------- the simulator from the grid
def _field(ctx, state: RaceState, order: list, pace: dict[str, float], total: int):
    from ..pitwall.engineers.strategy import analysis
    from ..pitwall.engineers.strategy.priors import LIFE
    from ..pitwall.engineers.strategy.sim import FieldIn

    pri = analysis.priors_for(ctx)
    C = len(order)
    names = [d.number for d in order]
    z = lambda: np.zeros(C)  # noqa: E731
    F = FieldIn(A=0, total=total, cars=names, x0=z(), pace0=z(), deg=z(), fuel=FUEL_S, age0=z(), stint_age=z(),
                comp0=np.ones(C, dtype=np.int64), fresh=np.zeros((C, 3)), fresh_ref=z(), degc=np.zeros((C, 3)),
                pp=np.zeros((C, 3)), must=np.ones(C, dtype=bool), used=np.zeros((C, 3), dtype=bool),
                cliff_risk=z(), team_off=z(), penalty=z())
    have = [pace[n] * RACE_PACE_K for n in names if n in pace]
    if have:
        med = stats.median(have)
        mid = {n: (pace[n] * RACE_PACE_K if n in pace else med + 0.5) for n in names}
        src = "qualifying"
    else:
        mid = {n: 90.0 + 0.04 * i for i, n in enumerate(names)}  # no pace data: grid order only
        src = "assumed"
    deg = [min(0.3, DEG_K / pri.life[c]) for c in DRY]
    for i, d in enumerate(order):
        c0 = DRY.index(d.compound) if d.compound in DRY else 1
        F.comp0[i] = c0
        F.used[i, c0] = True
        F.x0[i] = GRID_GAP_S * i
        base = mid[d.number]
        F.deg[i] = deg[c0]
        F.pace0[i] = base + OFFSETS[c0] + FUEL_S * (total / 2 - 1)
        F.fresh_ref[i] = 3.0
        for j in range(3):
            F.fresh[i, j] = base + OFFSETS[j] + FUEL_S * (total / 2 - 3)
            F.degc[i, j] = deg[j]
        F.pp[i] = (0.0, 0.0, 0.01)
    F.loss = {"green": ctx.prior.green, "sc": ctx.prior.sc, "vsc": ctx.prior.vsc, "sd": 1.2}
    F.life = np.array([pri.life[c] for c in DRY])
    F.sc_rate, F.vsc_rate, F.sc_len, F.vsc_len = pri.sc_rate, pri.vsc_rate, pri.sc_len, pri.vsc_len
    F.pass_p = tuple(pri.pass_p)
    F.ref_pace = float(stats.median(mid.values()))
    F.stints = []
    for c in DRY:
        done, opened = pri.stints[c], pri.stints_open[c]
        if len(done) < 5:
            done = [int(round(LIFE[c] * f)) for f in np.linspace(0.75, 1.3, 11)]
            opened = []
        F.stints.append((done, opened))
    allnum = sorted(state.drivers, key=lambda x: (int(x) if x.isdigit() else 999, x))
    slot = {n: i for i, n in enumerate(allnum)}
    F.slots = np.array([slot[n] for n in names], dtype=np.int64)
    F.n_slots = len(allnum)
    info = {"stops_done": {n: 0 for n in names}, "stint_left": {n: None for n in names}, "rules": {}, "idx": {n: i for i, n in enumerate(names)}}
    return F, info, src


def _sig(r) -> tuple:
    return (len(r.stops), tuple(c for _, c in r.stops))


def _plan_dict(name: str, r, extra: dict | None = None) -> dict:
    pos = r.pos
    d = {"name": name, "stops": [[int(l), c] for l, c in r.stops], "exp_pos": round(float(r.exp_pos), 2),
         "exp_pts": round(float(r.exp_pts), 2), "pos_sd": round(float(r.pos_sd), 2),
         "p_points": round(float((pos <= 10).mean()), 3) if pos is not None else None,
         "p_podium": round(float((pos <= 3).mean()), 3) if pos is not None else None}
    d.update(extra or {})
    return d


def simulate_plans(ctx, state: RaceState, cars: list[str], pace: dict[str, float], total: int, sims: int = SIMS) -> dict:
    """Plan A/B/C (distinct tyre strategies, best first) and the safety-car play for each car, from the grid."""
    from ..pitwall.engineers.strategy import analysis
    from ..pitwall.engineers.strategy.run import evaluate_plans, simulate_field
    from ..pitwall.engineers.strategy.sim import Draws

    order = sorted((d for d in state.drivers.values() if d.running), key=lambda d: (d.grid if d.grid else (d.position or 99), d.number))
    if len(order) < 2 or total < 5:
        return {"ok": False, "why": "no field or race distance known"}
    F, info, src = _field(ctx, state, order, pace, total)
    seed = analysis.seed_for(ctx, "", 0, "prerace")
    D = Draws(F, sims, seed)
    run = simulate_field(F, D)
    out: dict = {"ok": True, "pace_source": src, "sims": sims, "cars": {}}
    names = F.cars
    for car in cars:
        if car not in names:
            continue
        c = names.index(car)
        a = analysis.analyse_focus(F, info, run, D, car, ctx, min(96, sims), 14)
        entry: dict = {"grid": order[c].grid or c + 1, "ok": a.ok, "why": a.why, "plans": [], "sc_play": None}
        if a.ok:
            ranked = a.ranked
            chosen = [ranked[0]]
            for r in ranked[1:]:
                if len(chosen) == 3:
                    break
                if all(_sig(r) != _sig(x) for x in chosen):
                    chosen.append(r)
            entry["plans"] = [_plan_dict("ABC"[i], r, {"loss_vs_a": round(float(r.exp_pos - chosen[0].exp_pos), 2)}) for i, r in enumerate(chosen)]
            entry["no_stop"] = _plan_dict("no stop", a.nostop) if a.nostop is not None else None
            sc = analysis.sc_scenario(F, info, car, ctx, a.plan_a, min(160, sims), analysis.seed_for(ctx, "", 0, "prerace-sc"))
            if sc[0] is not None:
                entry["sc_play"] = {"stop": [int(sc[0].stops[0][0]), sc[0].stops[0][1]], "trigger": sc[1], "places_gained": round(float(sc[2]), 2),
                                    "exp_pos": round(float(sc[0].exp_pos), 2)}
            # who can hurt us: the chance each rival finishes ahead of this car when it runs plan A
            stop, comp = analysis._arrays([tuple((l, DRY.index(cn)) for l, cn in chosen[0].stops)], D.S)
            _, ours, _ = evaluate_plans(F, D, run, c, stop, comp, D.S)
            entry["rival_ahead_p"] = {n: round(float(np.mean(run.final[:, j] < ours[0])), 3) for j, n in enumerate(names) if j != c}
        out["cars"][car] = entry
    return out


# ----------------------------------------------------------------------------- the whole pre-race picture
def _weather_series(pre: EventLog, state_meta: dict) -> tuple[RaceState, list]:
    st = RaceState(pre.meta)
    series = []
    for e in pre.events:
        st.apply(e)
        if e.topic == "WeatherData":
            series.append([round(e.t, 1), {k: v for k, v in st.weather.items() if v is not None}])
    return st, series


def rival_rows(state: RaceState, plans: dict, pace: dict, car: str) -> dict:
    """Rivals to watch for one car: the cars that could finish ahead of it from behind on the grid, and targets ahead."""
    me = state.drivers[car]
    ahead_p = (plans.get("cars", {}).get(car) or {}).get("rival_ahead_p") or {}
    rows = []
    for n, d in state.drivers.items():
        if n == car or d.team == me.team:
            continue
        gd = (d.grid or d.position or 99) - (me.grid or me.position or 99)
        dp = (pace[n] - pace[car]) if n in pace and car in pace else None
        rows.append({"car": n, "tla": d.tla, "team": d.team, "grid": d.grid or d.position, "grid_delta": gd,
                     "pace_delta_s": None if dp is None else round(dp, 3), "start_tyre": d.compound,
                     "p_ahead_at_flag": ahead_p.get(n)})
    threats = [r for r in rows if r["grid_delta"] > 0 and r["p_ahead_at_flag"] is not None]
    threats.sort(key=lambda r: (-r["p_ahead_at_flag"], r["grid"] or 99, r["car"]))
    targets = [r for r in rows if r["grid_delta"] < 0 and r["p_ahead_at_flag"] is not None]
    targets.sort(key=lambda r: (r["p_ahead_at_flag"], -(r["grid"] or 0), r["car"]))
    near = sorted((r for r in rows if abs(r["grid_delta"]) <= 2), key=lambda r: r["grid_delta"])
    return {"threats": threats[:4], "targets": targets[:3], "neighbours": near}


def write_brief(car: str, state: RaceState, entry: dict, rain_p: float | None, voice=None) -> dict:
    """The voice's pre-race brief for a car (its own guard checks the text; the template is the fallback)."""
    try:
        from ..voice.api import Voice
        from ..voice.facts import from_snapshot

        plans = entry.get("plans") or []
        if not plans:
            return {"text": "", "source": "none"}
        def mk(p, name, trigger=None):
            return Plan(name, tuple(PlanStop(int(l), c) for l, c in p["stops"]), p["exp_pos"], p["exp_pts"], trigger)
        b = None
        if len(plans) > 1:
            b = mk(plans[1], "B", (entry.get("sc_play") or {}).get("trigger"))
        call = Call(0.0, car, "NO_CALL", None, None, (), mk(plans[0], "A"), b)  # no call before the start: the brief carries the plan
        order = sorted((d for d in state.drivers.values() if d.position is not None), key=lambda d: d.position)
        tower = [{"position": d.grid or d.position, "car": d.number, "tla": d.tla, "team": d.team, "compound": d.compound,
                  "tyre_age": d.tyre_age or None, "pit_stops": 0} for d in order]
        idx = next((i for i, r in enumerate(tower) if r["car"] == car), None)
        vals: dict = {}
        if idx is not None:
            if idx > 0:
                vals["rivals__ahead"] = tower[idx - 1]["car"]
            if idx + 1 < len(tower):
                vals["rivals__behind"] = tower[idx + 1]["car"]
        vals["weather__rain_prob_10min"] = rain_p
        snap = Snapshot(0.0, 1, state.total_laps, "1", state.session_status, tower, {car: vals}, {"rules__sc_phase": "none"}, [], [], [car])
        if voice is None:
            use = os.environ.get("PITSENSE_VOICE", "") not in ("off", "template")
            voice = Voice(use_model=use)
        res = voice.write(from_snapshot(call, snap), "brief")
        return {"text": res.text, "source": res.source, "guard_fallback": res.fallback}
    except Exception as exc:  # the voice must never block a briefing
        return {"text": "", "source": "none", "error": type(exc).__name__}


def prerace(log: EventLog, team_text: str, *, history="auto", sims: int = SIMS, voice=None) -> dict:
    """The pre-race picture for ``team_text`` from everything published up to the lights out."""
    from ..pitwall import PitWall
    from ..pitwall.engineers.tyresets import TyreSetsEngineer
    from ..pitwall.engineers.weather import WeatherEngineer
    from ..pitwall.engineers.rules import RulesEngineer

    t0 = session_start(log)
    pre = log.until(t0) if t0 is not None else log
    team = team_config(team_text)
    if history == "auto":
        history = load_history()
    ctx = build_context(pre.meta, team, history)
    state, wseries = _weather_series(pre, pre.meta)
    cars = team.focus(state)
    if not cars:
        raise SystemExit(f"no car matches team {team_text!r}; teams here: {sorted({d.team for d in state.drivers.values() if d.team})}")
    total = state.total_laps
    if not total:
        laps = [s.laps for s in ctx.past_races if s.circuit_key == ctx.meta.get("circuit_key") and s.laps]
        total = int(stats.median(laps)) if laps else 0
    wall = PitWall(ctx, engineers=[RulesEngineer, WeatherEngineer, TyreSetsEngineer])
    st2 = RaceState(pre.meta)
    for e in pre.events:
        st2.apply(e)
        wall.observe(st2)
    try:
        race_v = wall.race_values(st2)
    except Exception:  # e.g. scikit-learn missing for the rain model
        race_v = {}
    pace, pace_name = weekend_pace(ref_from_meta(pre.meta))
    plans = simulate_plans(ctx, state, cars, pace, total, sims)
    grid = []
    for d in sorted(state.drivers.values(), key=lambda d: (d.grid or d.position or 99, d.number)):
        grid.append({"pos": d.grid or d.position, "car": d.number, "tla": d.tla, "team": d.team, "tyre": d.compound,
                     "tyre_new": d.tyre_new, "tyre_age": d.tyre_age, "quali_s": pace.get(d.number), "mine": d.number in cars})
    pole = min((g["quali_s"] for g in grid if g["quali_s"]), default=None)
    for g in grid:
        g["gap_to_best_s"] = round(g["quali_s"] - pole, 3) if g["quali_s"] and pole else None
    tyres = {c: {k.split("__", 1)[1]: v for k, v in wall.car_values(st2, c).items() if k.startswith("tyresets__")} for c in cars}
    rain_p = race_v.get("weather__rain_prob_10min")
    out = {
        "race": {k: pre.meta.get(k) for k in ("year", "meeting_name", "circuit", "location", "country", "session_name", "start_local")},
        "team": team_text, "team_name": team.match({d.team for d in state.drivers.values() if d.team}) or team_text, "cars": cars,
        "as_of": {"session_start_t": t0, "events_used": len(pre), "cut": t0 is not None},
        "total_laps": total, "grid": grid, "pace": {"source": plans.get("pace_source") if plans.get("ok") else None, "session": pace_name},
        "circuit": circuit_facts(ctx, total), "tyres": tyres,
        "weather": {"now": {k: v for k, v in state.weather.items() if v is not None}, "series": wseries[-40:], "rain_prob_10min": rain_p,
                    "crossover": race_v.get("weather__crossover"), "risk_of_rain": race_v.get("rules__risk_of_rain")},
        "plans": plans,
        "rivals": {c: rival_rows(state, plans, pace, c) for c in cars},
        "model": {"pace_k": RACE_PACE_K, "deg_k": DEG_K, "fuel_s": FUEL_S, "grid_gap_s": GRID_GAP_S},
    }
    out["brief"] = {c: write_brief(c, state, (plans.get("cars") or {}).get(c) or {}, rain_p, voice) for c in cars}
    return out
