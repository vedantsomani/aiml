"""The strategy engineer: simulates the rest of the race and ranks plans for the focus cars.

Values (per focus car, ``strategy__<key>``):

* ``plan_a`` / ``plan_b``: text such as ``"L33 MEDIUM, L54 SOFT"``; ``plan_a_pos`` / ``plan_a_pts`` expected finishing
  position and points; ``plan_b_pos``; ``next_stop_lap`` (plan A's first stop, in-lap);
* ``now_cost``: places lost by stopping this lap rather than following plan A; ``sc_gain``: places gained by
  stopping under a safety car that comes within 5 laps, over staying on plan A;
* ``ok`` / ``why``: False with a reason when no plan could be made (too early, wet race with no wet-tyre model, ...), ``sim_ms``.
  In a wet or mixed race the plans are tyre-class switches ("L33 INTERMEDIATE"), ranked on race time by the wet
  simulator (``wetsim``): ``plan_a_pos`` / ``plan_a_pts`` are None and ``now_cost`` is the seconds lost by not
  switching this lap.

Race: ``sc_prob_5`` / ``vsc_prob_5`` (chance of a neutralisation within 5 laps, from this circuit's history).
Plans are refreshed when a focus car completes a lap, the track status changes, a rule fact changes, the car's tyres
change (compound, stops made, stint: the feed can confirm a stop a few laps late) or the weather flag changes.
"""

from __future__ import annotations

from ...engineer import Engineer
from . import analysis, priors


def focus_cars(ctx, state) -> list[str]:
    cars = ctx.team.focus(state)
    return cars or [d.number for d in state.running_order() if d.running]


def _weather_flag(view, rules) -> tuple:
    """The coarse weather state the planner reads: rain flag up, wet tyres on track, the crossover call and whether
    the race is still dry. Never the nowcast values (probability, minutes, temperatures): they move on every
    message and the cache would never hit."""
    w = view.race("weather")
    return ((w.get("rainfall") or 0) > 0 or bool(w.get("rain_now")), bool(w.get("wet_running")), w.get("crossover"),
            bool(rules.get("race_dry", True)))


def get_analysis(ctx, memory, state, view) -> dict:
    """Plans for the focus cars. Each car's plan is refreshed when it completes a lap, the track status or the
    weather flag changes, one of its rule facts changes, or its tyres change (compound, stops made, stint: the feed
    confirms a tyre change up to a few laps after the stop); otherwise the cached plan stands."""
    cars = [n for n in focus_cars(ctx, state) if n in state.drivers and state.drivers[n].running]
    rules = view.race("rules")
    light = len(cars) > 4
    glob = (state.track_status, rules.get("sc_phase"), rules.get("pit_lane_open"), light, _weather_flag(view, rules), ctx.team.risk)
    cache = ctx.__dict__.setdefault("_strategy_cache", {})
    out, todo = {}, []
    for n in cars:
        d, rl = state.drivers[n], view.car("rules", n)
        key = (glob, d.laps, d.compound, d.pit_stops, d.stint, bool(rl.get("must_stop")), rl.get("penalty_s_pending"))
        hit = cache.get(n)
        if hit is not None and hit[0] == key:
            out[n] = hit[1]
        else:
            todo.append((n, key))
    if todo:
        res = analysis.analyse(state, view, ctx, memory, [n for n, _ in todo], light=light)
        for n, key in todo:
            if n in res:
                cache[n] = (key, res[n])
                out[n] = res[n]
    return out


def plan_text(stops) -> str:
    return ", ".join(f"L{lap} {comp}" for lap, comp in stops) if stops else "no stop"


OPTIONS = 5  # plans shown side by side on the dashboard


def _opt(p, base, tags, np) -> dict:
    lo = hi = None
    if p.pos is not None and len(p.pos):
        lo, hi = (float(x) for x in np.percentile(p.pos, [10, 90]))
    return {"plan": plan_text(p.stops), "stops": [[int(l), c] for l, c in p.stops],
            "exp_pos": round(float(p.exp_pos), 2), "exp_pts": round(float(p.exp_pts), 2), "pos_sd": round(float(p.pos_sd), 2),
            "p10": None if lo is None else round(lo, 1), "p90": None if hi is None else round(hi, 1),
            "delta": None if base is None else round(float(p.exp_pos - base), 2), "tags": tags}


def options(a) -> dict:
    """The strategy comparison view for one car.

    ``ranked``: the best few distinct plans for the race as it stands, plan A first, then by the simulator's
    utility; each has ``plan`` text, ``stops`` [[in-lap, compound]], expected position / points, ``p10`` / ``p90``
    of the simulated finishing positions (the spread of outcomes, 1 = best), ``delta`` places vs plan A, and its
    role (``A``, ``now`` = best plan that stops this lap, ``later``, ``no stop``).
    ``if_sc``: plan B, simulated in futures where a safety car comes, so it is not comparable with the others
    (no ``delta``), with its ``trigger`` and ``sc_gain`` (places gained by stopping under that SC)."""
    import numpy as np

    tags: dict[int, list[str]] = {}
    for tag, p in (("A", a.plan_a), ("now", a.now_best), ("later", a.later_best), ("no stop", a.nostop)):
        if p is not None:
            tags.setdefault(id(p), []).append(tag)
    named = [p for p in (a.plan_a, a.now_best, a.later_best, a.nostop) if p is not None]
    seen, uniq = set(), []
    for p in named + sorted(a.ranked, key=lambda q: (q.util, q.exp_pos)):
        key = tuple(p.stops)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    named = [p for p in uniq if id(p) in tags]
    rest = [p for p in uniq if id(p) not in tags][: max(0, OPTIONS - len(named))]
    base = a.plan_a.exp_pos
    rows = sorted(named + rest, key=lambda q: (q is not a.plan_a, q.util, q.exp_pos))
    out = {"ranked": [_opt(p, base, tags.get(id(p), []), np) for p in rows], "if_sc": None}
    if a.plan_b is not None:
        b = _opt(a.plan_b, None, ["B"], np)
        b.update(trigger=a.plan_b_trigger, sc_gain=None if a.gain_sc is None else round(float(a.gain_sc), 2))
        out["if_sc"] = b
    return out


class StrategyEngineer(Engineer):
    name = "strategy"
    requires = ("tyre", "pitstop", "rivals", "rules", "weather", "models", "tyresets")
    in_bench = False

    @classmethod
    def summarize_race(cls, final, meta):
        return priors.learn(final, meta)

    def car(self, state, number, view):
        if number not in focus_cars(self.ctx, state):
            return {}
        a = get_analysis(self.ctx, self.memory, state, view).get(number)
        if a is None:
            return {}
        out = {"ok": a.ok, "why": a.why or None, "sim_ms": round(a.ms), "plan_a": None, "plan_a_pos": None,
               "plan_a_pts": None, "plan_b": None, "plan_b_pos": None, "next_stop_lap": None,
               "now_cost": None, "sc_gain": None}
        if a.ok and a.wet is not None:  # wet simulator: plans are tyre-class switches
            wr = a.wet
            out.update(plan_a=plan_text(wr.best.switches), plan_a_pos=None, plan_a_pts=None,
                       next_stop_lap=wr.best.switches[0][0] if wr.best.switches else None,
                       now_cost=round(-wr.gain_now_s, 2) if wr.now is not None else None)
            if wr.plan_b is not None:
                out["plan_b"] = plan_text(wr.plan_b.switches)
        if a.ok and a.plan_a is not None:
            A = a.plan_a
            out.update(plan_a=plan_text(A.stops), plan_a_pos=round(A.exp_pos, 2), plan_a_pts=round(A.exp_pts, 2),
                       next_stop_lap=A.stops[0][0] if A.stops else None)
            if a.now_best is not None:
                out["now_cost"] = round(a.now_best.exp_pos - A.exp_pos, 3)
            if a.plan_b is not None:
                out.update(plan_b=plan_text(a.plan_b.stops), plan_b_pos=round(a.plan_b.exp_pos, 2))
            if a.gain_sc is not None:
                out["sc_gain"] = round(a.gain_sc, 3)
        return out

    def details(self, state, view):
        """Per focus car: ``ranked`` / ``if_sc`` (see :func:`options`) and ``stop_dist`` (share of simulated futures whose
        first stop is 0, 1, ... laps from now) for the dashboard's strategy comparison."""
        out = {}
        for n, a in get_analysis(self.ctx, self.memory, state, view).items():
            if a.ok and a.plan_a is not None:
                out[n] = {**options(a), "sets_note": a.sets_note, "stop_dist": [round(float(x), 3) for x in a.stop_p]}
        return out

    def race(self, state, view):
        pri = analysis.priors_for(self.ctx)
        return {"sc_prob_5": round(1 - (1 - pri.sc_rate) ** 5, 4), "vsc_prob_5": round(1 - (1 - pri.vsc_rate) ** 5, 4),
                "strategy_history_races": pri.n_races}
