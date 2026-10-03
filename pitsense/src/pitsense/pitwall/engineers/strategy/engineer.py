"""The strategy engineer: simulates the rest of the race and ranks plans for the focus cars.

Values (per focus car, ``strategy__<key>``):

* ``plan_a`` / ``plan_b``: text such as ``"L33 MEDIUM, L54 SOFT"``; ``plan_a_pos`` / ``plan_a_pts`` expected finishing
  position and points; ``plan_b_pos``; ``next_stop_lap`` (plan A's first stop, in-lap);
* ``now_cost``: places lost by stopping this lap rather than following plan A; ``sc_gain``: places gained by
  stopping under a safety car that comes within 5 laps, over staying on plan A;
* ``ok`` / ``why``: False with a reason when no plan could be made (wet race, too early, ...), ``sim_ms``.

Race: ``sc_prob_5`` / ``vsc_prob_5`` (chance of a neutralisation within 5 laps, from this circuit's history).
Plans are refreshed when a focus car completes a lap, the track status changes, or a rule fact changes.
"""

from __future__ import annotations

from ...engineer import Engineer
from . import analysis, priors


def focus_cars(ctx, state) -> list[str]:
    cars = ctx.team.focus(state)
    return cars or [d.number for d in state.running_order() if d.running]


def get_analysis(ctx, memory, state, view) -> dict:
    """Plans for the focus cars, cached on the context until a lap, the track status or a rule fact changes."""
    cars = [n for n in focus_cars(ctx, state) if n in state.drivers and state.drivers[n].running]
    rules = view.race("rules")
    key = (
        tuple((n, state.drivers[n].laps, bool(view.car("rules", n).get("must_stop")),
               view.car("rules", n).get("penalty_s_pending")) for n in cars),
        state.track_status, rules.get("sc_phase"), rules.get("pit_lane_open"),
    )
    hit = ctx.__dict__.get("_strategy_cache")
    if hit is not None and hit[0] == key:
        return hit[1]
    res = analysis.analyse(state, view, ctx, memory, cars, light=len(cars) > 4)
    ctx.__dict__["_strategy_cache"] = (key, res)
    return res


def plan_text(stops) -> str:
    return ", ".join(f"L{lap} {comp}" for lap, comp in stops) if stops else "no stop"


class StrategyEngineer(Engineer):
    name = "strategy"
    requires = ("tyre", "pitstop", "rivals", "rules", "weather", "models")
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

    def race(self, state, view):
        pri = analysis.priors_for(self.ctx)
        return {"sc_prob_5": round(1 - (1 - pri.sc_rate) ** 5, 4), "vsc_prob_5": round(1 - (1 - pri.vsc_rate) ** 5, 4),
                "strategy_history_races": pri.n_races}
