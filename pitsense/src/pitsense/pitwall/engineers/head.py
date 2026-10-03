"""Head of strategy: one call per car from the strategy engineer's ranked plans, with the reasons.

The decision (see ``decide``):

* BOX: stopping this lap costs at most ``tol_box`` places against the best plan (the best plan's first stop is
  this lap, or waiting is no better); the pit lane must be open.
* PREPARE_BOX: the best plan stops within ``prep_laps`` laps.
* BOX_IF_SC: no stop is due, but a safety car / VSC in the next 5 laps would make stopping worth at least
  ``gain_sc`` places (Plan B names the tyre).
* STAY_OUT otherwise. NO_CALL when no plan could be made (wet race, too early); the reason says why.

Hysteresis: a call made on the previous lap is kept unless the evidence moves by more than the tolerance
(a BOX stays BOX at twice the tolerance; PREPARE_BOX is not dropped to STAY_OUT until the plan's stop is 2
laps further away than the preparation window). Decisions are refreshed when a car completes a lap or the
track status changes, so a call is a function of that moment and the call before it.
"""

from __future__ import annotations

import math

from ..engineer import Engineer
from ..types import Call, Plan, Reason
from .strategy.analysis import SETTINGS
from .strategy.engineer import focus_cars, get_analysis, plan_text


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def decide(a, prev: str | None, *, pit_open: bool = True, sc_phase: str = "none") -> tuple[str, str | None, float, str]:
    """(action, compound, confidence, rule) from one car's analysis and its call on the previous lap."""
    S = SETTINGS
    A = a.plan_a
    if A is None:
        return "NO_CALL", None, 0.0, "no_plan"
    cost_now = (a.now_best.exp_pos - A.exp_pos) if a.now_best is not None else math.inf
    tol = S["tol_box"] * (2.0 if prev == "BOX" else 1.0)
    d, se = a.diff_now_later if a.diff_now_later else (0.0, 0.5)
    se = max(se, 0.03)
    prep_window = S["prep_laps"] + (2 if prev in ("PREPARE_BOX", "BOX") else 0)
    comp = A.stops[0][1] if A.stops else None
    if not A.stops and cost_now > tol:
        return "STAY_OUT", None, min(0.97, max(0.5, _phi(d / se))), "no_stop_needed"
    if cost_now <= tol:
        c = a.now_best.stops[0][1]
        conf = min(0.97, max(0.5, _phi(-d / se)))
        if not pit_open:
            return "PREPARE_BOX", c, conf, "pit_lane_closed"
        return "BOX", c, conf, "box_now"
    if A.first_offset is not None and A.first_offset <= prep_window:
        return "PREPARE_BOX", comp, min(0.95, max(0.5, _phi(d / se))), "stop_soon"
    if a.gain_sc is not None and a.gain_sc >= S["gain_sc"] and sc_phase == "none" and A.stops:
        return "BOX_IF_SC", a.sc_best_comp, min(0.9, 0.5 + 0.2 * a.gain_sc), "sc_gain"
    return "STAY_OUT", None, min(0.97, max(0.5, _phi(d / se))), "stay"


class HeadOfStrategy(Engineer):
    name = "head"
    requires = ("pitstop", "rules", "strategy")
    in_bench = False

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self._hist: dict[str, tuple[int, str | None, str | None]] = {}  # car -> (lap, action, action before)

    def calls(self, state, view):
        cars = [n for n in focus_cars(self.ctx, state) if state.drivers.get(n) is not None and state.drivers[n].running]
        res = get_analysis(self.ctx, self.memory, state, view)
        out = []
        rules = view.race("rules")
        weather = view.race("weather")
        for n in cars:
            a = res.get(n)
            d = state.drivers[n]
            t = round(state.t, 3)
            if a is None or not a.ok:
                why = a.why if a is not None else "no plan"
                out.append(Call(t=t, car=n, action="NO_CALL", reasons=(Reason("no_plan", why),)))
                continue
            lap0, act0, before = self._hist.get(n, (None, None, None))
            prev_action = before if lap0 == d.laps else act0  # the call on the previous lap
            action, comp, conf, rule = decide(a, prev_action, pit_open=rules.get("pit_lane_open") is not False,
                                              sc_phase=str(rules.get("sc_phase") or "none"))
            self._hist[n] = (d.laps, action, before if lap0 == d.laps else act0)
            reasons = self._reasons(n, a, action, rule, view, rules, weather, state)
            rain = weather.get("rain_prob_10min")
            if isinstance(rain, (int, float)) and rain >= 0.4 or weather.get("crossover") not in (None, "none"):
                conf *= 0.8
            if d.laps < 8:
                conf *= 0.85
            plan_a = a.plan_a.to_plan("A")
            plan_b = None
            if a.plan_b is not None:
                plan_b = a.plan_b.to_plan("B", a.plan_b_trigger)
            elif a.ranked:
                alt = next((r for r in a.ranked if r.stops != a.plan_a.stops and r.first_offset != a.plan_a.first_offset), None)
                if alt is not None:
                    plan_b = alt.to_plan("B", "if Plan A cannot be followed")
            out.append(Call(t=t, car=n, action=action, compound=comp if action != "STAY_OUT" else None,
                            confidence=round(conf, 2), reasons=tuple(reasons), plan_a=plan_a, plan_b=plan_b))
        return out

    def _reasons(self, n, a, action, rule, view, rules, weather, state) -> list[Reason]:
        r: list[Reason] = []
        A = a.plan_a
        phase = str(rules.get("sc_phase") or "none")
        pit = view.race("pitstop")
        tyre, riv, rl, ps = view.car("tyre", n), view.car("rivals", n), view.car("rules", n), view.car("pitstop", n)
        if phase in ("sc", "vsc", "sc_ending", "vsc_ending", "red"):
            r.append(Reason("sc_phase", f"{phase.replace('_', ' ')}: a stop costs {pit.get('loss_now')} s instead of {pit.get('loss_green')} s",
                            pit.get("loss_now")))
        if rule == "pit_lane_closed":
            r.append(Reason("pit_lane_closed", "pit lane is closed: box as soon as it opens"))
        if rl.get("must_stop"):
            r.append(Reason("must_stop", "still has to fit a second dry compound", True))
        if (rl.get("penalty_s_pending") or 0) > 0:
            r.append(Reason("penalty", f"{rl['penalty_s_pending']:.0f} s penalty to serve", rl["penalty_s_pending"]))
        cl = tyre.get("cliff_risk")
        if isinstance(cl, (int, float)) and cl >= 0.15:
            r.append(Reason("tyre_cliff", f"{cl:.0%} chance the tyres fall off within 3 laps", cl))
        thr, cha = riv.get("undercut_threat"), riv.get("undercut_chance")
        if isinstance(thr, (int, float)) and thr >= 0.35:
            r.append(Reason("undercut_threat", f"{thr:.0%} chance the car behind undercuts", thr))
        if isinstance(cha, (int, float)) and cha >= 0.35:
            r.append(Reason("undercut_chance", f"{cha:.0%} chance of undercutting the car in front", cha))
        if rule == "box_now" and A.stops:
            alt = a.later_best
            if alt is not None:
                gap = alt.exp_pos - a.now_best.exp_pos
                r.insert(0, Reason("plan_gain", f"stopping now averages P{a.now_best.exp_pos:.1f}; waiting P{alt.exp_pos:.1f}", round(gap, 2)))
        elif action == "BOX_IF_SC":
            r.insert(0, Reason("sc_gain", f"a safety car within 5 laps would be worth {a.gain_sc:.1f} places if we stop under it", round(a.gain_sc, 2)))
        elif action == "PREPARE_BOX":
            r.insert(0, Reason("plan_stop", f"best plan stops on lap {A.stops[0][0]} ({A.stops[0][1]})", A.stops[0][0]))
        elif action == "STAY_OUT" and A.stops:
            r.insert(0, Reason("plan_wait", f"best plan stops on lap {A.stops[0][0]}, {A.first_offset} laps from now", A.first_offset))
        elif action == "STAY_OUT":
            r.insert(0, Reason("no_stop", "no further stop is needed"))
        rj = ps.get("rejoin_if_box_now")
        if rj is not None and action in ("BOX", "PREPARE_BOX"):
            r.append(Reason("rejoin", f"boxing now rejoins P{rj}", rj))
        r.append(Reason("expected_finish", f"Plan A: {plan_text(A.stops)}; expected P{A.exp_pos:.1f}, {A.exp_pts:.1f} points", round(A.exp_pos, 2)))
        rain = weather.get("rain_prob_10min")
        if isinstance(rain, (int, float)) and rain >= 0.4:
            r.append(Reason("rain_risk", f"{rain:.0%} chance of rain within 10 minutes: dry-tyre plans may not hold", rain))
        return r
