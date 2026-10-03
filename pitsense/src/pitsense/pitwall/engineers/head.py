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


def pick_plan(a, prev_target: int | None):
    """Plan A: the best plan, unless last lap's target stop lap is still nearly as good (hysteresis)."""
    best = a.ranked[0]
    if prev_target is not None and prev_target > a.anchor:
        cand = next((r for r in a.ranked if r.stops and r.stops[0][0] == prev_target), None)
        if cand is not None and cand.util - best.util <= SETTINGS["tol_keep"]:
            return cand
    return best


def _gap(a, A, lo: int, hi: int) -> float | None:
    """Places lost by stopping within offsets lo..hi laps from now instead of following plan A (None: no such plan)."""
    u = [r.util for r in a.ranked if r.stops and r.first_offset is not None and lo <= r.first_offset <= hi]
    return min(u) - A.util if u else None


def decide(a, prev_target: int | None, *, pit_open: bool = True, sc_phase: str = "none", prev_call: str | None = None):
    """(action, compound, confidence, rule, plan A) for one car's analysis and last lap's target stop lap.

    The plan says where a stop is worth most; the pit-probability model (``a.pp``, the models bundle's
    ``pit_prob_1/3`` or the rivals' hazard) says when the team will really stop. A box call needs both:
    BOX: stopping this lap costs at most ``tol_box`` places against plan A *and* the stop is likely now
    (``p1_box`` within this lap or ``p3_box`` within 3 laps). PREPARE_BOX: stopping within ``prep_laps`` laps costs
    at most ``tol_prep`` and a stop within 3 laps has probability ``p3_prep``. A call already made last lap is
    held down to ``hold`` x those thresholds. Without pit probabilities the plan alone decides (as before).
    """
    S = SETTINGS
    if a.plan_a is None or not a.ranked:
        return "NO_CALL", None, 0.0, "no_plan", None
    A = pick_plan(a, prev_target)
    d, se = a.diff_now_later if a.diff_now_later else (0.0, 0.5)
    se = max(se, 0.03)
    if not A.stops:
        return "STAY_OUT", None, min(0.97, max(0.5, _phi(d / se))), "no_stop_needed", A
    comp = A.stops[0][1]
    pp = getattr(a, "pp", None)
    if pp is None or not S["use_hazard"]:
        if A.first_offset == 0:
            if not pit_open:
                return "PREPARE_BOX", comp, 0.6, "pit_lane_closed", A
            return "BOX", comp, min(0.97, max(0.5, _phi(-d / se))), "box_now", A
        if A.first_offset <= S["prep_laps"]:
            return "PREPARE_BOX", comp, min(0.95, max(0.5, _phi(d / se))), "stop_soon", A
    else:
        h = S["hold"] if prev_call in ("BOX", "PREPARE_BOX") else 1.0  # thresholds are lower for a call already made
        p1, p3 = pp[0], max(pp[0], pp[1])
        g_now, g_win = _gap(a, A, 0, 0), _gap(a, A, 0, S["prep_laps"])
        if g_now is not None and g_now <= S["tol_box"] / h and (p1 >= S["p1_box"] * h or p3 >= S["p3_box"] * h):
            if not pit_open:
                return "PREPARE_BOX", comp, 0.6, "pit_lane_closed", A
            return "BOX", comp, min(0.97, max(0.5, _phi(-d / se))), "box_now", A
        near = S["prep_near"] is None or (g_now is not None and g_now <= S["prep_near"] / h)  # BOX conditions nearly met
        if (g_win is not None and g_win <= S["tol_prep"] / h and p3 >= S["p3_prep"] * h and p1 >= S["p1_prep"] * h
                and near):
            return "PREPARE_BOX", comp, min(0.95, max(0.5, _phi(d / se))), "stop_soon", A
    if a.gain_sc is not None and a.gain_sc >= S["gain_sc"] and sc_phase == "none":
        return "BOX_IF_SC", a.sc_best_comp, min(0.9, 0.5 + 0.2 * a.gain_sc), "sc_gain", A
    return "STAY_OUT", None, min(0.97, max(0.5, _phi(d / se))), "stay", A


class HeadOfStrategy(Engineer):
    name = "head"
    requires = ("pitstop", "rules", "strategy")
    in_bench = False

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self._hist: dict[str, tuple[int, int | None, int | None]] = {}  # car -> (lap, target stop lap, target before)
        self._last: dict[str, str] = {}  # car -> last call's action

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
            lap0, tgt0, before = self._hist.get(n, (None, None, None))
            prev_target = before if lap0 == d.laps else tgt0  # target stop lap decided on the previous lap
            action, comp, conf, rule, chosen = decide(a, prev_target, pit_open=rules.get("pit_lane_open") is not False,
                                                      sc_phase=str(rules.get("sc_phase") or "none"), prev_call=self._last.get(n))
            self._last[n] = action
            target = chosen.stops[0][0] if chosen is not None and chosen.stops else None
            self._hist[n] = (d.laps, target, before if lap0 == d.laps else tgt0)
            if chosen is None:
                out.append(Call(t=t, car=n, action="NO_CALL", reasons=(Reason("no_plan", "no plan could be ranked"),)))
                continue
            reasons = self._reasons(n, a, chosen, action, rule, view, rules, weather, state)
            rain = weather.get("rain_prob_10min")
            if (isinstance(rain, (int, float)) and rain >= 0.4) or weather.get("rain_now") or weather.get("crossover") not in (None, "none"):
                conf *= 0.8
            if d.laps < 8:
                conf *= 0.85
            plan_a = chosen.to_plan("A")
            plan_b = None
            if a.plan_b is not None:
                plan_b = a.plan_b.to_plan("B", a.plan_b_trigger)
            elif a.ranked:
                alt = next((r for r in a.ranked if r.stops != chosen.stops and r.first_offset != chosen.first_offset), None)
                if alt is not None:
                    plan_b = alt.to_plan("B", "if Plan A cannot be followed")
            out.append(Call(t=t, car=n, action=action, compound=comp if action != "STAY_OUT" else None,
                            confidence=round(conf, 2), reasons=tuple(reasons), plan_a=plan_a, plan_b=plan_b))
        return out

    def _reasons(self, n, a, A, action, rule, view, rules, weather, state) -> list[Reason]:
        r: list[Reason] = []
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
        if weather.get("rain_now"):
            r.append(Reason("rain_now", "rain is falling: dry-tyre plans may not hold", True))
        elif isinstance(rain, (int, float)) and rain >= 0.4:
            r.append(Reason("rain_risk", f"{rain:.0%} chance of rain within 10 minutes: dry-tyre plans may not hold", rain))
        return r
