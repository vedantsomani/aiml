"""Head of strategy: one call per car from the strategy engineer's ranked plans, with the reasons.

The decision (see ``decide``):

* BOX: stopping this lap costs at most ``tol_box`` places against the best plan (the best plan's first stop is
  this lap, or waiting is no better); the pit lane must be open.
* PREPARE_BOX: the best plan stops within ``prep_laps`` laps.
* BOX_IF_SC: no stop is due, but a safety car / VSC in the next 5 laps would make stopping worth at least
  ``gain_sc`` places (Plan B names the tyre).
* STAY_OUT otherwise. NO_CALL when no plan could be made (too early, wet race without a wet-tyre model); the reason says why.
* Wet or mixed race with a wet-tyre model: tyre-class calls (BOX for INTERS / SLICKS / WETS), see ``strategy/wethead.py``.

Hysteresis: a call made on the previous lap is kept unless the evidence moves by more than the tolerance
(a BOX stays BOX at twice the tolerance; PREPARE_BOX is not dropped to STAY_OUT until the plan's stop is 2
laps further away than the preparation window). Decisions are refreshed when a car completes a lap or the
track status changes, so a call is a function of that moment and the call before it.
"""

from __future__ import annotations

import math

from ..engineer import Engineer
from ..types import Call, Plan, PlanStop, Reason
from .strategy import wethead
from .strategy.analysis import SETTINGS, priors_for
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


SC_ALERT_P = 0.08  # the safetycar engineer's alert level (pitwall/engineers/safetycar.py ALERT_P)


def _sc_prob(view) -> float | None:
    """P(SC or VSC within 2 laps) from the safetycar engineer, None when it is not on this pit wall."""
    try:
        v = view.race("safetycar").get("neutral_prob_2laps")
    except KeyError:
        return None
    return float(v) if isinstance(v, (int, float)) else None


def _core(a, prev_target: int | None, *, pit_open: bool = True, sc_phase: str = "none", prev_call: str | None = None,
          sc_prob: float | None = None):
    """(action, compound, confidence, rule, plan A) for one car's analysis and last lap's target stop lap.

    The plan says where a stop is worth most; the pit-probability model (``a.pp``, the models bundle's
    ``pit_prob_1/3`` or the rivals' hazard) says when the team will really stop. A box call needs both:
    BOX: stopping this lap costs at most ``tol_box`` places against plan A *and* the stop is likely now
    (``p1_box`` within this lap or ``p3_box`` within 3 laps). PREPARE_BOX: stopping within ``prep_laps`` laps costs
    at most ``tol_prep`` and a stop within 3 laps has probability ``p3_prep``. A call already made last lap is
    held down to ``hold`` x those thresholds. Without pit probabilities the plan alone decides (as before).

    With the models bundle's laps-to-stop distribution (``a.ps``; ``use_stop_dist``) the gates are its CDF instead:
    BOX needs the plan gain within ``tol_box`` and ``p_stop_le_1 >= q1_box``; PREPARE_BOX needs the gain within
    ``tol_prep`` for a stop within ``prep_laps`` and ``p_stop_le_2 >= q2_prep``.
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
    ps = getattr(a, "ps", None)
    if ps is not None and S["use_stop_dist"]:
        # the laps-to-stop model separates "a stop in 1-2 laps" from "in 4-8 laps": gate on its CDF, then on the plan's gain
        h = S["hold"] if prev_call in ("BOX", "PREPARE_BOX") else 1.0
        q1, q2 = ps[0], max(ps[0], ps[1])
        g_now, g_win = _gap(a, A, 0, 0), _gap(a, A, 0, S["prep_laps"])
        if g_now is not None and g_now <= S["tol_box"] / h and q1 >= S["q1_box"] * h:
            if not pit_open:
                return "PREPARE_BOX", comp, 0.6, "pit_lane_closed", A
            return "BOX", comp, min(0.97, max(0.5, _phi(-d / se))), "box_now", A
        if g_win is not None and g_win <= S["tol_prep"] / h and q2 >= S["q2_prep"] * h and q1 >= S["q1_prep"] * h:
            return "PREPARE_BOX", comp, min(0.95, max(0.5, _phi(d / se))), "stop_soon", A
    elif pp is None or not S["use_hazard"]:
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
    bar = S["gain_sc"]
    if sc_prob is not None and S["use_sc_prob"] and sc_prob >= SC_ALERT_P:  # the safetycar engineer warns: a smaller gain is worth the call
        bar = min(bar, S["sc_alert_gain"])
    if a.gain_sc is not None and a.gain_sc >= bar and sc_phase == "none":
        return "BOX_IF_SC", a.sc_best_comp, min(0.9, 0.5 + 0.2 * a.gain_sc), "sc_gain", A
    return "STAY_OUT", None, min(0.97, max(0.5, _phi(d / se))), "stay", A


def _now_plan(a):
    """The best plan that stops this lap (the compound a BOX call names), None when there is none."""
    return next((r for r in a.ranked if r.stops and r.first_offset == 0), None)


def _note(ctl: dict, code: str, text: str, value=None) -> None:
    ctl.setdefault("notes", []).append((code, text, value))


def _q1(a) -> float | None:
    ps, pp = getattr(a, "ps", None), getattr(a, "pp", None)
    return ps[0] if ps is not None else (pp[0] if pp is not None else None)


def _lower(a, A, prev_action: str | None):
    """What a BOX call that cannot stand becomes: PREPARE_BOX while the plan stops within a lap or two, else STAY_OUT."""
    if A.stops and A.first_offset is not None and A.first_offset <= SETTINGS["prep_laps"]:
        return "PREPARE_BOX", A.stops[0][1]
    return "STAY_OUT", None


def decide(a, prev_target: int | None, *, pit_open: bool = True, sc_phase: str = "none", prev_call: str | None = None,
           sc_prob: float | None = None, ctl: dict | None = None, stops_done: int | None = None, pos: int | None = None):
    """(action, compound, confidence, rule, plan A): ``_core`` (the plan and the stop-time gate), then

    * the SC / VSC reaction: a neutralisation deployed and the car in its window: BOX now ("cheap stop under SC");
    * the sanity guard: a forecast finish far from the current position is not trusted for a BOX;
    * call stability: a BOX or PREPARE_BOX keeps its compound unless the plan for it is clearly worse, a BOX not acted
      on within ``box_ttl`` laps decays (PREPARE_BOX / STAY_OUT) and is not repeated for ``box_cool`` laps unless the
      stop became clearly more likely, and a stop (``stops_done`` goes up) resets all of it.

    ``ctl`` is the car's memory between calls (a plain dict the caller keeps); ``ctl["notes"]`` returns what was changed.
    """
    S = SETTINGS
    ctl = {} if ctl is None else ctl
    key = (a.anchor, sc_phase, pit_open, stops_done, prev_target, a.ranked[0].util if getattr(a, "ranked", None) else None)
    if ctl.get("memo", (None,))[0] == key:  # asked again with nothing new (same lap, status, plans): same answer
        ctl["notes"] = list(ctl["memo"][2])
        return ctl["memo"][1]
    out = _decide(a, prev_target, pit_open, sc_phase, prev_call, sc_prob, ctl, stops_done, pos)
    ctl["memo"] = (key, out, list(ctl["notes"]))
    return out


def _decide(a, prev_target, pit_open, sc_phase, prev_call, sc_prob, ctl, stops_done, pos):
    S = SETTINGS
    ctl["notes"] = []
    if stops_done is not None:
        if ctl.get("stops") is not None and stops_done > ctl["stops"]:  # the car stopped: a new stint, a new call history
            for k in ("comp", "comp_lap", "box_lap", "box_last", "cool", "cool_q"):
                ctl.pop(k, None)
            prev_call = None
        ctl["stops"] = stops_done
    action, comp, conf, rule, A = _core(a, prev_target, pit_open=pit_open, sc_phase=sc_phase, prev_call=prev_call, sc_prob=sc_prob)
    if A is None or action == "NO_CALL":
        return action, comp, conf, rule, A
    lap = a.anchor
    d, se = a.diff_now_later if a.diff_now_later else (0.0, 0.5)
    se = max(se, 0.03)
    now = _now_plan(a)
    # the compound of a BOX is the one for stopping this lap (plan A's stop may be later)
    if action == "BOX" and now is not None:
        comp = now.stops[0][1]
    # sanity guard
    if action == "BOX" and pos is not None and A.exp_pos is not None and abs(A.exp_pos - pos) > S["sane_places"]:
        _note(ctl, "forecast_implausible", f"forecast P{A.exp_pos:.1f} against P{pos} now: not trusted, no BOX", round(A.exp_pos, 1))
        action, rule, conf = "PREPARE_BOX", "forecast_implausible", min(conf, 0.5)
    # SC / VSC deployed and the car in its window
    if S["sc_event"] and sc_phase in ("sc", "vsc") and A.stops and action != "BOX" and now is not None:
        g_now = _gap(a, A, 0, 0)
        ps = getattr(a, "ps", None)
        in_window = (A.first_offset is not None and A.first_offset <= S["sc_window"]) or (ps is not None and ps[4] >= S["sc_p8"])
        sane = pos is None or A.exp_pos is None or abs(A.exp_pos - pos) <= S["sane_places"]
        if in_window and sane and g_now is not None and g_now <= S["sc_tol"]:
            action, comp, rule = ("BOX" if pit_open else "PREPARE_BOX"), now.stops[0][1], "sc_cheap_stop" if pit_open else "pit_lane_closed"
            conf = min(0.9, max(0.5, _phi(-d / se)))
    # compound kept
    if action in ("BOX", "PREPARE_BOX") and comp is not None:
        lock, lock_lap = ctl.get("comp"), ctl.get("comp_lap")
        if lock is not None and lock_lap is not None and lap - lock_lap > S["comp_ttl"]:
            lock = None
        if lock is not None and comp != lock:
            pool = [r for r in a.ranked if r.stops and (r.first_offset == 0 if action == "BOX" else r.first_offset == A.first_offset)]
            best = min((r.util for r in pool), default=None)
            cand = next((r for r in pool if r.stops[0][1] == lock), None)
            if cand is not None and best is not None and cand.util - best <= S["comp_tol"]:
                _note(ctl, "compound_kept", f"keeping {lock}: {comp} is no better than {cand.util - best:+.2f} places", lock)
                comp = lock
        ctl["comp"], ctl["comp_lap"] = comp, lap
    # a BOX that is not acted on decays
    if action == "BOX":
        last = ctl.get("box_last")
        if last is None or lap - last > S["box_ttl"] + 2:  # a new episode
            ctl["box_lap"] = lap
        ctl["box_last"] = lap
        q1 = _q1(a)
        if lap - ctl["box_lap"] >= S["box_ttl"]:
            _note(ctl, "box_decay", f"BOX called on lap {ctl['box_lap']} and not taken: holding it", ctl["box_lap"])
            action, comp = _lower(a, A, prev_call)
            rule = "box_not_taken"
            ctl["cool"], ctl["cool_q"] = lap + S["box_cool"], q1
            ctl["box_lap"] = lap
        elif ctl.get("cool") is not None and lap < ctl["cool"] and rule != "sc_cheap_stop"                 and not (q1 is not None and ctl.get("cool_q") is not None and q1 - ctl["cool_q"] >= S["box_q"]):
            _note(ctl, "box_decay", "BOX was not taken last time and nothing has changed: holding it", ctl.get("cool"))
            action, comp = _lower(a, A, prev_call)
            rule = "box_not_taken"
    return action, comp, conf, rule, A


class HeadOfStrategy(Engineer):
    name = "head"
    requires = ("pitstop", "rules", "strategy")
    in_bench = False

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self._hist: dict[str, tuple[int, int | None, int | None]] = {}  # car -> (lap, target stop lap, target before)
        self._last: dict[str, str] = {}  # car -> last call's action
        self._ctl: dict[str, dict] = {}  # car -> call memory (compound kept, BOX episode, stops seen)

    def calls(self, state, view):
        cars = [n for n in focus_cars(self.ctx, state) if state.drivers.get(n) is not None and state.drivers[n].running]
        res = get_analysis(self.ctx, self.memory, state, view)
        out = []
        rules = view.race("rules")
        weather = view.race("weather")
        sc_prob = _sc_prob(view)
        for n in cars:
            a = res.get(n)
            d = state.drivers[n]
            t = round(state.t, 3)
            ctl = self._ctl.setdefault(n, {})
            ev = wethead.field_event(self.memory, state, view, d)
            if a is not None and not a.ok and a.why.startswith("wet conditions"):  # no wet model (or engine off): the naive rain-flag rule
                action, comp, rs = wethead.naive_call(view, state, n, priors_for(self.ctx))
                action, comp, rs = wethead.apply_event(ev, action, comp, rs, ctl, d.laps, priors_for(self.ctx), state)
                self._last[n] = action
                out.append(Call(t=t, car=n, action=action, compound=comp, confidence=0.4, reasons=tuple(rs)))
                continue
            if a is None or not a.ok:
                why = a.why if a is not None else "no plan"
                out.append(Call(t=t, car=n, action="NO_CALL", reasons=(Reason("no_plan", why),)))
                continue
            if a.wet is not None:  # wet or mixed race: tyre-class calls from the wet simulator
                pri = priors_for(self.ctx)
                action, comp, cls, conf, rule = wethead.decide_wet(a, pri, self._last.get(n))
                evr = []
                if ev is not None and ev["target"] is not None and action != "BOX":
                    action, comp, evr = wethead.apply_event(ev, action, comp, [], ctl, d.laps, pri, state)[:3]
                    cls, conf, rule = ev["target"], max(conf, 0.7), "field_switch"
                elif action == "BOX":
                    action, comp, evr = wethead.apply_event(None, action, comp, [], ctl, d.laps, pri, state)
                self._last[n] = action
                pa, pb = wethead.plans(a, pri)
                if d.laps < 8:
                    conf *= 0.85
                out.append(Call(t=t, car=n, action=action, compound=comp if action != "STAY_OUT" else None,
                                confidence=round(conf, 2), reasons=tuple(list(evr) + wethead.reasons(a, action, cls, view, state)),
                                plan_a=pa, plan_b=pb))
                continue
            lap0, tgt0, before = self._hist.get(n, (None, None, None))
            prev_target = before if lap0 == d.laps else tgt0  # target stop lap decided on the previous lap
            action, comp, conf, rule, chosen = decide(a, prev_target, pit_open=rules.get("pit_lane_open") is not False,
                                                      sc_phase=str(rules.get("sc_phase") or "none"), prev_call=self._last.get(n),
                                                      sc_prob=sc_prob, ctl=ctl, stops_done=max(d.pit_stops, d.stint - 1), pos=d.position)
            self._last[n] = action
            target = chosen.stops[0][0] if chosen is not None and chosen.stops else None
            self._hist[n] = (d.laps, target, before if lap0 == d.laps else tgt0)
            if chosen is None:
                out.append(Call(t=t, car=n, action="NO_CALL", reasons=(Reason("no_plan", "no plan could be ranked"),)))
                continue
            reasons = self._reasons(n, a, chosen, action, rule, view, rules, weather, state)
            for code, text, val in ctl.get("notes", ()):
                reasons.insert(0, Reason(code, text, val))
            if rule == "sc_cheap_stop":
                pit = view.race("pitstop")
                reasons.insert(0, Reason("sc_cheap_stop", f"cheap stop under {str(rules.get('sc_phase')).upper()}: "
                                         f"{pit.get('loss_now')} s instead of {pit.get('loss_green')} s", pit.get("loss_now")))
            rain = weather.get("rain_prob_10min")
            if (isinstance(rain, (int, float)) and rain >= 0.4) or weather.get("rain_now") or weather.get("crossover") not in (None, "none"):
                conf *= 0.8
            if d.laps < 8:
                conf *= 0.85
            plan_a = chosen.to_plan("A")
            plan_b = None
            if getattr(a, "rain_b", None) is not None:  # rain likely and the car is on slicks: the reaction to a shower
                rb = a.rain_b
                pri = priors_for(self.ctx)
                stops = tuple(PlanStop(int(l), c if c != "SLICKS" else wethead.slick_compound(pri, max((state.total_laps or 60) - l, 1)))
                              for l, c in rb.plan_b.switches[:1])
                plan_b = Plan("B", stops, None, None, rb.plan_b_trigger)
                p10 = weather.get("rain_prob_10min")
                reasons.append(Reason("rain_plan_b", f"rain in 10 min {p10:.0%}: {rb.plan_b_trigger}", round(float(p10), 2)))
            elif a.plan_b is not None:
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
