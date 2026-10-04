"""Head of strategy in a wet or mixed race: tyre-class calls ("BOX for INTERS", "BOX for SLICKS") from the wet
simulator's plans (``wetsim``), with reasons. Called by ``pitwall/engineers/head.py``."""

from __future__ import annotations

from ...types import Plan, PlanStop, Reason
from .wetsim import CLASS_NAMES

# decision settings (grid on the 2018-2024 wet races, F1 of switch-call precision/recall; docs/engineers/strategy.md)
HEAD_WET = {
    "g_box": 10.0,  # BOX: switching this lap beats the best later-or-never plan by this many seconds over the race
    "p_box": 0.85,  # ... in at least this share of futures
    "g_prep": 3.0,  # PREPARE_BOX: the best plan switches within prep_laps laps and gains this over staying
    "p_prep": 0.60,
    "prep_laps": 2,
    "hold": 0.5,  # a box call made last lap is held down to this share of the thresholds
}
NAME = {"SLICKS": "SLICKS", "INTERMEDIATE": "INTERS", "WET": "WETS"}


def slick_compound(pri, laps_left: int) -> str:
    """Softest dry compound that lasts the rest of the race (typical life x 1.1), else the hardest."""
    for c in ("SOFT", "MEDIUM", "HARD"):
        if pri.life[c] * 1.1 >= laps_left:
            return c
    return "HARD"


def decide_wet(a, pri, prev_call: str | None = None):
    """(action, compound, class name, confidence, rule) for one car's wet analysis."""
    S = HEAD_WET
    wr, inp = a.wet, a.wet_in
    h = S["hold"] if prev_call in ("BOX", "PREPARE_BOX") else 1.0
    best = wr.best
    left = max(inp.total - inp.A, 1)

    def comp_for(cls: str) -> str:
        return slick_compound(pri, left) if cls == "SLICKS" else cls

    if wr.now is not None and wr.now.switches and wr.gain_now_s >= S["g_box"] * h and wr.p_now >= S["p_box"] * h:
        cls = wr.now.switches[0][1]
        return "BOX", comp_for(cls), cls, min(0.95, max(0.5, wr.p_now)), "switch_now"
    if best.first_offset is not None and 0 <= best.first_offset <= S["prep_laps"] and wr.gain_best_s >= S["g_prep"] * h:
        cls = best.switches[0][1]
        p = wr.p_now if best.first_offset == 0 else 0.6
        if p >= S["p_prep"] * h or best.first_offset > 0:
            return "PREPARE_BOX", comp_for(cls), cls, min(0.9, max(0.5, p)), "switch_soon"
    conf = min(0.95, max(0.5, 1.0 - 0.5 * wr.p_now)) if wr.now is not None else 0.8
    return "STAY_OUT", None, None, conf, "stay"


def reasons(a, action: str, cls: str | None, view, state) -> list[Reason]:
    wr, inp = a.wet, a.wet_in
    weather, rules = view.race("weather"), view.race("rules")
    out: list[Reason] = []
    cross, d_s = weather.get("crossover"), weather.get("inters_vs_slicks_s")
    if cross not in (None, "none") and isinstance(d_s, (int, float)):
        who = "inters" if d_s < 0 else "slicks"
        out.append(Reason("crossover", f"crossover reached: {who} {abs(d_s):.1f} s/lap faster", round(float(d_s), 2)))
    else:
        dl = wr.delta_s
        who = "inters" if dl < 0 else "slicks"
        out.append(Reason("wet_pace", f"model: {who} about {abs(dl):.1f} s/lap faster ({inp.notes.get('w_source')})", round(dl, 2)))
    p = weather.get("rain_prob_10min")
    if weather.get("rain_now"):
        out.append(Reason("rain_now", "rain is falling", True))
    elif isinstance(p, (int, float)) and p >= 0.3:
        out.append(Reason("rain_risk", f"rain in 10 min {p:.0%}", round(float(p), 2)))
    cur = NAME[CLASS_NAMES[inp.c0]]
    if action in ("BOX", "PREPARE_BOX") and cls:
        verb = "BOX" if action == "BOX" else "prepare to box"
        out.insert(0, Reason("tyre_switch", f"{verb} for {NAME[cls]}: saves {wr.gain_best_s:.0f} s over the rest of the race "
                                            f"against staying on {cur}", round(wr.gain_best_s, 1)))
        out.append(Reason("pit_cost", f"the stop costs about {inp.loss_green:.0f} s", round(inp.loss_green, 1)))
    elif action == "STAY_OUT":
        out.insert(0, Reason("stay", f"staying on {cur}: no switch pays for the stop (best plan gains {wr.gain_best_s:.0f} s)",
                             round(wr.gain_best_s, 1)))
    if cls == "SLICKS" and rules.get("race_dry") is False:
        out.append(Reason("two_compounds_void", "wets have been used: the two-dry-compound rule is void", True))
    run = [d for d in state.drivers.values() if d.running]
    if run:
        n_i = sum(d.compound == "INTERMEDIATE" for d in run)
        n_w = sum(d.compound == "WET" for d in run)
        n_s = sum(d.compound in ("SOFT", "MEDIUM", "HARD") for d in run)
        out.append(Reason("field_tyres", f"field: {n_s} on slicks, {n_i} on inters, {n_w} on wets", n_s))
    if inp.c0 == 1 and inp.age0 >= 8:
        out.append(Reason("inter_age", f"inters {inp.age0} laps old", inp.age0))
    return out


def plans(a, pri) -> tuple[Plan, Plan | None]:
    wr, inp = a.wet, a.wet_in

    def conv(sw):
        return tuple(PlanStop(int(l), slick_compound(pri, max(inp.total - l, 1)) if c == "SLICKS" else c) for l, c in sw)

    pa = Plan("A", conv(wr.best.switches), None, None, None)
    pb = None
    if wr.plan_b is not None:
        pb = Plan("B", conv(wr.plan_b.switches), None, None, wr.plan_b_trigger)
    elif wr.later is not None and wr.later.switches != wr.best.switches:
        pb = Plan("B", conv(wr.later.switches), None, None, "if Plan A cannot be followed")
    return pa, pb


# naive rule (fallback when the wet engine is off or has no model): switch when the rain flag says so
NAIVE = {"rain_min": 2.0, "dry_min": 5.0}


def naive_call(view, state, number: str, pri):
    """(action, compound, reasons) for the rule "BOX for INTERS once the rain flag has been up 2 min with the car on
    slicks; BOX for SLICKS once it has been down 5 min, the car is on inters/wets and slicks have become faster"."""
    d = state.drivers[number]
    w = view.race("weather")
    run, since = w.get("rain_minutes"), w.get("minutes_since_rain")
    on_slick = d.compound in ("SOFT", "MEDIUM", "HARD")
    if on_slick and w.get("rain_now") and isinstance(run, (int, float)) and run >= NAIVE["rain_min"]:
        return "BOX", "INTERMEDIATE", [Reason("rain_flag", f"BOX for INTERS: rain flag up for {run:.0f} min and no wet-tyre model", round(float(run), 1))]
    if (d.compound in ("INTERMEDIATE", "WET") and not w.get("rain_now") and isinstance(since, (int, float)) and since >= NAIVE["dry_min"]
            and w.get("crossover") == "to_slicks"):
        left = max((state.total_laps or 60) - d.laps, 1)
        d_s = w.get("inters_vs_slicks_s")
        txt = f"crossover reached: slicks {abs(d_s):.1f} s/lap faster; " if isinstance(d_s, (int, float)) else ""
        return "BOX", slick_compound(pri, left), [Reason("rain_flag", f"BOX for SLICKS: {txt}rain flag down for {since:.0f} min", round(float(since), 1))]
    return "STAY_OUT", None, [Reason("rain_flag", "no tyre-class switch signalled by the rain flag")]
