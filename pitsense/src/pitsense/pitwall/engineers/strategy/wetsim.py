"""Wet-weather simulator: one car's tyre-class plans on simulated futures of a drying / raining track.

The state is the slick penalty ``w`` (see ``wetmodel``): it rises on rain laps and falls on dry laps, with
uncertain rates. Rain starts and stops at random (onset hazard from the weather engineer's ``rain_prob_10min``).
A plan is a list of tyre-class switches ``(in-lap offset, class)`` with class 0 slicks / 1 intermediates /
2 wets; each switch costs a pit stop. Plans are scored on total race time (seconds) over the laps still to
run on the same futures, so differences between plans are paired.

Not simulated: other cars (a plan is judged on time, not position), traffic, safety cars (a stop under a
neutralisation just costs ``loss_now`` on the first lap), tyre sets running out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .wetmodel import W, WetModel

SLK_WEAR = 0.0006  # slick wear per lap on a damp track (ratio)
CLASS_NAMES = ("SLICKS", "INTERMEDIATE", "WET")

# search settings (tuned on 2018-2024 wet races)
WSET = {
    "S": 200,
    "H_max": 40,  # laps simulated ahead at most
    "stop_extra_s": 3.0,  # on top of the pitstop engineer's loss: rejoin traffic, cold tyres
    "stop_sd": 1.2,
    "rate_sd": 0.35,  # log-sd of the drying / wetting rate in a future
    "late_hazard": 0.5,  # onset hazard after the weather engineer's 10-minute horizon, as a share of it
    "min_w_sd": 0.02,
    "trend_gain": 0.8,  # share of the recent trend of w carried into the simulated rain / dry laps
}
OFFS1 = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 28)
OFFS2 = (0, 1, 2, 3, 4, 6, 8, 10)
GAPS2 = (4, 8, 12, 18)


@dataclass
class WetIn:
    A: int
    total: int
    ref_s: float  # dry lap (s): ratio -> seconds
    c0: int  # current tyre class
    age0: int  # laps on the current set
    w0: float  # slick penalty now
    w0_sd: float
    rain_now: bool
    p10: float  # P(rain within 10 min)
    lap_s: float  # typical lap time now (s): turns 10 minutes into laps
    loss_green: float
    loss_now: float
    must_slick_pair: bool = False  # two-compound rule still binds (race_dry and the car has one dry compound)
    run_laps: float = 0.0  # laps the current rain has lasted
    trend: float = 0.0  # recent change of w per lap (from the field's lap times)
    notes: dict = field(default_factory=dict)

    @property
    def H(self) -> int:
        return int(max(1, min(self.total - self.A, WSET["H_max"])))


def simulate(model: WetModel, I: WetIn, S: int, seed: int, force: str | None = None):
    """Slick penalty [S,H] and rain flags [S,H] for the next laps. ``force``: 'rain' (a shower starts within 4
    laps) or 'dry' (the rain stops within 4 laps and does not come back soon)."""
    rng = np.random.default_rng(seed)
    H = I.H
    zr = rng.normal(0, 1, S)
    zd = rng.normal(0, 1, S)
    w0 = np.clip(I.w0 + max(I.w0_sd, WSET["min_w_sd"]) * rng.normal(0, 1, S), W["w_min"], W["w_max"])
    rr = model.rate_rain * np.exp(WSET["rate_sd"] * zr)
    rd = model.rate_dry * np.exp(WSET["rate_sd"] * zd)
    if I.trend > 0:  # the track has been getting wetter: rain laps keep doing at least that
        rr = np.maximum(rr, WSET["trend_gain"] * I.trend)
    elif I.trend < 0:
        rd = np.maximum(rd, -WSET["trend_gain"] * I.trend)
    n10 = max(600.0 / max(I.lap_s, 30.0), 1.0)
    p = float(np.clip(I.p10, 0.0, 0.97))
    h_on = 1.0 - (1.0 - p) ** (1.0 / n10)
    mean_run = max(W["mean_rain_run_laps"], 2.0)
    u_on = rng.random((S, H))
    u_off = rng.random((S, H))
    start = rng.integers(0, 4, S)
    rain = np.zeros((S, H), dtype=bool)
    on = np.full(S, bool(I.rain_now))
    if force == "rain":
        on = np.zeros(S, dtype=bool)
    for k in range(H):
        if force == "rain":
            begin = k == start
            on = np.where(begin, True, on)
            ended = on & (k > start) & (u_off[:, k] < 1.0 / mean_run)
            on = np.where(ended, False, on)
        elif force == "dry":
            on = np.where(k >= start, False, on)
            if k == 0:
                pass
        else:
            h = h_on if k < n10 else h_on * WSET["late_hazard"]
            ended = on & (u_off[:, k] < 1.0 / mean_run)
            began = ~on & (u_on[:, k] < h)
            if k > 0:
                on = np.where(ended, False, np.where(began, True, on))
            else:  # the lap in progress: rain now stays up unless it stops
                on = np.where(on, ~ended, began)
        rain[:, k] = on
    w = np.empty((S, H))
    cur = w0
    for k in range(H):
        cur = np.clip(cur + np.where(rain[:, k], rr, -rd), W["w_min"], W["w_max"])
        w[:, k] = cur
    return w, rain


def class_costs(model: WetModel, w: np.ndarray):
    """Per-class lap-time ratios [3,S,H] and wear-per-lap [3,S,H]."""
    T = np.stack([w, model.inter(w), model.wet_tyre(w)])
    slope_i = model.wear_slope(w)
    wear = np.stack([np.full_like(w, SLK_WEAR), slope_i, 0.8 * slope_i])
    return T, wear


def candidates(c0: int, H: int) -> list[tuple]:
    """Plans as tuples of (in-lap offset, class). () stays on the current tyre."""
    others = [c for c in range(3) if c != c0]
    plans: list[tuple] = [()]
    for o in OFFS1:
        if o >= H - 1:
            break
        for c in others:
            plans.append(((o, c),))
    for o in OFFS2:
        for g in GAPS2:
            o2 = o + g
            if o2 >= H - 1:
                continue
            for a in others:
                for b in [x for x in range(3) if x != a]:
                    plans.append(((o, a), (o2, b)))
    return plans


def evaluate(model: WetModel, I: WetIn, w: np.ndarray, plans: list[tuple]) -> np.ndarray:
    """Total time [P,S] in seconds (relative: only differences between plans mean anything)."""
    S, H = w.shape
    T, wear = class_costs(model, w)
    PT = np.concatenate([np.zeros((3, S, 1)), np.cumsum(T, 2)], 2)  # PT[c,s,k] = sum of T[:k]
    cs = np.concatenate([np.zeros((3, S, 1)), np.cumsum(wear, 2)], 2)  # cs[c,s,k] = wear added before lap k
    PC = np.concatenate([np.zeros((3, S, 1)), np.cumsum(cs[:, :, 1:], 2)], 2)  # PC[c,s,k] = sum_{j<k} cs[j+1]
    # wear of the set on the car now
    if I.c0 == 1:
        w_init = model.wear_slope(np.full(S, I.w0)) * max(I.age0 - 2, 0)
    elif I.c0 == 2:
        w_init = 0.8 * model.wear_slope(np.full(S, I.w0)) * max(I.age0 - 2, 0)
    else:
        w_init = np.full(S, SLK_WEAR * I.age0)
    rng = np.random.default_rng(7)
    pit_noise = rng.normal(0, WSET["stop_sd"], S)

    def seg(c: int, k0: int, k1: int, init) -> np.ndarray:
        n = k1 - k0
        if n <= 0:
            return np.zeros(S)
        base = PT[c, :, k1] - PT[c, :, k0]
        wr = (PC[c, :, k1] - PC[c, :, k0]) - n * cs[c, :, k0]
        return base + wr + n * init

    out = np.empty((len(plans), S))
    for i, plan in enumerate(plans):
        tot = np.zeros(S)
        cls, k0, init = I.c0, 0, w_init
        pit_s = 0.0
        for o, c in plan:
            tot += seg(cls, k0, o + 1, init)  # the in-lap is on the old tyre
            loss = (I.loss_now if o == 0 else I.loss_green) + WSET["stop_extra_s"]
            pit_s += max(loss, 3.0)
            cls, k0, init = c, o + 1, 0.0
        tot += seg(cls, k0, H, init)
        used_wet = any(c > 0 for _, c in plan) or I.c0 > 0
        if I.must_slick_pair and not used_wet:  # rule still binds: a dry stop is owed
            pit_s += max(I.loss_green, 3.0) + WSET["stop_extra_s"]
        out[i] = I.ref_s * tot + pit_s + (pit_noise * min(len(plan), 2) if plan else 0.0)
    return out


@dataclass
class WetPlan:
    switches: tuple  # ((in-lap, class name), ...)
    exp_s: float
    first_offset: int | None
    util_s: np.ndarray | None = None


@dataclass
class WetResult:
    best: WetPlan
    stay: WetPlan
    now: WetPlan | None  # best plan switching this lap
    later: WetPlan  # best plan whose first switch is later (or none)
    ranked: list
    gain_now_s: float  # seconds the best "switch this lap" plan gains over the best "later or never" plan
    p_now: float  # share of futures where it does
    gain_best_s: float  # best plan over staying
    w0: float
    w0_sd: float
    delta_s: float  # expected inters-minus-slicks lap time now (s; negative: inters faster)
    plan_b: WetPlan | None = None
    plan_b_gain_s: float | None = None
    plan_b_trigger: str | None = None
    n_plans: int = 0


def _wp(plan: tuple, I: WetIn, tot: np.ndarray) -> WetPlan:
    return WetPlan(tuple((I.A + 1 + o, CLASS_NAMES[c]) for o, c in plan), float(tot.mean()), plan[0][0] if plan else None, tot)


def analyse_car(model: WetModel, I: WetIn, seed: int, S: int | None = None, with_b: bool = True) -> WetResult:
    S = S or WSET["S"]
    w, _ = simulate(model, I, S, seed)
    plans = candidates(I.c0, I.H)
    tot = evaluate(model, I, w, plans)
    util = tot.mean(1)
    order = np.argsort(util, kind="stable")
    ranked = [_wp(plans[i], I, tot[i]) for i in order[:20]]
    best = ranked[0]
    stay = _wp((), I, tot[0])
    nows = [i for i in order if plans[i] and plans[i][0][0] == 0]
    laters = [i for i in order if not plans[i] or plans[i][0][0] > 0]
    now = _wp(plans[nows[0]], I, tot[nows[0]]) if nows else None
    later = _wp(plans[laters[0]], I, tot[laters[0]])
    gain_now = float(later.exp_s - now.exp_s) if now else 0.0
    p_now = float(np.mean(later.util_s > now.util_s)) if now else 0.0
    d_s = float((model.inter(np.array([I.w0])) - I.w0)[0] * I.ref_s)
    res = WetResult(best, stay, now, later, ranked, gain_now, p_now, float(stay.exp_s - best.exp_s), I.w0, I.w0_sd, d_s, n_plans=len(plans))
    if with_b and I.H >= 6:
        force = "rain" if I.c0 == 0 and not I.rain_now else "dry" if I.c0 > 0 else None
        if force:
            wb, _ = simulate(model, I, max(S // 2, 64), seed + 1, force=force)
            pb = candidates(I.c0, I.H)
            tb = evaluate(model, I, wb, pb)
            ub = tb.mean(1)
            j = int(np.argmin(ub))
            if pb[j]:
                res.plan_b = _wp(pb[j], I, tb[j])
                res.plan_b_gain_s = float(ub[0] - ub[j])
                cls = CLASS_NAMES[pb[j][0][1]]
                res.plan_b_trigger = ("if rain starts, box for " + ("INTERS" if cls == "INTERMEDIATE" else cls)
                                      if force == "rain" else "if the track dries out, box for " + ("INTERS" if cls == "INTERMEDIATE" else cls))
    return res
