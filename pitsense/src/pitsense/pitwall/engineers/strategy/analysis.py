"""From the engineers' values to plans: build the simulated field, generate candidate plans, rank them.

``analyse`` is the entry point (the strategy engineer and the head of strategy both call it; the result is
cached on the context per decision moment). Everything is as of ``state.t``: the field comes from ``state``,
``memory`` and the other engineers' values in ``view``; history comes from ``ctx.past_races``.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from ...types import Plan, PlanStop
from .priors import DRY, LIFE, Priors
from .run import FieldRun, evaluate_plans, simulate_field
from .sim import K_STOPS, NOSTOP, Draws, FieldIn

POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
COMP_NAMES = DRY  # index 0 soft, 1 medium, 2 hard
CIDX = {c: i for i, c in enumerate(DRY)}

# search and decision settings (tuned on 2025; see docs/engineers/strategy.md)
SETTINGS = {
    "S": 288,  # simulated futures
    "S1": 96,  # futures used to screen the candidate plans
    "S_B": 160,  # futures in the safety-car scenario
    "keep": 14,  # plans re-evaluated on all futures
    "tol_box": 0.30,  # box now if it costs at most this many places versus the best plan
    "tol_prep": 0.50,
    "prep_laps": 2,  # plan's first stop within this many laps -> PREPARE_BOX
    "gain_sc": 0.6,  # places gained by stopping under a safety car to call BOX_IF_SC
    "p_sc": 0.7,  # share of surprise neutralisations that are full safety cars
    "w_time": 0.001,  # places per second of race time: breaks ties between plans with equal expected position
    "lam_prior": 0.02,  # places per lap between a plan's first stop and the typical stop timing (rivals' pit probabilities)
    "use_hazard": True,  # head: time box calls with the pit probabilities (False: the plan alone, as before)
    "p1_box": 0.25,  # BOX needs a stop probability this lap of at least this ...
    "p3_box": 0.35,  # ... or within 3 laps of at least this
    "p3_prep": 0.20,  # PREPARE_BOX needs a stop probability within 3 laps of at least this
    "p1_prep": 0.05,  # PREPARE_BOX also needs a stop probability this lap of at least this
    "prep_near": None,  # PREPARE_BOX also needs stopping now to cost at most this many places (None: no such condition)
    "hold": 0.7,  # a box call made last lap is held down to this share of the thresholds
    "tol_keep": 0.05,  # keep last lap's target stop lap unless the best plan is better by more than this
}
OFFS1 = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 24, 28, 33, 38, 44, 50, 58)
OFFS2 = (0, 3, 6, 10, 15, 21, 27)
GAPS2 = (8, 12, 16, 21, 27)
PHASES = {"none": (0, 0), "sc": (2, 3), "sc_ending": (2, 1), "vsc": (1, 2), "vsc_ending": (1, 1), "red": (2, 3)}


def _num(x, default=None):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else default


def priors_for(ctx) -> Priors:
    p = ctx.__dict__.get("_strategy_priors")
    if p is None:
        p = Priors(ctx.past_races, ctx.meta.get("circuit_key"))
        ctx.__dict__["_strategy_priors"] = p
    return p


def race_key(ctx) -> str:
    m = ctx.meta
    return f"{m.get('year')}|{m.get('session_key')}|{m.get('meeting_name')}"


def seed_for(ctx, car: str, anchor: int, tag: str = "") -> int:
    return zlib.crc32(f"{race_key(ctx)}|{car}|{anchor}|{tag}".encode()) & 0xFFFFFFFF


@dataclass
class PlanResult:
    stops: tuple  # ((in-lap, compound name), ...)
    exp_pos: float
    exp_pts: float
    pos_sd: float
    first_offset: int | None  # laps from the next lap to the first stop (0 = this lap)
    pos: np.ndarray | None = None  # per-simulation finishing positions (not exported)
    util: float = 0.0  # expected position plus a small price on race time: breaks ties between equal positions
    util_s: np.ndarray | None = None  # per-simulation utility

    def to_plan(self, name: str, trigger: str | None = None) -> Plan:
        return Plan(name, tuple(PlanStop(int(l), c) for l, c in self.stops), round(float(self.exp_pos), 2),
                    round(float(self.exp_pts), 2), trigger)


@dataclass
class CarAnalysis:
    car: str
    ok: bool
    why: str = ""
    anchor: int = 0
    total: int = 0
    plan_a: PlanResult | None = None
    plan_b: PlanResult | None = None
    plan_b_trigger: str | None = None
    ranked: list = field(default_factory=list)  # PlanResult, best first (all futures)
    now_best: PlanResult | None = None  # best plan that stops this lap
    later_best: PlanResult | None = None  # best plan whose first stop is later (or none)
    nostop: PlanResult | None = None
    diff_now_later: tuple | None = None  # mean, standard error of (now - later) in places; negative: now is better
    gain_sc: float | None = None  # places gained by stopping under an SC vs staying on plan A, given one soon
    sc_prob5: float = 0.0
    sc_best_comp: str | None = None
    pp: tuple | None = None  # the car's pit probabilities within 1 / 3 / 5 laps as the simulator was given them
    stop_p: tuple = ()  # share of simulated futures (default strategy) with the first stop at offset 0..11 laps from the next lap
    ms: float = 0.0
    n_sims: int = 0
    n_plans: int = 0


# --------------------------------------------------------------------------- the field
def _anchor_time(memory, number: str, A: int, pace: float):
    laps = memory.index.by_driver.get(number, {})
    rec = laps.get(A)
    if rec is not None:
        return rec.t_end, 0
    if not laps:
        return None, 0
    last = laps[max(laps)]
    return last.t_end + (A - last.lap) * pace, max(0, A - last.lap)


def build_field(state, view, memory, ctx, pri: Priors, A: int) -> tuple[FieldIn, list[str], dict]:
    """The running field at the end of lap ``A``, as arrays."""
    tyre_r = view.race("tyre")
    pit_r = view.race("pitstop")
    rule_r = view.race("rules")
    total = state.total_laps or (int(np.median(pri.laps_seen)) if pri.laps_seen else 58)
    fuel = min(0.08, max(0.01, _num(tyre_r.get("fuel_s_per_lap"), 0.035)))
    field_deg = [_num(tyre_r.get(f"field_deg_{c.lower()}")) for c in DRY]
    cars = [d for d in state.running_order() if d.running and state.drivers[d.number].laps >= 1]
    raw = []
    paces = []
    for d in cars:
        t = view.car("tyre", d.number)
        p = _num(t.get("pace_s"))
        if p is None:
            p = _num(t.get("stint_pace")) or _num(d.last_lap_time)
        raw.append((d, t, p))
        if p is not None:
            paces.append(p)
    ref_pace = float(np.median(paces)) if paces else 90.0
    rows = []
    for d, t, p in raw:
        x, lag_est = _anchor_time(memory, d.number, A, p or ref_pace)
        if x is None:
            continue
        rows.append((d, t, p, x))
    C = len(rows)
    names = [r[0].number for r in rows]
    z = lambda: np.zeros(C)  # noqa: E731
    F = FieldIn(A=A, total=total, cars=names, x0=z(), pace0=z(), deg=z(), fuel=fuel, age0=z(), stint_age=z(),
                comp0=np.ones(C, dtype=np.int64), fresh=np.zeros((C, 3)), fresh_ref=z(), degc=np.zeros((C, 3)),
                pp=np.zeros((C, 3)), must=np.zeros(C, dtype=bool), used=np.zeros((C, 3), dtype=bool),
                cliff_risk=z(), team_off=z(), penalty=z())
    off_s, off_h = _num(tyre_r.get("off_soft_s"), -0.6), _num(tyre_r.get("off_hard_s"), 0.5)
    offs = [off_s, 0.0, off_h]
    info: dict = {"stops_done": {}, "stint_left": {}, "rules": {}, "idx": {n: i for i, n in enumerate(names)}}
    for i, (d, t, p, x) in enumerate(rows):
        n = d.number
        lag = int(np.clip(A - d.laps, -3, 6))
        rv, rl, pv, md = view.car("rivals", n), view.car("rules", n), view.car("pitstop", n), view.car("models", n)
        c0 = CIDX.get(d.compound or "", 1)
        deg = min(0.3, max(0.0, _num(t.get("deg_s_per_lap"), field_deg[c0] if field_deg[c0] is not None else 0.08)))
        pace = p if p is not None else ref_pace
        F.x0[i] = x
        F.deg[i] = deg
        F.pace0[i] = pace + (deg - fuel) * lag
        age = d.tyre_age if d.tyre_age is not None else max(0, d.laps - d.stint_start_lap)
        F.age0[i] = max(0, age + lag)
        F.stint_age[i] = max(0, A - d.stint_start_lap)
        F.comp0[i] = c0
        F.fresh_ref[i] = d.laps + 3
        ratio = 1.0
        if field_deg[c0]:
            ratio = float(np.clip(deg / field_deg[c0], 0.6, 1.6))
        for j in range(3):
            fj = _num(t.get(f"fresh_{DRY[j].lower()}_s"))
            if fj is None:  # no tyre fit yet: this set's pace shifted by the compound offsets
                fj = pace - deg * max(F.age0[i] - 1, 0) - 2 * fuel + (offs[j] - offs[c0])
            F.fresh[i, j] = fj
            fd = field_deg[j]
            F.degc[i, j] = deg if j == c0 else min(0.3, max(0.0, (fd if fd is not None else deg * (0.7, 1.0, 1.3)[j]) * ratio))
        pr = [_num(md.get("pit_prob_1")), _num(md.get("pit_prob_3"))]
        pp1 = pr[0] if pr[0] is not None else _num(rv.get("pit_prob_1"), 0.05)
        pp3 = pr[1] if pr[1] is not None else _num(rv.get("pit_prob_3"), 0.15)
        pp5 = _num(rv.get("pit_prob_5"), 0.25)
        F.pp[i] = (pp1, max(pp1, pp3), max(pp1, pp3, pp5))
        F.must[i] = bool(rl.get("must_stop"))
        for comp in (d.compounds_used or []) + [d.compound]:
            if comp in CIDX:
                F.used[i, CIDX[comp]] = True
        F.cliff_risk[i] = _num(t.get("cliff_risk"), 0.0)
        pen = _num(rl.get("penalty_s_pending"), 0.0) + (18.0 if rl.get("drive_through_pending") else 0.0)
        F.penalty[i] = pen
        lin, ln = _num(pv.get("loss_if_box_now")), _num(pit_r.get("loss_now"))
        F.team_off[i] = float(np.clip(lin - ln - _num(pv.get("penalty_s"), 0.0), -3.0, 6.0)) if lin is not None and ln is not None else 0.0
        info["stops_done"][n] = d.pit_stops
        info["stint_left"][n] = _num(rl.get("stint_laps_left"))
        info["rules"][n] = rl
    pr_ = ctx.prior
    F.loss = {
        "green": _num(pit_r.get("loss_green"), pr_.green), "sc": _num(pit_r.get("loss_sc"), pr_.sc),
        "vsc": _num(pit_r.get("loss_vsc"), pr_.vsc), "sd": max(0.8, _num(pit_r.get("loss_now_sd"), 1.2)),
    }
    F.life = np.array([pri.life[c] for c in DRY])
    F.sc_rate, F.vsc_rate, F.sc_len, F.vsc_len = pri.sc_rate, pri.vsc_rate, pri.sc_len, pri.vsc_len
    F.sc_now, F.sc_now_left = PHASES.get(str(rule_r.get("sc_phase") or "none"), (0, 0))
    F.pass_p = tuple(pri.pass_p)
    F.ref_pace = ref_pace
    F.max_stint = _num(rule_r.get("reg_max_stint_laps"), 0.0) or 0.0
    F.reg_min_stops = int(_num(rule_r.get("reg_min_stops"), 0) or 0)
    F.stints = []
    for j, c in enumerate(DRY):
        done, opened = pri.stints[c], pri.stints_open[c]
        if len(done) < 5:  # no history: spread around the default life
            done = [int(round(LIFE[c] * f)) for f in np.linspace(0.75, 1.3, 11)]
            opened = []
        F.stints.append((done, opened))
    allnum = sorted(state.drivers, key=lambda x: (int(x) if x.isdigit() else 999, x))
    slot = {n: i for i, n in enumerate(allnum)}
    F.slots = np.array([slot[n] for n in names], dtype=np.int64)
    F.n_slots = len(allnum)
    F.x0 -= F.x0.min() if C else 0.0
    return F, names, info


# --------------------------------------------------------------------------- candidate plans
def candidates(F: FieldIn, c: int, info: dict, number: str) -> list[tuple]:
    """Plans as tuples of (in-lap, compound index), legal under the rules and plausible for tyre life."""
    A, total, life = F.A, F.total, F.life
    last = total - 3
    used = {j for j in range(3) if F.used[c, j]}
    stops_done = info["stops_done"].get(number, 0)
    stint_left = info["stint_left"].get(number)
    max_stint = F.max_stint or None

    def legal(plan: tuple) -> bool:
        comps = {cj for _, cj in plan}
        if F.must[c] and len(used | comps) < 2:
            return False
        if stops_done + len(plan) < F.reg_min_stops:
            return False
        laps = [l for l, _ in plan]
        if plan and stint_left is not None and laps[0] > A + max(stint_left, 0):
            return False
        if not plan and stint_left is not None and stint_left < F.R:
            return False
        prev = laps[0] if plan else None
        bounds = laps + [total]
        for k, (l, cj) in enumerate(plan):
            nxt = bounds[k + 1]
            if nxt - l > 1.35 * life[cj] or nxt - l < 4 and nxt != total:
                return False
            if max_stint and nxt - l > max_stint:
                return False
        if plan and plan[-1][0] > last:
            return False
        return all(b > a for a, b in zip(laps, laps[1:]))

    plans: list[tuple] = []
    if legal(()):
        plans.append(())
    for o in OFFS1:
        j = A + 1 + o
        if j > last:
            break
        for cj in range(3):
            p = ((j, cj),)
            if legal(p):
                plans.append(p)
    for o in OFFS2:
        j1 = A + 1 + o
        for g in GAPS2:
            j2 = j1 + g
            if j2 > last:
                continue
            for c1 in range(3):
                for c2 in range(3):
                    p = ((j1, c1), (j2, c2))
                    if legal(p):
                        plans.append(p)
    if F.R >= 30:
        for o in (2, 8):
            for g1 in (8, 14):
                for g2 in (8, 14):
                    j1 = A + 1 + o
                    j2, j3 = j1 + g1, j1 + g1 + g2
                    if j3 > last:
                        continue
                    for combo in ((0, 1, 1), (1, 0, 0), (2, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 2)):
                        p = ((j1, combo[0]), (j2, combo[1]), (j3, combo[2]))
                        if legal(p):
                            plans.append(p)
    return plans


def _arrays(plans: list[tuple], S: int) -> tuple[np.ndarray, np.ndarray]:
    P = len(plans)
    stop = np.full((P, K_STOPS), NOSTOP, dtype=np.int64)
    comp = np.zeros((P, K_STOPS), dtype=np.int64)
    for i, p in enumerate(plans):
        for k, (l, cj) in enumerate(p[:K_STOPS]):
            stop[i, k], comp[i, k] = l, cj
    return np.repeat(stop[:, None, :], S, 1), np.repeat(comp[:, None, :], S, 1)


def _points(pos: np.ndarray) -> np.ndarray:
    idx = np.clip(pos.astype(int) - 1, 0, 10)
    return np.concatenate([POINTS, [0.0]])[idx]


def prior_offset(F: FieldIn, c: int) -> float:
    """Typical laps until this car's next stop from the stop probabilities (as the rivals' hazard reads them)."""
    p1, p3, p5 = F.pp[c]
    if p1 >= 0.5:
        return 0.0
    if p3 >= 0.5:
        return 2 * (0.5 - p1) / max(p3 - p1, 1e-6)
    if p5 >= 0.5:
        return 2 + 2 * (0.5 - p3) / max(p5 - p3, 1e-6)
    return max(5.0, F.life[F.comp0[c]] - F.stint_age[c] - 1)


def _prior_pen(F: FieldIn, c: int, plan: tuple) -> float:
    lam = SETTINGS["lam_prior"]
    if not lam:
        return 0.0
    o = prior_offset(F, c)
    if plan:
        return lam * min(15.0, abs(plan[0][0] - (F.A + 1) - o))
    return lam * min(15.0, max(0.0, F.R - o))


def _result(plan: tuple, pos: np.ndarray, time: np.ndarray, soft: np.ndarray, A: int, tref: float, pen: float = 0.0) -> PlanResult:
    u = soft + SETTINGS["w_time"] * (time - tref) + pen
    return PlanResult(tuple((int(l), DRY[cj]) for l, cj in plan), float(pos.mean()), float(_points(pos).mean()),
                      float(pos.std()), (plan[0][0] - (A + 1)) if plan else None, pos, float(u.mean()), u)


# --------------------------------------------------------------------------- one focus car
def analyse_focus(F: FieldIn, info: dict, run: FieldRun, D: Draws, number: str, ctx, S1: int, keep: int) -> CarAnalysis:
    c = F.cars.index(number)
    A = F.A
    out = CarAnalysis(number, True, anchor=A, total=F.total)
    plans = candidates(F, c, info, number)
    out.n_plans = len(plans)
    out.pp = tuple(float(x) for x in F.pp[c])
    s0 = run.stop[:, c, 0] - (A + 1)
    out.stop_p = tuple(float(np.mean(s0 == o)) for o in range(12))
    if not plans:
        out.ok, out.why = False, "no legal plan"
        return out
    S1 = min(S1, D.S)
    stop, comp = _arrays(plans, S1)
    pos1, time1, soft1 = evaluate_plans(F, D, run, c, stop, comp, S1)
    means = (soft1 + SETTINGS["w_time"] * (time1 - time1.mean())).mean(1) + np.array([_prior_pen(F, c, p) for p in plans])
    order = np.argsort(means, kind="stable")
    chosen = list(order[:keep])
    first = np.array([p[0][0] - (A + 1) if p else -1 for p in plans])
    # keep the best plan for each of the next few stop laps, for no stop and for each stop count
    for o in range(8):
        for i in order:
            if first[i] == o:
                chosen.append(i)
                break
    for n in (0, 1, 2, 3):
        for i in order:
            if len(plans[i]) == n:
                chosen.append(i)
                break
    chosen = sorted(set(int(i) for i in chosen), key=lambda i: means[i])
    sub = [plans[i] for i in chosen]
    stop, comp = _arrays(sub, D.S)
    pos, time, soft = evaluate_plans(F, D, run, c, stop, comp, D.S)
    tref = float(time.mean())
    res = [_result(p, pos[i], time[i], soft[i], A, tref, _prior_pen(F, c, p)) for i, p in enumerate(sub)]
    out.ranked = sorted(res, key=lambda r: (r.util, r.exp_pos))
    out.plan_a = out.ranked[0]
    now = [r for r in out.ranked if r.first_offset == 0]
    later = [r for r in out.ranked if r.first_offset is None or r.first_offset > 0]
    out.now_best = now[0] if now else None
    out.later_best = later[0] if later else None
    ns = [r for r in out.ranked if not r.stops]
    out.nostop = ns[0] if ns else None
    if out.now_best is not None and out.later_best is not None:
        d = out.now_best.util_s - out.later_best.util_s
        out.diff_now_later = (float(d.mean()), float(d.std() / math.sqrt(len(d))))
    return out


def sc_scenario(F: FieldIn, info: dict, number: str, ctx, plan_a: PlanResult, S_B: int, seed: int) -> tuple:
    """Plan B: what to do if a safety car / VSC comes within the next few laps.

    Returns (best reactive PlanResult or None, trigger text, gain in places over staying on plan A, best compound).
    """
    c = F.cars.index(number)
    A, total = F.A, F.total
    if F.R < 10 or F.sc_now:
        return None, None, None, None
    DB = Draws(F, S_B, seed, force_sc=(0, 4, SETTINGS["p_sc"]))
    runB = simulate_field(F, DB)
    P = 1 + 3
    stop = np.full((P, S_B, K_STOPS), NOSTOP, dtype=np.int64)
    comp = np.zeros((P, S_B, K_STOPS), dtype=np.int64)
    for k, (l, cn) in enumerate(plan_a.stops[:K_STOPS]):
        stop[0, :, k], comp[0, :, k] = l, DRY.index(cn)
    l0 = A + 1 + DB.sc_start  # the lap the neutralisation starts: the stop is at its end
    late = l0 > total - 3
    for j in range(3):
        stop[1 + j, :, 0] = np.where(late, NOSTOP, l0)
        comp[1 + j, :, 0] = j
        rem = total - l0
        gap = max(8, int(0.9 * F.life[j]))
        second = np.minimum(l0 + gap, total - 3)
        use = (rem > 1.1 * F.life[j]) & (second > l0 + 5) & ~late
        stop[1 + j, :, 1] = np.where(use, second, NOSTOP)
        comp[1 + j, :, 1] = np.where(rem - gap > F.life[1], 2, 1)
    pos, time, soft = evaluate_plans(F, DB, runB, c, stop, comp, S_B)
    mean = (soft + SETTINGS["w_time"] * (time - time.mean())).mean(1)
    posm = pos.mean(1)
    best = 1 + int(np.argmin(mean[1:]))
    gain = float(posm[0] - posm[best])
    stops_txt = (int(l0.mean().round()), DRY[best - 1])
    res = PlanResult(((stops_txt[0], DRY[best - 1]),), float(posm[best]), float(_points(pos[best]).mean()), float(pos[best].std()), 0, pos[best])
    trigger = f"SC/VSC within 5 laps: box that lap for {DRY[best - 1]}"
    return res, trigger, gain, DRY[best - 1]


# --------------------------------------------------------------------------- entry point
def analyse(state, view, ctx, memory, cars: list[str], *, S: int | None = None, light: bool = False) -> dict[str, CarAnalysis]:
    """Plans for ``cars`` as of now. Cars on the same lap share one simulated field."""
    import time

    pri = priors_for(ctx)
    st = SETTINGS
    S = S or (96 if light else st["S"])
    S1 = min(st["S1"], S)
    out: dict[str, CarAnalysis] = {}
    wet = not bool(view.race("rules").get("race_dry", True)) or bool(view.race("weather").get("wet_running"))
    by_anchor: dict[int, list[str]] = {}
    for n in cars:
        d = state.drivers.get(n)
        if d is None or not d.running:
            continue
        if d.laps < 2:
            out[n] = CarAnalysis(n, False, "too early: no laps to read pace from")
            continue
        if wet:
            out[n] = CarAnalysis(n, False, "wet conditions: the simulator only models dry tyres")
            continue
        by_anchor.setdefault(d.laps, []).append(n)
    for A, group in sorted(by_anchor.items()):
        t0 = time.perf_counter()
        F, names, info = build_field(state, view, memory, ctx, pri, A)
        if F.R <= 0 or F.C < 2:
            for n in group:
                out[n] = CarAnalysis(n, False, "race is over", anchor=A)
            continue
        seed = seed_for(ctx, "", 0)
        D = Draws(F, S, seed)
        run = simulate_field(F, D)
        for n in group:
            if n not in names:
                out[n] = CarAnalysis(n, False, "no timing for the car yet", anchor=A)
                continue
            a = analyse_focus(F, info, run, D, n, ctx, S1, 8 if light else st["keep"])
            if a.ok and not light:
                sc = sc_scenario(F, info, n, ctx, a.plan_a, min(st["S_B"], S), seed_for(ctx, "", 0, "sc"))
                a.plan_b, a.plan_b_trigger, a.gain_sc, a.sc_best_comp = sc
            a.sc_prob5 = 1 - (1 - min(0.5, F.sc_rate + F.vsc_rate)) ** 5
            a.n_sims = S
            a.ms = (time.perf_counter() - t0) * 1000
            out[n] = a
    return out
