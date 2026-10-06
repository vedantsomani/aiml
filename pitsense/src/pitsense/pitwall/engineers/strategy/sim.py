"""Lap-by-lap Monte Carlo of the rest of the race, vectorised over simulated futures.

Two parts, both on the same random numbers (common random numbers):

* ``simulate_field``: every running car follows a sampled default strategy (stop timing from the rivals'
  pit probabilities, then stint lengths from past races); cars queue behind slower cars and pass with a
  chance that depends on the pace advantage; safety cars and VSCs bunch the field and cheapen stops.
* ``evaluate_plans``: one focus car is replayed against those simulated opponents with each candidate plan
  (stop laps and tyres). Each lap only needs the nearest car ahead and behind (a sorted search), so
  hundreds of plans cost about the same as a few.

Times are seconds relative to the leader at the anchor lap ``A`` (the lap the focus car last completed);
laps are indexed 0..R-1 for race laps A+1 .. A+R = total.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .priors import PASS_BINS

NOSTOP = 10**6  # "no stop" lap
K_STOPS = 3
N_EV = 4  # neutralisations tracked per future (for stop pulls and the red flag); more may run
SC_LAP_FACTOR = 1.45  # lap time under the safety car vs a normal lap
VSC_LAP_FACTOR = 1.30
SC_CAP_S = 1.0  # gap between cars after the field bunches behind the safety car
FOLLOW_S = 0.45  # a car stuck behind another arrives this far behind it
RETIRED = 1e6  # time of a retired car (sorts after every finisher)
PLACEHOLDER = 5e6  # sorts after everything; the focus car's own slot among opponents
OFF = 1e7  # row offset that makes a flattened per-simulation search possible
PASS_TOP = 0.98  # pass chance when the car ahead is slowed by a stop

# knobs fitted/tuned on 2025 (see docs/engineers/strategy.md)
PARAMS = {
    "level_sd": 0.10,  # s: uncertainty of a car's pace level
    "noise_sd": 0.20,  # s: lap-to-lap noise
    "deg_sd": 0.20,  # log-sd of the degradation slope
    "cliff_mult": 1.7,  # tyre cliff at this multiple of the typical stint
    "cliff_sd": 0.18,
    "cliff_slope": 0.25,  # s per lap beyond the cliff
    "cliff_cap": 6.0,
    "retire_rate": 0.0003,  # per car-lap
    "sc_pull": 0.85,  # share of cars due to stop soon that come in under a safety car
    "vsc_pull": 0.45,
    "sc_pull_window": 20,
    "stop_extra_s": 3.0,  # s added to every green-flag stop: cold-tyre laps (rejoin traffic is simulated)
    "stop_extra_neutral_s": 0.0,  # the same under a safety car / VSC (the field runs slowly on cold tyres too)
    "air_scale": 0.5,  # share of the learnt within-1-s lap-time loss charged as dirty air (the rest is the queue)
    "restart_gap": 2,  # green laps at least between two neutralisations
    "stint_scale": 1.12,  # stints drawn from past races are stretched by this factor (opponents' later stops)
    "hold_gain": 1.0,  # most a defending car can gain (s) per lap by holding a rival off
}


@dataclass
class FieldIn:
    """Everything the simulator needs about the field at the anchor lap (arrays are per car)."""

    A: int
    total: int
    cars: list[str]
    x0: np.ndarray  # s behind the leader at the end of lap A
    pace0: np.ndarray  # expected lap time of lap A+1 on the current set
    deg: np.ndarray  # lap-time increase per lap on the current set (fuel removed)
    fuel: float  # lap-time gain per lap from fuel burn
    age0: np.ndarray  # laps on the current set when lap A+1 starts
    stint_age: np.ndarray  # laps run since the current set was fitted
    comp0: np.ndarray  # current compound 0 soft / 1 medium / 2 hard
    fresh: np.ndarray  # [C,3] first flying lap on a new set fitted at the end of lap fresh_ref
    fresh_ref: np.ndarray  # lap number the fresh_* values refer to
    degc: np.ndarray  # [C,3] degradation per lap by compound
    pp: np.ndarray  # [C,3] pit probability within 1 / 3 / 5 laps
    must: np.ndarray  # bool: still has to fit another compound
    used: np.ndarray  # [C,3] bool compounds used so far (incl. current)
    cliff_risk: np.ndarray
    team_off: np.ndarray  # s: team's stop time vs field
    penalty: np.ndarray  # s of time penalty still to serve
    loss: dict = field(default_factory=dict)  # green / vsc / sc / sd (s)
    life: np.ndarray = field(default_factory=lambda: np.array([18.0, 28.0, 38.0]))
    sc_rate: float = 0.0075  # safety-car starts per lap
    vsc_rate: float = 0.0045
    sc_near: float | None = None  # per-lap SC start rate for the next 2 laps from the safetycar engineer (None: sc_rate)
    vsc_near: float | None = None
    sc_len: float = 4.0
    vsc_len: float = 2.0
    haz: np.ndarray | None = None  # [3,R] per-lap start probability of red flag / SC / VSC (None: sc_rate, vsc_rate)
    sc_lens: tuple = ()  # past SC lengths in laps (sorted); empty: geometric around sc_len
    vsc_lens: tuple = ()
    air: float = 0.0  # s/lap lost within 1 s of the car ahead (learnt; scaled by PARAMS["air_scale"])
    sc_now: int = 0  # 0 green, 1 VSC, 2 SC in force at the anchor
    sc_now_left: int = 0  # laps of it still to run (including the one in progress)
    stints: list = field(default_factory=list)  # per compound: (sorted done lengths, sorted open lengths)
    pass_p: tuple = (0.02, 0.05, 0.12, 0.25, 0.40, 0.60)
    max_stint: float = 0.0  # regulation stint limit in laps (0 = none)
    reg_min_stops: int = 0
    ref_pace: float = 90.0
    slots: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # random-number slot per car
    n_slots: int = 0

    @property
    def R(self) -> int:
        return max(0, self.total - self.A)

    @property
    def C(self) -> int:
        return len(self.cars)


class Draws:
    """The random numbers of S simulated futures. Every plan and every car sees the same ones."""

    def __init__(self, F: FieldIn, S: int, seed: int, *, force_sc: tuple | None = None) -> None:
        """Arrays are drawn for every driver slot and every lap of the race, then cut to the cars and laps
        still to run, so the same seed gives the same futures lap after lap (calls do not flicker on noise)."""
        rng = np.random.default_rng(seed)
        n, Rt, sl, A = F.n_slots, max(F.total, 1), F.slots, F.A
        P = PARAMS
        self.S = S
        cut = lambda x: x[:, sl]  # noqa: E731
        self.level = cut(rng.normal(0, P["level_sd"], (S, n)))
        self.degmul = cut(np.exp(rng.normal(0, P["deg_sd"], (S, n))))
        self.noise = rng.normal(0, P["noise_sd"], (S, n, Rt))[:, sl, A:Rt]
        self.u_stop = cut(rng.random((S, n)))
        self.u_after = cut(rng.random((S, n, K_STOPS)))
        self.u_open = cut(rng.random((S, n, K_STOPS)))
        self.u_comp = cut(rng.random((S, n, K_STOPS)))
        self.u_pull = cut(rng.random((S, n)))
        self.u_pass = rng.random((S, n, Rt))[:, sl, A:Rt]
        self.z_pit = cut(rng.normal(0, 1, (S, n, K_STOPS)))
        self.z_cliff = cut(rng.normal(0, 1, (S, n, 3)))
        self.u_cliff = cut(rng.random((S, n, 2)))
        self.u_def = rng.random((S, Rt))[:, A:Rt]
        self.u_force = cut(rng.random((S, n)))
        u_ret = cut(rng.random((S, n)))
        self.ret_lap = np.floor(np.log(np.maximum(u_ret, 1e-12)) / np.log(1 - P["retire_rate"])).astype(np.int64)
        self.u_sc = rng.random((S, 4))  # start, kind, length draws for the safety-car scenario
        self.u_sc_lap = rng.random((S, 2, Rt))[:, :, A:Rt]  # per-lap start draws: VSC, SC
        self.u_len = rng.random((S, 2))
        self.u_ev = rng.random((S, Rt))[:, A:Rt]  # neutralisation start draw per lap
        self.u_dur = rng.random((S, Rt))[:, A:Rt]  # its length
        self.u_pull2 = cut(rng.random((S, n, N_EV)))
        self.status, self.sc_start = self._status(F, rng, force_sc)

    def _status(self, F: FieldIn, rng, force_sc):
        """Track status per simulated lap: 0 green, 1 VSC, 2 SC, 3 red flag (its lap: field bunched, free tyre
        change; then a lap behind the safety car). Any number of neutralisations per future: a per-lap start
        hazard (circuit and lap phase, ``F.haz``), lengths drawn from past ones, at least ``restart_gap`` green laps
        between two. Returns the status and the first neutralisation's start lap index (R if none); also sets
        ``ev_start`` / ``ev_kind`` [S,N_EV] (all starts, R if none) and ``red_lap`` [S] (absolute lap of the first
        red flag, NOSTOP if none)."""
        S, R, A = self.S, max(F.R, 1), F.A
        haz = np.zeros((3, R))
        if F.haz is not None and F.haz.shape[1] >= F.R:
            haz[:, :F.R] = F.haz[:, :F.R]
        else:
            haz[1], haz[2] = F.sc_rate, F.vsc_rate
        if F.sc_near is not None:
            haz[1, :2] = F.sc_near
        if F.vsc_near is not None:
            haz[2, :2] = F.vsc_near
        cum = np.cumsum(haz, 0)  # one draw per lap: < red, < red + sc, < red + sc + vsc
        lens = {2: (F.sc_lens, F.sc_len), 1: (F.vsc_lens, F.vsc_len)}

        def dur(kind: np.ndarray, u: np.ndarray) -> np.ndarray:
            out = np.ones(kind.shape, dtype=np.int64)
            for k, (tab, mean) in lens.items():
                sel = kind == k
                if not sel.any():
                    continue
                if len(tab) >= 5:
                    t = np.asarray(tab, dtype=np.int64)
                    out[sel] = t[np.minimum((u[sel] * len(t)).astype(int), len(t) - 1)]
                else:
                    uu = np.clip(u[sel], 1e-9, 1 - 1e-9)
                    out[sel] = 1 + np.floor(-np.log(1 - uu) * max(mean - 1, 0.3)).astype(np.int64)
            out[kind == 3] = 2
            return out

        status = np.zeros((S, R), dtype=np.int8)
        left = np.zeros(S, dtype=np.int64)  # laps of the running neutralisation still to go
        kind_now = np.zeros(S, dtype=np.int64)
        cool = np.zeros(S, dtype=np.int64)
        gap = int(PARAMS["restart_gap"])
        ev_start = np.full((S, N_EV), R, dtype=np.int64)
        ev_kind = np.zeros((S, N_EV), dtype=np.int64)
        n_ev = np.zeros(S, dtype=np.int64)
        forced_at = np.full(S, -1, dtype=np.int64)
        forced_kind = np.zeros(S, dtype=np.int64)
        if force_sc is not None:  # scenario: a safety car / VSC starting within the next few laps
            lo, hi, p_sc = force_sc
            forced_at = lo + np.minimum((self.u_sc[:, 0] * (hi - lo + 1)).astype(int), hi - lo)
            forced_kind = np.where(self.u_sc[:, 1] < p_sc, 2, 1)
        elif F.sc_now:  # a safety car / VSC is out now: it covers the laps still to run
            u = np.clip(self.u_len[:, 0], 1e-9, 1 - 1e-9)
            left = 1 + np.floor(-np.log(1 - u) * max(F.sc_now_left, 0.3)).astype(np.int64)
            kind_now[:] = min(int(F.sc_now), 2)
            ev_start[:, 0], ev_kind[:, 0], n_ev[:] = 0, kind_now, 1
        rows = np.arange(S)
        nu = self.u_ev.shape[1]
        for l in range(R):
            u = self.u_ev[:, l] if l < nu else np.ones(S)
            k_new = np.where(u < cum[0, l], 3, np.where(u < cum[1, l], 2, np.where(u < cum[2, l], 1, 0)))
            free = (left <= 0) & (cool <= 0)
            if force_sc is not None:
                k_new = np.where(l < forced_at, 0, np.where(l == forced_at, forced_kind, k_new))
                free = free | ((left <= 0) & (l == forced_at))
            start = free & (k_new > 0)
            if start.any():
                kn = np.where(start, k_new, 0)
                d = dur(kn, self.u_dur[:, l] if l < nu else np.full(S, 0.5))
                left = np.where(start, d, left)
                kind_now = np.where(start, kn, kind_now)
                keep = start & (n_ev < N_EV)
                ev_start[rows[keep], n_ev[keep]] = l
                ev_kind[rows[keep], n_ev[keep]] = kn[keep]
                n_ev = n_ev + start
            on = left > 0
            # a red flag's first lap is the stoppage; the second runs behind the safety car
            status[:, l] = np.where(on, np.where((kind_now == 3) & (left == 1), 2, kind_now), 0)
            ending = on & (left == 1)
            left = np.maximum(left - 1, 0)
            cool = np.where(ending, gap, np.maximum(cool - 1, 0))
        self.ev_start, self.ev_kind = ev_start, ev_kind
        red_i = np.where(ev_kind == 3, ev_start, R).min(1)
        self.red_lap = np.where(red_i < F.R, A + 1 + red_i, NOSTOP).astype(np.int64)
        return status, np.minimum(ev_start[:, 0], R)


def red_tyres(F: FieldIn, red_lap: np.ndarray, stop: np.ndarray, comp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A red flag gives every car a free tyre change: the first stop planned after the red-flag lap moves to it; a
    car with no stop left takes a set that reaches the flag (if more than 8 laps remain and a stop slot is free).
    ``red_lap`` (absolute lap, NOSTOP if none) broadcasts against ``stop[..., 0]``."""
    if not (red_lap < NOSTOP).any():
        return stop, comp
    stop, comp = stop.copy(), comp.copy()
    r = np.broadcast_to(red_lap, stop.shape[:-1])
    has_red = r < NOSTOP
    after = (stop > r[..., None]) & (stop < NOSTOP)
    any_after = after.any(-1)
    j = after.argmax(-1)[..., None]
    mv = has_red & any_after
    np.put_along_axis(stop, j, np.where(mv, r, np.take_along_axis(stop, j, -1)[..., 0])[..., None], -1)
    n_used = (stop < NOSTOP).sum(-1)
    at_red = (stop == r[..., None]).any(-1)
    rem = F.total - r
    add = has_red & ~any_after & ~at_red & (n_used < stop.shape[-1]) & (rem > 8)
    if add.any():
        life = F.life
        cj = np.where(rem <= 0.9 * life[0], 0, np.where(rem <= 1.1 * life[1], 1, 2))
        slot = np.minimum(n_used, stop.shape[-1] - 1)[..., None]
        np.put_along_axis(stop, slot, np.where(add, r, np.take_along_axis(stop, slot, -1)[..., 0])[..., None], -1)
        np.put_along_axis(comp, slot, np.where(add, cj, np.take_along_axis(comp, slot, -1)[..., 0])[..., None], -1)
    return stop, comp


# --------------------------------------------------------------------------- helpers
def pass_prob(delta: np.ndarray, table: tuple) -> np.ndarray:
    """Chance to get past per lap given the chaser's pace advantage; saturates when the car ahead stops."""
    k = np.searchsorted(PASS_BINS, delta, side="right")
    p = np.asarray(table)[k]
    return np.where(delta > 3.0, PASS_TOP, p)


def _loss_by_status(F: FieldIn, status_lap: np.ndarray) -> np.ndarray:
    L = F.loss
    return np.where(status_lap == 2, L["sc"], np.where(status_lap == 1, L["vsc"], L["green"]))


def _cliff_lap(F: FieldIn, D: Draws, c: int, age: np.ndarray, cliff_age: np.ndarray) -> np.ndarray:
    P = PARAMS
    return P["cliff_slope"] * np.clip(age - cliff_age, 0.0, P["cliff_cap"] / P["cliff_slope"])


def cliff_ages(F: FieldIn, D: Draws, c: int) -> tuple[np.ndarray, np.ndarray]:
    """Per simulation: age at which the current set falls off [S], and for each new compound [S,3]."""
    P = PARAMS
    now_age = F.age0[c] + 1.0
    typical = F.life[F.comp0[c]] * P["cliff_mult"]
    drawn = typical * np.exp(P["cliff_sd"] * D.z_cliff[:, c, 0])
    r = F.cliff_risk[c]
    soon = D.u_cliff[:, c, 0] < r
    cur = np.where(soon, now_age + 1.0 + 2.0 * D.u_cliff[:, c, 1], np.maximum(drawn, now_age + 3.5))
    new = F.life[None, :] * P["cliff_mult"] * np.exp(P["cliff_sd"] * D.z_cliff[:, c, :])
    return cur, new


def lap_times_for(F: FieldIn, D: Draws, c: int, stop_lap: np.ndarray, comp: np.ndarray, cur_cliff, new_cliff) -> np.ndarray:
    """Free lap times [P,S,R] of car ``c`` under plans.

    stop_lap [P,S,K] absolute in-laps (NOSTOP if none); comp [P,S,K] compound index fitted at each stop.
    """
    A, R, fuel = F.A, F.R, F.fuel
    Pn, S, K = stop_lap.shape
    i = np.arange(R)
    lap = A + 1 + i  # [R]
    lvl = D.level[None, :, c, None]
    dm = D.degmul[None, :, c, None]
    noise = D.noise[None, :, c, :R]
    # stint 0: the current set
    age = F.age0[c] + 1.0 + i  # age during each lap
    t0 = F.pace0[c] + (F.deg[c] * dm - fuel) * i[None, None, :]
    t0 = t0 + _cliff_lap(F, D, c, age[None, None, :], cur_cliff[None, :, None])
    m = (stop_lap[:, :, :, None] < lap[None, None, None, :]).sum(2)  # [P,S,R] number of stops done before each lap
    out = np.where(m == 0, t0, 0.0)
    for k in range(min(K, K_STOPS)):
        sel = m == (k + 1)
        if not sel.any():
            continue
        sl = stop_lap[:, :, k][:, :, None]  # in-lap of the stop that started this stint
        cj = comp[:, :, k]  # [P,S]
        ag = lap[None, None, :] - sl  # age during lap; out-lap = 1
        fresh = F.fresh[c][cj]  # [P,S]
        deg = F.degc[c][cj]
        ca = np.take_along_axis(new_cliff[None, :, :].repeat(Pn, 0), cj[:, :, None], 2)  # [P,S,1]
        t = fresh[:, :, None] + deg[:, :, None] * dm * (ag - 2.0) - fuel * (lap[None, None, :] - F.fresh_ref[c])
        t = t + _cliff_lap(F, D, c, ag, ca)
        out = np.where(sel, t, out)
    return out + lvl + noise


# --------------------------------------------------------------------------- opponents' strategies
def _empirical(F: FieldIn, comp: np.ndarray, min_len: np.ndarray, u: np.ndarray, u_open: np.ndarray):
    """Stint length (laps) drawn from past stints of ``comp`` that lasted at least ``min_len``; NOSTOP if the
    past stint ran to the flag (censored)."""
    out = np.full(comp.shape, NOSTOP, dtype=np.int64)
    for j in range(3):
        done, opened = F.stints[j]
        sel = comp == j
        if not sel.any():
            continue
        d = np.asarray(done, dtype=float)
        o = np.asarray(opened, dtype=float)
        ml = min_len[sel]
        if d.size == 0:
            continue
        i0 = np.searchsorted(d, ml, side="left")
        n_done = d.size - i0
        n_open = o.size - np.searchsorted(o, ml, side="left") if o.size else np.zeros_like(n_done)
        pick = np.minimum(i0 + (u[sel] * np.maximum(n_done, 1)).astype(int), d.size - 1)
        p_open = n_open / np.maximum(n_done + n_open, 1)
        length = np.where((n_done == 0) | (u_open[sel] < p_open), NOSTOP, d[pick])
        out[sel] = length
    return out


def sample_schedules(F: FieldIn, D: Draws):
    """Opponents' (and baseline) stop plan: stop_lap [S,C,K] (absolute in-lap, NOSTOP if none), comp [S,C,K]."""
    S, C, R, A, total = D.S, F.C, F.R, F.A, F.total
    stop = np.full((S, C, K_STOPS), NOSTOP, dtype=np.int64)
    comp = np.zeros((S, C, K_STOPS), dtype=np.int64)
    pp = np.maximum.accumulate(np.clip(F.pp, 0.0, 0.995), axis=1)  # [C,3]
    u = D.u_stop
    p1, p3, p5 = pp[None, :, 0], pp[None, :, 1], pp[None, :, 2]
    k = np.where(u < p1, 1.0,
                 np.where(u < p3, 1.0 + 2.0 * (u - p1) / np.maximum(p3 - p1, 1e-9),
                          3.0 + 2.0 * (u - p3) / np.maximum(p5 - p3, 1e-9)))
    k = np.ceil(k).astype(np.int64)
    first_in = A + k  # in-lap of the first stop when sampled within 5 laps
    beyond = u >= p5
    # later: stint length from past races, at least 6 laps from now
    cur_c = np.broadcast_to(F.comp0[None, :], (S, C))
    ml = np.broadcast_to(F.stint_age[None, :] + 6.0, (S, C))
    length = _empirical(F, cur_c, ml, D.u_after[:, :, 0], D.u_open[:, :, 0])
    later = np.where(length >= NOSTOP, NOSTOP, A + (length * PARAMS["stint_scale"] - F.stint_age[None, :]).astype(np.int64))
    s0 = np.where(beyond, later, first_in)
    # safety car / VSC: cars due to stop soon come in (each neutralisation in turn; red flags are free changes)
    in_prog = bool(F.sc_now)
    win = PARAMS["sc_pull_window"]
    ev_start = getattr(D, "ev_start", D.sc_start[:, None])
    ev_kind = getattr(D, "ev_kind", None)
    for e in range(ev_start.shape[1]):
        l0 = ev_start[:, e]  # [S] start lap index of this neutralisation (R if none)
        if not (l0 < R).any():
            break
        if ev_kind is None:
            kind = np.take_along_axis(D.status, np.minimum(l0, max(R, 1) - 1)[:, None], 1)[:, 0]
        else:
            kind = ev_kind[:, e]
        if e == 0 and in_prog:
            continue
        pull_p = np.where(kind == 2, PARAMS["sc_pull"], np.where(kind == 1, PARAMS["vsc_pull"], 0.0))[:, None]
        lap0 = (A + 1 + l0)[:, None]  # absolute lap of its first lap
        due = (s0 >= lap0) & (s0 <= lap0 + win)
        age_at = F.stint_age[None, :] + (lap0 - A)
        ratio = age_at / np.maximum(F.life[F.comp0][None, :], 5.0)
        extra = (s0 > lap0 + win) & (ratio > 0.55) & ((total - lap0) > 8)  # no stop due: old tyres, take the cheap one
        u_p = D.u_pull if e == 0 else D.u_pull2[:, :, e]
        pull = (u_p < pull_p * np.where(due, 1.0, np.where(extra, 0.55 * np.minimum(ratio, 1.2), 0.0))) & (l0 < R)[:, None]
        s0 = np.where(pull & (lap0 <= total - 3), lap0, s0)
    # must-stop cars always make one
    need = F.must[None, :] & (s0 >= total - 2)
    forced = A + 3 + (D.u_force * np.maximum(R - 6, 1)).astype(np.int64)
    s0 = np.where(need, np.minimum(forced, total - 3), s0)
    s0 = np.where(s0 > total - 2, NOSTOP, s0)
    s0 = np.where(s0 <= A, A + 1, s0)
    stop[:, :, 0] = s0

    def choose(prev_comp, rem, used_mask, uc):
        w = np.array([0.25, 0.45, 0.30])[None, None, :] * np.ones((S, C, 3))
        same = prev_comp[:, :, None] == np.arange(3)[None, None, :]
        w = np.where(same, w * 0.35, w)
        short = rem[:, :, None] > 1.1 * F.life[None, None, :]
        w = np.where(short, w * 0.15, w)
        unused = ~used_mask
        has_unused = unused.any(2, keepdims=True) & F.must[None, :, None] & True
        w = np.where(has_unused & used_mask, 1e-6, w)  # a second compound is mandatory: pick one not used yet
        w = w / w.sum(2, keepdims=True)
        cw = np.cumsum(w, 2)
        return (uc[:, :, None] > cw).sum(2).clip(0, 2)

    used = np.broadcast_to(F.used[None, :, :], (S, C, 3)).copy()
    prev = cur_c.copy()
    for j in range(K_STOPS):
        has = stop[:, :, j] < NOSTOP
        rem = np.where(has, total - stop[:, :, j], 0.0)
        c_j = choose(prev, rem, used, D.u_comp[:, :, j])
        comp[:, :, j] = np.where(has, c_j, 0)
        used = used | (has[:, :, None] & (c_j[:, :, None] == np.arange(3)[None, None, :]))
        if j + 1 < K_STOPS:
            length = _empirical(F, c_j, np.full(c_j.shape, 6.0), D.u_after[:, :, j + 1], D.u_open[:, :, j + 1])
            nxt = np.where(has & (length < NOSTOP), stop[:, :, j] + length, NOSTOP)
            stop[:, :, j + 1] = np.where(nxt <= total - 2, nxt, NOSTOP)
        prev = np.where(has, c_j, prev)
    red = getattr(D, "red_lap", None)
    if red is not None:
        stop, comp = red_tyres(F, red[:, None], stop, comp)
    return stop, comp


