"""Running the simulation: the whole field with default strategies, and one car's candidate plans against it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import (
    FOLLOW_S, K_STOPS, NOSTOP, OFF, PARAMS, PLACEHOLDER, RETIRED, SC_CAP_S, SC_LAP_FACTOR, VSC_LAP_FACTOR,
    Draws, FieldIn, cliff_ages, lap_times_for, pass_prob, sample_schedules,
)


def _loss_by_status(F: FieldIn, status_lap: np.ndarray) -> np.ndarray:
    L = F.loss
    return np.where(status_lap == 2, L["sc"], np.where(status_lap == 1, L["vsc"], L["green"]))


@dataclass
class FieldRun:
    cum: np.ndarray  # [S,C,R+1] time at each lap end (anchor lap, then each simulated lap), retirements ignored
    final: np.ndarray  # [S,C] finishing time incl. penalties; retired cars sort last
    final_alive: np.ndarray  # [S,C] the same ignoring the car's own retirement
    retired: np.ndarray  # [S,C] bool
    stop: np.ndarray  # [S,C,K] stop schedules used
    comp: np.ndarray
    pos: np.ndarray  # [S,C] finishing position if the car finishes (others' retirements counted)


def _pit_losses(F: FieldIn, D: Draws, stop: np.ndarray, c: int | None = None) -> np.ndarray:
    """Time added at the in-lap of every stop, [S,C,R] (or [P,S,R] for one car's plans when c is given)."""
    S, R, A = D.S, F.R, F.A
    if c is None:
        Cn = stop.shape[1]
        loss = np.zeros((S, Cn, R))
        sidx = np.broadcast_to(np.arange(S)[:, None], (S, Cn))
        cidx = np.broadcast_to(np.arange(Cn)[None, :], (S, Cn))
        off = F.team_off[None, :]
        for k in range(K_STOPS):
            li = stop[:, :, k] - (A + 1)
            ok = (li >= 0) & (li < R)
            lis = np.clip(li, 0, R - 1)
            st = np.take_along_axis(D.status, lis, 1)
            amt = np.maximum(_loss_by_status(F, st) + F.loss["sd"] * np.where(st == 0, 1.0, 0.6) * D.z_pit[:, :, k] + off, 3.0)
            np.add.at(loss, (sidx[ok], cidx[ok], lis[ok]), amt[ok])
        return loss
    P = stop.shape[0]
    loss = np.zeros((P, S, R))
    pidx = np.broadcast_to(np.arange(P)[:, None], (P, S))
    sidx = np.broadcast_to(np.arange(S)[None, :], (P, S))
    for k in range(K_STOPS):
        li = stop[:, :, k] - (A + 1)
        ok = (li >= 0) & (li < R)
        lis = np.clip(li, 0, R - 1)
        st = D.status[sidx, lis]
        amt = np.maximum(_loss_by_status(F, st) + F.loss["sd"] * np.where(st == 0, 1.0, 0.6) * D.z_pit[None, :, c, k] + F.team_off[c], 3.0)
        np.add.at(loss, (pidx[ok], sidx[ok], lis[ok]), amt[ok])
    return loss


def simulate_field(F: FieldIn, D: Draws, stop: np.ndarray | None = None, comp: np.ndarray | None = None) -> FieldRun:
    S, C, R, A = D.S, F.C, F.R, F.A
    if stop is None:
        stop, comp = sample_schedules(F, D)
    lt = np.empty((S, C, R))
    for c in range(C):
        cur, new = cliff_ages(F, D, c)
        lt[:, c, :] = lap_times_for(F, D, c, stop[None, :, c, :], comp[None, :, c, :], cur, new)[0]
    loss = _pit_losses(F, D, stop)
    sc_t = F.ref_pace * np.array([1.0, VSC_LAP_FACTOR, SC_LAP_FACTOR])
    cum = np.empty((S, C, R + 1))
    cum[:, :, 0] = F.x0[None, :]
    cur = np.broadcast_to(F.x0[None, :], (S, C)).copy()
    for l in range(R):
        st = D.status[:, l]
        green, sc = st == 0, st == 2
        free = np.where(st[:, None] > 0, sc_t[st][:, None], lt[:, :, l])
        order = np.argsort(cur, axis=1, kind="stable")
        cp = np.take_along_axis(cur, order, 1)
        fr = np.take_along_axis(free, order, 1)
        ls = np.take_along_axis(loss[:, :, l], order, 1)
        up = np.take_along_axis(D.u_pass[:, :, l], order, 1)
        arrive = cp + fr
        out = np.empty_like(arrive)  # arrival time including this lap's stop
        nl = np.empty_like(arrive)  # the same without it (safety-car bunching ignores stops)
        out[:, 0] = arrive[:, 0] + ls[:, 0]
        nl[:, 0] = arrive[:, 0]
        for r in range(1, C):
            a_out = out[:, r - 1]
            b = arrive[:, r] + ls[:, r]
            delta = (a_out - cp[:, r - 1]) - (fr[:, r] + ls[:, r])
            keep = up[:, r] < pass_prob(delta, F.pass_p)
            held = np.maximum(b, a_out + FOLLOW_S)
            close = (cp[:, r] - cp[:, r - 1]) < 1.0
            g = np.where(b < a_out, np.where(keep, b, held), np.where(close, held, b))
            nb = np.minimum(arrive[:, r], nl[:, r - 1] + SC_CAP_S)
            nl[:, r] = np.where(sc, nb, arrive[:, r])
            out[:, r] = np.where(green, g, nl[:, r] + ls[:, r])
        newc = np.empty_like(out)
        np.put_along_axis(newc, order, out, 1)
        cum[:, :, l + 1] = newc
        cur = newc
    return _finish(F, D, cum, stop, comp)


def _finish(F: FieldIn, D: Draws, cum: np.ndarray, stop, comp) -> FieldRun:
    S, C, R = D.S, F.C, F.R
    ret = D.ret_lap < R
    final_alive = cum[:, :, R] + F.penalty[None, :]
    final = np.where(ret, RETIRED + np.arange(C)[None, :], final_alive)
    pos = np.empty((S, C))
    for c in range(C):
        pos[:, c] = 1 + (np.delete(final, c, axis=1) < final_alive[:, c:c + 1]).sum(1)
    return FieldRun(cum, final, final_alive, ret, stop, comp, pos)


# --------------------------------------------------------------------------- the focus car's plans
def evaluate_plans(F: FieldIn, D: Draws, run: FieldRun, c: int, stop_lap: np.ndarray, comp: np.ndarray, S_use: int | None = None):
    """Finishing position [P,S] of car ``c`` under each plan, against the simulated opponents.

    stop_lap [P,S,K] absolute in-laps (NOSTOP if none), comp [P,S,K] compound fitted at each stop. Uses the
    first ``S_use`` simulations. Each lap needs only the nearest opponents ahead and behind, found by a sorted search.
    """
    S = S_use or D.S
    Pn = stop_lap.shape[0]
    C, R, A = F.C, F.R, F.A
    stop_lap, comp = stop_lap[:, :S], comp[:, :S]
    Dv = _Slice(D, S)
    cur_cliff, new_cliff = cliff_ages(F, Dv, c)
    tfree = lap_times_for(F, Dv, c, stop_lap, comp, cur_cliff, new_cliff)  # [P,S,R]
    ploss = _pit_losses(F, Dv, stop_lap, c)  # [P,S,R]
    cum = run.cum[:S].copy()
    cum[:, c, :] = PLACEHOLDER  # the car's own baseline trajectory is replaced by the plan
    idx = np.arange(R + 1)[None, None, :]
    gone = (D.ret_lap[:S, :, None] < idx) & (np.arange(C)[None, :, None] != c)
    cum = np.where(gone, RETIRED, cum)
    srow = (np.arange(S) * OFF)[None, :]
    base_k = (np.arange(S) * C)[None, :]
    si = np.arange(S)[None, :]
    sp_all, nw_all, flat_all = [], [], []
    for l in range(R):
        o = np.argsort(cum[:, :, l], axis=1, kind="stable")
        sp = np.take_along_axis(cum[:, :, l], o, 1)
        sp_all.append(sp)
        nw_all.append(np.take_along_axis(cum[:, :, l + 1], o, 1))
        flat_all.append((sp + srow.T).ravel())
    sc_t = F.ref_pace * np.array([1.0, VSC_LAP_FACTOR, SC_LAP_FACTOR])
    ours = np.full((Pn, S), F.x0[c])
    hold = PARAMS["hold_gain"]
    for l in range(R):
        st = D.status[:S, l]
        sp, nw = sp_all[l], nw_all[l]
        k = np.searchsorted(flat_all[l], (ours + srow).ravel()).reshape(Pn, S) - base_k  # opponents ahead of us
        hasA = k > 0
        ia = np.clip(k - 1, 0, C - 1)
        a_prev, a_new = sp[si, ia], nw[si, ia]
        ib = np.clip(k, 0, C - 1)
        b_prev, b_new = sp[si, ib], nw[si, ib]
        hasB = (k < C) & (b_prev < RETIRED / 2)
        lo = ploss[:, :, l]
        free = np.where(st[None, :] > 0, sc_t[st][None, :], tfree[:, :, l])
        arrive = ours + free
        total = arrive + lo
        green, sc = (st == 0)[None, :], (st == 2)[None, :]
        delta = (a_new - a_prev) - (free + lo)
        keep = D.u_pass[:S, c, l][None, :] < pass_prob(delta, F.pass_p)
        held = np.maximum(total, a_new + FOLLOW_S)
        close = hasA & ((ours - a_prev) < 1.0)
        g = np.where(hasA & (total < a_new), np.where(keep, total, held), np.where(close, held, total))
        # a faster car right behind may be held off at the cost of nothing here (bounded gain)
        delta_b = (free + lo) - (b_new - b_prev)  # negative when the car behind is faster
        pb = pass_prob(-delta_b, F.pass_p)
        near = hasB & ((b_prev - ours) < 1.5) & (b_new < g)
        defend = near & (D.u_def[:S, l][None, :] >= pb)
        g = np.where(defend, np.maximum(np.minimum(g, b_new - 0.2), g - hold), g)
        scg = np.where(hasA, np.minimum(arrive, a_new + SC_CAP_S), arrive) + lo
        ours = np.where(green, g, np.where(sc, scg, total))
    ours = ours + F.penalty[c]
    fin = run.final[:S].copy()
    fin[:, c] = PLACEHOLDER
    flat = (np.sort(fin, axis=1) + srow.T).ravel()
    pos = np.searchsorted(flat, (ours + srow).ravel()).reshape(Pn, S) - base_k + 1
    return pos.astype(float)


class _Slice:
    """Draws restricted to the first S simulations."""

    def __init__(self, D: Draws, S: int) -> None:
        self.S = S
        self.level, self.degmul, self.noise = D.level[:S], D.degmul[:S], D.noise[:S]
        self.status, self.z_pit = D.status[:S], D.z_pit[:S]
        self.z_cliff, self.u_cliff = D.z_cliff[:S], D.u_cliff[:S]
