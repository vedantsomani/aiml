"""Tyre & pace estimation: pure functions over lap data (no race state, no I/O).

Lap-time model for a clean lap of car ``d`` on lap ``L`` on compound ``c`` at tyre age ``a``:

    t = base_d + off_c + deg_c * a - fuel * L + noise

* Inside one stint tyre age and lap number move together, so a stint only shows the
  *net* slope ``deg_c - fuel``. The field fit therefore works in two steps:
  1. **within stints** (one level per stint): the net slope of each dry compound,
     robust and shrunk toward the prior;
  2. **across stops** (the age resets, the fuel load doesn't): each stint's level is
     ``base_d + off_c - fuel * (lap the set was fitted - its age then)``, which gives
     ``fuel`` (fuel burn plus track evolution) and the compound offsets ``off_c``.
  ``deg_c = net_c + fuel``. A bias in step 2 moves fuel and deg together but can't
  bend the within-stint predictions, which only use the net slope.
* A car's pace is tracked over its clean laps by a two-part filter: a slow level
  (a random walk: set-up, pace management) plus a fast deviation that decays
  (traffic, energy deployment, a push lap). The next lap leans on the fast part,
  laps further ahead fall back to the slow level.
* The car's net slope on its current set is its own, shrunk toward the field's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DRY = ("SOFT", "MEDIUM", "HARD")
CI = {c: i for i, c in enumerate(DRY)}


@dataclass(frozen=True)
class Priors:
    """What we expect before seeing this race (defaults: 2025 field averages)."""

    fuel: float = 0.055  # s/lap gained from fuel burn + track evolution
    fuel_sd: float = 0.02
    net: tuple[float, float, float] = (0.005, -0.002, -0.008)  # within-stint slope deg - fuel, s/lap (S, M, H)
    net_sd: float = 0.02
    off: tuple[float, float, float] = (-0.4, 0.0, 0.35)  # s vs MEDIUM
    off_sd: float = 0.4
    car_net_sd: float = 0.01  # one car's slope on one set around the field's
    # pace filter (chosen on 2025: next-lap and 5-laps-ahead MAE)
    noise: float = 0.3  # sd of a clean lap around the car's current pace (white part)
    slow: float = 0.2  # random-walk sd of a car's slow level, per lap
    fast: float = 0.7  # sd of the fast deviation (stationary)
    phi: float = 0.8  # per-lap persistence of the fast deviation
    stop_jump: float = 0.25  # extra sd of the slow level across a stop
    clip: float = 5.0  # residuals are clipped at this many sd (only freak laps)
    stint_sd: float = 0.25  # stint level around base + offset - fuel * start (step 2)
    first_lap: float = 0.0  # first flying lap on a new set vs the model, s
    source: str = "default"
    # per-compound sds of ``net`` (S, M, H) and of the SOFT / HARD ``off``, when the race-start prior
    # knows some compounds better than others (circuit history, practice); else ``net_sd`` / ``off_sd``
    net_sds: tuple[float, float, float] | None = None
    off_sds: tuple[float, float] | None = None

    @property
    def deg(self) -> tuple[float, float, float]:
        return tuple(n + self.fuel for n in self.net)


@dataclass(frozen=True)
class FieldFit:
    fuel: float
    fuel_sd: float
    net: tuple[float, float, float]  # within-stint slope deg - fuel, by compound
    off: tuple[float, float, float]
    base: dict  # driver -> base pace (MEDIUM, new tyre, no fuel), s
    noise: float  # robust residual sd within stints
    n: int  # laps used

    @property
    def deg(self) -> tuple[float, float, float]:
        return tuple(n + self.fuel for n in self.net)


def prior_fit(pri: Priors) -> FieldFit:
    return FieldFit(pri.fuel, pri.fuel_sd, pri.net, pri.off, {}, pri.noise, 0)


def _huber_w(r: np.ndarray, k: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-9))


def fit_field(drivers: list[str], lap: np.ndarray, age: np.ndarray, comp: np.ndarray, t: np.ndarray,
              w: np.ndarray, stint: np.ndarray, pri: Priors, min_laps: int = 40, iters: int = 3) -> FieldFit:
    """Two-step robust fit of the field model on the race's clean dry laps so far.

    ``drivers[i]`` is the car of lap i, ``comp`` holds compound indices (``CI``) and
    ``stint`` labels each car's sets (any integers; laps of one set share a label).
    Priors enter as pseudo-observations, so with little data the fit is the prior.
    """
    n = len(t)
    if n < min_laps:
        return prior_fit(pri)
    keys = [f"{d}#{s}" for d, s in zip(drivers, stint)]
    names = sorted(set(keys))
    index = {k: i for i, k in enumerate(names)}
    sid = np.array([index[k] for k in keys])
    ns = len(names)
    s2 = pri.noise ** 2
    p_net = 1 / np.array(pri.net_sds or (pri.net_sd,) * 3) ** 2
    net = np.array(pri.net, dtype=float)
    rw = np.ones(n)
    # ---- step 1: net slope by compound, one free level per stint
    for it in range(iters + 1):
        ww = w * rw
        sw = np.bincount(sid, ww, ns)
        am = np.bincount(sid, ww * age, ns) / np.maximum(sw, 1e-12)
        tm = np.bincount(sid, ww * t, ns) / np.maximum(sw, 1e-12)
        da, dt = age - am[sid], t - tm[sid]
        for c in range(3):
            m = comp == c
            sxx = float((ww[m] * da[m] ** 2).sum()) / s2
            sxy = float((ww[m] * da[m] * dt[m]).sum()) / s2
            net[c] = (sxy + p_net[c] * pri.net[c]) / (sxx + p_net[c])
        r = dt - net[comp] * da
        if it < iters:
            rw = _huber_w(r, 1.5 * pri.noise)
    mad = float(np.median(np.abs(r - np.median(r)))) * 1.4826
    # ---- step 2: stint levels at age 0 -> base_d + off_c - fuel * (fit lap - age then)
    ww = w * rw
    sw = np.bincount(sid, ww, ns)
    level = np.bincount(sid, ww * (t - net[comp] * age), ns) / np.maximum(sw, 1e-12)
    start = np.bincount(sid, ww * (lap - age), ns) / np.maximum(sw, 1e-12)  # constant within a stint
    s_comp = np.zeros(ns, dtype=int)
    s_comp[sid] = comp
    s_drv = sorted(set(drivers))
    di = {d: i for i, d in enumerate(s_drv)}
    s_d = np.zeros(ns, dtype=int)
    for k, name in enumerate(names):
        s_d[k] = di[name.rsplit("#", 1)[0]]
    nd = len(s_drv)
    X = np.zeros((ns, nd + 3))
    X[np.arange(ns), s_d] = 1.0
    X[:, nd] = s_comp == 0
    X[:, nd + 1] = s_comp == 2
    X[:, nd + 2] = -start
    prec_w = 1 / (s2 / np.maximum(sw, 1e-12) + pri.stint_sd ** 2)
    mu = np.zeros(nd + 3)
    prec = np.full(nd + 3, 1e-8)
    mu[nd], mu[nd + 1], mu[nd + 2] = pri.off[0] - pri.off[1], pri.off[2] - pri.off[1], pri.fuel
    prec[nd:nd + 2] = 1 / np.array(pri.off_sds or (pri.off_sd,) * 2) ** 2
    prec[nd + 2] = 1 / pri.fuel_sd ** 2
    H = X.T @ (X * prec_w[:, None]) + np.diag(prec)
    beta = np.linalg.solve(H, X.T @ (prec_w * level) + prec * mu)
    cov = np.linalg.inv(H)
    return FieldFit(
        fuel=float(beta[nd + 2]),
        fuel_sd=float(math.sqrt(max(cov[nd + 2, nd + 2], 0.0))),
        net=tuple(float(x) for x in net),
        off=(float(beta[nd]), 0.0, float(beta[nd + 1])),
        base={d: float(beta[i]) for d, i in di.items()},
        noise=mad,
        n=n,
    )


@dataclass(frozen=True)
class CarFit:
    slow: float  # slow level now (MEDIUM, new tyre, no fuel units), s
    fast: float  # fast deviation now, s (decays by phi per lap)
    level_sd: float  # sd of slow + fast now
    net: float  # within-stint slope on the current set, s/lap
    net_sd: float
    n_stint: int  # clean laps on the current set used
    resid_last: float  # last clean lap minus the model's expectation before it (s), NaN if none
    resid_trend: float  # mean of the last 2 residuals minus the 3 before (s), NaN if < 5 on this set
    last_lap: int  # the last clean lap the filter has seen

    def level(self, lap: float, phi: float) -> float:
        """Base pace expected on race lap ``lap`` (the fast part decays from the last clean lap)."""
        return self.slow + phi ** max(lap - self.last_lap, 0) * self.fast


def expected(fit: CarFit, ff: FieldFit, pri: Priors, comp: int, net: float, age: float, lap: float,
             fast: bool = True) -> float:
    """Expected clean lap time on race lap ``lap`` at tyre age ``age`` on compound ``comp``.

    ``net`` is the within-stint slope of that set; ``lap - age`` is when the set was fitted,
    so ``fuel * (lap - age)`` is the fuel burnt before it. ``fast=False`` drops the
    short-term deviation (the trend to project several laps ahead).
    """
    base = fit.level(lap, pri.phi) if fast else fit.slow
    return base + ff.off[comp] + net * age - ff.fuel * (lap - age)


def stint_net(age: np.ndarray, t: np.ndarray, w: np.ndarray, prior: float, prior_sd: float, noise: float,
              iters: int = 2) -> tuple[float, float]:
    """Within-stint slope of lap time vs age on one set, shrunk toward ``prior``."""
    if len(t) < 2:
        return prior, prior_sd
    rw = np.ones(len(t))
    s2 = noise ** 2
    p0 = 1 / prior_sd ** 2
    for i in range(iters + 1):
        ww = w * rw
        sw = ww.sum()
        am, tm = (ww * age).sum() / sw, (ww * t).sum() / sw
        sxx = (ww * (age - am) ** 2).sum() / s2
        sxy = (ww * (age - am) * (t - tm)).sum() / s2
        b = (sxy + prior * p0) / (sxx + p0)
        if i < iters:
            rw = _huber_w(t - (tm + b * (age - am)), 1.5 * noise)
    return float(b), float(1 / math.sqrt(sxx + p0))


def fit_car(lap: np.ndarray, age: np.ndarray, comp: np.ndarray, t: np.ndarray, w: np.ndarray,
            stint: np.ndarray, current: int, ff: FieldFit, pri: Priors) -> CarFit | None:
    """Filter one car's base pace over all its clean laps; net slope from its current set.

    Arrays are the car's clean dry laps in lap order; ``stint`` labels each lap's set and
    ``current`` is the set on the car now.
    """
    if len(t) == 0:
        return None
    cur = stint == current
    if cur.any():
        c_now = int(comp[cur][0])
        net_now, net_sd = stint_net(age[cur], t[cur], w[cur], ff.net[c_now], pri.car_net_sd, pri.noise)
    else:
        net_now, net_sd = float("nan"), float("nan")
    net = np.where(cur, net_now, np.array(ff.net)[comp])
    # base-pace units: remove the slope on the set and the fuel since the set was fitted
    z = t - net * age - np.array(ff.off)[comp] + ff.fuel * (lap - age)
    R = pri.noise ** 2
    q = pri.slow ** 2
    phi = pri.phi
    vx_inf = pri.fast ** 2  # stationary variance of the fast part
    # state (mu, x) with covariance [[Pm, Pmx], [Pmx, Px]]
    mu, x = float(z[0]), 0.0
    Pm, Pmx, Px = R + 1.0, 0.0, vx_inf
    first = True
    prev_lap, prev_stint = lap[0], stint[0]
    res = []
    for k in range(len(z)):
        if not first:
            g = float(lap[k] - prev_lap)
            f = phi ** g
            Pm += q * g + (pri.stop_jump ** 2 if stint[k] != prev_stint else 0.0)
            Pmx *= f
            Px = f * f * Px + vx_inf * (1 - f * f)
            x *= f
        pred = mu + x
        r = float(z[k]) - pred
        res.append(r)
        Rk = R / max(float(w[k]), 1e-3)
        S = Pm + 2 * Pmx + Px + Rk
        s = math.sqrt(S)
        # Clip only freak laps: an "off" lap (traffic, a push) says a lot about the next
        # one, so skipping or tightly clipping laps made next-lap predictions worse on 2025.
        rc = max(-pri.clip * s, min(pri.clip * s, r))
        Km, Kx = (Pm + Pmx) / S, (Pmx + Px) / S
        mu += Km * rc
        x += Kx * rc
        Pm, Pmx, Px = Pm - Km * (Pm + Pmx), Pmx - Km * (Pmx + Px), Px - Kx * (Pmx + Px)
        prev_lap, prev_stint = lap[k], stint[k]
        first = False
    tail =[r for r, s in zip(res, stint) if s == current]
    last = tail[-1] if tail else float("nan")
    trend = (float(np.mean(tail[-2:])) - float(np.mean(tail[-5:-2]))) if len(tail) >= 5 else float("nan")
    sd = math.sqrt(max(Pm + 2 * Pmx + Px, 0.0))
    return CarFit(mu, x, sd, net_now, net_sd, int(cur.sum()), last, trend, int(lap[-1]))
