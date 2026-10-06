"""What a team knows about the tyres before the race starts: the weekend's practice long runs.

Practice 1-3 of the race's own meeting all finish before the race, so their streams are known at
the start. ``practice_prior`` reads those sessions (only sessions that started before the race;
only from disk, never the network), keeps the long runs (``MIN_RUN`` or more consecutive clean
laps on one set), and fits per dry compound

* the within-run slope of lap time against tyre age (``net`` = wear minus fuel burn, the same
  quantity the tyre engineer's field model works in), and
* the pace of each compound against MEDIUM (``off``), from runs of the same driver in the same
  session, so car and driver cancel.

Practice is a noisy stand-in for the race (track evolution, engine modes, unknown fuel), so the
estimates are returned with standard deviations that already include a bias allowance
(``NET_INFLATE`` / ``OFF_INFLATE``, chosen on 2025), and ``blend`` combines them with the
circuit's history and the defaults by precision. Missing practice (sprint weekend, wet FP, not
downloaded) gives ``None`` and the other sources carry on.

``fit_race`` does the same job on a finished race (for ``summarize_race``): history of earlier
races at the circuit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import raw_dir

DRY = ("SOFT", "MEDIUM", "HARD")
MIN_RUN = 5  # consecutive clean laps on one set that make a long run
RACING = 1.07  # a lap slower than this times the run's median isn't part of the run
SLOW_LAP = 1.12  # laps slower than this times the session's fastest lap are installation/cool-down laps
MIN_LAPS = {"net": 12, "off": 25}  # laps of a compound before its estimate is used
NET_INFLATE = 1.5  # practice slope sd is this many times its statistical sd (bias allowance)
OFF_INFLATE = 1.5
USE_PRACTICE_OFF = False  # practice compound offsets mislead (fuel and run order differ by compound): 2025 corr < 0
NET_FLOOR = 0.006  # s/lap: no source is trusted better than this
OFF_FLOOR = 0.12  # s


# ----------------------------------------------------------------------------- reading sessions
def earlier_practice(meta: dict) -> list[Path]:
    """Practice session folders of the race's meeting that started before the race (from disk)."""
    path, start = meta.get("path"), _start_utc(meta)
    if not path or start is None:
        return []
    meeting = raw_dir() / str(path).rstrip("/").rsplit("/", 1)[0]
    out = []
    for d in sorted(meeting.glob("*Practice*")) if meeting.is_dir() else []:
        s = _session_start(d)
        if s is not None and s < start and (d / "TimingData.jsonStream").exists():
            out.append(d)
    return out


def _utc(local: str, offset: str) -> datetime:
    sign = -1 if offset.startswith("-") else 1
    h, m, s = (int(x) for x in offset.lstrip("-+").split(":"))
    return (datetime.fromisoformat(local) - sign * timedelta(hours=h, minutes=m, seconds=s)).replace(tzinfo=timezone.utc)


def _start_utc(meta: dict) -> datetime | None:
    try:
        return _utc(meta["start_local"], meta["gmt_offset"])
    except (KeyError, ValueError):
        return None


def _session_start(d: Path) -> datetime | None:
    try:
        return _start_utc(json.loads((d / "session.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=64)
def _session_laps(d: str) -> tuple:
    from .events import load_archive_session
    from .state import replay

    return tuple(replay(load_archive_session(Path(d))).laps)


# ----------------------------------------------------------------------------- long runs
def long_runs(laps, min_run: int = MIN_RUN) -> list[list]:
    """Runs of >= ``min_run`` consecutive clean, representative laps by one driver on one set."""
    laps = [x for x in laps if x.lap_time is not None and x.compound in DRY and x.tyre_age is not None]
    if not laps:
        return []
    best = min(x.lap_time for x in laps if not x.is_in_lap and not x.is_out_lap and x.track_status == "1"
               ) if any(not x.is_in_lap and not x.is_out_lap and x.track_status == "1" for x in laps) else None
    if best is None:
        return []
    by: dict[tuple[str, int], list] = {}
    for x in laps:
        if x.lap > 1 and not x.is_in_lap and not x.is_out_lap and x.track_status == "1" and x.lap_time < SLOW_LAP * best:
            by.setdefault((x.driver, x.stint), []).append(x)
    runs = []
    for xs in by.values():
        xs.sort(key=lambda x: x.lap)
        cur = [xs[0]]
        for x in xs[1:] + [None]:
            if x is not None and x.lap == cur[-1].lap + 1:
                cur.append(x)
                continue
            if len(cur) >= min_run:
                med = float(np.median([y.lap_time for y in cur]))
                keep = [y for y in cur if y.lap_time <= RACING * med]
                # a lap dropped inside the run breaks it; keep the longest unbroken piece
                piece, best_piece = [], []
                for y in keep:
                    piece = piece + [y] if piece and y.lap == piece[-1].lap + 1 else [y]
                    if len(piece) > len(best_piece):
                        best_piece = piece
                if len(best_piece) >= min_run:
                    runs.append(best_piece)
            cur = [x] if x is not None else []
    return runs


# ----------------------------------------------------------------------------- estimates
@dataclass(frozen=True)
class Estimate:
    """Per-compound (SOFT, MEDIUM, HARD) slope and offset estimates with their sds; None = unknown."""

    net: tuple[float | None, float | None, float | None]
    net_sd: tuple[float | None, float | None, float | None]
    off: tuple[float | None, float | None, float | None]  # vs MEDIUM; MEDIUM is 0.0 when others are known
    off_sd: tuple[float | None, float | None, float | None]
    n: tuple[int, int, int]  # laps by compound
    sessions: int = 0
    source: str = ""

    def to_json(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @staticmethod
    def from_json(d: dict) -> "Estimate":
        return Estimate(tuple(d["net"]), tuple(d["net_sd"]), tuple(d["off"]), tuple(d["off_sd"]), tuple(d["n"]),
                        d.get("sessions", 0), d.get("source", ""))


def _huber(r: np.ndarray, k: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-9))


def fit_runs(runs: list[list], groups: list | None = None, *, noise: float = 0.3) -> Estimate | None:
    """Slope by compound (pooled within runs) and offsets vs MEDIUM (same driver, same session)."""
    rows = []  # (group, run id, comp, age, time, lap)
    for k, run in enumerate(runs):
        for x in run:
            rows.append((f"{x.driver}", k, DRY.index(x.compound), float(x.tyre_age), float(x.lap_time)))
    if not rows:
        return None
    run_id = np.array([r[1] for r in rows])
    comp = np.array([r[2] for r in rows])
    age = np.array([r[3] for r in rows])
    t = np.array([r[4] for r in rows])
    n = tuple(int((comp == c).sum()) for c in range(3))
    # ---- slope: one free level per run, Huber-weighted
    nr = int(run_id.max()) + 1
    w = np.ones(len(t))
    net = np.zeros(3)
    sxx = np.zeros(3)
    sd_noise = noise
    for it in range(4):
        sw = np.bincount(run_id, w, nr)
        am = np.bincount(run_id, w * age, nr) / np.maximum(sw, 1e-9)
        tm = np.bincount(run_id, w * t, nr) / np.maximum(sw, 1e-9)
        da, dt = age - am[run_id], t - tm[run_id]
        for c in range(3):
            m = comp == c
            sxx[c] = float((w[m] * da[m] ** 2).sum())
            net[c] = float((w[m] * da[m] * dt[m]).sum()) / sxx[c] if sxx[c] > 1e-9 else 0.0
        r = dt - net[comp] * da
        if m.any():
            mad = float(np.median(np.abs(r - np.median(r)))) * 1.4826
            sd_noise = max(mad, 0.1)
        w = _huber(r, 1.5 * sd_noise)
    net_v: list[float | None] = []
    net_sd: list[float | None] = []
    for c in range(3):
        ok = n[c] >= MIN_LAPS["net"] and sxx[c] > 20
        net_v.append(float(net[c]) if ok else None)
        net_sd.append(max(NET_INFLATE * sd_noise / math.sqrt(sxx[c]), NET_FLOOR) if ok else None)
    # ---- offsets: run level (at its mean age, slope removed) = driver-session level + compound offset
    lvl = np.array([(w[run_id == k] * (t - net[comp] * age)[run_id == k]).sum() / max(w[run_id == k].sum(), 1e-9)
                    for k in range(nr)])
    rc = np.array([int(comp[run_id == k][0]) for k in range(nr)])
    seen: dict = {}
    groups = groups or [run[0].driver for run in runs]
    rg = np.array([seen.setdefault(g, len(seen)) for g in groups])
    off = [None, None, None]
    off_sd = [None, None, None]
    # driver-demeaned levels, so the offset of compound c vs MEDIUM is a within-driver contrast
    ng = len(seen)
    X = np.zeros((nr, ng + 2))
    X[np.arange(nr), rg] = 1.0
    X[:, ng] = rc == 0
    X[:, ng + 1] = rc == 2
    ok_c = [n[0] >= MIN_LAPS["off"], n[1] >= MIN_LAPS["off"], n[2] >= MIN_LAPS["off"]]
    if ok_c[1] and (ok_c[0] or ok_c[2]) and nr > ng + 1:
        ridge = np.concatenate([np.full(ng, 1e-6), [1e-3, 1e-3]])
        H = X.T @ X + np.diag(ridge)
        beta = np.linalg.solve(H, X.T @ lvl)
        resid = lvl - X @ beta
        s = max(float(np.sqrt((resid ** 2).sum() / max(nr - ng - 2, 1))), 0.15)
        cov = np.linalg.inv(H) * s * s
        for j, c in ((ng, 0), (ng + 1, 2)):
            if ok_c[c]:
                off[c] = float(beta[j])
                off_sd[c] = max(OFF_INFLATE * math.sqrt(max(cov[j, j], 0.0)), OFF_FLOOR)
        off[1], off_sd[1] = 0.0, 0.0
    return Estimate(tuple(net_v), tuple(net_sd), tuple(off), tuple(off_sd), n, 0, "")


def practice_estimate(meta: dict) -> Estimate | None:
    """Estimate from this weekend's earlier practice sessions, or None when there is nothing usable."""
    runs: list[list] = []
    groups: list = []
    sessions = 0
    for d in earlier_practice(meta):
        try:
            r = long_runs(_session_laps(str(d)))
        except Exception:  # a broken stream is unknown practice, never a guess
            continue
        if r:
            sessions += 1
            runs.extend(r)
            groups.extend((d.name, x[0].driver) for x in r)
    if not runs:
        return None
    e = fit_runs(runs, groups)
    if e is None:
        return None
    return Estimate(e.net, e.net_sd, e.off, e.off_sd, e.n, sessions, "practice")


# ----------------------------------------------------------------------------- finished races
def fit_race(final) -> dict | None:
    """What a finished race says about its tyres (for ``summarize_race``): field fit of the clean dry laps."""
    from .pitwall.engineers.tyre.model import CI, Priors, fit_field
    from .pitwall.memory import is_clean

    laps = [x for x in final.laps if is_clean(x) and x.compound in CI and x.tyre_age is not None]
    if len(laps) < 60:
        return None
    fastest = sorted(x.lap_time for x in laps)[:5]
    cap = 1.07 * fastest[-1]
    laps = [x for x in laps if x.lap_time <= cap]
    weak = Priors(net_sd=0.2, off_sd=3.0, fuel_sd=0.2, net=(0.0, 0.0, 0.0), off=(0.0, 0.0, 0.0), fuel=0.05)
    ff = fit_field([x.driver for x in laps], np.array([x.lap for x in laps], float),
                   np.array([x.tyre_age for x in laps], float), np.array([CI[x.compound] for x in laps]),
                   np.array([x.lap_time for x in laps], float), np.ones(len(laps)),
                   np.array([x.stint for x in laps]), weak)
    n = [sum(x.compound == c for x in laps) for c in DRY]
    runs = [[x for x in laps if x.driver == d and x.stint == s] for d, s in
            sorted({(x.driver, x.stint) for x in laps})]
    est = fit_runs([r for r in runs if len(r) >= MIN_RUN])
    return {
        "net": [round(v, 4) if n[i] >= 30 else None for i, v in enumerate(ff.net)],
        "off": [round(v, 3) if n[i] >= 30 and n[1] >= 30 else None for i, v in enumerate(ff.off)],
        "fuel": round(ff.fuel, 4),
        "n": n,
        "net_sd": [None if est is None or est.net_sd[i] is None else round(est.net_sd[i] / NET_INFLATE, 4)
                   for i in range(3)],
    }


# ----------------------------------------------------------------------------- blending
def _combine(parts: list[tuple[float, float]]) -> tuple[float, float]:
    """Precision-weighted mean and sd of (value, sd) parts."""
    p = [1 / max(s, 1e-6) ** 2 for _, s in parts]
    m = sum(w * v for w, (v, _) in zip(p, parts)) / sum(p)
    return m, math.sqrt(1 / sum(p))


def circuit_history(past, circuit_key) -> dict:
    """Per-compound slope and offset of earlier races at this circuit (most recent weighted most)."""
    mine = [s.extra["tyre"] for s in reversed(past) if s.circuit_key == circuit_key
            and isinstance(s.extra.get("tyre"), dict) and s.extra["tyre"].get("net")]
    out = {"net": [None] * 3, "off": [None] * 3, "n_races": len(mine)}
    for key in ("net", "off"):
        for c in range(3):
            vals = [r[key][c] for r in mine if r[key][c] is not None]
            if vals:
                w = [0.6 ** i for i in range(len(vals))]
                out[key][c] = sum(a * b for a, b in zip(w, vals)) / sum(w)
    return out


def blend(default, hist: dict | None, prac: Estimate | None, *, hist_net_sd: float = 0.012,
          hist_off_sd: float = 0.25, net_sd: float = 0.02, off_sd: float = 0.4):
    """Race-start prior: defaults, shrunk toward circuit history, shrunk toward practice.

    ``default`` is a tyre-model ``Priors``; returns a copy with ``net`` / ``off`` and their sds
    replaced where practice or history knows better. Returns (priors, source string).
    """
    from dataclasses import replace

    if prac is None and not (hist and hist["n_races"]):
        return default, default.source
    net, off = list(default.net), list(default.off)
    net_s, off_s = [net_sd] * 3, [off_sd] * 3
    src = []
    for c in range(3):
        parts = [(default.net[c], net_sd)]
        if hist and hist["net"][c] is not None:
            parts.append((hist["net"][c], max(hist_net_sd, NET_FLOOR)))
        if prac and prac.net[c] is not None:
            parts.append((prac.net[c], prac.net_sd[c]))
        net[c], net_s[c] = _combine(parts)
        if c != 1:  # MEDIUM is the reference
            parts = [(default.off[c], off_sd)]
            if hist and hist["off"][c] is not None:
                parts.append((hist["off"][c] - (hist["off"][1] or 0.0), max(hist_off_sd, OFF_FLOOR)))
            if prac and USE_PRACTICE_OFF and prac.off[c] is not None:
                parts.append((prac.off[c], prac.off_sd[c]))
            off[c], off_s[c] = _combine(parts)
    if hist and hist["n_races"]:
        src.append(f"circuit x{hist['n_races']}")
    if prac:
        src.append(f"practice x{prac.sessions}")
    pri = replace(default, net=tuple(net), off=tuple(off), net_sd=float(np.mean(net_s)),
                  off_sd=float(np.mean([off_s[0], off_s[2]])), net_sds=tuple(net_s), off_sds=(off_s[0], off_s[2]),
                  source="+".join(src) or default.source)
    return pri, pri.source
