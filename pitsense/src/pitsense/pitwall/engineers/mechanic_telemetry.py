"""Mechanic (telemetry): is this car behaving like itself, and like the rest of the field?

What the feed can and cannot show. CarData has RPM, speed, gear, throttle, brake (on/off) and
DRS per car, ~4 Hz, plus Position X/Y. There are no temperatures, pressures, ERS state or brake
pressure. A fault is visible only through what it does to the car's motion:

* ``power_loss``   speed (and RPM) at full throttle falls against the car's own earlier laps.
                   Speed is compared with what the *field* does at the same place on track
                   (150 m grid cell from Position), so corner, fuel and tyre effects cancel.
* ``gearbox``      neutral while moving, upshifts that skip a gear, RPM that does not match
                   the gear (slip), gear held far outside its normal speed range.
* ``brake_issue``  braking from speed sheds less speed per second than it used to; brakes held
                   on far too long.
* ``slow_car``     speed collapses against the field in the same cell, or the car coasts
                   (throttle closed, no brake) down a straight.

Everything is learned online from the car's own earlier laps (a baseline that lags 150 s and is
frozen while the car is flagged, so a failure cannot become "normal") and from the field this
race. Nothing is read before ``state.t``: samples are ingested in ``observe`` from the feed's
as-of store, in a fixed 4 s rhythm, so values at ``t`` are the same however often they are asked.
Samples while the car is in the pits, under SC/VSC/red/yellow, on lap 1 or after the flag are
ignored.

Values per car: ``mech_risk`` (0-1), ``mech_issue`` (str, "" when none), ``mech_since``,
``power_loss`` / ``gearbox`` / ``brake_issue`` / ``slow_car`` (scores 0-1, None until there is
enough data), ``<check>_since`` (when the level-triggered alert became true, else None).
Docs: docs/engineers/mechanics.md.
"""

from __future__ import annotations

import numpy as np

from ...state import RaceState
from ..engineer import Engineer
from ..types import Alert

CHECKS = ("power_loss", "gearbox", "brake_issue", "slow_car")

INGEST_S = 4.0  # state seconds between ingests
CELL = 1500.0  # position units (1/10 m): 150 m grid
LAG_S = 150.0  # the baseline only sees samples at least this old ...
SPAN_S = 1500.0  # ... and at most this old
WIN_S = 75.0  # "now" = the last this-many seconds of the car's own clock
MIN_BASE = 40  # samples before the own baseline is trusted (else the field, 1.0)
CLEAR_GAP_S = 1.8  # full-throttle speed is only compared with the car's own when no car is within this gap ahead
MIN_REF = 40  # samples in a grid cell before it is a reference

# (on, hold_s, clear_s): level-triggered alert thresholds. ``on`` was chosen on 2025 by
# bench/mechanics.py (docs/engineers/mechanics.md); the alert drops again below 0.6 * on for ``clear_s``.
THRESH: dict[str, tuple[float, float, float]] = {
    "power_loss": (0.9, 20.0, 30.0),
    "gearbox": (0.9, 8.0, 30.0),
    "brake_issue": (0.9, 12.0, 30.0),
    "slow_car": (0.6, 8.0, 20.0),
}
OFF_RATIO = 0.6


class Series:
    """Append-only (time, value) arrays with window reads. Times must be non-decreasing."""

    def __init__(self, cap: int = 4096) -> None:
        self.cap = cap
        self.t = np.empty(cap)
        self.v = np.empty(cap)
        self.n = 0

    def add(self, t: np.ndarray, v: np.ndarray) -> None:
        k = len(t)
        if k == 0:
            return
        if k > self.cap:
            t, v, k = t[-self.cap:], v[-self.cap:], self.cap
        if self.n + k > self.cap:
            keep = min(self.n, self.cap // 2)
            drop = self.n - keep
            self.t[:keep] = self.t[drop:self.n]
            self.v[:keep] = self.v[drop:self.n]
            self.n = keep
            if self.n + k > self.cap:
                k2 = self.cap - self.n
                t, v, k = t[-k2:], v[-k2:], k2
        self.t[self.n:self.n + k] = t
        self.v[self.n:self.n + k] = v
        self.n += k

    def between(self, t0: float, t1: float) -> np.ndarray:
        """Values with t0 < t <= t1."""
        t = self.t[:self.n]
        i = int(np.searchsorted(t, t0, side="right"))
        j = int(np.searchsorted(t, t1, side="right"))
        return self.v[i:j]

    def slice(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        t = self.t[:self.n]
        i = int(np.searchsorted(t, t0, side="right"))
        j = int(np.searchsorted(t, t1, side="right"))
        return self.t[i:j], self.v[i:j]


class Ref:
    """Fixed-size ring of recent values per key, with a cached median."""

    def __init__(self, cap: int = 400) -> None:
        self.cap = cap
        self.buf: dict = {}
        self.n: dict = {}
        self.cache: dict = {}

    def add(self, key, values: np.ndarray) -> None:
        if len(values) == 0:
            return
        b = self.buf.get(key)
        if b is None:
            b = self.buf[key] = np.empty(self.cap)
            self.n[key] = 0
        n = self.n[key]
        for x in values[-self.cap:]:
            b[n % self.cap] = x
            n += 1
        self.n[key] = n

    def count(self, key) -> int:
        return min(self.n.get(key, 0), self.cap)

    def stat(self, key, q: float = 50.0, min_n: int = MIN_REF) -> float | None:
        n = self.n.get(key, 0)
        if min(n, self.cap) < min_n:
            return None
        ck = (key, q)
        c = self.cache.get(ck)
        if c is not None and n - c[0] < 20:  # refresh every 20 new values
            return c[1]
        val = float(np.percentile(self.buf[key][:min(n, self.cap)], q))
        self.cache[ck] = (n, val)
        return val


class Level:
    """Level-triggered state with hysteresis: active once ``score >= on`` for ``hold`` seconds."""

    def __init__(self, on: float, off: float, hold: float, clear: float) -> None:
        self.on, self.off, self.hold, self.clear = on, off, hold, clear
        self.active = False
        self.since: float | None = None
        self._cand: float | None = None
        self._low: float | None = None

    def update(self, t: float, score: float | None) -> None:
        s = -1.0 if score is None else score
        if not self.active:
            if s >= self.on:
                if self._cand is None:
                    self._cand = t
                if t - self._cand >= self.hold:
                    self.active, self.since, self._low = True, self._cand, None
            else:
                self._cand = None
        else:
            if s < self.off:
                if self._low is None:
                    self._low = t
                if t - self._low >= self.clear:
                    self.active, self.since, self._cand, self._low = False, None, None, None
            else:
                self._low = None


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _med(a: np.ndarray) -> float:
    return float(np.median(a))


def _mad(a: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


class _Check:
    """One measured quantity of one car: recent window against its own lagged baseline."""

    def __init__(self) -> None:
        self.s = Series()
        self.b = Series()
        self.t_moved = -1e18

    def feed(self, t: np.ndarray, v: np.ndarray) -> None:
        self.s.add(t, v)

    def roll(self, now: float, frozen: bool) -> None:
        """Move samples that have aged past the lag into the baseline (unless the car is flagged)."""
        hi = now - LAG_S
        if hi > self.t_moved:
            if not frozen:
                ts, vals = self.s.slice(self.t_moved, hi)
                self.b.add(ts, vals)
            self.t_moved = hi

    def compare(self, now: float, min_cur: int = 12, min_span: float = 0.0):
        """(drop, z, n_cur): baseline median minus recent median, in baseline noise units.

        ``min_span``: the recent samples must cover at least this many seconds (a single corner is not a lap).
        """
        ts, cur = self.s.slice(now - WIN_S, now)
        if len(cur) < min_cur or (min_span and ts[-1] - ts[0] < min_span):
            return None
        base = self.b.between(now - SPAN_S, now)
        if len(base) >= MIN_BASE:
            bm, bs = _med(base), max(_mad(base), 1e-6)
        else:
            return None
        se = bs / np.sqrt(max(len(cur) / 3.0, 1.0))  # samples are correlated: ~3 per independent look
        drop = bm - _med(cur)
        return drop, drop / se, len(cur), bs


class CarTrack:
    def __init__(self) -> None:
        self.last_utc = -1.0  # newest sample measured
        self.seen_utc = -1.0  # newest sample received
        self.held = None  # the newest batch, measured at the next ingest
        self.pw = _Check()  # full-throttle speed / field speed in the cell
        self.rp = _Check()  # full-throttle rpm / field rpm in the cell
        self.gr = _Check()  # |rpm/speed| / field ratio of the gear (signed, stable gear)
        self.pc = _Check()  # pace: speed / field median speed of the cell, all samples
        self.bk = _Check()  # braking: decel / field decel for the entry speed
        self.still = Series()  # 1 when standing (< 15 km/h) on track
        self.slow = Series()  # 1 when speed < 0.6 * p20 of the cell, else 0
        self.coast = Series()  # 1 when coasting on a straight cell
        self.neutral = Series()  # 1 when gear 0 and moving
        self.skip = Series()  # 1 per upshift of 2+ gears, 0 per ordinary upshift
        self.bad_gear = Series()  # 1 when rpm/speed is off by > 12 % for the gear (stable, fast)
        self.long_brake = Series()  # brake time (s) of each braking phase from > 100 km/h
        self.brake_open: tuple[float, float, float] | None = None  # (utc0, v0, vlast)
        self.last_gear = np.nan
        self.last_throttle = np.nan
        self.scores: dict[str, float | None] = dict.fromkeys(CHECKS)
        self.levels = {c: Level(THRESH[c][0], THRESH[c][0] * OFF_RATIO, THRESH[c][1], THRESH[c][2]) for c in CHECKS}
        self.detail: dict[str, float] = {}
        self.n_ok = 0.0  # seconds of usable data seen


def _positions_agree(data: dict, pos: dict, u: np.ndarray) -> np.ndarray:
    """For each CarData sample time ``u``: does the position feed around it agree with CarData?

    Speed implied by consecutive positions (1/10 m) against the CarData speed. A healthy feed gives
    a ratio of 1.00; a frozen or scrambled one (2026 Hungary after t = 4500 s) gives ~0.
    Where there is too little to judge (slow, or few samples) the sample counts as agreeing.
    """
    ok_all = np.ones(len(u), bool)
    pu = pos["utc"]
    if len(pu) < 4 or len(data["utc"]) < 2:
        return ok_all
    o = np.argsort(pu, kind="stable")
    pu, x, y = pu[o], pos["x"][o], pos["y"][o]
    dt = np.diff(pu)
    good = (dt > 0.05) & (dt < 2.0)
    vp = np.hypot(np.diff(x), np.diff(y)) / 10.0 / np.maximum(dt, 1e-3) * 3.6
    mid = 0.5 * (pu[1:] + pu[:-1])
    vt = np.interp(mid, data["utc"], data["speed"])
    judge = good & (vt > 80)
    agree = np.ones(len(mid), bool)
    r = vp[judge] / vt[judge]
    agree[judge] = (r >= 0.5) & (r <= 1.8)
    idx = np.clip(np.searchsorted(mid, u), 0, len(mid) - 1)
    near = np.minimum(idx, np.maximum(idx - 1, 0))
    # a sample is trusted only if the pairs on both sides agree (a freeze starts between two of them)
    return agree[idx] & agree[near] & agree[np.clip(idx + 1, 0, len(mid) - 1)]


class MechanicTelemetry(Engineer):
    name = "mechanic_telemetry"
    requires = ()
    features = ()
    in_bench = False

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self.cars: dict[str, CarTrack] = {}
        self._last_ingest = -1e18
        self.ref_speed = Ref()  # (cell, drs) -> speed, full throttle
        self.ref_rpm = Ref()  # (cell, drs) -> rpm, full throttle
        self.ref_all = Ref(600)  # cell -> speed, any throttle
        self.ref_full_share = Ref(600)  # cell -> 1 if full throttle else 0
        self.ref_gear = Ref(600)  # gear -> rpm/speed
        self.ref_brake = Ref(300)  # entry-speed bin -> decel (km/h per s)
        self.ref_gear_speed = Ref(600)  # gear -> speed (for the stuck-gear envelope)

    # ------------------------------------------------------------------ ingest
    def observe(self, state: RaceState) -> None:
        if state.t - self._last_ingest < INGEST_S:
            return
        tel = state.feeds.telemetry
        if tel.n_messages == 0:
            return
        last, self._last_ingest = self._last_ingest, state.t
        span = min(INGEST_S + 2.0 + (state.t - last if last > -1e17 else 0.0), 600.0)
        gate = self._gate_state(state)
        loose = self._loose_gate(state)
        batches = []
        for car in tel.cars():
            d = state.drivers.get(car)
            tr = self.cars.get(car)
            if tr is None:
                tr = self.cars[car] = CarTrack()
            data = tel.telemetry(car, last_s=span)
            utc = data["utc"]
            if len(utc) == 0:
                continue
            new = utc > tr.seen_utc
            if not new.any():
                continue
            pos = tel.position_history(car, last_s=span + 4.0)
            b_new = self._prepare(tr, data, pos, new)
            tr.seen_utc = float(utc[-1])
            car_ok = d is not None and self._car_ok(state, d)
            clear = d is not None and (d.interval is None or d.interval >= CLEAR_GAP_S)  # no slipstream
            # A batch is measured one ingest late, and only if the car was still fine afterwards: the timing
            # feed flags "in pit" a few seconds after the car has slowed for the pit entry.
            held, tr.held = tr.held, (b_new, gate and car_ok, loose and car_ok, clear)
            if held is None:
                continue
            hb, hok, hlo, hclear = held
            batches.append((car, tr, hb, hok and car_ok, hlo and car_ok, hclear))
        # all cars are measured against the reference as it stood before this ingest
        pending = []
        stills = []
        for car, tr, b, ok, lo, clear_air in batches:
            if b is not None and lo:
                stills.append((tr, *self._still(b)))
            if b is not None and ok:
                pending.append((tr, self._measure(tr, b, clear_air)))
            if b is not None:
                tr.last_utc = float(b["utc"][-1])
        # a car standing still while the field is too (a restart grid after a red flag) is not a fault
        field_still = [float(v.mean()) > 0.7 for _, _, v in stills if len(v)]
        crowd = len(field_still) >= 6 and sum(field_still) / len(field_still) >= 0.3
        for tr, ts, v in stills:
            tr.still.add(ts, np.zeros(len(v)) if crowd else v)
        # conditions that move every car together (rain, a restart, wind) are divided out
        for key in ("pw", "rp", "bk", "pc"):
            meds = [float(np.median(p[key][1])) for _, p in pending if len(p[key][1]) >= 2]
            common = float(np.median(meds)) if len(meds) >= 5 else 1.0
            for tr, p in pending:
                if len(p[key][0]):
                    getattr(tr, key).feed(p[key][0], p[key][1] / common)
        for car, tr, b, ok, lo, _ in batches:
            if b is not None and ok:
                self._learn(b)
        for car, tr, b, ok, lo, _ in batches:
            self._score(tr, state.t)

    @staticmethod
    def _gate_state(state: RaceState) -> bool:
        return (
            state.track_status == "1"
            and state.session_status == "Started"
            and state.finished_t is None
            and state.current_lap >= 2
            and not (state.total_laps and max((d.laps for d in state.drivers.values()), default=0) >= state.total_laps)
        )

    @staticmethod
    def _loose_gate(state: RaceState) -> bool:
        """Racing, but not necessarily green: a car standing still on track is wrong under any flag but red."""
        return (
            state.track_status != "5"
            and state.session_status == "Started"
            and state.finished_t is None
            and state.current_lap >= 2
            and not (state.total_laps and max((d.laps for d in state.drivers.values()), default=0) >= state.total_laps)
        )

    @staticmethod
    def _car_ok(state: RaceState, d) -> bool:
        return d.running and not d.in_pit and not d.pit_out and d.laps >= 1 and d.last_out_lap != d.laps + 1

    @staticmethod
    def _prepare(tr: CarTrack, data: dict, pos: dict, new: np.ndarray) -> dict | None:
        """Cut the new samples, attach grid cell and flags."""
        u = data["utc"][new]
        speed, rpm, gear = data["speed"][new], data["rpm"][new], data["gear"][new]
        thr, brk, drs = data["throttle"][new], data["brake"][new], data["drs"][new]
        # previous sample of the car (for shift detection) is carried in the track
        pu = pos["utc"]
        cell = np.full(len(u), -1, dtype=np.int64)
        if len(pu) >= 2:
            o = np.argsort(pu, kind="stable")
            pu_s = pu[o]
            x = np.interp(u, pu_s, pos["x"][o])
            y = np.interp(u, pu_s, pos["y"][o])
            idx = np.clip(np.searchsorted(pu_s, u), 0, len(pu_s) - 1)
            near = np.minimum(np.abs(pu_s[idx] - u), np.abs(pu_s[np.maximum(idx - 1, 0)] - u))
            on = pos["on_track"][o][idx] > 0.5
            cell = np.floor(x / CELL).astype(np.int64) * 100003 + np.floor(y / CELL).astype(np.int64)
            cell[(near > 1.5) | ~on] = -1
            cell[~_positions_agree(data, pos, u)] = -1  # frozen or scrambled positions: no place on track
        valid = np.isfinite(speed) & np.isfinite(rpm) & (thr <= 100) & (brk <= 100)
        return {"utc": u, "speed": speed, "rpm": rpm, "gear": gear, "thr": thr, "brk": brk, "drs": drs,
                "cell": cell, "valid": valid}

    def _drs_key(self, drs: np.ndarray) -> np.ndarray:
        return (drs >= 8).astype(np.int64)

    @staticmethod
    def _still(b: dict) -> tuple[np.ndarray, np.ndarray]:
        mv = b["valid"] & (b["cell"] >= 0)
        return b["utc"][mv], (b["speed"][mv] < 15).astype(float)

    def _measure(self, tr: CarTrack, b: dict, clear_air: bool = True) -> dict:
        u, speed, rpm, gear, thr, brk = b["utc"], b["speed"], b["rpm"], b["gear"], b["thr"], b["brk"]
        cell, valid = b["cell"], b["valid"]
        dk = self._drs_key(b["drs"])
        tr.n_ok += float(u[-1] - u[0]) if len(u) > 1 else 0.0
        # ---- full-throttle speed / rpm against the field in the same cell
        full = valid & (thr >= 99) & (brk == 0) & (speed > 100) & (cell >= 0)
        ts, rs, rr = [], [], []
        for i in np.flatnonzero(full if clear_air else full & False):
            key = (int(cell[i]), int(dk[i]))
            m = self.ref_speed.stat(key)
            if m is None:
                continue
            ts.append(u[i])
            rs.append(speed[i] / m)
            mr = self.ref_rpm.stat(key)
            rr.append(rpm[i] / mr if mr else np.nan)
        rr_a = np.array(rr)
        keep = np.isfinite(rr_a)
        pend = {"pw": (np.array(ts), np.array(rs)), "rp": (np.array(ts)[keep], rr_a[keep])}
        # ---- slow / coasting against the field
        sl_t, sl_v, co_v = [], [], []
        pc_t, pc_v = [], []
        mv = valid & (cell >= 0) & (speed >= 0)
        for i in np.flatnonzero(mv):
            c = int(cell[i])
            lo = self.ref_all.stat(c, 20.0)
            if lo is None or lo < 60:
                continue
            md = self.ref_all.stat(c, 50.0)
            if speed[i] > 60 and md and md > 80:
                pc_t.append(u[i])
                pc_v.append(speed[i] / md)
            sl_t.append(u[i])
            sl_v.append(1.0 if speed[i] < 0.6 * lo else 0.0)
            share = self.ref_full_share.stat(c, 50.0, 80)
            co_v.append(1.0 if (share is not None and share >= 1.0 and thr[i] < 10 and brk[i] == 0 and speed[i] > 50
                                and speed[i] < 0.9 * (self.ref_speed.stat((c, 0)) or 1e9)) else 0.0)
        if sl_t:
            tr.slow.add(np.array(sl_t), np.array(sl_v))
            tr.coast.add(np.array(sl_t), np.array(co_v))
        # ---- gearbox: neutral, skipped upshifts, rpm/speed against the gear
        mov = valid & (speed > 40)
        tr.neutral.add(u[mov], (gear[mov] == 0).astype(float))
        g = np.concatenate([[tr.last_gear], gear])
        dg = np.diff(g)
        du = np.diff(np.concatenate([[tr.last_utc], u]))
        ups = np.isfinite(dg) & (dg > 0) & valid & (gear >= 2) & (thr >= 50) & (du <= 0.45)
        if ups.any():
            tr.skip.add(u[ups], (dg[ups] >= 2).astype(float))
        g_prev = g[:-1]
        stable = valid & (gear >= 4) & (dg == 0) & (speed > 150) & (rpm > 3000)
        if stable.any():
            sel = np.flatnonzero(stable)
            rat = rpm[sel] / speed[sel]
            vals, tt = [], []
            for k, i in enumerate(sel):
                m = self.ref_gear.stat(int(gear[i]), 50.0, 100)
                if m is None:
                    continue
                vals.append(rat[k] / m)
                tt.append(u[i])
            if vals:
                vals_a = np.array(vals)
                tr.bad_gear.add(np.array(tt), (np.abs(vals_a - 1.0) > 0.12).astype(float))
                tr.gr.feed(np.array(tt), vals_a)
        # ---- stuck gear: speed far outside the gear's usual range at speed under power
        # (kept in the same series as bad_gear: both say "gear and speed do not belong together")
        mid = valid & (gear >= 2) & (thr >= 50) & (dg == 0)
        if mid.any():
            sel = np.flatnonzero(mid)
            vals, tt = [], []
            for i in sel:
                hi = self.ref_gear_speed.stat(int(gear[i]), 97.0, 100)
                lo = self.ref_gear_speed.stat(int(gear[i]), 3.0, 100)
                if hi is None or lo is None:
                    continue
                vals.append(1.0 if (speed[i] > 1.15 * hi or speed[i] < 0.6 * lo) else 0.0)
                tt.append(u[i])
            if vals:
                tr.bad_gear.add(np.array(tt), np.array(vals))
        tr.last_gear = float(gear[-1])
        # ---- braking phases (brake on from speed): decel against the field
        pend["bk"] = self._brake_phases(tr, b)
        pend["pc"] = (np.array(pc_t), np.array(pc_v))
        return pend

    def _brake_phases(self, tr: CarTrack, b: dict) -> tuple[np.ndarray, np.ndarray]:
        u, speed, brk, valid = b["utc"], b["speed"], b["brk"], b["valid"]
        ts, vs = [], []
        op = tr.brake_open
        for i in range(len(u)):
            if not valid[i]:
                continue
            on = brk[i] == 100 and b["thr"][i] < 30
            if on:
                if op is None:
                    op = (float(u[i]), float(speed[i]), float(speed[i]))
                else:
                    op = (op[0], op[1], float(speed[i]))
            elif op is not None:
                dur = float(u[i]) - op[0]
                v0, v1 = op[1], op[2]
                op = None
                if v0 >= 140 and 0.4 <= dur <= 12 and v0 - v1 >= 40:
                    dec = (v0 - v1) / max(dur, 0.1)
                    key = int(v0 // 40)
                    tr_ref = self.ref_brake.stat(key, 50.0, 30)
                    self.ref_brake.add(key, np.array([dec]))
                    if tr_ref:
                        ts.append(float(u[i]))
                        vs.append(dec / tr_ref)
                    tr.long_brake.add(np.array([float(u[i])]), np.array([dur]))
        if op is not None and float(u[-1]) - op[0] > 9.0 and op[2] > 60:  # held on for 8 s at speed
            tr.long_brake.add(np.array([float(u[-1])]), np.array([float(u[-1]) - op[0]]))
        tr.brake_open = op
        return np.array(ts), np.array(vs)

    def _learn(self, b: dict) -> None:
        u, speed, rpm, gear, thr, brk, cell, valid = (b[k] for k in ("utc", "speed", "rpm", "gear", "thr", "brk", "cell", "valid"))
        dk = self._drs_key(b["drs"])
        m = valid & (cell >= 0)
        full = m & (thr >= 99) & (brk == 0) & (speed > 100)
        for i in np.flatnonzero(full):
            key = (int(cell[i]), int(dk[i]))
            self.ref_speed.add(key, speed[i:i + 1])
            self.ref_rpm.add(key, rpm[i:i + 1])
        for i in np.flatnonzero(m & (speed >= 0)):
            c = int(cell[i])
            self.ref_all.add(c, speed[i:i + 1])
            self.ref_full_share.add(c, np.array([1.0 if (thr[i] >= 99 and brk[i] == 0) else 0.0]))
        stable = valid & (gear >= 2) & (speed > 60) & (rpm > 3000)
        g_prev = np.concatenate([[np.nan], gear[:-1]])
        st = stable & (gear == g_prev)
        for i in np.flatnonzero(st & (gear >= 4) & (speed > 150)):
            self.ref_gear.add(int(gear[i]), np.array([rpm[i] / speed[i]]))
        for i in np.flatnonzero(valid & (gear >= 2) & (thr >= 50) & (gear == g_prev)):
            self.ref_gear_speed.add(int(gear[i]), speed[i:i + 1])

    # ------------------------------------------------------------------ scores
    def _score(self, tr: CarTrack, t: float) -> None:
        now = tr.last_utc
        for ch in (tr.pw, tr.rp, tr.gr, tr.bk, tr.pc):
            ch.roll(now, frozen=any(l.active for l in tr.levels.values()))
        sc: dict[str, float | None] = {}
        # power: speed at full throttle, backed by rpm
        p = tr.pw.compare(now, min_span=20.0)
        s_pow = None
        if p is not None:
            drop, z, n, _ = p
            s_pow = _clip01((drop - 0.02) / 0.05) * _clip01(z / 4.0)
            tr.detail["pw_drop"] = round(float(drop), 4)
            tr.detail["pw_z"] = round(float(z), 2)
        r = tr.rp.compare(now)
        if r is not None and s_pow is not None:
            drop, z, n, _ = r
            if drop > 0.015 and z > 4:  # rpm down too: both channels agree
                s_pow = max(s_pow, min(1.0, s_pow + 0.15))
        sc["power_loss"] = s_pow
        # gearbox
        sc["gearbox"] = self._gearbox(tr, now)
        # brake
        sc["brake_issue"] = self._brake(tr, now)
        # slow car
        sc["slow_car"] = self._slow(tr, now)
        tr.scores = sc
        for c in CHECKS:
            tr.levels[c].update(t, sc[c])

    def _gearbox(self, tr: CarTrack, now: float) -> float | None:
        win = 20.0
        neu = tr.neutral.between(now - win, now)
        s = 0.0
        have = False
        if len(neu) >= 10:
            have = True
            s = max(s, _clip01((float(neu.mean()) - 0.10) / 0.30))  # neutral in motion for a third of the time
        sk = tr.skip.between(now - 300.0, now)
        if len(sk) >= 40:
            have = True
            base = tr.skip.between(now - SPAN_S, now - LAG_S)
            bf = float(base.mean()) if len(base) >= 100 else 0.03
            s = max(s, _clip01((float(sk.mean()) - max(3.0 * bf, 0.10)) / 0.2))  # share of upshifts that skip a gear
        bg = tr.bad_gear.between(now - 40.0, now)
        if len(bg) >= 40:
            have = True
            base = tr.bad_gear.between(now - SPAN_S, now - LAG_S)
            bm = float(base.mean()) if len(base) >= 100 else 0.03
            s = max(s, _clip01((float(bg.mean()) - max(bm * 3.0, 0.15)) / 0.4))
        return s if have else None

    def _brake(self, tr: CarTrack, now: float) -> float | None:
        s = None
        c = tr.bk.compare(now, min_cur=4)
        if c is not None:
            drop, z, n, _ = c
            s = _clip01((drop - 0.10) / 0.25) * _clip01(z / 4.0)
        return s

    def _slow(self, tr: CarTrack, now: float) -> float | None:
        out = []
        st = tr.still.between(now - 12.0, now)
        if len(st) >= 20:
            out.append(_clip01((float(st.mean()) - 0.7) / 0.25))  # standing on track for most of 12 s
        sl = tr.slow.between(now - 12.0, now)
        if len(sl) >= 20:
            co = tr.coast.between(now - 12.0, now)
            base = tr.slow.between(now - SPAN_S, now - LAG_S)
            # a car that is "slow" in its own early laps has a position/cell mismatch, not a fault
            lo = max(0.6, float(base.mean()) + 0.35) if len(base) >= 100 else 0.6
            out.append(_clip01((float(sl.mean()) - lo) / 0.3))
            out.append(_clip01((float(co.mean()) - 0.5) / 0.4) * 0.9)
        pc = tr.pc.compare(now, min_cur=60, min_span=45.0)
        if pc is not None:  # sudden pace collapse: speed against the field, all samples, 8 % down for a lap
            drop, z, n, _ = pc
            out.append(_clip01((drop - 0.04) / 0.10) * _clip01(z / 4.0))
            tr.detail["pc_drop"] = round(float(drop), 4)
        return max(out) if out else None

    # ------------------------------------------------------------------ outputs
    def car(self, state, number, view):
        tr = self.cars.get(number)
        out: dict = {"mech_risk": None, "mech_issue": "", "mech_since": None, "mech_active": False, "tel_s": 0.0}
        for c in CHECKS:
            out[c] = None
            out[f"{c}_since"] = None
        if tr is None:
            return out
        sc = tr.scores
        have = [(v, c) for c, v in sc.items() if v is not None]
        if have:
            v, c = max(have)
            out["mech_risk"] = round(v, 3)
            out["mech_issue"] = c if v >= 0.5 else ""
        for c in CHECKS:
            out[c] = None if sc[c] is None else round(sc[c], 3)
            out[f"{c}_since"] = tr.levels[c].since
        act = [tr.levels[c].since for c in CHECKS if tr.levels[c].active]
        out["mech_since"] = min(act) if act else None
        out["mech_active"] = bool(act)
        out["tel_s"] = round(tr.n_ok, 1)
        return out

    def alerts(self, state, view):
        out = []
        for n in sorted(self.cars, key=lambda x: int(x) if x.isdigit() else 999):
            tr = self.cars[n]
            tla = state.drivers[n].tla if n in state.drivers else ""
            for c in CHECKS:
                lv = tr.levels[c]
                if lv.active:
                    s = tr.scores[c] or 0.0
                    out.append(Alert(
                        state.t, self.name, f"mech_{c}", "critical" if s >= 0.9 else "warn",
                        f"Car {n}{f' ({tla})' if tla else ''}: {c.replace('_', ' ')} suspected from telemetry (score {s:.2f})",
                        car=n, since=lv.since, data={"score": round(s, 3)}))
        return out
