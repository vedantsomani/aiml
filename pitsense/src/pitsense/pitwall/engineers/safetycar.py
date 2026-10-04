"""Safety-car engineer: "SC / VSC likely within the next 1-2 laps", as of now.

Signals (all read from what has been published by ``state.t``):

* race control: yellow and double-yellow sectors (and when they started / cleared), "car stopped", off-track
  and spun cars, recovery vehicle / marshals / medical car / debris, incidents noted or under investigation;
* the track status itself: how long status 2 (yellow) has lasted;
* timing: a car losing a lot of time on a lap against its own recent clean laps;
* lap 1 chaos (messages of the first lap) and how recently a neutralisation ended (restarts);
* wet running (inters / wets on the last laps);
* the circuit's history: SC / VSC starts per lap at this circuit in earlier races (``ctx.past_races``);
* telemetry / position feeds when the race was loaded with them: a car stopped on track, or off track.

Two small logistic models (SC and VSC) turn the signals into probabilities. They are fitted at the start of
the race on the lap-end samples that :meth:`SafetyCarEngineer.summarize_race` stored for every earlier race
(``ctx.past_races`` only holds races that ended before this one started). The circuit's history enters as a
shift of the log-odds. With too little history a fixed default weighting is used.

Values (race scope, ``safetycar__<key>``): ``sc_prob_2laps``, ``vsc_prob_2laps``, ``neutral_prob_2laps``
(either), ``sc_reason`` and the signal counts below. Docs: docs/engineers/safetycar.md.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from statistics import median

import numpy as np

from ..engineer import Engineer
from ..types import Alert

HORIZON_LAPS = 2

# --- signal windows (seconds of race time)
W_FLAG = 90.0
W_STOP = 180.0
W_VEH = 300.0
W_LOSS = 150.0
YELLOW_EXPIRE = 420.0  # a yellow sector with no "clear" message this long is treated as cleared
POST_NEUTRAL_S = 400.0

# --- model
FEATURES = (
    "yellow_now", "yellow_age", "dy_active", "y_active", "yellow_old", "dy_recent", "y_recent",
    "stopped", "offtrack", "vehicle", "incident", "loss_max", "loss_n", "lap1_chaos", "early",
    "wet", "post_neutral", "frac",
)
NF = len(FEATURES)
L2 = 5.0
MIN_POSITIVES = 25  # below this many positives in earlier races, use DEFAULT_W
CIRCUIT_BETA = 0.25  # weight of the circuit's log-odds shift
CIRCUIT_SHRINK_LAPS = 120.0
DEFAULT_RATE = {"sc": 0.0075, "vsc": 0.0045}  # starts per lap (all circuits, no history)
ALERT_P = 0.08  # warn when P(SC or VSC within 2 laps) reaches this (tuned on 2025, see docs)
BIG_LOSS_S = 5.0
FEED_STOP_KMH = 8.0
FEED_STOP_S = 6.0

# Used with fewer than MIN_POSITIVES earlier positives: hand-set weights, intercept from the default rate.
DEFAULT_W = {
    "sc": [0.0] * NF,
    "vsc": [0.0] * NF,
}
for _k, _v in {"yellow_now": 1.6, "yellow_age": 0.8, "y_active": 1.0, "y_recent": 0.8, "dy_active": 1.0, "dy_recent": 0.8, "stopped": 1.2,
               "offtrack": 0.4, "vehicle": 0.8, "incident": 0.5, "loss_max": 0.6, "lap1_chaos": 0.6,
               "post_neutral": 0.3, "wet": 0.4}.items():
    DEFAULT_W["sc"][FEATURES.index(_k)] = _v
    DEFAULT_W["vsc"][FEATURES.index(_k)] = _v * 0.7
del _k, _v

# --------------------------------------------------------------------------- message classes
_SECTOR = re.compile(r"SECTOR (\d+)")
_STOPPED = re.compile(r"\bCAR \d+(?: \([A-Z]{2,4}\))? (?:STOPPED|STOPPING)\b|\bSTOPPED (?:ON|AT|IN) ")
_OFF = re.compile(r"\b(?:OFF TRACK|SPUN|SPIN|CRASH|COLLISION|COLLIDED)\b")
_VEH = re.compile(r"RECOVERY VEHICLE|MARSHALS ON TRACK|MEDICAL CAR|DEBRIS|FIRE|OIL ON TRACK|OBJECT ON TRACK")
_INC = re.compile(r"\bINCIDENT\b")
_SLIP = re.compile(r"SLIPPERY")


def _classify(category: str, flag: str, message: str) -> tuple[str, int | None]:
    """(kind, sector) of one race-control message: ``dy``, ``y``, ``clear``, ``clear_all``, ``stopped``,
    ``off``, ``vehicle``, ``incident``, ``slip`` or ``other``."""
    text = re.sub(r"\s+", " ", message.upper()).strip()
    fl = flag.upper()
    m = _SECTOR.search(text)
    sector = int(m.group(1)) if m else None
    if fl == "DOUBLE YELLOW" or text.startswith("DOUBLE YELLOW"):
        return "dy", sector
    if fl == "YELLOW" or text.startswith("YELLOW"):
        return "y", sector
    if text.startswith("CLEAR IN TRACK SECTOR") or (fl == "CLEAR" and sector is not None):
        return "clear", sector
    if text == "TRACK CLEAR" or (fl == "CLEAR" and sector is None):
        return "clear_all", None
    if "PENALTY" in text or "BLUE FLAG" in text or "TRACK LIMITS" in text:
        return "other", None
    if _STOPPED.search(text):
        return "stopped", None
    if _OFF.search(text):
        return "off", None
    if _VEH.search(text):
        return "vehicle", None
    if _INC.search(text):
        return "incident", None
    if _SLIP.search(text):
        return "slip", None
    return "other", None


class Tracker:
    """Race-control and timing bookkeeping behind the signals. Feed it in time order; ask it about ``t``."""

    def __init__(self) -> None:
        self.events: list[tuple[float, str, int | None]] = []  # (t, kind, sector), time order
        self.active: dict[int, tuple[float, str]] = {}  # sector -> (since, "y" | "dy")
        self.lap1_events = 0
        self.laps: list = []  # LapRecord references (lap time may be filled in later)
        self.by_driver: dict[str, dict[int, object]] = {}
        self.green_t: float | None = None  # when a neutralisation last ended
        self._status = "1"

    # ------------------------------------------------------------------ feeding
    def add_rc(self, t: float, category: str, flag: str, message: str, lap: int | None) -> None:
        kind, sector = _classify(category, flag, message)
        if kind == "other":
            return
        self.events.append((t, kind, sector))
        if kind in ("dy", "y") and sector is not None:
            cur = self.active.get(sector)
            if cur is None:
                self.active[sector] = (t, kind)
            elif kind == "dy" and cur[1] == "y":
                self.active[sector] = (cur[0], "dy")
        elif kind == "clear" and sector is not None:
            self.active.pop(sector, None)
        elif kind == "clear_all":
            self.active.clear()
        if kind in ("dy", "y", "stopped", "off", "incident") and (lap or 0) <= 1:
            self.lap1_events += 1

    def add_lap(self, rec) -> None:
        self.laps.append(rec)
        self.by_driver.setdefault(rec.driver, {})[rec.lap] = rec

    def set_status(self, t: float, status: str) -> None:
        if status != self._status:
            if self._status in ("4", "6", "7") and status in ("1", "2"):
                self.green_t = t
            self._status = status

    # ------------------------------------------------------------------ signals
    def _count(self, t: float, kinds: tuple[str, ...], window: float) -> int:
        n = 0
        for et, k, _ in reversed(self.events):
            if et > t:
                continue
            if et < t - window:
                break
            n += k in kinds
        return n

    def _baseline(self, rec) -> float | None:
        prev = self.by_driver.get(rec.driver, {})
        times = []
        for lap in range(rec.lap - 1, max(rec.lap - 9, 1), -1):
            p = prev.get(lap)
            if (p is not None and p.lap_time is not None and not p.is_in_lap and not p.is_out_lap
                    and p.track_status == "1"):
                times.append(p.lap_time)
                if len(times) == 5:
                    break
        return median(times) if len(times) >= 3 else None

    def losses(self, t: float) -> list[float]:
        """Lap-time loss (s) of cars whose lap time was published in the last ``W_LOSS`` s, against their own clean laps."""
        out = []
        for rec in reversed(self.laps):
            if rec.t_end > t:
                continue
            if rec.t_end < t - W_LOSS - 120.0:
                break
            if (rec.lap_time is None or rec.lap_time_t is None or rec.lap_time_t > t or rec.lap_time_t < t - W_LOSS
                    or rec.lap <= 1 or rec.is_in_lap or rec.is_out_lap):
                continue
            base = self._baseline(rec)
            if base is not None:
                out.append(rec.lap_time - base)
        return out

    def wet_share(self, t: float) -> float:
        n = w = 0
        for rec in reversed(self.laps):
            if rec.t_end > t:
                continue
            if rec.t_end < t - 200.0:
                break
            n += 1
            w += rec.compound in ("INTERMEDIATE", "WET")
        return w / n if n else 0.0

    def features(self, t: float, status: str, status_since: float, lap: int, total_laps: int | None) -> dict[str, float]:
        act = {s: v for s, v in self.active.items() if t - v[0] <= YELLOW_EXPIRE}
        dy = sum(1 for _, k in act.values() if k == "dy")
        yy = len(act) - dy
        losses = self.losses(t)
        big = [x for x in losses if x >= BIG_LOSS_S]
        age = (t - status_since) if status == "2" else 0.0
        oldest = max((t - v[0] for v in act.values()), default=0.0)
        post = 0.0
        if self.green_t is not None and t >= self.green_t:
            post = max(0.0, 1.0 - (t - self.green_t) / POST_NEUTRAL_S)
        return {
            "yellow_now": 1.0 if status == "2" else 0.0,
            "yellow_age": min(age / 120.0, 1.0),
            "dy_active": min(dy, 3) / 3.0,
            "y_active": min(yy, 3) / 3.0,
            "yellow_old": min(oldest / 180.0, 1.0),
            "dy_recent": min(self._count(t, ("dy",), W_FLAG), 4) / 4.0,
            "y_recent": min(self._count(t, ("y",), W_FLAG), 4) / 4.0,
            "stopped": min(self._count(t, ("stopped",), W_STOP), 2) / 2.0,
            "offtrack": min(self._count(t, ("off",), W_STOP), 3) / 3.0,
            "vehicle": min(self._count(t, ("vehicle",), W_VEH), 2) / 2.0,
            "incident": min(self._count(t, ("incident",), W_STOP), 3) / 3.0,
            "loss_max": min(max(losses, default=0.0), 30.0) / 30.0 if losses else 0.0,
            "loss_n": min(len(big), 4) / 4.0,
            "lap1_chaos": min(self.lap1_events, 6) / 6.0 if lap <= 4 else 0.0,
            "early": 1.0 if lap <= 3 else 0.0,
            "wet": self.wet_share(t),
            "post_neutral": post,
            "frac": min(lap / total_laps, 1.0) if total_laps else 0.5,
        }

    def since(self, t: float) -> float | None:
        """When the evidence behind a warning began: the oldest active yellow sector, else the latest stop / incident."""
        act = [v[0] for v in self.active.values() if t - v[0] <= YELLOW_EXPIRE]
        if act:
            return min(act)
        for et, k, _ in reversed(self.events):
            if et <= t and k in ("stopped", "off", "vehicle", "incident"):
                return et
        return None


# --------------------------------------------------------------------------- the models
def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def _logit(p: float) -> float:
    p = min(max(p, 1e-5), 1 - 1e-5)
    return math.log(p / (1 - p))


def fit_logit(X: np.ndarray, y: np.ndarray, l2: float | None = None, iters: int = 30) -> np.ndarray:
    """L2-regularised logistic regression by Newton steps (deterministic). Returns [intercept, w...]."""
    l2 = L2 if l2 is None else l2
    n, k = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(k + 1)
    w[0] = _logit(float(np.clip(y.mean(), 1e-4, 0.5)))
    reg = np.eye(k + 1) * l2
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y) + reg @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + reg + 1e-6 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if float(np.abs(step).max()) < 1e-7:
            break
    return w


def two_lap(rate: float) -> float:
    return 1.0 - (1.0 - rate) ** HORIZON_LAPS


def vector(f: dict[str, float]) -> list[float]:
    return [f[k] for k in FEATURES]


def sample_labels(starts: dict[str, list[float]], lap_t: list[float], i: int) -> tuple[int, int]:
    """(SC, VSC) onset in (lap_t[i], lap_t[i + HORIZON_LAPS]] (race end when the lap does not exist)."""
    t0 = lap_t[i]
    t1 = lap_t[i + HORIZON_LAPS] if i + HORIZON_LAPS < len(lap_t) else float("inf")
    return tuple(int(any(t0 < s <= t1 for s in starts[k])) for k in ("sc", "vsc"))  # type: ignore[return-value]


def status_starts(status_log: list[tuple[float, str]]) -> dict[str, list[float]]:
    """Times at which an SC (status 4) or a VSC (status 6) began."""
    out: dict[str, list[float]] = {"sc": [], "vsc": []}
    prev = "1"
    for t, code in status_log:
        if code == "4" and prev != "4":
            out["sc"].append(t)
        elif code == "6" and prev != "6":
            out["vsc"].append(t)
        prev = code
    return out


def status_at(status_log: list[tuple[float, str]], t: float) -> str:
    times = [x[0] for x in status_log]
    i = bisect_right(times, t) - 1
    return status_log[i][1] if i >= 0 else "1"


def race_samples(final) -> dict:
    """Training samples and event times from a finished race: one per leader lap end (neutralised moments left out)."""
    first: dict[int, float] = {}
    for rec in final.laps:
        if rec.lap not in first or rec.t_end < first[rec.lap]:
            first[rec.lap] = rec.t_end
    laps_sorted = sorted(first)
    lap_t = [first[lap] for lap in laps_sorted]
    starts = status_starts(list(final.status_log))
    total = final.total_laps or (laps_sorted[-1] if laps_sorted else 0)

    tr = Tracker()
    rc_i = lap_i = st_i = 0
    rc = sorted(final.rc, key=lambda m: m.t)
    recs = sorted(final.laps, key=lambda r: r.t_end)
    log = list(final.status_log)
    samples = []
    status, since = "1", 0.0
    for i, lap in enumerate(laps_sorted):
        t = lap_t[i]
        while rc_i < len(rc) and rc[rc_i].t <= t:
            m = rc[rc_i]
            tr.add_rc(m.t, m.category, m.flag, m.message, m.lap)
            rc_i += 1
        while lap_i < len(recs) and recs[lap_i].t_end <= t:
            tr.add_lap(recs[lap_i])
            lap_i += 1
        while st_i < len(log) and log[st_i][0] <= t:
            status, since = log[st_i][1], log[st_i][0]
            tr.set_status(log[st_i][0], status)
            st_i += 1
        if status in ("4", "5", "6", "7") or lap >= total:
            continue
        f = tr.features(t, status, since, lap, total)
        ysc, yvsc = sample_labels(starts, lap_t, i)
        samples.append([round(x, 4) for x in vector(f)] + [ysc, yvsc])
    return {"samples": samples, "sc": [round(x, 1) for x in starts["sc"]], "vsc": [round(x, 1) for x in starts["vsc"]],
            "laps": int(total), "sc_laps": _start_laps(starts["sc"], lap_t), "vsc_laps": _start_laps(starts["vsc"], lap_t)}


def _start_laps(times: list[float], lap_t: list[float]) -> list[int]:
    return [bisect_right(lap_t, t) + 1 for t in times]  # the (leader) lap the neutralisation began on


class Model:
    """What earlier races teach: signal weights for SC and VSC, and per-lap start rates by circuit."""

    def __init__(self, past_races, circuit) -> None:
        rows, pool_laps, circ_laps = [], 0, 0
        pool = {"sc": 0, "vsc": 0}
        circ = {"sc": 0, "vsc": 0}
        self.n_races = 0
        for race in past_races:
            ex = (race.extra or {}).get("safetycar")
            if not ex:
                continue
            self.n_races += 1
            rows += ex.get("samples") or []
            laps = int(ex.get("laps") or race.laps or 0)
            pool_laps += laps
            pool["sc"] += len(ex.get("sc_laps") or [])
            pool["vsc"] += len(ex.get("vsc_laps") or [])
            if circuit is not None and race.circuit_key == circuit:
                circ_laps += laps
                circ["sc"] += len(ex.get("sc_laps") or [])
                circ["vsc"] += len(ex.get("vsc_laps") or [])
        self.rate, self.pool_rate = {}, {}
        for k in ("sc", "vsc"):
            pr = (pool[k] + 5 * DEFAULT_RATE[k] * 60) / (pool_laps + 300.0)
            self.pool_rate[k] = pr
            self.rate[k] = (circ[k] + CIRCUIT_SHRINK_LAPS * pr) / (circ_laps + CIRCUIT_SHRINK_LAPS)
        self.circuit_laps = circ_laps
        self.w: dict[str, np.ndarray] = {}
        self.fitted = False
        if rows:
            A = np.asarray(rows, dtype=float)
            X = A[:, :NF]
            for j, k in enumerate(("sc", "vsc")):
                y = A[:, NF + j]
                if y.sum() >= MIN_POSITIVES:
                    self.w[k] = fit_logit(X, y)
            self.fitted = len(self.w) == 2
        if not self.fitted:
            for k in ("sc", "vsc"):
                self.w[k] = np.array([_logit(two_lap(self.pool_rate[k]))] + DEFAULT_W[k])

    def prob(self, kind: str, f: dict[str, float]) -> float:
        w = self.w[kind]
        z = float(w[0] + np.dot(w[1:], vector(f)))
        z += CIRCUIT_BETA * (_logit(two_lap(self.rate[kind])) - _logit(two_lap(self.pool_rate[kind])))
        return _sigmoid(z)


# --------------------------------------------------------------------------- reasons
_REASONS = (
    ("dy_active", "double yellow", 0.3),
    ("stopped", "car stopped", 0.3),
    ("vehicle", "recovery vehicle / marshals", 0.3),
    ("offtrack", "car off / spun", 0.3),
    ("incident", "incident noted", 0.3),
    ("yellow_age", "yellow flag lasting", 0.3),
    ("y_active", "yellow flag", 0.3),
    ("loss_max", "car losing time", 0.3),
    ("lap1_chaos", "lap 1 trouble", 0.3),
    ("post_neutral", "restart", 0.4),
)


def reason_text(f: dict[str, float], extras: dict[str, float], base_p: float, circuit_high: bool) -> str:
    parts = []
    if extras.get("feed_stopped", 0) > 0:
        parts.append(f"{int(extras['feed_stopped'])} car stopped on track (telemetry)")
    if extras.get("feed_off", 0) > 0:
        parts.append(f"{int(extras['feed_off'])} car off track (position)")
    for key, label, thr in _REASONS:
        if f[key] >= thr:
            parts.append(label)
    if f["wet"] >= 0.3:
        parts.append("wet track")
    if not parts:
        return "circuit history" if circuit_high else "none"
    return ", ".join(parts[:3])


# --------------------------------------------------------------------------- the engineer
class SafetyCarEngineer(Engineer):
    name = "safetycar"
    requires = ()
    # Declared inputs for the cross-race models (docs/engineers/safetycar.md).
    features = ("sc_prob_2laps", "vsc_prob_2laps")
    in_bench = True

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self.tr = Tracker()
        self._n_rc = 0
        self._n_laps = 0
        self._model: Model | None = None
        self._cache: tuple | None = None

    @classmethod
    def summarize_race(cls, final, meta):
        return race_samples(final)

    # ------------------------------------------------------------------ model
    @property
    def model(self) -> Model:
        if self._model is None:
            key = "_safetycar_model"
            m = self.ctx.__dict__.get(key)
            if m is None:
                m = Model(self.ctx.past_races, self.ctx.meta.get("circuit_key"))
                self.ctx.__dict__[key] = m
            self._model = m
        return self._model

    # ------------------------------------------------------------------ bookkeeping
    def _sync(self, state) -> None:
        rc = state.rc
        while self._n_rc < len(rc):
            m = rc[self._n_rc]
            self._n_rc += 1
            self.tr.add_rc(m.t, m.category, m.flag, m.message, m.lap)
        laps = state.laps
        while self._n_laps < len(laps):
            self.tr.add_lap(laps[self._n_laps])
            self._n_laps += 1
        self.tr.set_status(state.track_status_since, state.track_status)

    def observe(self, state) -> None:
        self._sync(state)

    # ------------------------------------------------------------------ feeds
    def _feed_signals(self, state) -> dict[str, float]:
        """Cars stopped on track (telemetry speed ~0 for several seconds) or off track (position feed); 0 without feeds."""
        out = {"feed_present": 0.0, "feed_stopped": 0.0, "feed_off": 0.0}
        feeds = getattr(state, "feeds", None)
        if feeds is None or feeds.telemetry.n_messages == 0 or not state.started or state.track_status in ("4", "5", "6", "7"):
            return out
        out["feed_present"] = 1.0
        stopped = off = 0
        tel = feeds.telemetry
        pos = tel.latest_position()
        for n, d in state.drivers.items():
            if d.in_pit or d.pit_out or d.laps < 1 or d.retired:
                continue
            p = pos.get(n)
            if p is not None and not p["on_track"]:
                off += 1
            h = tel.telemetry(n, last_s=FEED_STOP_S)
            sp = h["speed"]
            if len(sp) >= 8 and float(np.nanmax(sp)) < FEED_STOP_KMH and float(h["utc"][-1] - h["utc"][0]) >= FEED_STOP_S - 1.5:
                stopped += 1
        out["feed_stopped"], out["feed_off"] = float(stopped), float(off)
        return out

    # ------------------------------------------------------------------ values
    def _compute(self, state) -> dict:
        key = (state.t, len(state.rc), len(state.laps), state.track_status)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        self._sync(state)
        t, status = state.t, state.track_status
        total = state.total_laps
        lap = max(state.current_lap, 1)
        f = self.tr.features(t, status, state.track_status_since, lap, total)
        ex = self._feed_signals(state)
        m = self.model
        if status in ("4", "5", "6", "7"):
            sc = vsc = 0.0
            reason = {"4": "safety car out", "5": "red flag", "6": "VSC out", "7": "VSC ending"}[status]
        else:
            sc, vsc = m.prob("sc", f), m.prob("vsc", f)
            # feeds: a car stopped on track / off track raises the odds (multipliers measured on 2025, see docs)
            if ex["feed_stopped"] > 0:
                sc, vsc = _odds(sc, FEED_STOP_ODDS), _odds(vsc, FEED_STOP_ODDS)
            elif ex["feed_off"] > 0:
                sc, vsc = _odds(sc, FEED_OFF_ODDS), _odds(vsc, FEED_OFF_ODDS)
            circuit_high = m.rate["sc"] + m.rate["vsc"] > 1.5 * (m.pool_rate["sc"] + m.pool_rate["vsc"])
            reason = reason_text(f, ex, sc, circuit_high)
        any_p = 1.0 - (1.0 - sc) * (1.0 - vsc)
        f0 = {k: (v if k in ("early", "frac") else 0.0) for k, v in f.items()}  # no live signal at all: circuit and race stage only
        base = 1.0 - (1.0 - m.prob("sc", f0)) * (1.0 - m.prob("vsc", f0))
        out = {"f": f, "ex": ex, "sc": sc, "vsc": vsc, "any": any_p, "base": base, "reason": reason,
               "dy_n": sum(1 for s, v in self.tr.active.items() if v[1] == "dy" and t - v[0] <= YELLOW_EXPIRE),
               "y_n": sum(1 for s, v in self.tr.active.items() if v[1] == "y" and t - v[0] <= YELLOW_EXPIRE)}
        self._cache = (key, out)
        return out

    def race(self, state, view):
        c = self._compute(state)
        f, ex = c["f"], c["ex"]
        return {
            "sc_prob_2laps": round(c["sc"], 4),
            "vsc_prob_2laps": round(c["vsc"], 4),
            "neutral_prob_2laps": round(c["any"], 4),
            "sc_reason": c["reason"],
            "neutral_prob_base": round(c["base"], 4),
            "sc_warn": warn_on(c),
            "dy_sectors": c["dy_n"],
            "y_sectors": c["y_n"],
            "yellow_age_s": round(f["yellow_age"] * 120.0, 1),
            "stopped_msgs": int(round(f["stopped"] * 2)),
            "incident_msgs": int(round(f["incident"] * 3)),
            "vehicle_msgs": int(round(f["vehicle"] * 2)),
            "loss_max_s": round(f["loss_max"] * 30.0, 2),
            "lap1_chaos": round(f["lap1_chaos"] * 6.0, 1),
            "circuit_sc_rate": round(self.model.rate["sc"], 5),
            "circuit_vsc_rate": round(self.model.rate["vsc"], 5),
            "feed_present": int(ex["feed_present"]),
            "feed_stopped": int(ex["feed_stopped"]),
            "feed_off_track": int(ex["feed_off"]),
            "history_races": self.model.n_races,
        }

    def alerts(self, state, view):
        c = self._compute(state)
        if not warn_on(c):
            return []
        kind = "SC" if c["sc"] >= c["vsc"] else "VSC"
        since = self.tr.since(state.t)
        return [Alert(state.t, self.name, "sc_likely", "critical" if c["any"] >= 3 * ALERT_P else "warn",
                      f"{kind} likely in the next 1-2 laps ({c['any']:.0%}): {c['reason']}", since=since,
                      data={"sc_prob_2laps": round(c["sc"], 4), "vsc_prob_2laps": round(c["vsc"], 4)})]


# Measured on 2025 (docs): a slow car in telemetry has no usable lift (positive share 0.044 vs 0.029; multiplier 3 worsened log loss
# 0.127 -> 0.133), so feed signals are exposed and named in reasons but do not move the probability.
FEED_STOP_ODDS = 1.0
FEED_OFF_ODDS = 1.0


def warn_on(c: dict) -> bool:
    """Warn when P(SC or VSC within 2 laps) reaches ALERT_P *and* is clearly above what the circuit alone gives
    (a circuit with a high base rate must not warn on history alone)."""
    return c["any"] >= ALERT_P and c["any"] >= WARN_OVER_BASE * c["base"]


WARN_OVER_BASE = 1.5


def _odds(p: float, mult: float) -> float:
    o = p / max(1.0 - p, 1e-9) * mult
    return o / (1.0 + o)
