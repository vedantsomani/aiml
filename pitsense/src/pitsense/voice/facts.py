"""The facts a voice message may cite, and their compact, deterministic text form.

`Facts` is built from a `Call` plus the values the engineers report (a pit-wall
`Snapshot` or a benchmark row). `Facts.text(task)` is the model input: one line,
fields separated by " | ". Every number, car, tyre or lap the voice may say is in
that line, so a guard can check the output against it.
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass
from typing import Any, Mapping

INTENTS = ("gap", "tyre_age", "pit_window", "plan_b", "why")
TASKS = ("radio", "brief") + tuple(f"ask_{i}" for i in INTENTS) + ("free",)
N_STYLE = 6
COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
ACTIONS = ("BOX", "STAY_OUT", "PREPARE_BOX", "BOX_IF_SC", "NO_CALL")
KNOWN_REASONS = (
    "undercut_gain", "tyre_cliff", "must_stop", "rejoin_if_box_now", "sc_window", "penalty_serve",
    "rain_onset", "undercut_threat", "clean_air", "tyre_life", "pit_loss_high", "rival_stopped",
)


def _num(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _i(x: Any) -> int | None:
    f = _num(x)
    return None if f is None else int(round(f))


def _d1(x: Any) -> str | None:
    f = _num(x)
    return None if f is None else f"{f:.1f}"


def _pct(x: Any) -> int | None:
    f = _num(x)
    return None if f is None else max(0, min(100, int(round(100 * f))))


def _word(s: Any) -> str:
    """A free-text clause kept to plain lowercase words, digits and single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()[:60].strip()


def compound_name(c: Any) -> str | None:
    c = str(c or "").upper()
    return c if c in COMPOUNDS else None


@dataclass(frozen=True)
class Neighbour:
    car: str
    tla: str
    gap: str | None  # seconds, one decimal


@dataclass(frozen=True)
class PlanF:
    stops: tuple[tuple[int, str], ...]  # (in-lap, compound)
    pos: int | None = None
    trigger: str | None = None


@dataclass(frozen=True)
class Facts:
    action: str
    car: str
    tla: str
    lap: int | None = None
    total: int | None = None
    pos: int | None = None
    cmp: str | None = None
    age: int | None = None
    stops: int | None = None
    fit: str | None = None
    conf: int | None = None
    ahead: Neighbour | None = None
    behind: Neighbour | None = None
    loss: str | None = None
    rejoin: int | None = None
    cliff: int | None = None
    deg: str | None = None
    uc: int | None = None
    sc: str | None = None  # none | sc | vsc | sc_ending | vsc_ending | red
    pen: int | None = None
    dt: bool = False
    rain: int | None = None
    xover: str | None = None
    must: bool = False
    plan_a: PlanF | None = None
    plan_b: PlanF | None = None
    reasons: tuple[tuple[str, str | None, str | None], ...] = ()  # (code, value, text for unknown codes)
    style: tuple[int, ...] = (0,) * N_STYLE

    # ------------------------------------------------------------------ derived
    @property
    def left(self) -> int | None:
        return None if self.lap is None or self.total is None else max(0, self.total - self.lap)

    @property
    def plan_in(self) -> int | None:
        if self.plan_a and self.plan_a.stops and self.lap is not None:
            return max(0, self.plan_a.stops[0][0] - self.lap)
        return None

    # ------------------------------------------------------------------ text
    def text(self, task: str = "radio", target: str | None = None) -> str:
        """The model input. ``task``: radio | brief | ask_<intent>; ``target`` is the car asked about (ask_gap) or the question text (free)."""
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        head = task + (f" {target}" if target and task in ("ask_gap", "free") else "")
        parts = [f"{head} S{''.join(str(s % 10) for s in self.style)}", f"ACT {self.action}", f"CAR {self.car} {self.tla}"]

        def add(tag, *vals):
            if all(v is not None for v in vals):
                parts.append(" ".join([tag, *[str(v) for v in vals]]))

        add("LAP", self.lap, self.total, self.left)
        add("POS", self.pos)
        if self.cmp:
            parts.append(f"TY {self.cmp} {self.age if self.age is not None else 0} {self.stops if self.stops is not None else 0}")
        add("FIT", self.fit)
        add("CONF", self.conf)
        for tag, n in (("AH", self.ahead), ("BH", self.behind)):
            if n:
                parts.append(f"{tag} {n.car} {n.tla}" + (f" {n.gap}" if n.gap else ""))
        add("LOSS", self.loss)
        add("REJ", self.rejoin)
        add("CLF", self.cliff)
        add("DEG", self.deg)
        add("UC", self.uc)
        if self.sc and self.sc != "none":
            parts.append(f"SC {self.sc}")
        add("PEN", self.pen)
        if self.dt:
            parts.append("DT")
        add("RN", self.rain)
        if self.xover and self.xover != "none":
            parts.append(f"XO {self.xover}")
        if self.must:
            parts.append("MUST")
        for tag, p in (("PA", self.plan_a), ("PB", self.plan_b)):
            if p:
                parts.append(f"{tag} {len(p.stops)}" + "".join(f" {lap} {c}" for lap, c in p.stops))
                add(tag + "P", p.pos)
        add("IN", self.plan_in)
        if self.plan_b and self.plan_b.trigger:
            parts.append(f"TRG {self.plan_b.trigger}")
        for code, val, txt in self.reasons[:3]:
            parts.append(f"R {code}" + (f" {val}" if val is not None else "") + (f" ~ {txt}" if txt else ""))
        return " | ".join(parts)


def style_for(car: str, t: float, salt: str = "") -> tuple[int, ...]:
    """Deterministic style digits from (car, time): the wording variant, so the same call always reads the same."""
    h = zlib.crc32(f"{salt}|{car}|{round(float(t), 1)}".encode())
    return tuple((h >> (4 * k)) % 6 for k in range(N_STYLE))


# ---------------------------------------------------------------------- builders
def _plan(p: Any) -> PlanF | None:
    if p is None:
        return None
    stops = tuple((int(s.lap), str(s.compound).upper()) for s in p.stops)
    return PlanF(stops, _i(p.expected_position), (_word(p.trigger) or None) if p.trigger else None)


def _reasons(call) -> tuple:
    out = []
    for r in call.reasons:
        v = r.value
        if isinstance(v, bool) or v is None:
            sv = None
        elif isinstance(v, float):
            sv = _d1(v)
        else:
            sv = str(v)
        known = r.code in KNOWN_REASONS
        out.append((r.code if known else "other", sv, None if known else _word(r.text)))
    return tuple(out)


def from_values(call, v: Mapping[str, Any], tla_of: Mapping[str, str], style: tuple[int, ...] | None = None) -> Facts:
    """Facts from a Call and a flat dict of values.

    ``v`` uses the benchmark-row names: base columns (``position``, ``compound``, ``tyre_age``,
    ``lap``, ``total_laps``, ``pit_stops``) and ``<engineer>__<key>`` values. Missing or NaN values
    are simply left out of the facts.
    """
    g = v.get

    def nb(key_car, key_gap):
        c = g(key_car)
        if c in (None, "") or (isinstance(c, float) and math.isnan(c)):
            return None
        c = str(c)
        return Neighbour(c, tla_of.get(c, c), _d1(g(key_gap)))

    sc = g("rules__sc_phase") or None
    deg = _num(g("tyre__deg_s_per_lap"))
    return Facts(
        action=call.action,
        car=str(call.car),
        tla=tla_of.get(str(call.car), str(call.car)),
        lap=_i(g("lap")),
        total=_i(g("total_laps")),
        pos=_i(g("position")),
        cmp=compound_name(g("compound")),
        age=_i(g("tyre_age")),
        stops=_i(g("pit_stops")),
        fit=compound_name(call.compound) if call.action != "STAY_OUT" else None,
        conf=_pct(call.confidence),
        ahead=nb("rivals__ahead", "rivals__gap_ahead"),
        behind=nb("rivals__behind", "rivals__gap_behind"),
        loss=_d1(g("pitstop__loss_now")),
        rejoin=_i(g("pitstop__rejoin_if_box_now")),
        cliff=_pct(g("tyre__cliff_risk")),
        deg=None if deg is None else f"{deg:.2f}",
        uc=_pct(g("rivals__undercut_threat")),
        sc=str(sc) if sc else None,
        pen=_i(g("rules__penalty_s_pending")) or None,
        dt=bool(g("rules__drive_through_pending")),
        rain=_pct(g("weather__rain_prob_10min")),
        xover=g("weather__crossover") or None,
        must=bool(g("rules__must_stop")),
        plan_a=_plan(call.plan_a),
        plan_b=_plan(call.plan_b),
        reasons=_reasons(call),
        style=style if style is not None else style_for(call.car, g("t") or 0.0),
    )


def snapshot_values(snapshot, car: str) -> tuple[dict, dict]:
    """Flat values for one car from a pit-wall Snapshot, plus the car-number -> TLA map."""
    row = next((r for r in snapshot.tower if r["car"] == car), {})
    v: dict[str, Any] = {
        "t": snapshot.t, "lap": snapshot.lap, "total_laps": snapshot.total_laps,
        "position": row.get("position"), "compound": row.get("compound"),
        "tyre_age": row.get("tyre_age"), "pit_stops": row.get("pit_stops"),
    }
    v.update(snapshot.race)
    v.update(snapshot.cars.get(car, {}))
    tla = {r["car"]: r["tla"] for r in snapshot.tower if r.get("tla")}
    return v, tla


def from_snapshot(call, snapshot) -> Facts:
    v, tla = snapshot_values(snapshot, call.car)
    return from_values(call, v, tla, style_for(call.car, snapshot.t))


# ---------------------------------------------------------------------- vocabulary of an input (for the guard)
_NUM = re.compile(r"\d+(?:\.\d+)?")


def allowed_numbers(text: str) -> set[str]:
    return set(_NUM.findall(text))
