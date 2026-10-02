"""Values the pit wall passes around.

Everything an engineer hands out is a scalar or an immutable record of
scalars. A value can then never alias live state, so it can't change after
the moment it was produced - which is what keeps replays honest.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Union

Scalar = Union[float, int, str, bool, None]

SEVERITIES = ("info", "warn", "critical")
ACTIONS = ("BOX", "STAY_OUT", "PREPARE_BOX", "BOX_IF_SC", "NO_CALL")


def is_scalar(x: Any) -> bool:
    return x is None or isinstance(x, (bool, int, float, str))


def jsonable(x: Any) -> Any:
    """Plain JSON: NaN/inf become None (JSON has no NaN)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


@dataclass(frozen=True)
class Alert:
    """Something the pit wall must hear about, active as of ``t``.

    Alerts are level-triggered: an engineer reports the conditions that are true
    now (``since`` = when it became true), so the result never depends on how
    often anyone asks.
    """

    t: float
    engineer: str
    code: str  # machine-readable, e.g. "undercut_threat", "penalty", "rain_onset"
    severity: str  # info | warn | critical
    message: str  # one plain-English line
    car: str | None = None
    since: float | None = None
    data: dict[str, Scalar] = field(default_factory=dict)  # read-only by convention


@dataclass(frozen=True)
class Reason:
    code: str  # e.g. "undercut_gain", "tyre_cliff", "must_stop"
    text: str  # one plain-English clause
    value: Scalar = None


@dataclass(frozen=True)
class PlanStop:
    lap: int  # in-lap: the car boxes at the end of this lap
    compound: str  # tyre fitted


@dataclass(frozen=True)
class Plan:
    name: str  # "A", "B", ...
    stops: tuple[PlanStop, ...] = ()
    expected_position: float | None = None
    expected_points: float | None = None
    trigger: str | None = None  # when to switch to this plan, e.g. "SC before lap 30"


@dataclass(frozen=True)
class Call:
    """The head of strategy's decision for one car, as of ``t``."""

    t: float
    car: str
    action: str  # one of ACTIONS
    compound: str | None = None  # tyre to fit when boxing
    confidence: float | None = None  # 0..1
    reasons: tuple[Reason, ...] = ()  # most important first
    plan_a: Plan | None = None
    plan_b: Plan | None = None


@dataclass(frozen=True)
class TeamConfig:
    """Whose pit wall this is: a team name as in the DriverList, or explicit car numbers."""

    team: str | None = None
    cars: tuple[str, ...] = ()  # explicit car numbers win over ``team``

    def match(self, teams: set[str]) -> str | None:
        """The team meant by ``team``: exact name (any case), else the only name containing it."""
        if not self.team:
            return None
        key = self.team.strip().lower()
        exact = [t for t in teams if t.lower() == key]
        if exact:
            return exact[0]
        partial = [t for t in teams if key in t.lower()]
        return partial[0] if len(partial) == 1 else None  # "bull" -> Red Bull Racing or Racing Bulls?

    def focus(self, state) -> list[str]:
        if self.cars:
            return [c for c in self.cars if c in state.drivers]
        team = self.match({d.team for d in state.drivers.values() if d.team})
        return sorted((n for n, d in state.drivers.items() if d.team == team), key=_num) if team else []


def _num(n: str) -> int:
    return int(n) if n.isdigit() else 999


@dataclass(frozen=True)
class Snapshot:
    """Everything the pit wall shows at one moment."""

    t: float
    lap: int
    total_laps: int | None
    track_status: str
    session_status: str | None
    tower: list[dict]  # timing tower rows in running order (base fields)
    cars: dict[str, dict[str, Scalar]]  # engineer values per car, keys "<engineer>__<key>"
    race: dict[str, Scalar]  # race-level engineer values
    alerts: list[Alert]
    calls: list[Call]
    focus: list[str]  # the team's cars, if a team is set

    def to_dict(self) -> dict:
        return jsonable(asdict(self))
