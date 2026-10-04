"""The engineer contract.

An engineer is one role on a strategy team. It follows the race through the
same RaceState as everything else and answers with plain values as of now:

    observe(state)          after every applied event - keep it cheap
    car(state, n, view)     values about car n          -> {key: scalar}
    race(state, view)       values about the race       -> {key: scalar}
    alerts(state, view)     conditions active right now -> [Alert]
    calls(state, view)      decisions (head of strategy) -> [Call]

Values are flattened to "<engineer>__<key>" in benchmark rows and snapshots,
so the corrupted-future test covers every engineer automatically.

An engineer may read only: ``state``, ``self.ctx`` (what was known before the
race), ``self.memory`` (indexes of the past) and ``view`` (the other engineers'
values at this same moment, for engineers listed in ``requires``). Never the
event log, never ``bench.labels``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from ..asof import RaceSummary, assert_trained_before
from ..pitloss import PitLossPrior
from .memory import RaceMemory
from .types import Alert, Call, Scalar, TeamConfig, is_scalar

if TYPE_CHECKING:
    from ..asof import HistoryStore
    from ..state import RaceState
    from .wall import PitWall


@dataclass
class Context:
    """What the pit wall may know before the race starts."""

    prior: PitLossPrior = field(default_factory=PitLossPrior)
    past_races: tuple[RaceSummary, ...] = ()  # only races that ended before this one started
    race_start_utc: datetime | None = None
    meta: dict = field(default_factory=dict)  # session info (archive SessionRef fields / recording meta)
    team: TeamConfig = field(default_factory=TeamConfig)
    models: Any = None  # cross-race models; must expose ``train_end_utc`` (checked below)

    @staticmethod
    def for_race(
        prior: PitLossPrior,
        meta: dict | None = None,
        *,
        history: "HistoryStore | None" = None,
        race_start_utc: datetime | None = None,
        team: TeamConfig | None = None,
        models: Any = None,
    ) -> "Context":
        past = tuple(history.asof(race_start_utc)) if history is not None and race_start_utc else ()
        if models is not None and race_start_utc is not None:
            assert_trained_before([models.train_end_utc], race_start_utc)
        return Context(prior, past, race_start_utc, dict(meta or {}), team or TeamConfig(), models)


class Engineer:
    """Base class: override what your role needs. All methods must be as-of ``state.t``."""

    name: ClassVar[str] = ""
    requires: ClassVar[tuple[str, ...]] = ()  # engineers whose values this one reads via ``view``
    features: ClassVar[tuple[str, ...]] = ()  # per-car keys benchmark models may use as inputs
    in_bench: ClassVar[bool] = True  # False for slow engineers (simulation): not run per decision row

    def __init__(self, ctx: Context, memory: RaceMemory) -> None:
        self.ctx = ctx
        self.memory = memory

    @classmethod
    def summarize_race(cls, final: "RaceState", meta: dict) -> dict:
        """What this role learns from a finished race, for races that start after it ended.

        Runs on the complete race when the benchmark history is built and is stored
        in ``RaceSummary.extra[name]`` (JSON values only). Read it back through
        ``self.ctx.past_races``, which only holds races finished before this one started.
        """
        return {}

    def observe(self, state: "RaceState") -> None:
        pass

    def prefetch(self, state: "RaceState", numbers: list[str], view: "View") -> None:
        """Optional: called before a snapshot asks ``car`` for each of ``numbers``, so a batch can be prepared.
        Must not change what ``car`` returns."""

    def car(self, state: "RaceState", number: str, view: "View") -> dict[str, Scalar]:
        return {}

    def race(self, state: "RaceState", view: "View") -> dict[str, Scalar]:
        return {}

    def alerts(self, state: "RaceState", view: "View") -> list[Alert]:
        return []

    def calls(self, state: "RaceState", view: "View") -> list[Call]:
        return []


def _checked(engineer: str, what: str, values: dict) -> dict[str, Scalar]:
    """Copy and validate an engineer's answer: flat keys, scalar values only."""
    out = {}
    for k, v in values.items():
        if not isinstance(k, str) or "__" in k:
            raise ValueError(f"{engineer}.{what}: bad key {k!r} (plain strings, no '__')")
        if not is_scalar(v):
            raise ValueError(f"{engineer}.{what}[{k!r}] is {type(v).__name__}; engineers return scalars only")
        out[k] = v
    return out


class View:
    """The engineers' values at one moment, computed on demand and memoised."""

    def __init__(self, wall: "PitWall", state: "RaceState") -> None:
        self._wall = wall
        self._state = state
        self._car: dict[tuple[str, str], dict[str, Scalar]] = {}
        self._race: dict[str, dict[str, Scalar]] = {}

    def car(self, engineer: str, number: str) -> dict[str, Scalar]:
        key = (engineer, number)
        if key not in self._car:
            e = self._wall.engineer(engineer)
            self._car[key] = _checked(engineer, "car", e.car(self._state, number, self))
        return self._car[key]

    def race(self, engineer: str) -> dict[str, Scalar]:
        if engineer not in self._race:
            e = self._wall.engineer(engineer)
            self._race[engineer] = _checked(engineer, "race", e.race(self._state, self))
        return self._race[engineer]
