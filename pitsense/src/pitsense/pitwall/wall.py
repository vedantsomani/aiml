"""PitWall: runs the engineers in dependency order and assembles what the pit wall sees."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .engineer import Context, Engineer, View
from .memory import RaceMemory
from .types import Alert, Call, Scalar, Snapshot

if TYPE_CHECKING:
    from ..state import RaceState


def ordered(classes: list[type[Engineer]]) -> list[type[Engineer]]:
    """Dependency order (everything in ``requires`` first), otherwise as listed."""
    by_name: dict[str, type[Engineer]] = {}
    for c in classes:
        if not c.name or "__" in c.name:
            raise ValueError(f"{c.__name__}: engineer name must be set and contain no '__'")
        if c.name in by_name:
            raise ValueError(f"two engineers named {c.name!r}")
        by_name[c.name] = c
    out: list[type[Engineer]] = []
    mark: dict[str, int] = {}  # 0 = visiting, 1 = done

    def visit(c: type[Engineer]) -> None:
        if mark.get(c.name) == 1:
            return
        if mark.get(c.name) == 0:
            raise ValueError(f"engineer dependency cycle through {c.name!r}")
        mark[c.name] = 0
        for dep in c.requires:
            if dep not in by_name:
                raise ValueError(f"{c.name!r} requires {dep!r}, which isn't on this pit wall")
            visit(by_name[dep])
        mark[c.name] = 1
        out.append(c)

    for c in classes:
        visit(c)
    return out


class PitWall:
    """One race, one set of engineers. Call :meth:`observe` after every applied event."""

    def __init__(
        self,
        ctx: Context,
        memory: RaceMemory | None = None,
        engineers: list[type[Engineer]] | None = None,
    ) -> None:
        from .. import registry

        self.ctx = ctx
        self.memory = memory or RaceMemory()
        classes = registry.engineer_classes() if engineers is None else list(engineers)
        self.engineers = [cls(ctx, self.memory) for cls in ordered(classes)]
        self._by_name = {e.name: e for e in self.engineers}
        self._view: View | None = None

    def engineer(self, name: str) -> Engineer:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"no engineer {name!r} on this pit wall") from None

    def observe(self, state: "RaceState") -> None:
        self.memory.observe(state)
        for e in self.engineers:
            e.observe(state)
        self._view = None

    def view(self, state: "RaceState") -> View:
        """Values at the current moment; shared by everything asked before the next event."""
        if self._view is None or self._view._state is not state:
            self._view = View(self, state)
        return self._view

    # ------------------------------------------------------------------ values
    def car_values(self, state: "RaceState", number: str) -> dict[str, Scalar]:
        v = self.view(state)
        return {f"{e.name}__{k}": x for e in self.engineers for k, x in v.car(e.name, number).items()}

    def race_values(self, state: "RaceState") -> dict[str, Scalar]:
        v = self.view(state)
        return {f"{e.name}__{k}": x for e in self.engineers for k, x in v.race(e.name).items()}

    def row_values(self, state: "RaceState", number: str) -> dict[str, Scalar]:
        """Car and race values for one decision row (an engineer may not reuse a key for both)."""
        car, race = self.car_values(state, number), self.race_values(state)
        clash = car.keys() & race.keys()
        if clash:
            raise ValueError(f"keys used for both car and race values: {sorted(clash)}")
        return {**car, **race}

    def alerts(self, state: "RaceState") -> list[Alert]:
        v = self.view(state)
        return [_typed(a, Alert, e.name) for e in self.engineers for a in e.alerts(state, v)]

    def calls(self, state: "RaceState") -> list[Call]:
        v = self.view(state)
        return [_typed(c, Call, e.name) for e in self.engineers for c in e.calls(state, v)]

    def snapshot(self, state: "RaceState") -> Snapshot:
        order = state.running_order()
        tower = [
            {
                "position": d.position, "car": d.number, "tla": d.tla, "team": d.team,
                "gap_to_leader": d.gap_to_leader, "laps_down": d.laps_down, "interval": d.interval,
                "compound": d.compound, "tyre_age": d.tyre_age, "stint": d.stint, "pit_stops": d.pit_stops,
                "in_pit": d.in_pit, "running": d.running, "laps": d.laps,
                "last_lap_time": d.last_lap_time, "best_lap_time": d.best_lap_time,
            }
            for d in order
        ]
        running = [d.number for d in order if d.running]
        return Snapshot(
            t=round(state.t, 3),
            lap=state.current_lap,
            total_laps=state.total_laps,
            track_status=state.track_status,
            session_status=state.session_status,
            tower=tower,
            cars={n: self.car_values(state, n) for n in running},
            race=self.race_values(state),
            alerts=self.alerts(state),
            calls=self.calls(state),
            focus=self.ctx.team.focus(state),
        )


def _typed(x, cls, engineer: str):
    if not isinstance(x, cls):
        raise TypeError(f"{engineer} returned {type(x).__name__}, expected {cls.__name__}")
    return x
