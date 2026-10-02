"""Indexes built incrementally from the race state, shared by every engineer.

They are filled from what the state already holds, one event at a time, so
they are as-of by construction.
"""

from __future__ import annotations

from ..pitloss import LapIndex, measure_stops
from ..state import LapRecord, RaceState


def is_clean(lap: LapRecord | None) -> bool:
    """A lap that says something about pace: timed, not lap 1, no pit stop, green throughout."""
    return (
        lap is not None
        and lap.lap_time is not None
        and lap.lap > 1
        and not lap.is_in_lap
        and not lap.is_out_lap
        and lap.track_status == "1"
    )


class RaceMemory:
    """Laps by driver and lap number, the order at each pit entry, pit losses so far."""

    def __init__(self) -> None:
        self.index = LapIndex([])
        self.order_at_pit: dict[int, dict[str, int]] = {}  # pit event idx -> positions at entry
        self._n_laps_seen = 0
        self._n_pits_seen = 0
        self._loss_cache_key: tuple[int, int] | None = None
        self._loss_cache: dict[str, list[float]] = {"green": [], "sc": [], "vsc": []}

    def observe(self, state: RaceState) -> None:
        """Call after every applied event: indexes new laps and freezes order at pit entry."""
        if len(state.laps) > self._n_laps_seen:
            for lap in state.laps[self._n_laps_seen:]:
                self.index.add(lap)
            self._n_laps_seen = len(state.laps)
        if len(state.pit_events) > self._n_pits_seen:
            positions = {d.number: d.position for d in state.drivers.values() if d.position is not None}
            for i in range(self._n_pits_seen, len(state.pit_events)):
                pe = state.pit_events[i]
                pre = dict(positions)
                # the stopping car may already have been demoted; use its last lap's position
                last = self.index.by_driver.get(pe.driver, {}).get(pe.in_lap - 1)
                if last is not None and last.position is not None:
                    pre[pe.driver] = last.position
                self.order_at_pit[i] = pre
            self._n_pits_seen = len(state.pit_events)

    def in_race_losses(self, state: RaceState) -> dict[str, list[float]]:
        """Pit losses measured in this race so far, by condition. Treat as read-only."""
        key = (self._n_laps_seen, sum(1 for p in state.pit_events if p.out_lap is not None))
        if key != self._loss_cache_key:
            samples = measure_stops(self.index, state.pit_events)
            out: dict[str, list[float]] = {"green": [], "sc": [], "vsc": []}
            for s in samples:
                if s.available_at <= state.t:
                    out[s.condition].append(s.loss)
            self._loss_cache, self._loss_cache_key = out, key
        return self._loss_cache
