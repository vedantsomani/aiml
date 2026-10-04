"""Tyre-set planner: which new and used dry sets each car still has for the race.

Sets run in the weekend's earlier sessions (FP1-3, sprint qualifying, sprint, qualifying) are read
from those sessions' own streams, all published before this session starts. During the race the
sets fitted so far (TimingAppData stints) are taken off. Allocation assumption: see weekend.py.

Per car: new_soft_left, new_medium_left, new_hard_left (None when no earlier session is known),
used_sets ("S12 M20": used sets still free, compound letter + laps), sets_run (sets run so far).
"""

from __future__ import annotations

import copy

from ... import weekend
from ...archive import SessionRef
from ..engineer import Engineer


class TyreSetsEngineer(Engineer):
    name = "tyresets"
    in_bench = False  # reads earlier sessions from disk: scored by its own tests, not per row

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self.base: dict[str, list] | None = None
        self.alloc = None
        self._load()

    def _load(self) -> None:
        meta = self.ctx.meta
        pre = meta.get("weekend_sets")  # injected (tests, other tools): {car: [[compound, laps, first], ...]}
        sprint = bool(meta.get("sprint_weekend"))
        try:
            if pre is not None:
                self.base = {c: [weekend.TyreSet(*x[:3]) for x in v] for c, v in pre.items()}
            elif "meeting_key" in meta and "path" in meta:
                ref = SessionRef.from_dict({k: meta[k] for k in SessionRef.__dataclass_fields__})
                if weekend.have_sessions(ref):
                    self.base = weekend.weekend_sets(ref)
                sprint = weekend.is_sprint_weekend(ref)
        except Exception:  # missing index, odd files: unknown, never guessed
            self.base = None
        self.alloc = weekend.allocation(sprint, meta.get("tyre_returned"), meta.get("tyre_allocation"))

    def _sets(self, state, number):
        sets = copy.deepcopy(self.base.get(number, []))
        lines = (state.topics.get("TimingAppData") or {}).get("Lines") or {}
        stints = (lines.get(number) or {}).get("Stints") or []
        if stints and state.started:
            before = {id(t): t.laps for t in sets}
            weekend.add_stints(sets, stints, "Race", state.meta)
            for t in sets:
                if t.first == "Race" or before.get(id(t), t.laps) != t.laps:
                    t.mounted = True
        return sets

    def car(self, state, number, view):
        if self.base is None:
            return {"new_soft_left": None, "new_medium_left": None, "new_hard_left": None,
                    "used_sets": None, "sets_run": None}
        sets = self._sets(state, number)
        left = weekend.remaining(sets, self.alloc)
        return {"new_soft_left": left["SOFT"], "new_medium_left": left["MEDIUM"], "new_hard_left": left["HARD"],
                "used_sets": weekend.used_str(sets), "sets_run": sum(1 for t in sets if t.first)}
