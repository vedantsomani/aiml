"""Tyre-set planner: which new and used dry sets each car still has for the race.

Sets run in the weekend's earlier sessions (FP1-3, sprint qualifying, sprint, qualifying) are read
from those sessions' own streams, all published before this session starts. During the race the
sets fitted so far (TimingAppData stints) are taken off. Allocation assumption: see weekend.py.

Missing sessions: the book is built from the sessions on disk (a broken one is skipped). With none,
only the standard allocation is known: ``sets_basis`` is "allocation" and new_*_left stay None.

Per car: new_soft_left, new_medium_left, new_hard_left (None when no earlier session is known),
used_sets ("S12 M20": used sets still free, compound letter + laps), sets_run (sets run so far),
avail_<compound> (new + free used sets still fittable), used_<compound>_laps (laps on the freshest
free used set, None if none), sets_basis ("weekend" | "partial" | "allocation").

Used vs new pace: ``weekend.used_offset`` (seconds slower than a new set) is learned from a past season's
stints, see ``weekend.fit_used_offset``; the strategy engineer adds it to a compound's fresh pace when the
car has no new set of it left.
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
        self.basis = "allocation"
        self.alloc = None
        self._load()

    def _load(self) -> None:
        meta = self.ctx.meta
        pre = meta.get("weekend_sets")  # injected (tests, other tools): {car: [[compound, laps, first], ...]}
        sprint = bool(meta.get("sprint_weekend"))
        self.basis, self.base = "allocation", None
        try:
            if pre is not None:
                self.base = {c: [weekend.TyreSet(*x[:3]) for x in v] for c, v in pre.items()}
                self.basis = "weekend"
            elif "meeting_key" in meta and "path" in meta:
                ref = SessionRef.from_dict({k: meta[k] for k in SessionRef.__dataclass_fields__})
                self.basis = weekend.sets_basis(ref)[0]
                if self.basis != "allocation":
                    self.base = weekend.weekend_sets(ref)
                sprint = weekend.is_sprint_weekend(ref)
        except Exception:  # missing index, odd files: the standard allocation, no guess about earlier sessions
            self.base, self.basis = None, "allocation"
        self.alloc = weekend.allocation(sprint, meta.get("tyre_returned"), meta.get("tyre_allocation"))

    def _sets(self, state, number):
        sets = copy.deepcopy((self.base or {}).get(number, []))
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
        sets = self._sets(state, number)
        av = weekend.availability(sets, self.alloc)
        out = {"sets_basis": self.basis}
        for c in weekend.DRY:
            out[f"avail_{c.lower()}"] = av[c]["new"] + av[c]["used"]
            out[f"used_{c.lower()}_laps"] = av[c]["used_laps"]
        if self.base is None:  # no earlier session: new sets left are not known
            out.update(new_soft_left=None, new_medium_left=None, new_hard_left=None, used_sets=None, sets_run=None)
            return out
        left = weekend.remaining(sets, self.alloc)
        out.update(new_soft_left=left["SOFT"], new_medium_left=left["MEDIUM"], new_hard_left=left["HARD"],
                   used_sets=weekend.used_str(sets), sets_run=sum(1 for t in sets if t.first))
        return out
