"""Rival analyst: who is around each car, and what they are about to do.

STUB (owner: rival analyst). Contract keys (docs/ENGINEERING.md) are all
present; the stub fills the neighbours and gaps. To build: each car's chance of
pitting in the next laps from in-race and pre-race knowledge, undercut /
overcut threats and chances, team habits learned from past races.
"""

from __future__ import annotations

from ..engineer import Engineer


class RivalsEngineer(Engineer):
    name = "rivals"

    def car(self, state, number, view):
        order = [x for x in state.running_order() if x.running]
        i = next((k for k, x in enumerate(order) if x.number == number), None)
        if i is None:
            return {}
        ahead = order[i - 1] if i > 0 else None
        behind = order[i + 1] if i + 1 < len(order) else None
        return {
            "ahead": ahead.number if ahead else None,
            "behind": behind.number if behind else None,
            "gap_ahead": order[i].interval if ahead else None,
            "gap_behind": behind.interval if behind else None,
            "pit_prob_1": None,
            "pit_prob_3": None,
            "undercut_threat": None,
            "undercut_chance": None,
        }
