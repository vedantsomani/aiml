"""Head of strategy: one call per car, with the reasons, from what the engineers report.

STUB (owner: strategy engineer). Until the strategy engineer can rank plans it
makes no call; it only states the facts a call would rest on.
"""

from __future__ import annotations

from ..engineer import Engineer
from ..types import Call, Reason


class HeadOfStrategy(Engineer):
    name = "head"
    requires = ("pitstop", "rules", "strategy")
    in_bench = False

    def calls(self, state, view):
        cars = self.ctx.team.focus(state) or [d.number for d in state.running_order() if d.running]
        out = []
        for n in cars:
            d = state.drivers.get(n)
            if d is None or not d.running:
                continue
            reasons = []
            rejoin = view.car("pitstop", n).get("rejoin_if_box_now")
            if rejoin is not None:
                reasons.append(Reason("rejoin_if_box_now", f"boxing now rejoins P{rejoin}", rejoin))
            if view.car("rules", n).get("must_stop"):
                reasons.append(Reason("must_stop", "still has to use a second dry compound", True))
            out.append(Call(t=round(state.t, 3), car=n, action="NO_CALL", reasons=tuple(reasons)))
        return out
