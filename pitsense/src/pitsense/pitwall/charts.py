"""Chart data for the dashboard: gaps around our cars over the last laps, and every car's stints.

Built from what the race state already holds (as of now) and kept small: gaps are capped at
``GAP_LAPS`` laps and ``NEIGHBOURS`` cars each side for the focus cars only, stints are one
short row per stint. Sent in every snapshot's ``extra["charts"]`` (about 3 KB at most).
"""

from __future__ import annotations

from ..state import RaceState
from .memory import RaceMemory

GAP_LAPS = 15
NEIGHBOURS = 3
SHORT = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}


def _gap(a, b) -> float | None:
    """Seconds between two cars' lap records on the same lap (None if lapped or unknown)."""
    if a is None or b is None or a.gap_to_leader is None or b.gap_to_leader is None or a.laps_down or b.laps_down:
        return None
    return round(abs(b.gap_to_leader - a.gap_to_leader), 2)


def gaps(state: RaceState, memory: RaceMemory, car: str) -> dict | None:
    """Per recorded lap: our position and the gaps to the cars 1..3 places ahead / behind on that lap."""
    idx = memory.index
    mine = idx.by_driver.get(car)
    if not mine:
        return None
    last = max(mine)
    laps, pos, ahead, behind, tla_a, tla_b = [], [], [], [], [], []
    for lap in range(max(1, last - GAP_LAPS + 1), last + 1):
        me = mine.get(lap)
        if me is None or me.position is None:
            continue
        by_pos = {r.position: r for r in idx.by_lap.get(lap, ()) if r.position is not None}
        a, b, ta, tb = [], [], [], []
        for k in range(1, NEIGHBOURS + 1):
            ra, rb = by_pos.get(me.position - k), by_pos.get(me.position + k)
            a.append(_gap(me, ra))
            b.append(_gap(me, rb))
            ta.append(state.drivers[ra.driver].tla if ra and ra.driver in state.drivers else None)
            tb.append(state.drivers[rb.driver].tla if rb and rb.driver in state.drivers else None)
        laps.append(lap)
        pos.append(me.position)
        ahead.append(a)
        behind.append(b)
        tla_a.append(ta)
        tla_b.append(tb)
    return {"laps": laps, "pos": pos, "ahead": ahead, "behind": behind, "tla_ahead": tla_a, "tla_behind": tla_b}


def stints(state: RaceState, memory: RaceMemory) -> dict:
    """car -> [[compound letter, first lap, last lap recorded], ...] in race order."""
    out: dict[str, list] = {}
    for n, laps in memory.index.by_driver.items():
        rows: list[list] = []
        key = None
        for lap in sorted(laps):
            r = laps[lap]
            k = (r.stint, r.compound)
            if k != key:
                rows.append([SHORT.get(r.compound or "", "?"), lap, lap])
                key = k
            else:
                rows[-1][2] = lap
        out[n] = rows
    return out


def build(state: RaceState, memory: RaceMemory, focus) -> dict:
    return {
        "stints": stints(state, memory),
        "gaps": {c: g for c in focus if (g := gaps(state, memory, c)) is not None},
        "order": [d.number for d in state.running_order()],
    }
