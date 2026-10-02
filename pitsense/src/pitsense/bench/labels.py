"""Labels: what actually happened after each decision point.

This module reads the *complete* race (the future). It is only used to score
predictions and must never be imported by feature code.

Pit-lane entries under a red flag are not strategy stops (everyone goes in and
tyres are changed for free), so they are ignored here.
"""

from __future__ import annotations

from ..state import RaceState

HORIZONS = (1, 2, 3, 5)


def add_labels(rows: list[dict], final: RaceState) -> list[dict]:
    """Attach y_* columns from every registered labeler (``registry.LABELERS``)."""
    from .. import registry

    for label in registry.labelers():
        label(rows, final)
    return rows


def pit_labels(rows: list[dict], final: RaceState) -> None:
    """Pit stops and retirements after each lap end; where a car rejoined after each pit entry."""
    stops: dict[str, list[int]] = {}
    for pe in final.pit_events:
        if not pe.under_red:
            stops.setdefault(pe.driver, []).append(pe.in_lap)
    lap_pos = {(lap.driver, lap.lap): lap.position for lap in final.laps}
    last_lap: dict[str, int] = {}
    for lap in final.laps:
        last_lap[lap.driver] = max(last_lap.get(lap.driver, 0), lap.lap)
    # Out at the flag = retired. Lapped finishers complete fewer than the race
    # distance but are not retirements (this agrees with "took the chequered
    # flag" on every 2025-26 race).
    retired = {n for n, d in final.drivers.items() if not d.running}

    for row in rows:
        drv, L = row["driver"], row["lap"]
        done = last_lap.get(drv, 0)
        row["y_retire_3"] = int(drv in retired and done < L + 3)
        if row.get("kind", "lap_end") == "pit_entry":
            pe = final.pit_events[row["pit_event"]]
            pos = lap_pos.get((drv, pe.out_lap)) if pe.out_lap is not None else None
            row["y_pos_after_stop"] = float(pos) if pos is not None else float("nan")
            row["y_out_lap"] = float(pe.out_lap) if pe.out_lap is not None else float("nan")
            continue
        future = sorted(x for x in stops.get(drv, []) if x > L)
        nxt = future[0] if future else None
        row["y_next_in_lap"] = float(nxt) if nxt is not None else float("nan")
        for k in HORIZONS:
            row[f"y_pit_{k}"] = int(nxt is not None and nxt <= L + k)
