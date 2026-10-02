"""Text views of a race state (used by `pitsense replay`)."""

from __future__ import annotations

from .bench.features import FeatureBuilder
from .config import TRACK_STATUS
from .pitloss import cars_within, shrink
from .state import RaceState

_SHORT = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}


def _fmt_time(x: float | None) -> str:
    if x is None:
        return "--"
    m, s = divmod(x, 60)
    return f"{int(m)}:{s:06.3f}" if m else f"{s:.3f}"


def _fmt_gap(x: float | None, laps_down: int = 0) -> str:
    if laps_down:
        return f"+{laps_down}L"
    return "--" if x is None else f"+{x:.3f}"


def timing_tower(state: RaceState, fb: FeatureBuilder) -> str:
    """Timing tower plus a 'pit now' projection: where each car would rejoin if it boxed now.

    The projection is the gap-minus-pit-loss baseline using only what is known at state.t.
    """
    order = [d for d in state.running_order() if d.running]
    losses = fb._in_race_losses(state)
    status = state.track_status
    prior = fb.prior
    loss = (
        shrink(prior.sc, losses["sc"]) if status == "4"
        else shrink(prior.vsc, losses["vsc"]) if status in ("6", "7")
        else shrink(prior.green, losses["green"])
    )
    head = (
        f"Lap {state.current_lap}/{state.total_laps or '?'}   "
        f"track: {TRACK_STATUS.get(status, status)}   pit loss now ≈ {loss:.1f} s "
        f"(prior {prior.source}, {sum(len(v) for v in losses.values())} stops measured this race)"
    )
    lines = [head, "", " P  CAR  GAP        INT       TYRE   STOPS  LAST       PIT NOW →"]
    for i, d in enumerate(order):
        within, _ = cars_within(order, i, loss)
        tyre = f"{_SHORT.get(d.compound or '', '?')}{d.tyre_age if d.tyre_age is not None else '?':>3}"
        pit_now = "in pit" if d.in_pit else f"P{(d.position or 0) + within}"
        lines.append(
            f"{d.position or '-':>2}  {d.tla or d.number:<4} {_fmt_gap(d.gap_to_leader, d.laps_down):<10} "
            f"{_fmt_gap(d.interval):<9} {tyre:<6} {d.pit_stops:^5}  {_fmt_time(d.last_lap_time):<10} {pit_now}"
        )
    out = [d for d in state.drivers.values() if not d.running]
    if out:
        lines.append("")
        lines.append("Out: " + ", ".join(d.tla or d.number for d in out))
    recent_rc = [
        m for m in state.rc
        if "BLUE FLAG" not in m.message and (m.category != "Other" or "PENALTY" in m.message)
    ][-3:]
    if recent_rc:
        lines.append("")
        lines += [f"RC: {m.message}" for m in recent_rc]
    return "\n".join(lines)
