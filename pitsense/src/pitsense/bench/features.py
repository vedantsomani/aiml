"""Features for one decision point: driver ``d`` has just completed lap ``L`` at time ``t``.

Everything here reads only the live :class:`RaceState` at ``t`` plus the
pre-race prior. Nothing may import from ``labels`` or look at the event log.

Each row holds the v0.1 base features below plus every bench engineer's values
("<engineer>__<key>", see ``pitwall/engineer.py``).
"""

from __future__ import annotations

import math
from statistics import mean, median

import numpy as np

from ..config import DRY_COMPOUNDS
from ..pitloss import LapIndex, PitLossPrior, cars_within, shrink
from ..pitwall.engineer import Context, Engineer
from ..pitwall.memory import RaceMemory, is_clean
from ..pitwall.wall import PitWall
from ..state import LapRecord, RaceState

FEATURE_VERSION = "v0.1"

COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")


def _nan(x: float | None) -> float:
    return float("nan") if x is None else float(x)


def bench_engineers() -> list[type[Engineer]]:
    """Registered engineers that are cheap enough to run for every decision row."""
    from .. import registry

    return [c for c in registry.engineer_classes() if c.in_bench]


class FeatureBuilder:
    """Stateful helper that follows one replay and emits a row per decision point."""

    def __init__(
        self,
        prior: PitLossPrior,
        race_meta: dict | None = None,
        *,
        ctx: Context | None = None,
        engineers: list[type[Engineer]] | None = None,
    ) -> None:
        self.prior = prior
        self.meta = dict(race_meta or {})
        self.memory = RaceMemory()
        ctx = ctx or Context(prior=prior, meta=self.meta)
        self.wall = PitWall(ctx, self.memory, bench_engineers() if engineers is None else engineers)

    @property
    def index(self) -> LapIndex:
        return self.memory.index

    # -------------------------------------------------------------- bookkeeping
    def observe(self, state: RaceState) -> None:
        """Call after every applied event: updates the shared race memory and every engineer."""
        self.wall.observe(state)

    def _in_race_losses(self, state: RaceState) -> dict[str, list[float]]:
        return self.memory.in_race_losses(state)

    # -------------------------------------------------------------- features
    def row(self, state: RaceState, rec: LapRecord | None, *, driver: str | None = None,
            kind: str = "lap_end", in_lap: int | None = None) -> dict | None:
        """One decision row.

        kind="lap_end":   ``rec`` was just completed; question: stop in the next k laps?
        kind="pit_entry": the car just entered the pit lane on ``in_lap``; question: where
                          does it rejoin? (``rec`` is its previous lap, may be None)
        """
        number = driver or (rec.driver if rec else None)
        d = state.drivers.get(number) if number else None
        if d is None or not state.started or not d.running:
            return None
        if state.track_status == "5" or state.finished_t is not None:
            return None
        if kind == "lap_end" and d.in_pit:
            return None
        total = state.total_laps or 0
        L = rec.lap if kind == "lap_end" else int(in_lap or d.laps + 1)
        if kind == "lap_end" and total and L >= total:
            return None
        t = state.t

        order = [x for x in state.running_order() if x.running]
        pos_idx = {x.number: i for i, x in enumerate(order)}
        i = pos_idx.get(d.number)
        ahead = order[i - 1] if i is not None and i > 0 else None
        behind = order[i + 1] if i is not None and i + 1 < len(order) else None

        # pit loss now: prior blended with this race's measured stops so far
        losses = self._in_race_losses(state)
        loss_green = shrink(self.prior.green, losses["green"])
        loss_sc = shrink(self.prior.sc, losses["sc"])
        loss_vsc = shrink(self.prior.vsc, losses["vsc"])
        status = state.track_status
        loss_now = loss_sc if status == "4" else loss_vsc if status in ("6", "7") else loss_green

        # cars that would pass us if we stopped now
        within, margin = cars_within(order, i, loss_now) if i is not None else (0, float("nan"))

        # pace in this stint (as-of)
        mine = self.index.by_driver.get(d.number, {})
        stint_laps = [mine[k] for k in sorted(mine) if k > d.stint_start_lap and k <= L]
        clean = [x for x in stint_laps if is_clean(x)]
        times = [x.lap_time for x in clean]
        pace_drop = (mean(times[-3:]) - mean(times[:3])) if len(times) >= 6 else float("nan")
        slope = float(np.polyfit([x.lap for x in clean], times, 1)[0]) if len(times) >= 4 else float("nan")
        rel = []
        for x in reversed(clean[-3:]):
            field = [y.lap_time for y in self.index.by_lap.get(x.lap, []) if is_clean(y)]
            if len(field) >= 5:
                rel.append(x.lap_time - median(field))
        pace_vs_field = mean(rel) if rel else float("nan")

        # recent stops around us (cover / undercut signals)
        ref_lap = median(times) if times else (d.last_lap_time or 90.0)
        window = 2.0 * ref_lap
        recent = [(k, pe) for k, pe in enumerate(state.pit_events)
                  if t - window <= pe.in_t <= t and not pe.under_red and pe.driver != d.number]
        my_pos = d.position or 0

        def rival_pitted(offset: int) -> int:
            for k, pe in recent:
                pre = self.memory.order_at_pit.get(k, {})
                if pe.driver != d.number and not pe.under_red and pre.get(pe.driver) == pre.get(d.number, my_pos) + offset:
                    return 1
            return 0

        dry_used = {c for x in state.drivers.values() for c in x.compounds_used}
        race_dry = bool(dry_used) and dry_used <= set(DRY_COMPOUNDS)
        my_dry = set(d.compounds_used) & set(DRY_COMPOUNDS)

        w = state.weather
        row = {
            "kind": kind,
            "t": round(t, 3),
            "lap": L,
            "total_laps": total or float("nan"),
            "laps_remaining": (total - L) if total else float("nan"),
            "race_frac": (L / total) if total else float("nan"),
            "track_status": status,
            "sc": int(status == "4"),
            "vsc": int(status in ("6", "7")),
            "yellow": int(status == "2"),
            "status_age_s": round(t - state.track_status_since, 3),
            "track_temp": _nan(w.get("TrackTemp")),
            "air_temp": _nan(w.get("AirTemp")),
            "rainfall": _nan(w.get("Rainfall")),
            "n_running": len(order),
            "position": _nan(d.position),
            "gap_to_leader": _nan(d.gap_to_leader),
            "laps_down": d.laps_down,
            "interval_ahead": _nan(d.interval),
            "gap_behind": _nan(behind.interval if behind else None),
            "compound": d.compound or "UNKNOWN",
            "tyre_age": _nan(d.tyre_age),
            "tyre_new": float("nan") if d.tyre_new is None else int(d.tyre_new),
            "laps_in_stint": L - d.stint_start_lap,
            "stint": d.stint,
            "pit_stops": d.pit_stops,
            "must_stop": int(race_dry and len(my_dry) < 2),
            "race_dry": int(race_dry),
            "last_lap": _nan(rec.lap_time if rec else None),
            "stint_best": min(times) if times else float("nan"),
            "pace_drop": pace_drop,
            "pace_slope": slope,
            "pace_vs_field": pace_vs_field,
            "n_clean_stint": len(times),
            "pit_loss_green": round(loss_green, 3),
            "pit_loss_now": round(loss_now, 3),
            "pit_loss_source": self.prior.source,
            "cars_within_pitloss": within,
            "pitloss_margin": margin,
            "n_pitted_recent": len(recent),
            "ahead_pitted_recent": rival_pitted(-1),
            "behind_pitted_recent": rival_pitted(+1),
            "ahead_tyre_age": _nan(ahead.tyre_age if ahead else None),
            "ahead_compound": (ahead.compound if ahead else None) or "NONE",
            "behind_tyre_age": _nan(behind.tyre_age if behind else None),
        }
        for c in COMPOUNDS:
            row[f"cmp_{c.lower()}"] = int(d.compound == c)
        row.update(self.wall.row_values(state, d.number))
        return row


NUMERIC_FEATURES = [
    "lap", "total_laps", "laps_remaining", "race_frac", "sc", "vsc", "yellow", "status_age_s",
    "track_temp", "air_temp", "rainfall", "n_running", "position", "gap_to_leader", "laps_down",
    "interval_ahead", "gap_behind", "tyre_age", "tyre_new", "laps_in_stint", "stint", "pit_stops",
    "must_stop", "race_dry", "last_lap", "pace_drop", "pace_slope", "pace_vs_field", "n_clean_stint",
    "pit_loss_now", "cars_within_pitloss", "pitloss_margin", "n_pitted_recent",
    "ahead_pitted_recent", "behind_pitted_recent", "ahead_tyre_age", "behind_tyre_age",
    "cmp_soft", "cmp_medium", "cmp_hard", "cmp_intermediate", "cmp_wet",
]


def feature_columns() -> list[str]:
    """Model inputs: the base features plus what each bench engineer declares in ``features``."""
    return NUMERIC_FEATURES + [f"{c.name}__{k}" for c in bench_engineers() for k in c.features]


def is_finite(x: float) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))
