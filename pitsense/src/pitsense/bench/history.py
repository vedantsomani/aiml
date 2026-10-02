"""Cross-race priors (pit loss by circuit and track status) with a UTC cutoff."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

from .. import registry
from ..archive import SessionRef
from ..asof import HistoryStore, RaceSummary
from ..events import load_archive_session
from ..pitloss import DEFAULTS, LapIndex, PitLossPrior, measure_stops
from ..state import RaceState, replay


def summarize(ref: SessionRef, final: RaceState | None = None) -> RaceSummary:
    if final is None:
        final = replay(load_archive_session(ref.local_dir))
    samples = measure_stops(LapIndex(final.laps), final.pit_events)
    by = {"green": [], "sc": [], "vsc": []}
    for s in samples:
        by[s.condition].append(s.loss)
    leader_laps = [lap for lap in final.laps if lap.position == 1]
    sc_laps = sum("4" in lap.track_status.split(",") for lap in leader_laps)
    vsc_laps = sum(bool({"6", "7"} & set(lap.track_status.split(","))) for lap in leader_laps)
    start = ref.start_utc
    # the end is when the chequered flag was published (fallback: start + 3h)
    end = start + timedelta(hours=3)
    if final.finished_t is not None and final.started_t is not None:
        end = start + timedelta(seconds=final.finished_t - final.started_t)
    extra = {}
    for cls in registry.engineer_classes():  # what each role learns for later races
        learned = cls.summarize_race(final, ref.to_dict())
        if learned:
            extra[cls.name] = learned
    return RaceSummary(
        race_id=ref.slug,
        year=ref.year,
        circuit_key=ref.circuit_key,
        start_utc=start,
        end_utc=end,
        pit_loss_green=by["green"],
        pit_loss_sc=by["sc"],
        pit_loss_vsc=by["vsc"],
        laps=final.total_laps or 0,
        sc_laps=sc_laps,
        vsc_laps=vsc_laps,
        extra=extra,
    )


def _to_json(s: RaceSummary) -> dict:
    d = dict(s.__dict__)
    d["start_utc"] = s.start_utc.isoformat()
    d["end_utc"] = s.end_utc.isoformat()
    return d


def _from_json(d: dict) -> RaceSummary:
    d = dict(d)
    d["start_utc"] = datetime.fromisoformat(d["start_utc"])
    d["end_utc"] = datetime.fromisoformat(d["end_utc"])
    return RaceSummary(**d)


def save(summaries: list[RaceSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_to_json(s) for s in summaries], indent=1), encoding="utf-8")


def load(path: Path) -> HistoryStore:
    if not path.exists():
        return HistoryStore()
    return HistoryStore(_from_json(d) for d in json.loads(path.read_text(encoding="utf-8")))


def prior_for(history: HistoryStore, circuit_key: int, race_start_utc: datetime) -> PitLossPrior:
    """Pit-loss prior using only races that ended before this one started."""
    past = history.asof(race_start_utc)
    same = [s for s in past if s.circuit_key == circuit_key]

    def pick(attr: str, min_n: int) -> tuple[float, str]:
        for s in reversed(same):  # most recent visit to this circuit first
            values = getattr(s, attr)
            if len(values) >= min_n:
                return float(median(values)), f"circuit:{s.race_id}"
        pooled = [v for s in past for v in getattr(s, attr)]
        if len(pooled) >= 10:
            return float(median(pooled)), "global"
        return DEFAULTS[attr.removeprefix("pit_loss_")], "default"

    green, src = pick("pit_loss_green", 3)
    sc, _ = pick("pit_loss_sc", 3)
    vsc, _ = pick("pit_loss_vsc", 3)
    return PitLossPrior(green=green, sc=sc, vsc=vsc, source=src)
