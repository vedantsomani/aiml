"""Build the benchmark: one parquet file of frozen decision points per race."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..archive import SessionRef
from ..asof import HistoryStore
from ..config import bench_dir
from ..events import EventLog, load_archive_session
from ..pitloss import PitLossPrior
from ..pitwall.engineer import Context
from ..state import RaceState
from .features import FEATURE_VERSION, FeatureBuilder
from .history import prior_for
from .labels import add_labels


def decision_rows(
    log: EventLog, prior: PitLossPrior, *, with_hash: bool = True, builder=FeatureBuilder,
    ctx: Context | None = None,
) -> tuple[list[dict], RaceState]:
    """Replay ``log`` once and emit a feature row each time a car completes a lap.

    Rows are produced only after every event sharing that timestamp has been
    applied, so a row at time t reflects exactly ``log.until(t)``. ``ctx`` is
    what the engineers may know before the race (default: just the prior).
    """
    state = RaceState(log.meta)
    fb = builder(prior, log.meta, ctx=ctx)
    rows: list[dict] = []
    queued_laps = []
    queued_pits = []
    n_pits = 0
    events = log.events
    for k, e in enumerate(events):
        state.apply(e)
        fb.observe(state)
        queued_laps.extend(state.new_laps)
        if len(state.pit_events) > n_pits:
            queued_pits.extend(range(n_pits, len(state.pit_events)))
            n_pits = len(state.pit_events)
        last_at_t = k + 1 == len(events) or events[k + 1].t > e.t
        if (queued_laps or queued_pits) and last_at_t:
            fingerprint = state.fingerprint() if with_hash else ""
            for rec in queued_laps:
                row = fb.row(state, rec)
                if row is not None:
                    rows.append(_tag(row, state, rec.driver, fingerprint))
            for i in queued_pits:
                pe = state.pit_events[i]
                if pe.under_red:
                    continue
                prev = fb.index.by_driver.get(pe.driver, {}).get(pe.in_lap - 1)
                row = fb.row(state, prev, driver=pe.driver, kind="pit_entry", in_lap=pe.in_lap)
                if row is not None:
                    row["pit_event"] = i
                    rows.append(_tag(row, state, pe.driver, fingerprint))
            queued_laps, queued_pits = [], []
    return rows, state


def _tag(row: dict, state: RaceState, driver: str, fingerprint: str) -> dict:
    d = state.drivers[driver]
    row.update(driver=driver, tla=d.tla, team=d.team, state_hash=fingerprint)
    return row


def build_race(ref: SessionRef, history: HistoryStore, out_dir: Path | None = None) -> Path:
    log = load_archive_session(ref.local_dir)
    prior = prior_for(history, ref.circuit_key, ref.start_utc)
    ctx = Context.for_race(prior, log.meta, history=history, race_start_utc=ref.start_utc)
    rows, final = decision_rows(log, prior, ctx=ctx)
    rows = add_labels(rows, final)
    df = pd.DataFrame(rows)
    df.insert(0, "race_id", ref.slug)
    df.insert(1, "year", ref.year)
    df.insert(2, "round", ref.round_index)
    df.insert(3, "circuit_key", ref.circuit_key)
    df.insert(4, "start_utc", ref.start_utc.isoformat())
    out_dir = out_dir or bench_dir() / FEATURE_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ref.slug}.parquet"
    df.to_parquet(path, index=False)
    (out_dir / f"{ref.slug}.prior.json").write_text(json.dumps(prior.__dict__), encoding="utf-8")
    return path


MANIFEST = "manifest.json"


def write_manifest(slugs: list[str], out_dir: Path | None = None, **info) -> Path:
    """Record which races make up the benchmark, so a partial rebuild never mixes in stale files."""
    out_dir = out_dir or bench_dir() / FEATURE_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST
    path.write_text(json.dumps({"races": sorted(slugs), **info}, indent=1, default=str), encoding="utf-8")
    return path


def load_bench(version: str = FEATURE_VERSION) -> pd.DataFrame:
    folder = bench_dir() / version
    manifest = folder / MANIFEST
    if manifest.exists():
        races = json.loads(manifest.read_text(encoding="utf-8"))["races"]
        files = [folder / f"{slug}.parquet" for slug in races]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"{manifest} lists races with no file: {missing}. Run `pitsense bench build`.")
    else:  # built before manifests existed
        files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No benchmark files in {folder}. Run `pitsense bench build`.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    return df.sort_values(["start_utc", "t", "driver"]).reset_index(drop=True)
