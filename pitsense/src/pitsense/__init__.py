"""PitSense: leakage-safe F1 race strategy.

Layers (bottom-up):
    archive  -> download raw live-timing streams (same messages the live feed sends)
    events   -> one time-ordered event log per session (replay == live)
    state    -> deterministic reducer: events -> RaceState, lap records, pit stops
    asof     -> stores that refuse to return data that wasn't available yet
    bench    -> frozen car-lap snapshots, labels, baselines, scoring
"""

__version__ = "0.1.0"
