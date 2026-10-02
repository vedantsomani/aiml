"""Cross-check our reconstruction against FastF1's post-processed laps.

FastF1 builds its tables after the session with hindsight; we build ours one
message at a time. Where both exist they should agree. Differences we accept:
  * lap 1 time: FastF1 derives it after the fact; the feed never publishes it.
  * final-lap positions can differ by one when cars finish within the same update.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .archive import SessionRef
from .events import load_archive_session
from .state import replay


@dataclass
class ValidationReport:
    session: str
    n_ours: int
    n_fastf1: int
    n_both: int
    lap_time_exact: float  # share of laps (both timed) within 1 ms
    lap_time_missing: int  # laps >1 FastF1 times but we don't
    position_match: float
    compound_match: float
    tyre_age_match: float
    in_lap_match: float
    out_lap_match: float

    def ok(self) -> bool:
        return (
            self.lap_time_exact >= 0.999
            and self.position_match >= 0.99
            and self.in_lap_match >= 0.995
            and self.out_lap_match >= 0.99
            and self.tyre_age_match >= 0.97
        )

    def line(self) -> str:
        return (
            f"{self.session:<44} laps {self.n_both:>5}/{self.n_fastf1:<5} "
            f"time {self.lap_time_exact:6.1%}  pos {self.position_match:6.1%}  "
            f"cmp {self.compound_match:6.1%}  age {self.tyre_age_match:6.1%}  "
            f"in {self.in_lap_match:6.1%}  out {self.out_lap_match:6.1%}  "
            f"{'OK' if self.ok() else 'CHECK'}"
        )


def ours_laps(ref: SessionRef) -> pd.DataFrame:
    state = replay(load_archive_session(ref.local_dir))
    return pd.DataFrame([vars(lap) for lap in state.laps])


def fastf1_laps(ref: SessionRef, cache_dir: str | None = None) -> pd.DataFrame:
    import fastf1  # optional dependency

    if cache_dir:
        fastf1.Cache.enable_cache(cache_dir)
    fastf1.set_log_level("ERROR")
    # resolve by name: round numbers shift when races are cancelled or moved
    ses = fastf1.get_session(ref.year, ref.meeting_name, ref.session_name)
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    lap = ses.laps
    return pd.DataFrame(
        {
            "driver": lap["DriverNumber"].astype(str),
            "lap": lap["LapNumber"].astype(int),
            "ff_time": lap["LapTime"].dt.total_seconds(),
            "ff_pos": lap["Position"],
            "ff_comp": lap["Compound"],
            "ff_age": lap["TyreLife"],
            "ff_pin": lap["PitInTime"].notna(),
            "ff_pout": lap["PitOutTime"].notna(),
        }
    )


def compare(ref: SessionRef, cache_dir: str | None = None) -> tuple[ValidationReport, pd.DataFrame]:
    ours = ours_laps(ref)
    ff = fastf1_laps(ref, cache_dir)
    both = ours.merge(ff, on=["driver", "lap"], how="inner")

    def share(mask: pd.Series) -> float:
        return float(mask.mean()) if len(mask) else float("nan")

    timed = both.dropna(subset=["lap_time", "ff_time"])
    pos = both.dropna(subset=["position", "ff_pos"])
    cmp_ = both.dropna(subset=["compound", "ff_comp"])
    age = both.dropna(subset=["tyre_age", "ff_age"])
    report = ValidationReport(
        session=ref.slug,
        n_ours=len(ours),
        n_fastf1=len(ff),
        n_both=len(both),
        lap_time_exact=share((timed.lap_time - timed.ff_time).abs() <= 0.0011),
        lap_time_missing=int(((both.lap > 1) & both.lap_time.isna() & both.ff_time.notna()).sum()),
        position_match=share(pos.position == pos.ff_pos),
        compound_match=share(cmp_.compound == cmp_.ff_comp),
        tyre_age_match=share(age.tyre_age == age.ff_age),
        in_lap_match=share(both.is_in_lap == both.ff_pin),
        out_lap_match=share(both.is_out_lap == both.ff_pout),
    )
    return report, both
