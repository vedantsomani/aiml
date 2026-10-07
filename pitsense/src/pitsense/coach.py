"""Driving coach: one lap against a reference lap (another driver's, or the driver's own best).

Both laps are resampled onto a common distance grid. The cumulative time delta shows where time goes; corners are
found from the reference lap's speed minima, and for each one the coach reports the brake point, minimum speed and
throttle pick-up differences and the time lost through it. Corners are numbered in the order driven ("Turn 3" is
the third corner found, not necessarily the circuit's official Turn 3).

Telemetry comes through an adapter (``LapSource``): ``F1Adapter`` reads the pit wall's own feeds (CarData aligned
to distance through Position). A sim-racing source (F1 25, iRacing) only needs to return the same ``Lap``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

GRID_M = 5.0  # metres between resampled points
CORNER_GAP_M = 150.0  # two speed minima closer than this are one corner
CORNER_DROP_KMH = 25.0  # a minimum this far below the surrounding maxima counts as a corner
FULL_THROTTLE = 90.0  # % at which the throttle counts as picked up


@dataclass
class Lap:
    driver: str
    lap: int
    dist: np.ndarray  # m from the line
    t: np.ndarray  # s from the line
    speed: np.ndarray  # km/h
    throttle: np.ndarray  # %
    brake: np.ndarray  # 0 / >0
    x: np.ndarray | None = None
    y: np.ndarray | None = None


class LapSource(Protocol):
    def lap(self, driver: str, lap: int) -> Lap | None: ...


class F1Adapter:
    """Laps from a pit wall race state (``insights.lap_samples``): only what the feeds have published."""

    def __init__(self, state) -> None:
        self.state = state

    def lap(self, driver: str, lap: int) -> Lap | None:
        from .insights import lap_samples

        s = lap_samples(self.state, driver, lap)
        if s is None:
            return None
        return Lap(driver, lap, s["dist"], s["t"], s["speed"], s["throttle"], s["brake"], s["x"], s["y"])

    def best_lap(self, driver: str) -> int | None:
        """The driver's quickest clean lap that is still in the feeds' window."""
        from .insights import clean

        laps = sorted((r.lap_time, r.lap) for r in self.state.laps if r.driver == driver and clean(r))
        return next((n for _, n in laps if self.lap(driver, n) is not None), None)


def _grid(lap: Lap, grid: np.ndarray) -> dict[str, np.ndarray]:
    o = np.argsort(lap.dist, kind="stable")
    d = np.maximum.accumulate(lap.dist[o])  # monotone distance for interpolation
    out = {k: np.interp(grid, d, np.nan_to_num(getattr(lap, k)[o].astype(float))) for k in ("t", "speed", "throttle", "brake")}
    if lap.x is not None:
        out["x"], out["y"] = np.interp(grid, d, lap.x[o]), np.interp(grid, d, lap.y[o])
    return out


def corners(speed: np.ndarray) -> list[int]:
    """Indices of corner apexes: speed minima well below the surrounding maxima, at least CORNER_GAP_M apart."""
    k = max(1, int(round(40 / GRID_M)))
    sm = np.convolve(speed, np.ones(k) / k, mode="same")
    w = int(CORNER_GAP_M / GRID_M)
    out = []
    for i in range(w, len(sm) - w):
        seg = sm[i - w:i + w + 1]
        if sm[i] == seg.min() and max(sm[max(0, i - 4 * w):i].max(), sm[i:i + 4 * w].max()) - sm[i] >= CORNER_DROP_KMH:
            if not out or i - out[-1] >= w:
                out.append(i)
    return out


BRAKE_LOOK_M = 400.0  # a braking zone starts at most this far before the apex


def brake_point(brake: np.ndarray, apex: int, lo: int) -> int | None:
    """Start of the braking zone that leads into ``apex``: the nearest braking before it, then back to where it began."""
    first = max(lo, apex - int(BRAKE_LOOK_M / GRID_M))
    i = next((k for k in range(apex, first - 1, -1) if brake[k] > 0), None)
    if i is None:
        return None
    while i - 1 >= first and brake[i - 1] > 0:
        i -= 1
    return i


def _first(mask: np.ndarray, start: int, stop: int, step: int = 1) -> int | None:
    rng = range(start, stop, step)
    return next((i for i in rng if 0 <= i < len(mask) and mask[i]), None)


def compare(ref: Lap, lap: Lap) -> dict:
    """``lap`` against ``ref`` on a common distance grid: cumulative delta (positive: ``lap`` behind) and per-corner
    differences, corners sorted by time lost, plus the top three as sentences."""
    n = float(min(ref.dist.max(), lap.dist.max()))
    grid = np.arange(0.0, n, GRID_M)
    a, b = _grid(ref, grid), _grid(lap, grid)
    delta = b["t"] - a["t"]
    apex = corners(a["speed"])
    out = []
    for c, i in enumerate(apex):
        lo = (apex[c - 1] + i) // 2 if c else 0
        hi = (i + apex[c + 1]) // 2 if c + 1 < len(apex) else len(grid) - 1
        ba, bb = brake_point(a["brake"], i, lo), brake_point(b["brake"], i, lo)
        ta = _first(a["throttle"] >= FULL_THROTTLE, i, hi)
        tb = _first(b["throttle"] >= FULL_THROTTLE, i, hi)
        out.append({
            "turn": c + 1, "at_m": round(float(grid[i])),
            "brake_later_m": None if ba is None or bb is None else round(float(grid[bb] - grid[ba])),
            "min_speed_diff_kmh": round(float(b["speed"][lo:hi + 1].min() - a["speed"][lo:hi + 1].min()), 1),
            "throttle_later_m": None if ta is None or tb is None else round(float(grid[tb] - grid[ta])),
            "time_lost_s": round(float(delta[hi] - delta[lo]), 3),
        })
    worst = sorted(out, key=lambda r: -r["time_lost_s"])
    tips = [_say(r) for r in worst[:3] if r["time_lost_s"] > 0.01]
    seg = None
    if "x" in a:  # the reference line coloured by where time is lost (+) or gained (-) per grid step
        step = np.diff(delta, prepend=delta[0])
        seg = [[round(float(x)), round(float(y)), round(float(s), 4)] for x, y, s in zip(a["x"], a["y"], step)][::2]
    return {"ref": {"driver": ref.driver, "lap": ref.lap}, "lap": {"driver": lap.driver, "lap": lap.lap},
            "total_s": round(float(delta[-1]), 3), "dist": [round(float(x)) for x in grid[::4]],
            "delta": [round(float(x), 3) for x in delta[::4]], "corners": out, "tips": tips, "track": seg}


def _say(r: dict) -> str:
    bits = []
    if r["brake_later_m"]:
        bits.append(f"brake {abs(r['brake_later_m'])} m {'later' if r['brake_later_m'] > 0 else 'earlier'}")
    if r["min_speed_diff_kmh"]:
        bits.append(f"{r['min_speed_diff_kmh']:+.0f} km/h min speed")
    if r["throttle_later_m"]:
        bits.append(f"full throttle {abs(r['throttle_later_m'])} m {'later' if r['throttle_later_m'] > 0 else 'earlier'}")
    return f"Turn {r['turn']}: " + (", ".join(bits) + ", " if bits else "") + f"potential gain {r['time_lost_s']:.2f} s"
