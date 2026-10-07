"""What past wet and mixed races teach the strategy engineer: how fast intermediates and wets are against
slicks as a function of how wet the track is, how quickly it dries, and how inters wear on a drying track.

``learn_wet`` reads one finished race (called from ``priors.learn``; plain JSON in
``RaceSummary.extra["strategy"]["wet"]``). ``WetModel`` pools the races in ``ctx.past_races`` (only races that
ended before this one started) and is the only thing the wet simulator reads.

The latent state is the **slick penalty** ``w``: how much slower a slick lap is than a normal dry lap at the
circuit (ratio; the dry lap is the 10th percentile of clean green slick laps in the circuit's past races, stored
as ``ref10``). Everything else follows from it:

* slicks: ``w`` (ratio against the dry lap);
* intermediates: a floor ``c_I`` (inters are slow on a dry-ish track) that rises on a really wet track:
  ``I(w) = c_I + k_I * max(w - c_I, 0)``; the crossover is where ``I(w) = w``, i.e. ``w = c_I``;
* wets: inters plus ``z``, positive on a damp track and negative when it is very wet.

Why not model the inters-minus-slicks gap directly: inter pace is nearly flat from a damp to a dry-ish track
(about 13 % off the dry lap at every circuit with enough data), so the gap is almost all slick pace, and slick
pace is what moves with the weather.

Per lap row stored by ``learn_wet`` (every lap >= 2 of a race that had rain or wet tyres)::

    [lap, rain_now, since_rain_min, run_min, nS, nI, nW, medS, medI, medW, wearI, n_old]

``n*`` are clean green laps on slicks / intermediates / wets, ``med*`` their median lap time (None below 2
cars), ``wearI`` the lap-time loss per lap of age on intermediates (cars with 8+ laps on the set against cars
with up to 3, same lap), ``n_old`` the cars it rests on.
"""

from __future__ import annotations

from statistics import median

import numpy as np

SLICK = ("SOFT", "MEDIUM", "HARD")
WETC = ("INTERMEDIATE", "WET")
CLS = ("S", "I", "W")  # tyre classes

# knobs (tuned on 2018-2024 wet races; docs/engineers/strategy.md)
W = {
    "min_slick_rows": 12,  # laps with slick pace needed before w is regressed on the weather
    "min_pair_rows": 4,  # laps with slicks and inters together needed to fit the inter floor
    "min_wet_laps": 60,  # laps with a wet-class car in the history before the wet model is used at all
    "ridge": 3.0,
    "w_max": 0.60,
    "w_min": 0.0,
    "k_inter": 0.6,  # inter pace rise per unit of slick penalty beyond the crossover
    "since_cap": 60.0,
    "run_cap": 30.0,
    "mean_rain_run_laps": 7.0,  # length of a simulated shower (laps) when history has none
}

DEFAULT_BETA = (0.02, 0.30, -0.02, 0.05, 0.10)  # w on [1, rain_now, dry-for-long, run, rain_now*run]
DEFAULT_C_I = 0.13
DEFAULT_RATES = (0.045, 0.030)  # w rises per rain lap, falls per dry lap (ratio)
DEFAULT_Z = (0.09, -0.30)  # wets vs inters: z = z0 + z1 * r_inter
DEFAULT_WEAR = (0.0006, 0.0015, 0.0040)  # inter wear per lap (ratio): wet / drying / dry-ish


def _cls(c: str | None) -> str | None:
    if c in SLICK:
        return "S"
    if c == "INTERMEDIATE":
        return "I"
    if c == "WET":
        return "W"
    return None


def _med(v):
    return round(float(median(v)), 3) if len(v) >= 2 else None


def dry_reference(final) -> float | None:
    """Fast-but-normal dry pace of a race: 10th percentile of the clean green slick laps (None with < 60 of them)."""
    t = [r.lap_time for r in final.laps if r.compound in SLICK and r.lap_time is not None and r.lap > 1
         and not r.is_in_lap and not r.is_out_lap and r.track_status == "1"]
    if len(t) < 60:
        return None
    t.sort()
    return round(t[len(t) // 10], 3)


def learn_wet(final, meta: dict) -> dict:
    """Per-lap tyre-class pace and wetness features from a finished race ({} for a race with no rain and no wets)."""
    from ..weather import _read_weather, _raining, series_features

    laps = final.laps
    s = _read_weather(meta)
    rained = s is not None and any(_raining(v) for v in s.v)
    if not rained and not any(x.compound in WETC for x in laps):
        return {}
    by_lap: dict[int, list] = {}
    for r in laps:
        by_lap.setdefault(r.lap, []).append(r)
    rows = []
    for lap in sorted(by_lap):
        rs = by_lap[lap]
        if lap <= 1:
            continue
        grp: dict[str, list[float]] = {"S": [], "I": [], "W": []}
        fresh, old = [], []
        for r in rs:
            k = _cls(r.compound)
            if k is None or r.lap_time is None or r.is_in_lap or r.is_out_lap or r.track_status != "1":
                continue
            grp[k].append(r.lap_time)
            if k == "I" and r.tyre_age is not None:
                if r.tyre_age <= 3:
                    fresh.append((r.lap_time, r.tyre_age))
                elif r.tyre_age >= 8:
                    old.append((r.lap_time, r.tyre_age))
        allt = [t for v in grp.values() for t in v]
        if not allt:
            continue
        cap = 1.25 * min(median(v) for v in grp.values() if v)
        grp = {k: [t for t in v if t <= cap] for k, v in grp.items()}  # laps neutralised under green
        t_end = float(median([r.t_end for r in rs if r.t_end is not None] or [0.0]))
        f = series_features(s, t_end, None) if s is not None and s.t else {"rain_now": 0.0, "since_rain_min": -1.0, "run_min": 0.0}
        wear, n_old = None, 0
        if len(fresh) >= 3 and len(old) >= 2:
            ref = median(t for t, _ in fresh)
            dev = median(t / ref - 1.0 for t, _ in old)
            a_old = median(a for _, a in old)
            wear = round(dev / max(a_old - 2.0, 1.0), 5)
            n_old = len(old)
        rows.append([lap, f["rain_now"], round(f["since_rain_min"], 1), round(f["run_min"], 1),
                     len(grp["S"]), len(grp["I"]), len(grp["W"]), _med(grp["S"]), _med(grp["I"]), _med(grp["W"]), wear, n_old])
    return {"rows": rows}


# --------------------------------------------------------------------------- features
def features(rain_now: float, since: float, run: float) -> np.ndarray:
    """Inputs of the wetness regression (``since`` < 0: no rain seen yet, or raining now)."""
    now = 1.0 if rain_now else 0.0
    dry = 0.0 if now else (1.0 if since < 0 else min(since, W["since_cap"]) / W["since_cap"])
    r = min(max(run, 0.0), W["run_cap"]) / W["run_cap"]
    return np.array([1.0, now, dry, r, now * r])


class WetModel:
    """Tables from the wet laps of the races that ended before this one started."""

    def __init__(self, past_races, circuit) -> None:
        refs_c: list[float] = []
        races = []
        for race in past_races:
            ex = (race.extra or {}).get("strategy") or {}
            if "ref10" in ex and race.circuit_key == circuit:
                refs_c.append(ex["ref10"])
            if ex.get("wet", {}).get("rows"):
                races.append((race, ex))
        self.ref = float(median(refs_c)) if refs_c else None  # dry lap time at this circuit (s)
        self.n_wet_laps = 0
        X, y, wts = [], [], []  # w regressed on the weather
        cI: list[float] = []  # inter pace on laps where slicks and inters ran together
        steps_rain: list[float] = []
        steps_dry: list[float] = []
        zs: list[tuple[float, float]] = []  # (inter ratio, wets minus inters)
        wear: list[tuple[float, float, int]] = []  # (inter ratio, wear slope, n)
        for race, ex in races:
            own = ex.get("ref10")
            if own is None:
                same = [((r.extra or {}).get("strategy") or {}).get("ref10") for r in past_races if r.circuit_key == race.circuit_key]
                same = [v for v in same if v]
                own = float(median(same)) if same else None
            if own is None:
                continue
            prev = None
            for row in ex["wet"]["rows"]:
                lap, rn, since, run, nS, nI, nW, mS, mI, mW, wr, n_old = row
                if nI or nW:
                    self.n_wet_laps += 1
                fx = features(rn, since, run)
                rS = mS / own - 1.0 if mS else None
                rI = mI / own - 1.0 if mI else None
                if rS is not None and -0.1 < rS < 0.8:
                    X.append(fx)
                    y.append(rS)
                    wts.append(float(nS))
                    if prev is not None and 0 < lap - prev[0] <= 2:
                        step = (rS - prev[1]) / (lap - prev[0])
                        if rn:
                            steps_rain.append(step)
                        elif rS > 0.06 and prev[1] > 0.06:
                            steps_dry.append(step)
                    prev = (lap, rS)
                if rS is not None and rI is not None and 0.02 < rI < 0.4:
                    cI.append(rI)
                if rI is not None and mW and 0 < rI < 0.6 and abs(mW / mI - 1) < 0.2:
                    zs.append((rI, mW / mI - 1.0))
                if wr is not None and n_old >= 2 and rI is not None:
                    wear.append((rI, wr, n_old))
        self.n_slick_rows = len(y)
        self.n_pair_rows = len(cI)
        self.beta = np.array(DEFAULT_BETA)
        self.resid_sd = 0.08
        if len(y) >= W["min_slick_rows"]:
            Xa, ya, wa = np.array(X), np.array(y), np.sqrt(np.array(wts))
            A = (Xa * wa[:, None]).T @ (Xa * wa[:, None]) + W["ridge"] * np.eye(Xa.shape[1])
            b = (Xa * wa[:, None]).T @ (ya * wa) + W["ridge"] * np.array(DEFAULT_BETA)
            self.beta = np.linalg.solve(A, b)
            res = ya - Xa @ self.beta
            self.resid_sd = float(np.clip(np.sqrt(np.average(res ** 2, weights=np.array(wts))), 0.04, 0.2))
        self.c_I = float(np.clip(median(cI), 0.07, 0.2)) if len(cI) >= W["min_pair_rows"] else DEFAULT_C_I
        r_rain = float(np.median(steps_rain)) if len(steps_rain) >= 6 else DEFAULT_RATES[0]
        r_dry = -float(np.median(steps_dry)) if len(steps_dry) >= 6 else DEFAULT_RATES[1]
        self.rate_rain = float(np.clip(r_rain, 0.01, 0.12))
        self.rate_dry = float(np.clip(r_dry, 0.008, 0.08))
        self.z0, self.z1 = DEFAULT_Z
        if len(zs) >= 10:
            a = np.array([[1.0, x] for x, _ in zs])
            t = np.array([z for _, z in zs])
            lam = 3.0
            A = a.T @ a + lam * np.eye(2)
            self.z0, self.z1 = (float(v) for v in np.linalg.solve(A, a.T @ t + lam * np.array(DEFAULT_Z)))
            self.z1 = min(self.z1, 0.0)
        self.wear_bins = DEFAULT_WEAR
        if len(wear) >= 12:
            groups = ([], [], [])
            for rI, wr, n in wear:
                groups[0 if rI > 0.20 else 1 if rI > 0.14 else 2].append((wr, n))
            out = []
            for i, g in enumerate(groups):
                out.append(float(np.clip(np.average([a for a, _ in g], weights=[n for _, n in g]), 0.0, 0.01)) if len(g) >= 4 else DEFAULT_WEAR[i])
            self.wear_bins = (out[0], max(out[0], out[1]), max(out[0], out[1], out[2]))

    @property
    def available(self) -> bool:
        return self.ref is not None and self.n_wet_laps >= W["min_wet_laps"]

    def w_prior(self, fx: np.ndarray) -> float:
        return float(np.clip(fx @ self.beta, W["w_min"], W["w_max"]))

    def inter(self, w):
        """Inter pace as a ratio of the dry lap, given the slick penalty (array in, array out)."""
        w = np.asarray(w, dtype=float)
        return self.c_I + W["k_inter"] * np.maximum(w - self.c_I, 0.0)

    def inter_to_w(self, r_inter: float) -> float | None:
        """Slick penalty implied by inter pace when it is clearly above the floor (None: no information)."""
        if r_inter <= self.c_I + 0.03:
            return None
        return float(self.c_I + (r_inter - self.c_I) / W["k_inter"])

    def wet_tyre(self, w):
        r = self.inter(w)
        return r + np.maximum(self.z0 + self.z1 * r, -0.08)

    def wear_slope(self, w):
        """Inter wear per lap of age at this slick penalty (ratio per lap)."""
        r = self.inter(w)
        b = self.wear_bins
        return np.where(r > 0.20, b[0], np.where(r > 0.14, b[1], b[2]))

    def describe(self) -> dict:
        return {"ref_s": self.ref, "wet_laps": self.n_wet_laps, "slick_rows": self.n_slick_rows, "pair_rows": self.n_pair_rows,
                "beta": [round(float(b), 3) for b in self.beta], "resid_sd": round(self.resid_sd, 3), "c_I": round(self.c_I, 3),
                "rate_rain": round(self.rate_rain, 3), "rate_dry": round(self.rate_dry, 3), "z": (round(self.z0, 3), round(self.z1, 3)),
                "wear": tuple(round(b, 5) for b in self.wear_bins)}
