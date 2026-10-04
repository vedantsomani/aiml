"""From the pit wall's state to the wet simulator's inputs (as of ``state.t``): what the field's tyre classes are
doing now, the slick penalty ``w`` they imply, and the car's own situation."""

from __future__ import annotations

import math
import zlib
from statistics import median

import numpy as np

from .wetmodel import CLS, SLICK, W, WetModel, features
from .wetsim import WSET, WetIn, analyse_car as _plan

OBS_LAPS = 3  # laps looked back for the field's pace by tyre class
THRESH = {
    "dry_pace": 0.06,  # rain flag up but slick pace this close to dry (ratio): the dry planner stays in charge
    "rain_p_b": 0.25,  # rain probability from which a dry-mode car gets a rain Plan B
}


def model_for(ctx) -> WetModel:
    m = ctx.__dict__.get("_wet_model")
    if m is None:
        m = WetModel(ctx.past_races, ctx.meta.get("circuit_key"))
        ctx.__dict__["_wet_model"] = m
    return m


def _cls(c):
    return "S" if c in SLICK else "I" if c == "INTERMEDIATE" else "W" if c == "WET" else None


def observe(state, model: WetModel) -> dict:
    """Median clean lap time by class over the last laps, as ratios of the circuit's dry lap.

    ``w_slick`` / ``r_inter`` / ``r_wet``: mean of the last two laps' medians (None below 2 cars on that class);
    ``w_hist``: slick penalty per lap, oldest first (slick laps, else implied by inters) for the trend.
    """
    cur = state.current_lap
    lo = cur - OBS_LAPS
    ref = model.ref
    by: dict[int, dict[str, list[float]]] = {}
    misses = 0
    for rec in reversed(state.laps):
        if rec.lap < lo:
            misses += 1
            if misses > 60:
                break
            continue
        k = _cls(rec.compound)
        if k is None or rec.lap_time is None or rec.lap <= 1 or rec.is_in_lap or rec.is_out_lap or rec.track_status != "1":
            continue
        by.setdefault(rec.lap, {"S": [], "I": [], "W": []})[k].append(rec.lap_time)
    med = {"S": [], "I": [], "W": []}
    for lap in sorted(by):
        g = by[lap]
        floor = min((median(v) for v in g.values() if v), default=None)
        for k in med:
            v = [t for t in g[k] if floor is not None and t <= 1.25 * floor]
            if len(v) >= 2:
                med[k].append((lap, median(v) / ref - 1.0))
    out: dict = {"n": {k: len(v) for k, v in med.items()}}
    for k, name in (("S", "w_slick"), ("I", "r_inter"), ("W", "r_wet")):
        last = [x for _, x in med[k][-2:]]
        out[name] = float(np.mean(last)) if last else None
    hist = []
    for lap in sorted(by):
        s = next((x for l, x in med["S"] if l == lap), None)
        i = next((x for l, x in med["I"] if l == lap), None)
        if s is not None:
            hist.append(s)
        elif i is not None:
            wi = model.inter_to_w(i)
            if wi is not None:
                hist.append(wi)
    out["w_hist"] = hist
    return out


def estimate_w(model: WetModel, ob: dict, weather: dict, ref_s: float, fx: np.ndarray) -> tuple[float, float, str]:
    """(w now, its sd, where it came from)."""
    if ob["w_slick"] is not None:
        return ob["w_slick"], 0.03, "slick laps"
    delta = weather.get("inters_vs_slicks_s")
    if isinstance(delta, (int, float)) and weather.get("crossover") not in (None, "none"):
        return float(np.clip(model.c_I - delta / ref_s, 0.0, W["w_max"])), 0.03, "crossover"
    r_i = ob["r_inter"]
    if r_i is None and ob["r_wet"] is not None:
        r_i = ob["r_wet"] - max(model.z0 + model.z1 * ob["r_wet"], -0.08)
    if r_i is not None:
        wi = model.inter_to_w(r_i)
        if wi is not None:
            return wi, 0.07, "inter pace"
        return model.c_I + 0.02, 0.06, "inters at their floor"
    return model.w_prior(fx), model.resid_sd, "weather prior"


def seed_for(ctx, car: str, lap: int, tag: str = "") -> int:
    m = ctx.meta
    return zlib.crc32(f"{m.get('year')}|{m.get('session_key')}|{m.get('meeting_name')}|{car}|{lap}|wet{tag}".encode()) & 0xFFFFFFFF


def build_input(state, view, ctx, model: WetModel, number: str, ob: dict | None = None) -> tuple[WetIn, dict]:
    d = state.drivers[number]
    weather, rules, pit = view.race("weather"), view.race("rules"), view.race("pitstop")
    ref_s = model.ref
    ob = ob if ob is not None else observe(state, model)
    since = weather.get("minutes_since_rain")
    run = weather.get("rain_minutes")
    fx = features(float(bool(weather.get("rain_now"))), float(since) if isinstance(since, (int, float)) else -1.0,
                  float(run) if isinstance(run, (int, float)) else 0.0)
    w0, sd, src = estimate_w(model, ob, weather, ref_s, fx)
    rain_now = bool(weather.get("rain_now"))
    hist = ob["w_hist"]
    if rain_now and len(hist) >= 3 and hist[-1] - hist[0] < -0.01 * (len(hist) - 1):
        rain_now = False  # the flag is up but the track keeps drying: it does not tell us about the next laps
    trend = 0.0
    if len(hist) >= 3:
        h4 = hist[-4:]
        trend = float(np.polyfit(np.arange(len(h4)), h4, 1)[0])
    if rain_now:
        w0 = max(w0, model.c_I + 0.05) if src in ("inters at their floor", "weather prior") else w0
    p10 = weather.get("rain_prob_10min")
    lap_s = ref_s * (1.0 + (ob["w_slick"] if ob["w_slick"] is not None else ob["r_inter"] if ob["r_inter"] is not None else 0.1))
    c0 = {"S": 0, "I": 1, "W": 2}.get(_cls(d.compound), 0)
    rl = view.car("rules", number)
    must = bool(rules.get("race_dry", True)) and bool(rl.get("must_stop"))
    loss_g = pit.get("loss_green")
    loss_n = pit.get("loss_now")
    age = d.tyre_age if d.tyre_age is not None else max(0, d.laps - d.stint_start_lap)
    inp = WetIn(A=d.laps, total=state.total_laps or 60, ref_s=ref_s, c0=c0, age0=int(age), w0=float(w0), w0_sd=float(sd),
                rain_now=rain_now, p10=float(p10) if isinstance(p10, (int, float)) else 0.0, lap_s=lap_s,
                loss_green=float(loss_g) if isinstance(loss_g, (int, float)) else ctx.prior.green,
                loss_now=float(loss_n) if isinstance(loss_n, (int, float)) else ctx.prior.green, must_slick_pair=must,
                trend=trend, notes={"w_source": src})
    return inp, ob


def analyse_car(state, view, ctx, memory, model: WetModel, number: str, *, force: bool = True):
    """A ``CarAnalysis`` from the wet simulator. ``force`` False (race still dry, rain flag up): None when the
    slicks run at about dry pace, so the dry planner keeps the car."""
    import time

    from .analysis import CarAnalysis

    t0 = time.perf_counter()
    d = state.drivers[number]
    ob = observe(state, model)
    if not force and (ob["w_slick"] is None or ob["w_slick"] < THRESH["dry_pace"]):
        return None
    if not force and _cls(d.compound) != "S":
        return None
    inp, ob = build_input(state, view, ctx, model, number, ob)
    res = _plan(model, inp, seed_for(ctx, "", 0), WSET["S"])
    a = CarAnalysis(number, True, anchor=d.laps, total=inp.total)
    a.wet, a.wet_in, a.wet_obs = res, inp, ob
    a.n_sims = WSET["S"]
    a.n_plans = res.n_plans
    a.ms = (time.perf_counter() - t0) * 1000
    return a


def rain_plan_b(state, view, ctx, model: WetModel, number: str):
    """For a car on slicks in a still-dry race: the best reaction if rain starts soon (WetResult.plan_b* fields), or None."""
    weather = view.race("weather")
    p10 = weather.get("rain_prob_10min")
    if weather.get("rain_now") or not isinstance(p10, (int, float)) or p10 < THRESH["rain_p_b"]:
        return None
    d = state.drivers[number]
    if _cls(d.compound) != "S":
        return None
    inp, ob = build_input(state, view, ctx, model, number)
    inp.w0, inp.w0_sd = min(inp.w0, 0.04), 0.02  # dry track now
    res = _plan(model, inp, seed_for(ctx, "", 0, "b"), 120)
    return res if res.plan_b is not None and (res.plan_b_gain_s or 0) > 0 else None
