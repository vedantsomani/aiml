"""The training corpus: real race situations, plus sampled calls to talk about.

Situations (position, tyres, gaps, engineer values, safety car, weather) come from
benchmark rows of real races. The head of strategy is not final, so each situation
gets a sampled but coherent `Call`: an action that follows from the values, reasons,
Plan A / Plan B, and sometimes a penalty or rain the row did not have. Targets are
then written by the template composer. Splits are by race; 2026 is held out.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass
from typing import Any

from ..pitwall.types import Call, Plan, PlanStop, Reason
from . import composer
from .facts import INTENTS, Facts, from_values, style_for


def _f(v: dict, k: str, default: float = 0.0) -> float:
    x = v.get(k)
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else x


def _rng(*key) -> random.Random:
    return random.Random(zlib.crc32("|".join(map(str, key)).encode()))


def augment(v: dict, rng: random.Random) -> dict:
    """Add what real rows rarely have: a pending penalty, a rain threat."""
    v = dict(v)
    if rng.random() < 0.12 and not _f(v, "rules__penalty_s_pending"):
        v["rules__penalty_s_pending"] = rng.choice([5.0, 5.0, 10.0])
        if rng.random() < 0.15:
            v["rules__drive_through_pending"] = True
    if v.get("weather__rain_prob_10min") is None or math.isnan(_f(v, "weather__rain_prob_10min", math.nan)):
        if rng.random() < 0.15:
            v["weather__rain_prob_10min"] = round(rng.uniform(0.2, 0.9), 2)
    if rng.random() < 0.04:
        v["weather__crossover"] = rng.choice(["to_inters", "to_slicks"])
    return v


def sample_call(v: dict, rng: random.Random) -> Call:
    car = str(v["driver"])
    lap = int(_f(v, "lap", 1))
    total = int(_f(v, "total_laps", lap + 20)) or lap + 20
    left = max(0, total - lap)
    pos = int(_f(v, "position", 10))
    age = int(_f(v, "tyre_age", 0))
    cmp_ = str(v.get("compound") or "").upper()
    cliff = _f(v, "tyre__cliff_risk")
    uc = _f(v, "rivals__undercut_threat")
    pit1 = _f(v, "rivals__pit_prob_1")
    rain = _f(v, "weather__rain_prob_10min")
    sc = str(v.get("rules__sc_phase") or "none")
    must = bool(v.get("rules__must_stop"))
    loss = _f(v, "pitstop__loss_now", 20.0)
    rej = _f(v, "pitstop__rejoin_if_box_now", math.nan)
    frac = lap / total
    pen = _f(v, "rules__penalty_s_pending")
    reasons: list[Reason] = []

    def add(code, text, value=None):
        reasons.append(Reason(code, text, value))

    if sc in ("sc", "vsc", "sc_ending", "vsc_ending") and age >= 6 and left > 4:
        action = "BOX" if rng.random() < 0.65 else "PREPARE_BOX"
        add("sc_window", "cheap stop under the safety car", round(loss, 1))
    elif cliff > 0.35:
        action = "BOX" if cliff > 0.6 else "PREPARE_BOX"
        add("tyre_cliff", "tyres near the cliff", round(100 * cliff))
    elif must and frac > 0.4:
        action = "BOX" if rng.random() < 0.5 else "PREPARE_BOX"
        add("must_stop", "needs a second compound", True)
    elif uc > 0.3:
        action = "BOX"
        add("undercut_threat", "car behind can undercut", round(100 * uc))
    elif rain >= 0.5:
        action = "PREPARE_BOX"
        add("rain_onset", "rain coming", round(100 * rain))
    elif pit1 > 0.5 and left > 5:
        action = "PREPARE_BOX"
        add("rival_stopped", "rival stopping", None)
    elif sc == "none" and frac < 0.85 and age >= 8 and rng.random() < 0.12:
        action = "BOX_IF_SC"
        add("sc_window", "stop is cheap under a safety car", round(_f(v, "pitstop__loss_sc", loss), 1))
    else:
        action = "STAY_OUT"
        r = rng.random()
        if r < 0.35:
            add("clean_air", "clean air", None)
        elif r < 0.65 and age:
            add("tyre_life", "tyres have life left", age)
        elif r < 0.85 and not math.isnan(rej):
            add("rejoin_if_box_now", "boxing now loses places", int(rej))
        else:
            add("pit_loss_high", "stop is expensive", round(loss, 1))
    if rng.random() < 0.06:  # a little disagreement with the rule-of-thumb, as a real head will have
        action = rng.choice(["STAY_OUT", "PREPARE_BOX", "BOX"])
        if not reasons:
            add("clean_air", "clean air", None)
    if rng.random() < 0.03:
        action = "NO_CALL"
    if pen and action in ("BOX", "PREPARE_BOX") and rng.random() < 0.7:
        add("penalty_serve", "penalty to serve at the stop", int(pen))
    if rng.random() < 0.1 and action != "NO_CALL":
        reasons.insert(0, Reason("fuel_saving", "fuel saving is needed", None)) if rng.random() < 0.3 else None
    if left > 4 and not math.isnan(rej) and action in ("BOX", "PREPARE_BOX") and rng.random() < 0.5:
        add("rejoin_if_box_now", "rejoins in position", int(rej))
    reasons = reasons[: rng.choice([1, 2, 2, 3])]

    # tyre to fit
    if action in ("STAY_OUT", "NO_CALL"):
        fit = None
    elif rain >= 0.5 or v.get("weather__crossover") == "to_inters":
        fit = "INTERMEDIATE"
    else:
        fit = "HARD" if left > 28 else "MEDIUM" if left > 14 else "SOFT"
        if fit == cmp_ and must:
            fit = rng.choice([c for c in ("SOFT", "MEDIUM", "HARD") if c != cmp_])
    # plans
    gap = {"BOX": 0, "PREPARE_BOX": rng.randint(1, 3), "BOX_IF_SC": rng.randint(3, 10), "STAY_OUT": rng.randint(4, 18), "NO_CALL": rng.randint(4, 18)}[action]

    def mkplan(name, first_gap, compound, trigger=None):
        stops = []
        l1 = lap + first_gap
        if l1 <= total - 3 and (left > 6 or action == "BOX"):
            stops.append(PlanStop(l1, compound or ("HARD" if total - l1 > 25 else "MEDIUM")))
            if total - l1 > 32 and rng.random() < 0.25:
                l2 = l1 + rng.randint(12, 20)
                if l2 <= total - 3:
                    stops.append(PlanStop(l2, "SOFT" if total - l2 < 15 else "MEDIUM"))
        p = max(1, pos + rng.randint(-2, 2))
        return Plan(name, tuple(stops), expected_position=float(p), trigger=trigger)

    plan_a = mkplan("A", gap, fit)
    plan_b = None
    if rng.random() < 0.75 and left > 6:
        trig = rng.choice([
            f"if sc comes before lap {min(total - 3, lap + rng.randint(3, 15))}",
            f"if rain starts within {rng.choice([5, 10, 15])} minutes",
            f"if the undercut threat passes {rng.choice([30, 40, 50])} percent",
            f"if the tyres fall off before lap {min(total - 3, lap + rng.randint(3, 12))}",
            "if the car behind pits first",
            None,
        ])
        plan_b = mkplan("B", max(0, gap + rng.choice([-4, -2, 2, 5])), rng.choice(["SOFT", "MEDIUM", "HARD"]), trig)
    conf = round(rng.uniform(0.5, 0.95), 2) if rng.random() < 0.8 else None
    return Call(t=round(_f(v, "t"), 3), car=car, action=action, compound=fit, confidence=conf,
                reasons=tuple(reasons), plan_a=plan_a, plan_b=plan_b)


FREE_QUESTIONS = (
    "how far is the car ahead", "what is the gap to the car in front", "who is ahead of us", "gap ahead please",
    "how far is the car behind", "what is the gap behind", "who is chasing us", "gap to the car behind",
    "what tyres are we on", "how old are the tyres", "what is the stint age", "which compound is on the car",
    "what does a stop cost", "how much is the pit loss now", "how long would a stop take",
    "where would we rejoin", "what position do we come out if we box", "where do we rejoin after a stop",
    "what is the rain chance", "is rain coming", "how is the weather looking",
    "how many laps are left", "laps to go", "how long until the end",
    "what position are we in", "where are we running", "what place are we",
    "do we have a penalty", "any penalty to serve", "is there a drive through pending",
    "what is the cliff risk", "how are the tyres wearing", "what is the degradation",
    "is the undercut a threat", "how big is the undercut threat",
    "what is the plan", "when do we box", "what is the pit window", "what is plan b",
    "why this call", "what is the reason for the call", "why are we boxing",
    "what is the call", "what should we do", "do we box or stay out",
)


@dataclass(frozen=True)
class Sample:
    race_id: str
    task: str
    input: str
    target: str
    action: str
    facts: Facts
    ask: str | None = None


def samples_for_row(v: dict, tla_of: dict, k: int, seed: int = 0, alias: bool = False, free: bool = False) -> list[Sample]:
    """The radio, brief and two question samples for one real situation (call variant ``k``)."""
    rng = _rng(seed, v["race_id"], v["driver"], v["t"], k)
    v = augment(v, rng)
    if alias:  # invented three-letter names, so the model learns to copy names, not to know a grid
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        tla_of = {c: "".join(rng.choice(letters) for _ in range(3)) for c in tla_of}
    call = sample_call(v, rng)
    f = from_values(call, v, tla_of, style_for(call.car, v["t"], salt=f"{seed}-{k}"))
    out = []
    tasks = ["radio", "brief"] + [f"ask_{i}" for i in rng.sample(INTENTS, 2)]
    if free:
        tasks.append("free")
    for task in tasks:
        target = None
        if task == "ask_gap":
            nbrs = [n.car for n in (f.ahead, f.behind) if n]
            if nbrs and rng.random() < 0.9:
                target = rng.choice(nbrs)
            else:
                others = [c for c in tla_of if c not in (f.car, *nbrs)]
                target = rng.choice(sorted(others)) if others else None
            if target is None:
                continue
        if task == "free":
            target = rng.choice(FREE_QUESTIONS)
        out.append(Sample(v["race_id"], task, f.text(task, target), composer.compose(f, task, target), f.action, f, target))
    return out


def load_situations(years=(2025, 2026), per_race: int | None = None, seed: int = 0):
    """Benchmark rows as dicts, grouped by race: {race_id: (year, rows, tla_of)}."""
    from ..bench.dataset import load_bench

    df = load_bench()
    df = df[df["year"].isin(years) & df["position"].notna() & df["compound"].notna()]
    out = {}
    for race_id, g in df.groupby("race_id", sort=True):
        tla_of = dict(zip(g["driver"].astype(str), g["tla"]))
        if per_race and len(g) > per_race:
            g = g.sample(per_race, random_state=zlib.crc32(f"{seed}{race_id}".encode()) % (2**31)).sort_values(["t", "driver"])
        out[race_id] = (int(g["year"].iloc[0]), g.to_dict("records"), tla_of)
    return out


def build(years, per_race, calls_per_row=1, seed=0, stay_keep=0.4, races=None, alias=0.0, free=False) -> list[Sample]:
    """Samples for the given years (or race ids). STAY_OUT is the common call in real races, so
    only ``stay_keep`` of those situations are kept: the model must see the rare calls often."""
    out: list[Sample] = []
    for race_id, (_, rows, tla_of) in load_situations(years, per_race, seed).items():
        for v in rows:
            v["driver"] = str(v["driver"])
            if races is not None and race_id not in races:
                continue
            for k in range(calls_per_row):
                ss = samples_for_row(v, tla_of, k, seed, _rng(seed, 'alias', race_id, v['driver'], v['t'], k).random() < alias, free)
                if ss and ss[0].action == "STAY_OUT" and _rng(seed, "keep", race_id, v["driver"], v["t"], k).random() > stay_keep:
                    continue
                out += ss
    return out


def split_races(situations: dict[str, Any], val_races: int = 2) -> tuple[list[str], list[str], list[str]]:
    """train / validation / test race ids. Test = every 2026 race; validation = the last 2025 races."""
    ids = sorted(situations)
    test = [r for r in ids if situations[r][0] >= 2026]
    rest = [r for r in ids if situations[r][0] < 2026]
    return rest[:-val_races], rest[-val_races:], test
