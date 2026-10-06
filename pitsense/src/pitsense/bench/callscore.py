"""Fair prediction scoring of the head's calls: each action against what it actually claims.

``runtime.shadow_score`` (the legacy score) counts BOX, PREPARE_BOX and BOX_IF_SC alike as "a stop within
+-k laps". That is unfair in both directions: a BOX called two laps *after* the stop counts as right, and
BOX_IF_SC ("box if a safety car comes") is scored as an unconditional box even when no safety car came.
Here every action is scored on its own claim, from the call's lap L (``car_lap``: the lap the car is about to
drive, i.e. the in-lap if it boxes now):

* ``BOX``: a real stop (in-lap, red-flag stops excluded) in L .. L+2.
* ``PREPARE_BOX`` ("box within 2 laps"): a real stop in L .. L+3.
* ``STAY_OUT``: no real stop in L .. L+2.
* ``BOX_IF_SC``: scored **only when its trigger happened**: the car drove a SC / VSC lap within L .. L+4 (Plan B's
  "SC/VSC within 5 laps"). Triggered at the first such lap s, it is right when the car stopped in s .. s+2.
  Untriggered calls are counted, not scored (``untriggered``); whether the car then stayed out (no stop in
  L .. L+2) is reported as ``held``.

Recall: share of real stops covered by a BOX / PREPARE_BOX window or a triggered, right BOX_IF_SC, reported for
green-flag stops and stops on SC/VSC laps separately. Everything reads the finished race on purpose (labels);
nothing here feeds a call.

Uncertainty: per-race bootstrap (races resampled with replacement, ratio of sums), 90 % intervals, fixed seed.
"""

from __future__ import annotations

import numpy as np

WINDOWS = {"BOX": (0, 2), "PREPARE_BOX": (0, 3)}
STAY_K = 2
SC_WINDOW = 5  # BOX_IF_SC trigger window: laps L .. L+SC_WINDOW-1
SC_REACT = 2  # triggered at lap s: right with a stop in s .. s+SC_REACT
ACTIONS = ("BOX", "PREPARE_BOX", "STAY_OUT", "BOX_IF_SC")
NEUTRAL = {"4", "6", "7"}  # SC, VSC, VSC ending
WET = {"INTERMEDIATE", "WET"}
B_BOOT = 2000
PHASES = ((0.0, 1 / 3, "early"), (1 / 3, 2 / 3, "mid"), (2 / 3, 10.0, "late"))


def race_facts(final) -> dict:
    """What happened in a finished race, as the scoring needs it (reads the future: labels only)."""
    stops: dict[str, list[int]] = {}
    for p in final.pit_events:
        if not p.under_red:
            stops.setdefault(p.driver, []).append(int(p.in_lap))
    neutral: dict[str, set[int]] = {}
    wet = False
    for x in final.laps:
        if set(str(x.track_status or "").split(",")) & NEUTRAL:
            neutral.setdefault(x.driver, set()).add(int(x.lap))
        if (x.compound or "") in WET:
            wet = True
    codes = {c for _, c in final.status_log}
    return {"stops": {k: sorted(v) for k, v in stops.items()}, "neutral": {k: sorted(v) for k, v in neutral.items()},
            "total": int(final.total_laps or 0), "sc_race": bool(codes & NEUTRAL), "wet": wet}


def phase_of(lap: int, total: int) -> str:
    f = lap / total if total else 0.0
    return next(name for lo, hi, name in PHASES if lo <= f < hi)


def score_call(c: dict, facts: dict) -> dict:
    """One call -> {action, lap, scored, right, ...}. ``right`` is None when the call is not scored."""
    act, L = c["action"], int(c.get("car_lap") or c.get("lap") or 0)
    mine = facts["stops"].get(c["car"], [])
    neut = set(facts["neutral"].get(c["car"], []))
    out = {"car": c["car"], "lap": L, "action": act, "scored": True, "right": None, "triggered": None, "held": None,
           "under_sc": L in neut, "phase": phase_of(L, facts["total"]),
           "legacy_right": any(abs(s - L) <= 2 for s in mine) if act != "STAY_OUT" else not any(0 <= s - L <= 2 for s in mine)}
    if act in WINDOWS:
        lo, hi = WINDOWS[act]
        out["right"] = any(L + lo <= s <= L + hi for s in mine)
    elif act == "STAY_OUT":
        out["right"] = not any(0 <= s - L <= STAY_K for s in mine)
    elif act == "BOX_IF_SC":
        trig = [l for l in range(L, L + SC_WINDOW) if l in neut]
        out["triggered"] = bool(trig)
        out["held"] = not any(0 <= s - L <= STAY_K for s in mine)
        if trig:
            s0 = trig[0]
            out["right"] = any(s0 <= s <= s0 + SC_REACT for s in mine)
        else:
            out["scored"] = False
    else:
        out["scored"] = False
    return out


def score_race(calls: list[dict], facts: dict, cars=None) -> dict:
    """Per-call rows and the race's real stops (with whether each was covered) for ``cars``."""
    calls = [c for c in calls if c.get("kind", "call") == "call" and c.get("action") in ACTIONS and (not cars or c["car"] in cars)]
    rows = [score_call(c, facts) for c in calls]
    real = []
    for car, laps in facts["stops"].items():
        if cars and car not in cars:
            continue
        neut = set(facts["neutral"].get(car, []))
        for s in laps:
            cov = False
            for c, r in zip(calls, rows):
                if c["car"] != car:
                    continue
                if c["action"] in WINDOWS:
                    lo, hi = WINDOWS[c["action"]]
                    cov = cov or r["lap"] + lo <= s <= r["lap"] + hi
                elif c["action"] == "BOX_IF_SC" and r["triggered"] and r["right"]:
                    cov = cov or any(l <= s <= l + SC_REACT for l in range(r["lap"], r["lap"] + SC_WINDOW) if l in neut)
            real.append({"car": car, "lap": s, "under_sc": s in neut, "covered": cov})
    return {"calls": rows, "stops": real}


# --------------------------------------------------------------------------- aggregation with per-race bootstrap
def _ratio(num, den):
    return float(num.sum() / den.sum()) if den.sum() else None


def boot_ratio(num: np.ndarray, den: np.ndarray, B: int = B_BOOT, seed: int = 0) -> tuple:
    """(point, lo, hi): sum(num)/sum(den) over races, 90 % interval from resampling races."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    pt = _ratio(num, den)
    if pt is None or len(num) < 2:
        return pt, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (B, len(num)))
    d = den[idx].sum(1)
    v = np.where(d > 0, num[idx].sum(1) / np.maximum(d, 1e-12), np.nan)
    v = v[np.isfinite(v)]
    return pt, float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))


def boot_mean(values_by_race: list[np.ndarray], B: int = B_BOOT, seed: int = 0) -> tuple:
    """(mean, lo, hi) of all values, races resampled (a race's values travel together)."""
    sums = np.array([float(np.sum(v)) for v in values_by_race])
    ns = np.array([len(v) for v in values_by_race], float)
    return boot_ratio(sums, ns, B, seed)


def _cell(rows_by_race: list[list[dict]], pred) -> dict:
    num = np.array([sum(1 for r in rows if pred(r) and r["right"]) for rows in rows_by_race])
    den = np.array([sum(1 for r in rows if pred(r)) for rows in rows_by_race])
    pt, lo, hi = boot_ratio(num, den)
    return {"n": int(den.sum()), "right": int(num.sum()), "rate": pt, "ci90": [lo, hi]}


def summarise(per_race: list[dict]) -> dict:
    """``per_race``: [{"race", "facts", "calls", "stops"}] -> per-action rates with 90 % CIs, recall, legacy."""
    rows = [[r for r in x["calls"] if r["scored"]] for x in per_race]
    allrows = [x["calls"] for x in per_race]
    out = {"races": len(per_race), "by_action": {}}
    for a in ACTIONS:
        out["by_action"][a] = _cell(rows, lambda r, a=a: r["action"] == a)
    bis = [r for rr in allrows for r in rr if r["action"] == "BOX_IF_SC"]
    out["box_if_sc"] = {"calls": len(bis), "triggered": sum(1 for r in bis if r["triggered"]),
                        "untriggered": sum(1 for r in bis if not r["triggered"]),
                        "held_when_untriggered": (sum(1 for r in bis if not r["triggered"] and r["held"]) /
                                                  max(1, sum(1 for r in bis if not r["triggered"]))) if bis else None}
    out["box_or_prepare"] = _cell(rows, lambda r: r["action"] in WINDOWS)
    for name, pred in (("all", lambda s: True), ("green", lambda s: not s["under_sc"]), ("sc_vsc", lambda s: s["under_sc"])):
        num = np.array([sum(1 for s in x["stops"] if pred(s) and s["covered"]) for x in per_race])
        den = np.array([sum(1 for s in x["stops"] if pred(s)) for x in per_race])
        pt, lo, hi = boot_ratio(num, den)
        out.setdefault("recall", {})[name] = {"stops": int(den.sum()), "covered": int(num.sum()), "rate": pt, "ci90": [lo, hi]}
    # the legacy rule on the same calls (BOX_IF_SC as an unconditional box, +-2 laps)
    leg = [[dict(r, right=r["legacy_right"]) for r in rr if r["action"] in ("BOX", "PREPARE_BOX", "BOX_IF_SC")] for rr in allrows]
    out["legacy_box_precision"] = _cell(leg, lambda r: True)
    return out


def splits(per_race: list[dict]) -> dict:
    """Per-action rates by circuit, weather (dry / wet race), race phase and safety-car presence."""
    def table(key_of_race=None, key_of_call=None):
        keys = {}
        for x in per_race:
            for r in x["calls"]:
                if not r["scored"]:
                    continue
                k = key_of_race(x) if key_of_race else key_of_call(r)
                keys.setdefault(k, {})
        res = {}
        for k in sorted(keys):
            sel = [[r for r in x["calls"] if r["scored"] and (key_of_race(x) == k if key_of_race else key_of_call(r) == k)]
                   for x in per_race]
            res[k] = {a: _cell(sel, lambda r, a=a: r["action"] == a) for a in ACTIONS}
        return res

    return {
        "circuit": table(key_of_race=lambda x: x["race"]),
        "weather": table(key_of_race=lambda x: "wet" if x["facts"]["wet"] else "dry"),
        "phase": table(key_of_call=lambda r: r["phase"]),
        "safety_car": table(key_of_race=lambda x: "SC/VSC race" if x["facts"]["sc_race"] else "no SC/VSC"),
        "call_under_sc": table(key_of_call=lambda r: "under SC/VSC" if r["under_sc"] else "green"),
    }
