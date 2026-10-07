"""Post-race scoring of a pit wall's call log against the stops that really happened (``pitsense shadow-score``),
including the operator's accept / reject answers."""

from __future__ import annotations

import json
from pathlib import Path

from ..state import RaceState


# ----------------------------------------------------------------------------- shadow scoring
def load_call_log(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # a half-written last line
    return rows


def shadow_score(calls: list[dict], final: RaceState, k: int = 2, cars: set[str] | None = None) -> dict:
    """Compare logged calls with the stops that really happened.

    A BOX / PREPARE_BOX call from a car's lap L (``car_lap``) counts as right when that
    car's real stop (its in-lap, red-flag stops excluded) is within L-k .. L+k. A STAY_OUT call is
    right when the car has no stop in L .. L+k. Recall: share of real stops that had a box call within
    +-k laps. BOX_IF_SC is scored apart, only when its SC/VSC trigger happened (``box_if_sc``). NO_CALL is not scored. Calls are scored as logged; later calls never rewrite earlier ones.
    """
    stops: dict[str, list[int]] = {}
    for p in final.pit_events:
        if not p.under_red:
            stops.setdefault(p.driver, []).append(p.in_lap)
    box = [c for c in calls if c.get("kind") == "call" and c.get("action") in ("BOX", "PREPARE_BOX")]
    stay = [c for c in calls if c.get("kind") == "call" and c.get("action") == "STAY_OUT"]
    if cars:
        box, stay = [c for c in box if c["car"] in cars], [c for c in stay if c["car"] in cars]

    def lap_of(c) -> int:
        return int(c.get("car_lap") or c.get("lap") or 0)

    box_hit = [any(abs(s - lap_of(c)) <= k for s in stops.get(c["car"], [])) for c in box]
    stay_ok = [not any(0 <= s - lap_of(c) <= k for s in stops.get(c["car"], [])) for c in stay]
    real = [(car, s) for car, laps in stops.items() if not cars or car in cars for s in laps]
    covered = [any(c["car"] == car and abs(s - lap_of(c)) <= k for c in box) for car, s in real]

    def rate(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    by_action: dict[str, dict] = {}
    for c, ok in zip(box, box_hit):
        a = by_action.setdefault(c["action"], {"calls": 0, "right": 0})
        a["calls"] += 1
        a["right"] += ok
    return {
        "k": k, "calls_logged": len([c for c in calls if c.get("kind") == "call"]),
        "box_calls": len(box), "box_precision": rate(box_hit), "by_action": by_action,
        "stay_out_calls": len(stay), "stay_out_accuracy": rate(stay_ok),
        "real_stops": len(real), "stop_recall": rate(covered),
        "note": "NO_CALL not scored" if not box and not stay else "",
        "box_if_sc": _box_if_sc(calls, final, cars),
        "operator": _operator_score(calls, stops, k, cars),
    }


def _operator_score(calls, stops: dict, k: int, cars) -> dict:
    """The operator's accept / reject answers against what happened: a call counts as right by the rules above.
    ``reject``'s ``call_right`` are the calls overruled that turned out right."""
    out: dict[str, dict] = {}
    for r in calls:
        if r.get("kind") != "ack" or (cars and r.get("car") not in cars) or r.get("action") not in ("BOX", "PREPARE_BOX", "STAY_OUT"):
            continue
        lap, own = int(r.get("call_lap") or r.get("lap") or 0), stops.get(r["car"], [])
        right = (any(abs(s - lap) <= k for s in own) if r["action"] != "STAY_OUT" else not any(0 <= s - lap <= k for s in own))
        g = out.setdefault(r["decision"], {"answers": 0, "call_right": 0})
        g["answers"] += 1
        g["call_right"] += right
    for g in out.values():
        g["call_right_rate"] = round(g["call_right"] / g["answers"], 4)
    return out


def _box_if_sc(calls, final, cars) -> dict:
    """BOX_IF_SC is conditional: scored only when an SC/VSC came within its window (bench/callscore.py)."""
    from ..bench import callscore

    facts = callscore.race_facts(final)
    rows = [callscore.score_call(c, facts) for c in calls if c.get("kind") == "call" and c.get("action") == "BOX_IF_SC"
            and (not cars or c["car"] in cars)]
    trig = [r for r in rows if r["triggered"]]
    return {"calls": len(rows), "triggered": len(trig), "right": sum(bool(r["right"]) for r in trig),
            "precision_when_triggered": round(sum(bool(r["right"]) for r in trig) / len(trig), 4) if trig else None}
