"""Benchmark for the wet-weather strategy calls (reads finished races on purpose), wet and mixed races only.

For the top-10 finishers of a wet race the head's calls are logged every lap and compared with the real
tyre-class switches (a pit stop that changes slicks / intermediates / wets), as in ``bench/strategy.py``:

* recall: share of real switches with a BOX call for the right class within +-2 laps of the in-lap;
* precision: share of BOX class-switch call episodes with a real switch to that class within +-2 laps;
* ``naive``: the same scoring for the rule "switch when the rain flag flips" (flag up while on slicks: BOX for
  INTERS; flag down while on inters / wets: BOX for SLICKS), called on the lap the flag flips.

Settings are tuned on 2018-2024 wet races and 2025-2026 are scored once (see docs/engineers/strategy.md).
``python -m pitsense.bench.wetstrategy batch|race|report`` is the runner (needs a built bench: ``pitsense bench build``).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from ..archive import downloaded_sessions
from ..config import bench_dir, data_dir
from ..events import load_archive_session
from ..pitwall.engineer import Context
from ..pitwall.types import TeamConfig
from ..pitwall.wall import PitWall
from ..state import RaceState

TOL = 2  # laps
SLICK = ("SOFT", "MEDIUM", "HARD")


def klass(c: str | None) -> str | None:
    return "S" if c in SLICK else "I" if c == "INTERMEDIATE" else "W" if c == "WET" else None


def wet_races(first: int = 2018, last: int = 2026) -> list[str]:
    """Race slugs where the rain flag was up or wet tyres were used (from the benchmark history)."""
    hist = json.loads((bench_dir() / "history.json").read_text(encoding="utf-8"))
    out = []
    for r in hist:
        w = (r.get("extra") or {}).get("weather") or {}
        if first <= r["year"] <= last and (w.get("rained") or (w.get("wet_laps") or 0) > 0):
            out.append(r["race_id"])
    return out


def real_switches(final: RaceState, top: set[str]) -> list[dict]:
    laps: dict[str, dict[int, object]] = {}
    for x in final.laps:
        laps.setdefault(x.driver, {})[x.lap] = x
    out = []
    for pe in final.pit_events:
        if pe.under_red or pe.driver not in top:
            continue
        a = laps.get(pe.driver, {}).get(pe.in_lap)
        b = laps.get(pe.driver, {}).get(pe.in_lap + 1) or laps.get(pe.driver, {}).get(pe.in_lap + 2)
        if a is None or b is None:
            continue
        ka, kb = klass(a.compound), klass(b.compound)
        if ka and kb and ka != kb:
            out.append({"car": pe.driver, "in_lap": pe.in_lap, "from": ka, "to": kb})
    return out


def replay(slug: str, settings: dict | None = None, stride: int = 1) -> dict:
    """Replay one race; the head's calls and the naive rule's calls for the top-10 finishers."""
    ref = next(r for r in downloaded_sessions() if r.slug == slug)
    from . import history as H

    hist = H.load(bench_dir() / "history.json")
    prior = H.prior_for(hist, ref.circuit_key, ref.start_utc)
    log = load_archive_session(ref.local_dir)
    final = RaceState(log.meta)
    for e in log.events:
        final.apply(e)
    total = final.total_laps or 0
    classified = [d for d in final.running_order() if d.running and d.laps >= 0.9 * total]
    top = tuple(d.number for d in classified[:10])
    if settings:
        from ..pitwall.engineers.strategy import wethead, wetmodel, wetsim

        wethead.HEAD_WET.update(settings.get("head", {}))
        wetsim.WSET.update(settings.get("sim", {}))
        wetmodel.W.update(settings.get("model", {}))
    ctx = Context.for_race(prior, ref.to_dict(), history=hist, race_start_utc=ref.start_utc, team=TeamConfig(cars=top))
    wall = PitWall(ctx)
    state = RaceState(log.meta)
    calls, naive, last_key, prev_flag, trace = [], [], {}, {}, []
    model_ok = None
    for e in log.events:
        state.apply(e)
        wall.observe(state)
        for lap in state.new_laps:
            if lap.driver not in top or lap.lap < 2 or lap.lap % stride or lap.lap >= total - 1:
                continue
            d = state.drivers[lap.driver]
            out = wall.calls(state)
            view = wall.view(state)
            flag = int(bool(view.race("weather").get("rain_now")))
            k = klass(d.compound)
            pf = prev_flag.get(lap.driver)
            prev_flag[lap.driver] = flag
            if pf is not None and k:
                if pf == 0 and flag == 1 and k == "S":
                    naive.append({"car": lap.driver, "car_lap": lap.lap + 1, "to": "I"})
                elif pf == 1 and flag == 0 and k in ("I", "W"):
                    naive.append({"car": lap.driver, "car_lap": lap.lap + 1, "to": "S"})
            hit = ctx.__dict__.get("_strategy_cache", {}).get(lap.driver)
            wa = hit[1].wet if hit is not None else None
            if wa is not None:  # what the head decided on, for re-scoring the thresholds without a replay
                wi = hit[1].wet_in
                trace.append({"car": lap.driver, "car_lap": lap.lap + 1, "from": k, "left": wi.total - wi.A,
                              "gain_now": wa.gain_now_s, "p_now": wa.p_now, "gain_best": wa.gain_best_s,
                              "now": wa.now.switches[0][1] if wa.now is not None and wa.now.switches else None,
                              "best_off": wa.best.first_offset, "best_cls": wa.best.switches[0][1] if wa.best.switches else None,
                              "src": wi.notes.get("w_source"), "w0": wi.w0})
            for c in out:
                if c.car != lap.driver:
                    continue
                if model_ok is None and c.action != "NO_CALL":
                    model_ok = True
                key = (c.action, c.compound)
                if last_key.get(c.car) != key:
                    last_key[c.car] = key
                    calls.append({"car": c.car, "car_lap": lap.lap + 1, "action": c.action, "compound": c.compound,
                                  "cls": klass(c.compound), "from": k, "conf": c.confidence,
                                  "why": [r.text for r in c.reasons[:3]]})
    return {"slug": slug, "year": ref.year, "top": top, "switches": real_switches(final, set(top)), "calls": calls,
            "naive": naive, "total": total, "model_ok": bool(model_ok), "trace": trace}


def score(res: dict, tol: int = TOL) -> dict:
    """Recall / precision of class-switch calls and of the naive rule for one replayed race."""
    sw = res["switches"]
    box, prev = [], {}
    for c in res["calls"]:  # an episode starts when the car's call changes (a new compound of the same class is not new)
        p = prev.get(c["car"])
        prev[c["car"]] = (c["action"], c["cls"])
        if c["action"] == "BOX" and c["cls"] and c["cls"] != c["from"] and p != (c["action"], c["cls"]):
            box.append(c)

    def match(calls, to_key):
        hit = 0
        errs = []
        for s in sw:
            best = None
            for c in calls:
                if c["car"] == s["car"] and to_key(c) == s["to"] and abs(c["car_lap"] - s["in_lap"]) <= tol:
                    if best is None or abs(c["car_lap"] - s["in_lap"]) < abs(best):
                        best = c["car_lap"] - s["in_lap"]
            if best is not None:
                hit += 1
                errs.append(best)
        good = sum(any(s["car"] == c["car"] and to_key(c) == s["to"] and abs(c["car_lap"] - s["in_lap"]) <= tol for s in sw) for c in calls)
        return hit, len(calls), good, errs

    h, n, g, errs = match(box, lambda c: c["cls"])
    nh, nn, ng, nerrs = match(res["naive"], lambda c: c["to"])
    return {"slug": res["slug"], "year": res["year"], "model": bool(res.get("trace")), "real": len(sw), "calls": n, "hits": h, "good_calls": g,
            "false_calls": n - g, "recall": h / len(sw) if sw else None, "precision": g / n if n else None,
            "med_err": float(np.median(errs)) if errs else None,
            "naive_calls": nn, "naive_hits": nh, "naive_false": nn - ng, "naive_recall": nh / len(sw) if sw else None,
            "naive_precision": ng / nn if nn else None}


def rescore(res: dict, head: dict) -> dict:
    """Re-run the head's wet decision over the recorded trace with other thresholds (same hysteresis chain)."""
    last, calls, prev = {}, [], {}
    for r in res["trace"]:
        h = head["hold"] if last.get(r["car"]) in ("BOX", "PREPARE_BOX") else 1.0
        act, cls = "STAY_OUT", None
        if r["now"] and r["gain_now"] >= head["g_box"] * h and r["p_now"] >= head["p_box"] * h:
            act, cls = "BOX", r["now"]
        elif r["best_off"] is not None and 0 <= r["best_off"] <= head["prep_laps"] and r["gain_best"] >= head["g_prep"] * h:
            p = r["p_now"] if r["best_off"] == 0 else 0.6
            if p >= head["p_prep"] * h or r["best_off"] > 0:
                act, cls = "PREPARE_BOX", r["best_cls"]
        last[r["car"]] = act
        key = (act, cls)
        if prev.get(r["car"]) != key:
            prev[r["car"]] = key
            calls.append({"car": r["car"], "car_lap": r["car_lap"], "action": act, "cls": {"SLICKS": "S", "INTERMEDIATE": "I", "WET": "W"}.get(cls),
                          "from": r["from"], "compound": cls})
    return dict(res, calls=calls)


def totals(scores: list[dict]) -> dict:
    s = [x for x in scores if x["model"]]
    real = sum(x["real"] for x in s)
    return {"races": len(s), "real": real, "hits": sum(x["hits"] for x in s), "calls": sum(x["calls"] for x in s),
            "false": sum(x["false_calls"] for x in s),
            "recall": sum(x["hits"] for x in s) / max(real, 1), "precision": sum(x["good_calls"] for x in s) / max(sum(x["calls"] for x in s), 1),
            "naive_calls": sum(x["naive_calls"] for x in s), "naive_false": sum(x["naive_false"] for x in s),
            "naive_recall": sum(x["naive_hits"] for x in s) / max(real, 1),
            "naive_precision": 1 - sum(x["naive_false"] for x in s) / max(sum(x["naive_calls"] for x in s), 1)}


def _scratch() -> Path:
    d = data_dir() / "scratch" / "wet"
    d.mkdir(parents=True, exist_ok=True)
    return d


def race_cmd(slug: str, tag: str, settings: dict | None = None) -> None:
    """Replay one race into data/scratch/wet/<tag>__<slug>.pkl (one process per race: Windows-safe)."""
    import time

    t = time.time()
    r = replay(slug, settings)
    r["secs"] = time.time() - t
    (_scratch() / f"{tag}__{slug}.pkl").write_bytes(pickle.dumps(r))
    print(slug, round(r["secs"]), flush=True)


def batch_cmd(tag: str, first: int, last: int, settings: dict | None = None, jobs: int = 3) -> None:
    """Replay every wet race of the years, ``jobs`` processes at a time, skipping races already done for ``tag``."""
    import subprocess
    import time

    todo = [s for s in wet_races(first, last) if not (_scratch() / f"{tag}__{s}.pkl").exists()]
    running: list = []
    while todo or running:
        running = [p for p in running if p.poll() is None]
        while todo and len(running) < jobs:
            cmd = [sys.executable, "-m", "pitsense.bench.wetstrategy", "race", todo.pop(0), tag, json.dumps(settings)]
            running.append(subprocess.Popen(cmd))
        time.sleep(2)


def load_tag(tag: str, first: int, last: int) -> list[dict]:
    out = []
    for f in sorted(_scratch().glob(f"{tag}__*.pkl")):
        r = pickle.loads(f.read_bytes())
        if first <= r["year"] <= last:
            out.append(r)
    return out


def report(tag: str, first: int, last: int, head: dict | None = None, verbose: bool = True) -> dict:
    from ..pitwall.engineers.strategy.wethead import HEAD_WET

    res = load_tag(tag, first, last)
    if head:
        h = dict(HEAD_WET)
        h.update(head)
        res = [rescore(r, h) for r in res]
    sc = [score(r) for r in res]
    if verbose:
        for x in sc:
            print(x["slug"][:34].ljust(34), "wet-engine" if x["model"] else "no-model ", "real", x["real"], "calls", x["calls"], "hit", x["hits"],
                  "false", x["false_calls"], "| naive calls", x["naive_calls"], "hit", x["naive_hits"], "false", x["naive_false"], "err", x["med_err"])
    t = totals(sc)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})
    return t


if __name__ == "__main__":  # python -m pitsense.bench.wetstrategy race|batch|report ...
    cmd = sys.argv[1]
    if cmd == "race":
        race_cmd(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif cmd == "batch":
        batch_cmd(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), json.loads(sys.argv[5]) if len(sys.argv) > 5 else None)
    elif cmd == "report":
        report(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), json.loads(sys.argv[5]) if len(sys.argv) > 5 else None)
