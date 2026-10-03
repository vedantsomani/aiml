"""Score the voice on held-out 2026 situations: template, from-scratch model, fine-tuned LLM.

    pitsense voice eval [--n 150] [--out reports/voice_eval.json]

Measures, per task: fact errors before the guard, fallback rate, action stated correctly
before and after the guard, length limits, wording diversity, latency (GPU and CPU), size.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

from . import composer, guard, synth
from .api import Voice
from .train import SEED, test_set


def _distinct(texts: list[str]) -> dict:
    grams: Counter = Counter()
    total = 0
    for t in texts:
        w = t.lower().split()
        for a, b in zip(w, w[1:]):
            grams[(a, b)] += 1
            total += 1
    return {"unique_messages": round(len(set(texts)) / max(1, len(texts)), 4), "distinct_2gram": round(len(grams) / max(1, total), 4), "n_messages": len(texts)}


def _score(samples, outputs, served, fallbacks) -> dict:
    """Per-task numbers for model outputs (before the guard) and what is served (after it)."""
    res = {}
    for task in sorted({s.task for s in samples}):
        idx = [i for i, s in enumerate(samples) if s.task == task]
        n = len(idx)
        pre_err = sum(1 for i in idx if guard.fact_errors(samples[i].input, outputs[i]))
        post_err = sum(1 for i in idx if guard.fact_errors(samples[i].input, served[i]))
        act_idx = [i for i in idx if task in ("radio", "brief", "ask_why")]
        res[task] = {
            "n": n,
            "fact_error_rate_before_guard": round(pre_err / n, 4),
            "fact_error_rate_after_guard": round(post_err / n, 4),
            "length_ok_before_guard": round(sum(guard.length_ok(task, outputs[i]) for i in idx) / n, 4),
            "length_ok_after_guard": round(sum(guard.length_ok(task, served[i]) for i in idx) / n, 4),
            "fallback_rate": round(sum(fallbacks[i] for i in idx) / n, 4),
            "exact_match_with_template": round(sum(outputs[i] == samples[i].target for i in idx) / n, 4),
        }
        if act_idx:
            res[task]["action_ok_before_guard"] = round(sum(guard.action_ok(task, samples[i].action, outputs[i]) for i in act_idx) / len(act_idx), 4)
            res[task]["action_ok_after_guard"] = round(sum(guard.action_ok(task, samples[i].action, served[i]) for i in act_idx) / len(act_idx), 4)
    return res


def _latency(voice: Voice, device: str, items, reps: int) -> dict:
    import torch

    voice.backends[0].to(device)
    out = {}
    for task in ("radio", "brief"):
        its = [it for it in items if it[1] == task][:reps]
        voice.write(*its[0])  # warm-up
        ts = []
        for it in its:
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            voice.write(*it)
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t))
        ts.sort()
        out[task] = {"median_ms": round(statistics.median(ts), 1), "p95_ms": round(ts[int(0.95 * (len(ts) - 1))], 1)}
    return out


def _one(kind: str, path, test, items, free_items, log) -> dict | None:
    import torch

    voice = Voice(path, model=kind)
    if not voice.has_model:
        log(f"{kind}: not available (not trained, or packages/base model missing); skipped")
        return None
    be = voice.backends[0]
    t = time.time()
    results = voice.write_many(items)
    log(f"{kind}: generated {len(results)} messages in {time.time() - t:.0f} s")
    outputs = [r.raw if r.fallback else r.text for r in results]
    served = [r.text for r in results]
    fallbacks = [r.fallback for r in results]
    rep = {"tasks": _score(test, outputs, served, fallbacks), "overall_fallback_rate": round(sum(fallbacks) / len(fallbacks), 4),
           "fallback_reasons": dict(Counter(e.split()[0] for r in results if r.fallback for e in r.errors))}
    rep["diversity"] = {task: _distinct([served[i] for i, s in enumerate(test) if s.task == task]) for task in ("radio", "brief")}
    if free_items:
        fs = [s for s, _ in free_items]
        fr = voice.write_many([it for _, it in free_items])
        rep["free_form"] = _score(fs, [r.raw if r.fallback else r.text for r in fr], [r.text for r in fr], [r.fallback for r in fr])["free"]
    rep["model"] = {"params_million": round(be.n_params() / 1e6, 1)}
    lat = {}
    if torch.cuda.is_available():
        lat["gpu"] = _latency(voice, "cuda", items, 30)
    lat["cpu"] = _latency(voice, "cpu", items, 8)
    rep["latency"] = lat
    return rep


def _time_template(items) -> float:
    t = time.perf_counter()
    for f, task, tgt in items[:500]:
        composer.compose(f, task, tgt)
    return (time.perf_counter() - t) / min(500, len(items))


def evaluate(path=None, n_rows: int = 150, out: Path | None = None, log=print, backends=("scratch", "llm")) -> dict:
    test = test_set(n_rows)
    log(f"held-out 2026 samples: {len(test)} from {len({s.race_id for s in test})} races")
    items = [(s.facts, s.task, s.ask) for s in test]
    tmpl = [s.target for s in test]
    free = [s for s in synth.build((2025, 2026), n_rows, 1, SEED, races={s.race_id for s in test}, free=True) if s.task == "free"]
    report = {"test_races": sorted({s.race_id for s in test}),
              "template": {"tasks": _score(test, tmpl, tmpl, [False] * len(test)),
                           "latency_us": round(1e6 * _time_template(items), 1),
                           "diversity": {t: _distinct([tmpl[i] for i, s in enumerate(test) if s.task == t]) for t in ("radio", "brief")},
                           "free_form": _score(free, [s.target for s in free], [s.target for s in free], [False] * len(free))["free"]}}
    free_items = [(s, (s.facts, "free", s.ask)) for s in free]
    for kind in backends:
        r = _one(kind, path, test, items, free_items if kind == "llm" else [], log)
        if r:
            report[kind] = r
    out = Path(out) if out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def show(rep: dict, log=print) -> None:
    for kind in ("scratch", "llm"):
        r = rep.get(kind)
        if not r:
            continue
        log(f"== {kind}  fallback {r['overall_fallback_rate']}  {r['fallback_reasons']}")
        for task, x in r["tasks"].items():
            log(f"{task:14s} " + "  ".join(f"{k}={v}" for k, v in x.items() if k != "n"))
        if "free_form" in r:
            log(f"free_form      {r['free_form']}")
        log(f"diversity {json.dumps(r['diversity'])}  model {r['model']}  latency {json.dumps(r['latency'])}")
