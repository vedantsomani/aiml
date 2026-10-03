"""Score the voice on held-out 2026 situations, against the template baseline.

    pitsense voice eval [--n 150] [--out reports/voice_eval.json]

Measures: fact errors before the guard, action stated correctly before and after the
guard, length limits, wording diversity, fallback rate, latency (GPU and CPU), size.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

from . import composer, guard, synth
from .api import Voice
from .train import model_dir, test_set


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

    if voice.model is not None:
        voice.model.to(device)
        voice.device = device
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


def evaluate(path=None, n_rows: int = 150, out: Path | None = None, log=print) -> dict:
    voice = Voice(path)
    if not voice.has_model:
        raise SystemExit("no trained voice model: run `pitsense voice train` first")
    import torch

    test = test_set(n_rows)
    log(f"held-out 2026 samples: {len(test)} from {len({s.race_id for s in test})} races")
    items = [(s.facts, s.task, s.ask) for s in test]
    t = time.time()
    results = voice.write_many(items)
    log(f"generated {len(results)} messages in {time.time() - t:.0f} s")
    outputs = [r.raw if r.fallback else r.text for r in results]
    served = [r.text for r in results]
    fallbacks = [r.fallback for r in results]
    slm = _score(test, outputs, served, fallbacks)
    tmpl_out = [s.target for s in test]
    base = _score(test, tmpl_out, tmpl_out, [False] * len(test))
    div = {}
    for task in ("radio", "brief"):
        ix = [i for i, s in enumerate(test) if s.task == task]
        div[task] = {"slm": _distinct([served[i] for i in ix]), "template": _distinct([tmpl_out[i] for i in ix])}
    sizes = {"params_million": round(voice.model.n_params() / 1e6, 2), "weights_mb": round((model_dir(path) / "model.pt").stat().st_size / 1e6, 1)}
    lat_items = [(s.facts, s.task, s.ask) for s in test]
    latency = {"template_us": round(1e6 * _time_template(lat_items), 1)}
    if torch.cuda.is_available():
        latency["gpu"] = _latency(voice, "cuda", lat_items, 40)
    latency["cpu"] = _latency(voice, "cpu", lat_items, 15)
    report = {"slm": slm, "template_baseline": base, "diversity": div, "model": sizes, "latency": latency,
              "overall_fallback_rate": round(sum(fallbacks) / len(fallbacks), 4),
              "fallback_reasons": dict(Counter(e.split()[0] for r in results if r.fallback for e in r.errors)),
              "test_races": sorted({s.race_id for s in test})}
    out = Path(out) if out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def _time_template(items) -> float:
    t = time.perf_counter()
    for f, task, tgt in items[:500]:
        composer.compose(f, task, tgt)
    return (time.perf_counter() - t) / min(500, len(items))


def show(rep: dict, log=print) -> None:
    for task, r in rep["slm"].items():
        log(f"{task:14s} " + "  ".join(f"{k}={v}" for k, v in r.items() if k != "n"))
    log(f"overall fallback {rep['overall_fallback_rate']}  reasons {rep['fallback_reasons']}")
    log(f"diversity {json.dumps(rep['diversity'])}")
    log(f"model {rep['model']}  latency {json.dumps(rep['latency'])}")
