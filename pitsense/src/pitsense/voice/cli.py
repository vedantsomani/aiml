"""`pitsense voice train | eval | say`."""

from __future__ import annotations

import json

from .facts import INTENTS, TASKS


def cmd_train(a) -> None:
    if a.backend == "llm":
        from .llm import train_llm

        train_llm(out=a.out, base=a.base, n_samples=a.samples, epochs=a.epochs or 1)
        return
    from .train import train

    train(out=a.out, epochs=a.epochs or 4, per_race=a.per_race, calls=a.calls, d=a.d, layers=a.layers, heads=a.heads)


def cmd_eval(a) -> None:
    from .evaluate import evaluate, show

    rep = evaluate(a.model, a.n, a.out)
    show(rep)
    if a.out:
        print(f"wrote {a.out}")


def cmd_say(a) -> None:
    """Say a message for one real situation (a benchmark row) with a sampled call, or for a Call from JSON."""
    import random

    from . import synth
    from .api import Voice
    from .facts import from_values, style_for

    voice = Voice(a.model, model="template" if a.template else a.backend)
    sit = synth.load_situations((a.year,), None, 0)
    ids = [r for r in sit if a.race.lower() in r]
    if len(ids) != 1:
        raise SystemExit(f"--race {a.race!r} matches {ids or 'nothing'}")
    _, rows, tla_of = sit[ids[0]]
    rows = [r for r in rows if int(r["lap"]) == a.lap and str(r["driver"]) == str(a.car) and r["kind"] == "lap_end"]
    if not rows:
        raise SystemExit("no such lap/car in that race")
    v = rows[0]
    v["driver"] = str(v["driver"])
    rng = random.Random(0)
    v = synth.augment(v, rng)
    call = synth.sample_call(v, rng)
    f = from_values(call, v, tla_of, style_for(call.car, v["t"]))
    task = a.task if a.task in TASKS else f"ask_{a.task}"
    r = voice.write(f, task, a.target)
    print(f"call   {call.action} {call.compound or ''} (sampled; the head of strategy is not final)")
    print(f"input  {f.text(task, a.target)}")
    print(f"source {r.source}" + (f"  (model output failed the guard: {', '.join(r.errors)})" if r.fallback else ""))
    print(r.text)
    if a.audio:
        from .tts import speak

        print(f"audio  {speak(r.text, a.audio if a.audio != 'auto' else None)}")


def add_commands(sub) -> None:
    p = sub.add_parser("voice", help="the pit wall's voice: our own small language model (train, eval, say)")
    vs = p.add_subparsers(dest="voice_cmd", required=True)
    t = vs.add_parser("train", help="train the model from scratch on the template corpus (needs torch, uses the GPU)")
    t.add_argument("--out", help="folder (default: data/models/voice)")
    t.add_argument("--backend", choices=("scratch", "llm"), default="scratch", help="scratch: our own transformer; llm: LoRA fine-tune of SmolLM2-360M-Instruct (needs transformers, peft)")
    t.add_argument("--base", default="HuggingFaceTB/SmolLM2-360M-Instruct", help="llm: base model id (or Qwen/Qwen2.5-0.5B-Instruct)")
    t.add_argument("--samples", type=int, default=30000, help="llm: training samples")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--per-race", type=int, default=1000, help="situations sampled per 2025 race")
    t.add_argument("--calls", type=int, default=3, help="sampled calls per situation")
    t.add_argument("--d", type=int, default=512)
    t.add_argument("--layers", type=int, default=8)
    t.add_argument("--heads", type=int, default=8)
    t.set_defaults(fn=cmd_train)
    e = vs.add_parser("eval", help="score on held-out 2026 situations against the template baseline")
    e.add_argument("--model", help="model folder")
    e.add_argument("--n", type=int, default=150, help="situations per 2026 race")
    e.add_argument("--out", help="write the report as JSON")
    e.set_defaults(fn=cmd_eval)
    s = vs.add_parser("say", help="say one message for a real situation (race, lap, car) with a sampled call")
    s.add_argument("--year", type=int, default=2026)
    s.add_argument("--race", required=True, help="part of the race id, e.g. hungary")
    s.add_argument("--lap", type=int, required=True)
    s.add_argument("--car", required=True, help="car number")
    s.add_argument("--task", default="radio", help=f"radio | brief | {' | '.join(INTENTS)}")
    s.add_argument("--target", help="for the gap question: the other car's number")
    s.add_argument("--model", help="model folder")
    s.add_argument("--backend", choices=("auto", "llm", "scratch"), default="auto", help="auto: fine-tuned LLM, then scratch model, whichever is trained")
    s.add_argument("--template", action="store_true", help="template composer only")
    s.add_argument("--audio", nargs="?", const="auto", help="also speak it (Piper); optional wav path")
    s.set_defaults(fn=cmd_say)
