"""The voice API: `say(call, snapshot)`, `brief(...)`, `answer(...)`.

The model writes the message; the guard checks it against the facts; anything that
fails (or any machine without torch or trained weights) gets the template text.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import composer, guard
from .facts import Facts, from_snapshot, from_values


@dataclass(frozen=True)
class Result:
    text: str
    source: str  # "slm" or "template"
    fallback: bool  # the model was asked and failed the guard
    errors: tuple[str, ...] = ()
    raw: str | None = None  # what the model wrote, if it failed


class Voice:
    """The model (if available) plus the guard and the template fallback."""

    def __init__(self, path: str | Path | None = None, device: str | None = None, use_model: bool = True):
        self.model = self.tok = None
        self.device = device
        if use_model and os.environ.get("PITSENSE_VOICE", "") != "template":
            self._load(path, device)

    @property
    def has_model(self) -> bool:
        return self.model is not None

    def _load(self, path, device) -> None:
        try:
            import torch

            from .slm import Config, VoiceLM
            from .tokenizer import Tokenizer
            from .train import model_dir
        except ImportError:
            return
        d = model_dir(path)
        if not all((d / f).exists() for f in ("model.pt", "tokenizer.json", "meta.json")):
            return
        import json

        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = Tokenizer.load(d / "tokenizer.json")
        m = VoiceLM(Config(**meta["config"]))
        m.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
        self.model = m.to(self.device).eval()
        self.meta = meta

    # ------------------------------------------------------------------ writing
    def _decode(self, inputs: list[str]) -> list[str]:
        from .slm import BOS, SEP

        prompts = [[BOS, *self.tok.encode(t), SEP] for t in inputs]
        outs: list[str] = [""] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for s in range(0, len(order), 64):
            idx = order[s : s + 64]
            for i, ids in zip(idx, self.model.generate([prompts[i] for i in idx])):
                outs[i] = self.tok.decode(ids).strip()
        return outs

    def write_many(self, items: list[tuple[Facts, str, str | None]]) -> list[Result]:
        """items: (facts, task, target) -> results, in the same order."""
        inputs = [f.text(task, target) for f, task, target in items]
        raws = self._decode(inputs) if self.model is not None else [None] * len(items)
        out = []
        for (f, task, target), inp, raw in zip(items, inputs, raws):
            ref = composer.compose(f, task, target)
            if raw is None:
                out.append(Result(ref, "template", False))
                continue
            why = guard.check(task, inp, raw, f.action)
            out.append(Result(raw, "slm", False) if not why else Result(ref, "template", True, tuple(why), raw))
        return out

    def write(self, f: Facts, task: str, target: str | None = None) -> Result:
        return self.write_many([(f, task, target)])[0]


_DEFAULT: Voice | None = None


def default_voice() -> Voice:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Voice()
    return _DEFAULT


def say(call, snapshot, voice: Voice | None = None) -> str:
    """A radio message (at most 20 words) for ``call``, from the values in ``snapshot``."""
    return (voice or default_voice()).write(from_snapshot(call, snapshot), "radio").text


def brief(call, snapshot, voice: Voice | None = None) -> str:
    """A strategist brief: 2 to 4 sentences."""
    return (voice or default_voice()).write(from_snapshot(call, snapshot), "brief").text


def answer(call, snapshot, intent: str, target: str | None = None, voice: Voice | None = None) -> str:
    """Answer a fixed question: gap (``target`` = the other car's number), tyre_age, pit_window, plan_b, why."""
    from .facts import INTENTS

    if intent not in INTENTS:
        raise ValueError(f"intent must be one of {INTENTS}")
    if intent == "gap" and target is None:
        raise ValueError("a gap question needs the car (target)")
    return (voice or default_voice()).write(from_snapshot(call, snapshot), f"ask_{intent}", target).text


def timed(fn, n: int = 1):
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


__all__ = ["Voice", "Result", "say", "brief", "answer", "default_voice", "from_values"]
