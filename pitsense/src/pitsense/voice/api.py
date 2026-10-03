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


class ScratchBackend:
    """The from-scratch transformer (slm.py + our own tokenizer)."""

    name = "scratch"

    def __init__(self, path=None, device: str | None = None):
        import json

        import torch

        from .slm import Config, VoiceLM
        from .tokenizer import Tokenizer
        from .train import model_dir

        d = model_dir(path)
        if not all((d / f).exists() for f in ("model.pt", "tokenizer.json", "meta.json")):
            raise FileNotFoundError(f"no scratch voice model in {d}")
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = Tokenizer.load(d / "tokenizer.json")
        m = VoiceLM(Config(**self.meta["config"]))
        m.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
        self.model = m.to(self.device).eval()

    def n_params(self) -> int:
        return self.model.n_params()

    def to(self, device: str) -> None:
        self.device = device
        self.model.to(device)

    def decode(self, inputs: list[str]) -> list[str]:
        from .slm import BOS, SEP

        prompts = [[BOS, *self.tok.encode(t), SEP] for t in inputs]
        outs: list[str] = [""] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for s in range(0, len(order), 64):
            idx = order[s : s + 64]
            for i, ids in zip(idx, self.model.generate([prompts[i] for i in idx])):
                outs[i] = self.tok.decode(ids).strip()
        return outs


def _load(kind: str, path, device):
    try:
        if kind == "llm":
            from .llm import LLMBackend

            return LLMBackend(path, device)
        return ScratchBackend(path, device)
    except Exception:  # missing torch/transformers/peft, no weights, base model not cached
        return None


class Voice:
    """Model backends (best first), the guard, and the template fallback.

    ``model``: "auto" (fine-tuned LLM, then the scratch model, whichever are installed and
    trained), "llm", "scratch", or "template". Each message tries the backends in order; the
    first whose output passes the guard is used, else the template composer.
    """

    ORDER = {"auto": ("llm", "scratch"), "llm": ("llm",), "scratch": ("scratch",), "template": ()}

    def __init__(self, path: str | Path | None = None, device: str | None = None, model: str = "auto", use_model: bool = True):
        if os.environ.get("PITSENSE_VOICE", "") == "template" or not use_model:
            model = "template"
        if model not in self.ORDER:
            raise ValueError(f"model must be one of {tuple(self.ORDER)}")
        self.backends = [b for b in (_load(k, path, device) for k in self.ORDER[model]) if b is not None]

    @property
    def has_model(self) -> bool:
        return bool(self.backends)

    @property
    def model(self):  # the first backend (what `n_params` etc. refer to)
        return self.backends[0].model if self.backends else None

    def write_many(self, items: list[tuple[Facts, str, str | None]]) -> list[Result]:
        """items: (facts, task, target) -> results, in the same order."""
        inputs = [f.text(task, target) for f, task, target in items]
        res: list[Result | None] = [None] * len(items)
        first_raw: dict[int, tuple[str, tuple]] = {}
        pending = list(range(len(items)))
        for be in self.backends:
            if not pending:
                break
            raws = be.decode([inputs[i] for i in pending])
            still = []
            for i, raw in zip(pending, raws):
                f, task, _ = items[i]
                why = guard.check(task, inputs[i], raw, f.action)
                if not why:
                    res[i] = Result(raw, be.name, False)
                else:
                    first_raw.setdefault(i, (raw, tuple(why)))
                    still.append(i)
            pending = still
        for i in pending:
            f, task, target = items[i]
            raw, why = first_raw.get(i, (None, ()))
            res[i] = Result(composer.compose(f, task, target), "template", raw is not None, why, raw)
        return res  # type: ignore[return-value]

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


def ask(call, snapshot, question: str, voice: Voice | None = None) -> str:
    """A free-form question about the car, answered from the facts in ``snapshot`` only."""
    return (voice or default_voice()).write(from_snapshot(call, snapshot), "free", question.strip().lower().rstrip("?")).text


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


__all__ = ["Voice", "Result", "say", "brief", "answer", "ask", "default_voice", "from_values"]
