"""The guard: nothing reaches the pit wall that the input does not support.

Every number, car number, three-letter name, tyre compound and lap in the output
must appear in the input, the length limits must hold, and (for radio, briefs and
"why") the action stated must be the call. Otherwise the caller falls back to the
template composer, which writes only from the facts.
"""

from __future__ import annotations

import re

from . import composer

_NUM = re.compile(r"\d+(?:\.\d+)?")
_TLA = re.compile(r"\b[A-Z]{3}\b")
_CMP = re.compile(r"\b(soft|medium|hard|wet|inter(?:mediate)?)s?\b", re.I)
_NUMWORDS = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b", re.I)
_OK_CAPS = {"BOX", "VSC", "DRS", "SC"}
RADIO_MAX_WORDS = 20


def _canon(c: str) -> str:
    c = c.lower()
    return "inter" if c.startswith("inter") else c


def input_vocab(inp: str) -> tuple[set[str], set[str], set[str]]:
    """(numbers, car TLAs, compounds) that the input holds."""
    nums = set(_NUM.findall(inp))
    tlas: set[str] = set()
    for field in inp.split(" | "):
        parts = field.split()
        if parts and parts[0] in ("CAR", "AH", "BH") and len(parts) >= 3:
            tlas.add(parts[2])
    comps = {_canon(m.group(1)) for m in _CMP.finditer(inp)}
    return nums, tlas, comps


def fact_errors(inp: str, out: str) -> list[str]:
    """Things the output says that the input does not hold."""
    nums, tlas, comps = input_vocab(inp)
    errs = [f"number {n}" for n in _NUM.findall(out) if n not in nums]
    errs += [f"name {t}" for t in _TLA.findall(out) if t not in tlas and t not in _OK_CAPS]
    errs += [f"compound {_canon(m.group(1))}" for m in _CMP.finditer(out) if _canon(m.group(1)) not in comps]
    errs += [f"number word {m.group(0)}" for m in _NUMWORDS.finditer(out)]
    return errs


def n_sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def length_ok(task: str, out: str) -> bool:
    if task == "radio":
        return 0 < len(out.split()) <= RADIO_MAX_WORDS
    if task == "brief":
        return 2 <= n_sentences(out) <= 4
    return 0 < len(out.split()) <= 60


def action_ok(task: str, action: str, out: str) -> bool:
    if task not in ("radio", "brief", "ask_why"):
        return True
    return composer.stated_action(out) == action


def check(task: str, inp: str, out: str, action: str) -> list[str]:
    """Reasons the output fails the guard (empty = passes)."""
    why = fact_errors(inp, out)
    if not length_ok(task, out):
        why.append("length")
    if not action_ok(task, action, out):
        why.append("action")
    return why
