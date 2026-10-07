"""The guard: nothing reaches the pit wall that the input does not support.

Every number, car number, three-letter name, tyre compound and lap in the output
must appear in the input, the length limits must hold, and (for radio, briefs and
"why") the action stated must be the call. What the output says about a thing must
also be what the input says about that thing: a gap next to a driver is that
driver's gap, "ahead" / "behind" is the side he is on, a lap named for a stop is a
lap of the plan (or this lap, for a BOX), and a tyre to fit is the call's.
Otherwise the caller falls back to the template composer, which writes only from
the facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import composer

_NUM = re.compile(r"\d+(?:\.\d+)?")
_TLA = re.compile(r"\b[A-Z]{3}\b")
_CMP = re.compile(r"\b(soft|medium|hard|wet|inter(?:mediate)?)s?\b", re.I)
_NUMWORDS = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b", re.I)
_OK_CAPS = {"BOX", "VSC", "DRS", "SC"}
RADIO_MAX_WORDS = 20

# ------------------------------------------------------------------ how the output refers to things (see relation_errors)
_SENT = re.compile(r"(?<=[.!?])\s+")
# clauses that never share a driver: "VER is 1.2 seconds ahead and NOR is 2.4 seconds behind"
_SEG = re.compile(r";| - |\s+and\s+", re.I)
_AMOUNT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_UNIT = re.compile(r"\s*(?:seconds?|secs?|s)\b", re.I)
_SIDE = r"(ahead|behind|in front)"
_SIDE_AFTER = re.compile(rf"\s*(?:(?:seconds?|secs?|s)\s+)?{_SIDE}\b", re.I)  # "1.2 seconds ahead"
_SIDE_BEFORE = re.compile(rf"\b{_SIDE}\s+by\s+$", re.I)  # "ahead by 1.2"
_SIDE_WORD = re.compile(rf"\b{_SIDE}\b", re.I)
_PLAN = re.compile(r"\bplan ([ab])\b", re.I)
_LAP = re.compile(r"\blaps? (\d+)(?: of (\d+))?\b", re.I)
# the words before a lap named for a stop: "box on lap", "stops on lap", "then on lap"
_BOX_ON = re.compile(r"(?:\b(?:box(?:es|ing)?|pit(?:s|ting)?|stop(?:s|ping|ped)?)\b[^.;,]*|\bthen)\s+(?:on|at)\s*$", re.I)
_CPD = r"(soft|medium|hard|wet|inter(?:mediate)?)s?"
# the tyre being fitted: "for hards", "onto softs", "to the mediums", "take hards", "hards going on", "hards ready", "plan on hards"
_NEW_TYRE = re.compile(
    rf"\b(?:for|onto|to|takes?|taking|fit(?:ting)?|put on)\s+(?:(?:the|a|new|fresh|used)\s+)*{_CPD}\b"
    rf"|\b{_CPD}\s+(?:going on|(?:are\s+|is\s+)?ready|next)\b"
    rf"|\bplan on\s+{_CPD}\b", re.I)


def _canon(c: str) -> str:
    c = c.lower()
    return "inter" if c.startswith("inter") else c


def input_vocab(inp: str) -> tuple[set[str], set[str], set[str]]:
    """(numbers, car TLAs, compounds) that the input holds."""
    nums = set(_NUM.findall(inp))
    tlas: set[str] = set()
    for field_ in inp.split(" | "):
        parts = field_.split()
        if parts and parts[0] in ("CAR", "AH", "BH") and len(parts) >= 3:
            tlas.add(parts[2])
    comps = {_canon(m.group(1)) for m in _CMP.finditer(inp)}
    return nums, tlas, comps


@dataclass
class _Call:
    """What the relational checks read from the input line (see ``facts.Facts.text``)."""

    act: str = ""
    tla: str = ""  # the car the call is for
    fit: str = ""  # the compound the call fits (canonical)
    lap: str = ""  # the lap the call is made on, and the race length
    total: str = ""
    side: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # neighbour TLA -> (ahead | behind, gap as written)
    loose: set[str] = field(default_factory=set)  # neighbours not told apart by TLA: only the car number, or a TLA two cars share
    plans: dict[str, set[tuple[str, str]]] = field(default_factory=dict)  # "A" / "B" -> {(in-lap, compound)}
    quoted: list[str] = field(default_factory=list)  # free text the input carries (trigger, reasons): the output may repeat it


def _parse(inp: str) -> _Call:
    c = _Call()
    nbrs = []
    for f in inp.split(" | ")[1:]:
        tag, *p = f.split() or [""]
        if tag == "ACT" and p:
            c.act = p[0]
        elif tag == "FIT" and p:
            c.fit = _canon(p[0])
        elif tag == "CAR" and len(p) >= 2:
            c.tla = p[1]
        elif tag == "LAP" and p:  # "LAP lap total left", or "LAP lap" when the race length is unknown
            c.lap, c.total = p[0], p[1] if len(p) > 1 else None
        elif tag in ("AH", "BH") and len(p) >= 2:
            nbrs.append((p[1], "ahead" if tag == "AH" else "behind", p[2] if len(p) > 2 else None))
        elif tag in ("PA", "PB") and p and p[0].isdigit():
            s = p[1: 1 + 2 * int(p[0])]
            c.plans[tag[1]] = {(s[i], _canon(s[i + 1])) for i in range(0, len(s) - 1, 2)}
        elif tag == "TRG":
            c.quoted.append(f[3:].strip())
        elif tag == "R" and " ~ " in f:
            c.quoted.append(f.split(" ~ ", 1)[1].strip())
    for name, side, gap in nbrs:
        if _TLA.fullmatch(name) and name != c.tla and [n for n, _, _ in nbrs].count(name) == 1:
            c.side[name] = (side, gap)
        else:
            c.loose.add(name)
    return c


def _unquote(out: str, quoted: list[str]) -> str:
    """``out`` with the free text of the input blanked: repeating the input's own words is not a claim of the voice."""
    for q in sorted({q for q in quoted if len(q) >= 6}, key=len, reverse=True):
        out = re.sub(rf"(?<!\w){re.escape(q)}(?!\w)", lambda m: " " * len(m.group()), out, flags=re.I)
    return out


def _scope(marks: list[tuple[int, str]], pos: int) -> str | None:
    """The plan ("A" / "B") a sentence is talking about at ``pos``: the last "Plan X" before it."""
    before = [k for p, k in marks if p < pos]
    return before[-1] if before else None


def _owner(named: list[tuple[int, str]], pos: int) -> str | None:
    """The neighbour a number or side word is said of: the last one named before it, else the next."""
    near = [t for p, t in named if p < pos][-1:] or [t for p, t in named if p > pos][:1]
    return near[0] if near else None


def _gaps(c: _Call, sent: str, skip: set[str]) -> list[str]:
    errs = []
    for seg in _SEG.split(sent):
        named = [(m.start(), m.group()) for m in _TLA.finditer(seg) if m.group() in c.side]
        # a neighbour with no known TLA is written as its car number ("44 is 1.2 seconds ahead"): never checked, but named
        bare = any(re.search(rf"(?<![\w.]){re.escape(t)}(?!\w|\.\d)", seg) for t in c.loose)
        for m in _AMOUNT.finditer(seg):
            num = m.group()
            if num in skip:
                continue
            adj = _SIDE_AFTER.match(seg, m.end()) or _SIDE_BEFORE.search(seg, 0, m.start())
            who = _owner(named, m.start())
            if not (adj or _UNIT.match(seg, m.end()) or ("." in num and who)):
                continue  # not a gap: a lap, a position, a percentage, a cost with no driver next to it
            if who is None:
                if adj and not bare:
                    errs.append(f"gap {num} {adj.group(1).lower()} names no driver")
            elif num != c.side[who][1]:
                errs.append(f"gap {num} is not {who}'s")
        for m in _SIDE_WORD.finditer(seg):
            who, word = _owner(named, m.start()), m.group(1).lower()
            if who is not None and c.side[who][0] != ("behind" if word == "behind" else "ahead"):
                errs.append(f"{who} is not {word}")
    return errs


def _laps(c: _Call, sent: str, marks: list[tuple[int, str]], skip: set[str]) -> list[str]:
    errs = []
    for m in _LAP.finditer(sent):
        n, total = m.groups()
        if n in skip or (total and total in skip):
            continue
        if total:  # "lap 21 of 57": where the race is now
            ok = (n, total) == (c.lap, c.total)
        else:
            scope = _scope(marks, m.start())
            stops = {lap for lap, _ in c.plans.get(scope or "A", ())}
            if _BOX_ON.search(sent, 0, m.start()):  # a lap named for a stop: a lap of the plan, or now for a BOX call
                ok = n in stops or (scope is None and c.act == "BOX" and n == c.lap)
            else:
                ok = n in stops or n == c.lap
        if not ok:
            errs.append(f"lap {n} is not the call's lap")
    return errs


def _tyres(c: _Call, sent: str, marks: list[tuple[int, str]], skip: set[str]) -> list[str]:
    errs = []
    laps = [(m.start(), m.group(1)) for m in _LAP.finditer(sent) if not m.group(2) and m.group(1) not in skip]
    for m in _NEW_TYRE.finditer(sent):
        x = _canon(next(g for g in m.groups() if g))
        if x in skip:
            continue
        prior = [(p, n) for p, n in laps if p < m.start()]
        if prior:  # "lap 22 for hards": the tyre of that stop; a plan named in the sentence must hold that very stop
            p, n = prior[-1]
            scope = _scope(marks, p)
            ok = (n, x) in c.plans.get(scope or "A", ()) or (scope is None and x == c.fit)
        else:
            ok = x == c.fit
        if not ok:
            errs.append(f"compound {x} is not the call's tyre")
    return errs


def relation_errors(inp: str, out: str, skip: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """What the output says of a driver, lap or tyre that the input does not say of it.

    * a gap next to a driver is that driver's gap, as the input writes it; a gap with a side ("2.1 ahead") names the driver;
    * "ahead" / "behind" next to a driver is his side (AH / BH);
    * "box on lap N", "stops on lap N", "lap N": N is a lap of the plan being talked about, or (not for a stop) this lap;
    * "for X", "onto X", "X going on": X is the call's tyre, or the tyre of the plan stop named just before it.

    ``skip``: numbers and compounds the vocabulary check already rejected, so they are not reported twice. Text the
    input itself carries (trigger, reason texts) is not checked when the output repeats it. Rough on purpose: it may
    reject a correct sentence (the caller then serves the template), it must not pass a wrong one."""
    c, skip = _parse(inp), set(skip)
    errs: list[str] = []
    for sent in _SENT.split(_unquote(out, c.quoted)):
        marks = [(m.start(), m.group(1).upper()) for m in _PLAN.finditer(sent)]
        errs += _gaps(c, sent, skip) + _laps(c, sent, marks, skip) + _tyres(c, sent, marks, skip)
    return list(dict.fromkeys(errs))


def fact_errors(inp: str, out: str) -> list[str]:
    """Things the output says that the input does not hold: words and numbers it lacks, and things said of the wrong driver, lap or tyre."""
    nums, tlas, comps = input_vocab(inp)
    bad = [n for n in _NUM.findall(out) if n not in nums]
    errs = [f"number {n}" for n in bad]
    errs += [f"name {t}" for t in _TLA.findall(out) if t not in tlas and t not in _OK_CAPS]
    unknown = [_canon(m.group(1)) for m in _CMP.finditer(out) if _canon(m.group(1)) not in comps]
    errs += [f"compound {x}" for x in unknown]
    errs += [f"number word {m.group(0)}" for m in _NUMWORDS.finditer(out)]
    return errs + relation_errors(inp, out, skip={*bad, *unknown})


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
