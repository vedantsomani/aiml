"""Template composer: the reference voice, and the fallback when the model fails the guard.

A paraphrase grammar. Every choice is picked by one of the six style digits in the
`Facts`, so the same facts and style always give the same words. It states only what
the facts hold: a clause whose value is missing is left out.
"""

from __future__ import annotations

import re

from .facts import Facts

PLURAL = {"SOFT": "softs", "MEDIUM": "mediums", "HARD": "hards", "INTERMEDIATE": "inters", "WET": "wets"}
SING = {"SOFT": "soft", "MEDIUM": "medium", "HARD": "hard", "INTERMEDIATE": "intermediate", "WET": "wet"}


def _pick(f: Facts, k: int, options: list[str]) -> str:
    return options[f.style[k % len(f.style)] % len(options)]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _fmt(s: str, **kw) -> str:
    return s.format(**kw)


def n_words(s: str) -> int:
    return len(s.split())


# ------------------------------------------------------------------ reasons
# code -> (variants with a value, variants without). {v} is the value as given.
REASONS: dict[str, tuple[list[str], list[str]]] = {
    "undercut_gain": (["the undercut gains {v} seconds", "an undercut is worth {v} seconds", "we gain {v} seconds on the cars ahead"], []),
    "tyre_cliff": (["the cliff risk is {v} percent", "tyre cliff risk is {v} percent", "the tyres could fall off, {v} percent risk"], ["the tyres are close to the cliff"]),
    "must_stop": ([], ["we still need a second compound", "the rules need another compound from us", "we must still stop for the rules"]),
    "rejoin_if_box_now": (["boxing now rejoins P{v}", "a stop now puts us P{v}", "we would rejoin in P{v}"], []),
    "sc_window": (["a stop under the safety car costs {v} seconds", "the safety car makes the stop cheap at {v} seconds"], ["the safety car makes the stop cheap"]),
    "penalty_serve": (["there is a {v} second penalty to serve", "we have {v} seconds of penalty to serve"], ["there is a penalty to serve"]),
    "rain_onset": (["rain chance is {v} percent", "rain is {v} percent likely soon"], ["rain is coming"]),
    "undercut_threat": (["the undercut threat is {v} percent", "the car behind can undercut, {v} percent"], ["the car behind threatens the undercut"]),
    "clean_air": ([], ["we are in clean air", "clean air is worth staying in", "track position is good here"]),
    "tyre_life": (["the tyres have {v} laps on them", "the set is {v} laps old"], ["the tyres are worn"]),
    "pit_loss_high": (["the stop costs {v} seconds", "a stop costs {v} seconds now"], ["the stop is expensive now"]),
    "rival_stopped": (["car {v} has stopped", "car {v} pitted"], ["a rival has stopped"]),
}


def reason_clause(f: Facts, i: int, k: int = 3) -> str | None:
    if i >= len(f.reasons):
        return None
    code, val, txt = f.reasons[i]
    if code == "other":
        return txt or None
    with_v, no_v = REASONS[code]
    if val is not None and with_v:
        return _fmt(_pick(f, k + i, with_v), v=val)
    if no_v:
        return _pick(f, k + i, no_v)
    return None


# ------------------------------------------------------------------ action phrases
def action_phrase(f: Facts, k: int = 1, long: bool = False) -> str:
    fit = PLURAL.get(f.fit or "")
    a = f.action
    n = f.plan_in
    if a == "BOX":
        opts = ["box, box, box", "box this lap", "pit this lap", "come in at the end of this lap", "we box now", "in this lap"]
        s = _pick(f, k, opts)
        if fit:
            s += _pick(f, k + 1, [f" for {fit}", f", {fit} going on", f", {fit} are ready", f" for {fit}", f", {fit} next", f" and take {fit}"])
        return s
    if a == "STAY_OUT":
        return _pick(f, k, ["stay out", "stay out, stay out", "keep going, no stop yet", "hold position, stay out", "stay on track, no stop this lap", "remain out for now"])
    if a == "PREPARE_BOX":
        s = _pick(f, k, ["prepare to box", "get ready to box", "stand by to box", "be ready to box", "expect the box call", "prepare to box soon"])
        if n is not None and n > 0 and _pick(f, k + 2, ["y", "n"]) == "y":
            s += f" in {n} laps"
        if fit:
            s += _pick(f, k + 1, [f" for {fit}", f", {fit} ready", f" for {fit}", f", {fit} next", f", plan on {fit}", f" for {fit}"])
        return s
    if a == "BOX_IF_SC":
        s = _pick(f, k, ["box if the safety car comes out", "if we get a safety car, box", "box on a safety car", "if the safety car is called, box", "safety car, then box", "box if safety car"])
        if fit:
            s += _pick(f, k + 1, [f" for {fit}", f", {fit} ready", f" for {fit}", f", {fit} next", f" and take {fit}", f" for {fit}"])
        return s
    return _pick(f, k, ["no call right now", "no strategy call for now", "nothing to call, keep going", "no call yet, stay focused", "no call, we keep watching", "no call at the moment"])


# ------------------------------------------------------------------ radio
def radio(f: Facts) -> str:
    who = f.tla
    lead = _pick(f, 0, [f"{who},", f"Copy {who},", f"{who}, listen,", f"OK {who},", f"Radio for {who},", f"{who}, strategy,"])
    act = action_phrase(f, 1)
    clause = reason_clause(f, 0) if f.action != "NO_CALL" else None
    seps = [", {r}", " - {r}", ", because {r}", ". {R}", ", {r}", " as {r}"]
    forms = []
    if clause:
        sep = _pick(f, 3, seps)
        forms.append(f"{lead} {act}" + sep.format(r=clause, R=_cap(clause)))
    forms += [f"{lead} {act}", f"{act}"]
    for t in forms:
        t = _cap(t.strip().rstrip(",")) + ("" if t.rstrip().endswith(".") else ".")
        if n_words(t) <= 20:
            return t
    return _cap(act) + "."


# ------------------------------------------------------------------ brief
def _tyre(f: Facts) -> str:
    c = SING.get(f.cmp or "")
    if not c:
        return ""
    return f"{f.age} lap old {c}s" if f.age is not None else f"{c}s"


def plan_text(f: Facts, plan, name: str) -> str | None:
    if plan is None:
        return None
    if not plan.stops:
        return f"Plan {name} is to run to the flag with no more stops"
    stops = [f"lap {lap} for {PLURAL.get(c, c.lower())}" for lap, c in plan.stops]
    s = f"Plan {name} stops on " + ", then on ".join(stops)
    if plan.pos is not None:
        s += f", finishing around P{plan.pos}"
    return s


def brief(f: Facts) -> str:
    who = f.tla
    sents: list[str] = []
    # 1: where we are
    bits = []
    if f.pos is not None:
        bits.append(f"P{f.pos}")
    ly = f"lap {f.lap} of {f.total}" if f.lap is not None and f.total is not None else (f"lap {f.lap}" if f.lap is not None else "")
    ty = _tyre(f)
    base = _pick(f, 0, [
        "{w} is {p} on {l}" if bits and ly else "{w} is running on {l}",
        "On {l}, {w} sits {p}" if bits and ly else "{w} is on track on {l}",
        "{w} holds {p} at {l}" if bits and ly else "{w} is out on {l}",
        "{w} is {p}, {l}" if bits and ly else "{w}, {l}",
        "At {l}, {w} is {p}" if bits and ly else "{w} at {l}",
        "{w}: {p} on {l}" if bits and ly else "{w} is on {l}",
    ])
    if ly or bits:
        s1 = base.format(w=who, p=bits[0] if bits else "", l=ly or "this lap")
    else:
        s1 = f"{who} is on track"
    if ty:
        s1 += _pick(f, 1, [f", running {ty}", f" on {ty}", f", on {ty}", f" with {ty}", f", using {ty}", f", currently on {ty}"])
    sents.append(s1)
    # 2: the call
    act = action_phrase(f, 2)
    clause = reason_clause(f, 0, 4) if f.action != "NO_CALL" else None
    lead = _pick(f, 3, ["Our call: ", "The call: ", "We say: ", "Strategy says: ", "Call: ", "Decision: "])
    s2 = _cap(act) if f.action == "NO_CALL" else lead + act
    if clause:
        s2 += _pick(f, 4, [f", because {clause}", f" since {clause}", f" as {clause}", f", {clause}", f", the reason being that {clause}", f", mainly because {clause}"])
    sents.append(s2)
    # 3: neighbours, loss and rejoin
    n3 = []
    nbr = []
    if f.ahead and f.ahead.gap:
        nbr.append(f"{f.ahead.tla} is {f.ahead.gap} seconds ahead")
    if f.behind and f.behind.gap:
        nbr.append(f"{f.behind.tla} is {f.behind.gap} seconds behind")
    if nbr:
        n3.append(" and ".join(nbr))
    if f.loss:
        n3.append(f"a stop costs {f.loss} seconds" + (f" and rejoins P{f.rejoin}" if f.rejoin is not None else ""))
    elif f.rejoin is not None:
        n3.append(f"a stop now rejoins P{f.rejoin}")
    if n3:
        j = _pick(f, 5, ["; ", ", and ", "; "])
        s3 = _cap(n3[0]) + (j + n3[1] if len(n3) > 1 else "")
        sents.append(s3)
    # 4: plan, penalty, weather
    n4 = []
    pa = plan_text(f, f.plan_a, "A")
    if pa:
        n4.append(pa)
    if f.plan_b and f.plan_b.trigger:
        n4.append(f"Plan B applies {f.plan_b.trigger}")
    elif f.plan_b:
        n4.append(plan_text(f, f.plan_b, "B"))
    if f.pen:
        n4.append(f"{f.pen} seconds of penalty are still to serve")
    if f.dt:
        n4.append("a drive-through is pending")
    if f.rain is not None and f.rain >= 20:
        n4.append(f"rain chance is {f.rain} percent")
    if f.sc and f.sc not in ("none",):
        n4.append({"sc": "the safety car is out", "vsc": "the VSC is active", "sc_ending": "the safety car is ending", "vsc_ending": "the VSC is ending", "red": "the race is red flagged"}.get(f.sc, ""))
    n4 = [x for x in n4 if x]
    if n4:
        k = f.style[5] % 3
        take = n4[: 1 + k % 2] if len(n4) > 1 else n4
        sents.append(_cap(take[0]) + (_pick(f, 2, ["; ", ", and "]) + take[1] if len(take) > 1 else ""))
    if len(sents) > 4:
        sents = sents[:4]
    return " ".join(s.rstrip(".") + "." for s in sents)


# ------------------------------------------------------------------ answers
def _neighbour(f: Facts, target: str):
    for rel, n in (("ahead", f.ahead), ("behind", f.behind)):
        if n and target in (n.car, n.tla):
            return rel, n
    return None, None


def answer(f: Facts, intent: str, target: str | None = None) -> str:
    who = f.tla
    if intent == "gap":
        rel, n = _neighbour(f, target or "")
        if n is None:
            return _pick(f, 0, [f"Car {target} is not next to {who}, so I have no gap.", f"I have no gap between {who} and car {target}.", f"No gap available, car {target} is not adjacent to {who}."])
        if not n.gap:
            return _pick(f, 0, [f"I do not have the gap to {n.tla} yet.", f"The gap to {n.tla} is not known yet.", f"No gap to {n.tla} at the moment."])
        return _pick(f, 0, [
            f"{n.tla} is {n.gap} seconds {rel} of {who}.",
            f"The gap to {n.tla} is {n.gap} seconds, he is {rel}.",
            f"{n.gap} seconds to {n.tla}, {rel}.",
            f"{who} has {n.tla} {n.gap} seconds {rel}.",
            f"{n.tla} is {rel} by {n.gap} seconds.",
            f"Gap to {n.tla}: {n.gap} seconds, {rel}.",
        ])
    if intent == "tyre_age":
        if not f.cmp:
            return f"I have no tyre data for {who}."
        c = SING[f.cmp]
        s = _pick(f, 0, [
            f"{who} is on {f.age} lap old {c}s.",
            f"The {c}s on {who} are {f.age} laps old.",
            f"{who} has {f.age} laps on the {c} set.",
            f"Tyre age is {f.age} laps, {c}s.",
            f"{who}: {c}s, {f.age} laps old.",
            f"The current set is {c}s with {f.age} laps.",
        ])
        if f.stops is not None and _pick(f, 1, ["y", "n", "y"]) == "y":
            s += f" That is after {f.stops} stop" + ("" if f.stops == 1 else "s") + "."
        return s
    if intent == "pit_window":
        if f.action == "BOX":
            return _pick(f, 0, [f"The window is open, {who} boxes this lap" + (f" for {PLURAL[f.fit]}." if f.fit else "."), f"Box this lap, {who}" + (f", {PLURAL[f.fit]} ready." if f.fit else "."), f"It is open now, {who} comes in this lap."])
        if f.plan_a and f.plan_a.stops:
            lap, c = f.plan_a.stops[0]
            n = f.plan_in
            return _pick(f, 0, [
                f"Plan A has {who} stopping on lap {lap} for {PLURAL.get(c, c.lower())}.",
                f"The window is lap {lap}, for {PLURAL.get(c, c.lower())}" + (f", in {n} laps." if n else "."),
                f"We plan the stop on lap {lap}, {PLURAL.get(c, c.lower())} going on.",
                f"{who} pits on lap {lap} under Plan A, taking {PLURAL.get(c, c.lower())}.",
                f"Lap {lap} is the planned stop for {who}.",
                f"Plan A: box on lap {lap} for {PLURAL.get(c, c.lower())}.",
            ])
        if f.plan_a:
            return _pick(f, 0, [f"Plan A has no more stops for {who}.", f"No window, {who} runs to the flag.", f"There is no stop planned for {who}."])
        return _pick(f, 0, [f"There is no pit window set for {who} yet.", f"No window yet for {who}.", f"We have not set a window for {who}."])
    if intent == "plan_b":
        if f.plan_b is None:
            return _pick(f, 0, [f"There is no Plan B for {who} yet.", f"No Plan B is set for {who}.", f"{who} has no Plan B at the moment."])
        stops = plan_text(f, f.plan_b, "B")
        if f.plan_b.trigger:
            return _pick(f, 0, [f"Plan B applies {f.plan_b.trigger}. {stops}.", f"We switch to Plan B {f.plan_b.trigger}. {stops}.", f"Trigger for Plan B: {f.plan_b.trigger}. {stops}."])
        return _pick(f, 0, [f"Plan B has no trigger yet. {stops}.", f"No trigger is set for Plan B. {stops}."])
    if intent == "why":
        act = action_phrase(f, 1)
        cl = [c for c in (reason_clause(f, i, 3) for i in range(min(3, len(f.reasons)))) if c]
        head = _cap(_pick(f, 0, [f"{who}: ", f"For {who}, ", f"{who}, ", f"Call for {who}: ", f"Right now for {who}: ", f"Our call for {who}: "]) + act)
        if not cl:
            return head.rstrip(".") + "."
        join = _pick(f, 4, [" because ", ". Reason: ", " since ", ". The reasons: ", " as ", ". The cause: "])
        body = " and ".join(cl[:2]) if len(cl) > 1 else cl[0]
        if len(cl) > 2:
            body = ", ".join(cl[:2]) + " and " + cl[2]
        return head.rstrip(".") + join + (_cap(body) if join.startswith(".") else body) + "."
    raise ValueError(f"unknown intent {intent!r}")


# ------------------------------------------------------------------ free-form questions
# (regex, kind): the first match decides what is asked. Used as the template reference answer.
FREE_KINDS = (
    ("gap_ahead", r"(gap|how far|distance|interval).*(ahead|in front|front)|car (ahead|in front)|who.*(ahead|in front)"),
    ("gap_behind", r"(gap|how far|distance|interval).*(behind|back)|car behind|who.*behind|who.*chasing"),
    ("tyre", r"tyre|tire|compound|stint|set"),
    ("loss", r"pit loss|stop cost|cost of (a )?stop|lose.*stop|time.*(pit|stop)"),
    ("rejoin", r"rejoin|come out|where.*(box|pit|stop)|position after"),
    ("rain", r"rain|weather|wet|crossover"),
    ("left", r"laps?.*(left|to go|remaining)|how long|remaining"),
    ("position", r"position|where are we|running order|what place"),
    ("penalty", r"penalt|drive.?through"),
    ("cliff", r"cliff|deg|wear|grip"),
    ("undercut", r"undercut|threat"),
    ("plan", r"plan a|plan b|plan|window|when.*(box|pit|stop)|strategy"),
    ("why", r"why|reason|because"),
    ("call", r"call|what.*(do|should)|box|stay"),
)


def free_kind(question: str) -> str | None:
    q = question.lower()
    for kind, rx in FREE_KINDS:
        if re.search(rx, q):
            return kind
    return None


def free(f: Facts, question: str) -> str:
    """Template answer to a free-form question: route by keywords, answer only from the facts."""
    who, kind = f.tla, free_kind(question)
    no = _pick(f, 0, [f"I do not have that for {who}.", f"No data on that for {who} right now.", f"That is not available for {who}."])
    if kind == "gap_ahead":
        return answer(f, "gap", f.ahead.car) if f.ahead else no
    if kind == "gap_behind":
        return answer(f, "gap", f.behind.car) if f.behind else no
    if kind == "tyre":
        return answer(f, "tyre_age") if f.cmp else no
    if kind == "loss":
        return _pick(f, 0, [f"A stop costs {f.loss} seconds now.", f"The pit loss is {f.loss} seconds."]) if f.loss else no
    if kind == "rejoin":
        return _pick(f, 0, [f"A stop now rejoins P{f.rejoin}.", f"{who} would come out P{f.rejoin}."]) if f.rejoin is not None else no
    if kind == "rain":
        return _pick(f, 0, [f"Rain chance is {f.rain} percent.", f"We rate rain at {f.rain} percent."]) if f.rain is not None else no
    if kind == "left":
        return _pick(f, 0, [f"{f.left} laps to go.", f"There are {f.left} laps left."]) if f.left is not None else no
    if kind == "position":
        return _pick(f, 0, [f"{who} is P{f.pos}.", f"We are P{f.pos} on lap {f.lap}."]) if f.pos is not None and f.lap is not None else no
    if kind == "penalty":
        if f.pen:
            return _pick(f, 0, [f"{who} has {f.pen} seconds of penalty to serve.", f"There is a {f.pen} second penalty pending."])
        return "A drive-through is pending." if f.dt else _pick(f, 0, [f"No penalty pending for {who}.", f"{who} has no penalty to serve."])
    if kind == "cliff":
        if f.cliff is None:
            return no
        return f"The cliff risk is {f.cliff} percent" + (f" and the tyres lose {f.deg} seconds a lap." if f.deg else ".")
    if kind == "undercut":
        return _pick(f, 0, [f"The undercut threat is {f.uc} percent.", f"The car behind has a {f.uc} percent undercut chance."]) if f.uc is not None else no
    if kind == "plan":
        return answer(f, "pit_window") if "plan b" not in question.lower() else answer(f, "plan_b")
    if kind == "why":
        return answer(f, "why")
    if kind == "call":
        return _cap(action_phrase(f, 1)) + "."
    return no


def compose(f: Facts, task: str, target: str | None = None) -> str:
    if task == "free":
        return free(f, target or "")
    if task == "radio":
        return radio(f)
    if task == "brief":
        return brief(f)
    if task.startswith("ask_"):
        return answer(f, task[4:], target)
    raise ValueError(f"unknown task {task!r}")


# ------------------------------------------------------------------ what a text says (used by the guard)
_SC = r"(?:safety car|sc|vsc)"
_STATE_RES = (
    ("NO_CALL", re.compile(r"\bno (?:strategy )?call\b|\bnothing to call\b", re.I)),
    ("BOX_IF_SC", re.compile(rf"\bbox (?:if|on) (?:the |a )?{_SC}\b|\bif (?:we get a |the )?{_SC}\b[^,.]*, box\b|\b{_SC}, then box\b", re.I)),
    ("PREPARE_BOX", re.compile(r"\b(?:prepare to|get ready to|stand by to|be ready to|expect the) box\b", re.I)),
    ("STAY_OUT", re.compile(r"\bstay out\b|\bkeep going, no stop\b|\bstay on track, no stop\b|\bremain out\b|\bhold position\b", re.I)),
    ("BOX", re.compile(r"\bbox, box\b|\bbox this lap\b|\bpit this lap\b|\bcome in at the end\b|\bwe box now\b|\bin this lap\b|\bboxes this lap\b|\bcomes in this lap\b|\bbox on lap\b|\bbox now\b|\bstopping on lap\b", re.I)),
)


def stated_action(text: str) -> str | None:
    """The action a radio/brief/why text states (first match in a fixed precedence), or None."""
    for name, rx in _STATE_RES:
        if rx.search(text):
            return name
    return None
