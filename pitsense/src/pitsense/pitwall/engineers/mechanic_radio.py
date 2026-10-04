"""Mechanic (radio): problem reports in the driver's radio transcript.

Reads ``state.feeds.radio.messages(car)``. A transcript exists 15 s after the message (the
store gives ``text=None`` before that), so the answer at ``t`` never depends on later audio.

The model is a phrase table, not a classifier. Each pattern names an issue (``power``,
``brakes``, ``gearbox``, ``damage``, ``puncture``, ``vibration``, ``leak_fire``, ``electrical``,
``retire``, ``generic``) and a weight (how often that wording is a real fault, set by reading
the 2025 transcripts). A hit is discarded when it is negated or hypothetical ("no problem",
"not a problem", "nothing wrong", "no damage", "if there is damage"), halved when the sentence
is a question (the engineer asking), and halved when it is about a fix ("we fixed ..."). Hits
in one message combine as 1 - prod(1 - w), different issues in one message add up. Routine
settings talk ("brake balance", "engine mode", "power mode", "max power") never matches
because every pattern needs a fault word (lost, no, losing, failing, problem, broken, ...).

Values per car: ``radio_issue`` (str, "" when none), ``radio_risk`` (0-1; the message score,
decaying with a 15 min half-life), ``radio_score`` (undecayed), ``radio_quote``,
``radio_t`` (message time), ``radio_lap`` (the car's lap then), ``radio_n`` (problem messages so far).
Docs: docs/engineers/mechanics.md.
"""

from __future__ import annotations

import math
import re

from ..engineer import Engineer
from ..types import Alert

HALF_LIFE_S = 900.0
ALERT_RISK = 0.5

_FAULT = r"(?:lost|lose|losing|loss|no|not (?:working|enough)|failing|failure|failed|fail|broken|gone|dead|cut(?:ting)? out|stuck|weak|bad|problem|issue|wrong|damage|leak\w*|shut(?:ting)? down|inconsistent|long|flat|soft|spongy|critical)"

# (issue, weight, pattern). Patterns run on lower-case text with punctuation kept.
_PATTERNS: list[tuple[str, float, re.Pattern]] = [
    (i, w, re.compile(p)) for i, w, p in [
        # ---- power / engine
        ("power", 0.9, r"\b(?:lost|lose|losing|loss of|no|not (?:enough|got)|don't have (?:any |enough )?|have no|got no|dropped?) (?:a (?:lot|bit|quite a bit) of )?power\b"),
        ("power", 0.9, r"\bpower (?:loss|drop|cut|is gone|has gone)\b"),
        ("power", 0.85, r"\bengine (?:is |has )?(?:failing|failure|failed|dead|dying|blown|gone|cut|cutting|misfir\w+|losing|lost|issue|problem|not|really not)\b"),
        ("power", 0.85, r"\b(?:engine|pu|power unit|mgu\w*|turbo) (?:issue|problem|failure|trouble)\b"),
        ("power", 0.7, r"\b(?:issue|problem|trouble) (?:with|on) the (?:pu|engine|power unit|turbo)\b"),
        ("power", 0.8, r"\breliability (?:problem|issue)\b"),
        ("power", 0.85, r"\b(?:no|lost|losing|lose) (?:drive|acceleration|boost|torque|revs)\b"),
        ("power", 0.6, r"\b(?:car|it) (?:is|feels|'s) (?:so |really |very )?(?:slow|down on power)\b(?!ly)"),
        ("power", 0.5, r"\bsomething (?:is |'s )?(?:wrong|off) with the (?:pu|engine|power)\b"),
        ("power", 0.5, r"\bnot (?:getting|giving) (?:any |the )?power\b"),
        # ---- brakes
        ("brakes", 0.9, r"\b(?:can't|cannot|can not|couldn't|won't|unable to) brake\b"),
        ("brakes", 0.9, r"\b(?:no|lost|losing|lose|without) brakes?\b(?! balance| bias)"),
        ("brakes", 0.9, r"\bbrake (?:failure|fail|failing|issue|problem)\b"),
        ("brakes", 0.8, r"\bbrakes? (?:are |is |has |have |feel |feels |been |getting |going |gone )+(?:so |really |very |quite |a lot |getting |going |bit )*(?:bad|inconsistent|long|gone|weak|soft|spongy|critical|failing|fading|gone|worse|flat|low|odd|strange|funny)\b"),
        ("brakes", 0.8, r"\bbrake pedal(?:'s| is| has| was| gets| getting| going| went)? (?:going |getting |gone |a bit |quite |so |very |really )*(?:long|longer|soft|flat|to the floor|gone|weak|spongy|low)\b"),
        ("brakes", 0.6, r"\bbrake (?:disc|duct|pad)s? .{0,20}\b(?:gone|failed|problem|issue|broken|damaged|worn)\b"),
        ("brakes", 0.6, r"\bi (?:do )?need brakes\b"),
        # ---- gearbox
        ("gearbox", 0.9, r"\bstuck in (?:first|second|third|fourth|fifth|sixth|seventh|eighth|neutral|\w+ gear|gear)\b"),
        ("gearbox", 0.9, r"\bgearbox (?:issue|problem|failure|is|has|broken|stuck|failing)\b"),
        ("gearbox", 0.9, r"\b(?:something(?:'s| is)? (?:broken|wrong) (?:in|with) (?:my |the )?gearbox)\b"),
        ("gearbox", 0.85, r"\b(?:can't|cannot|won't|not|no) (?:shift|change gear|upshift|downshift|select (?:a )?gear)s?\b"),
        ("gearbox", 0.8, r"\b(?:lost|missing|no|losing) (?:a )?gears?\b"),
        ("gearbox", 0.7, r"\b(?:gear|shift|downshift|upshift|clutch) (?:problem|issue|failure|fault|not working|is stuck)\b"),
        ("gearbox", 0.6, r"\bvibrations? in the gearbox\b"),
        ("gearbox", 0.5, r"\bclutch\b.{0,15}\b(?:problem|issue|slipping|gone|failed|bad)\b"),
        # ---- damage / structural
        ("damage", 0.7, r"\b(?:significant|heavy|big|some|bit of|a lot of|lot of|serious|front|rear|floor|wing|car|bodywork) damage\b"),
        ("damage", 0.7, r"\b(?:have|got|has|with|took|see|see the|taking|suffered) (?:a )?damage\b"),
        ("damage", 0.7, r"\b(?:front wing|rear wing|floor|wing|suspension|wheel|nose|diffuser|brake duct|sidepod|engine cover)s? (?:is |are |has |have |got |looks? |was |'s )?(?:completely |totally |really |very |quite )?(?:broken|damaged|gone|bent|cracked|hanging|loose|missing)\b"),
        ("damage", 0.7, r"\b(?:car|it) (?:is|'s|feels|has been) (?:completely |totally |really |very )?broken\b"),
        ("damage", 0.7, r"\bsomething (?:fell|broke|came|has come|is hanging|is loose|has fallen|just fell|snapped)\b"),
        ("damage", 0.7, r"\bsuspension is broken\b"),
        ("damage", 0.6, r"\b(?:lost|lose|losing) (?:a |the |part of the |my )?(?:front wing|rear wing|wing|floor|mirror|endplate|bargeboard|wheel|nose|part)\b"),
        ("damage", 0.5, r"\bdamage (?:on|to|in) (?:the |my )?(?:car|front|rear|floor|wing|tyre|tire)\b"),
        # ---- puncture
        ("puncture", 0.9, r"\b(?:puncture|punctured|deflat\w+|blown tyre|blown tire|blow out|blowout)\b"),
        ("puncture", 0.8, r"\b(?:slow|flat) (?:puncture|tyre|tire)\b"),
        ("puncture", 0.7, r"\b(?:losing|lost) (?:air|pressure|tyre pressure)\b"),
        ("puncture", 0.6, r"\b(?:tyre|tire)s? (?:is|are) (?:flat|deflating|losing)\b"),
        # ---- vibration / noise
        ("vibration", 0.7, r"\b(?:big|huge|massive|heavy|strong|serious|lots of|some|bad|severe)? ?vibrations?\b(?! from the pit)"),
        ("vibration", 0.7, r"\b(?:shaking|vibrating|wobbl\w+|shudder\w*)\b"),
        ("vibration", 0.5, r"\b(?:strange|weird|funny|odd|metallic|rattling|grinding|banging|bad) (?:noise|sound)\b"),
        ("vibration", 0.5, r"\b(?:noise|sound) (?:from|in) (?:the |my )?(?:rear|front|back|engine|gearbox|car)\b"),
        # ---- leaks, smoke, fire, heat
        ("leak_fire", 0.95, r"\b(?:smoke|smoking|flames?)\b"),
        ("leak_fire", 0.9, r"\b(?:engine|gearbox|brakes?|rear|cockpit|pu) (?:is |are |'s )?(?:on )?fire\b|\bfire (?:in|from) the (?:engine|car|cockpit|rear|back)\b"),
        ("leak_fire", 0.25, r"\bfire\b(?! in the air)"),
        ("leak_fire", 0.9, r"\boil leak\b|\bleak(?:ing)? (?:oil|water|fluid|fuel)\b"),
        ("leak_fire", 0.6, r"\b(?:engine|brakes?|gearbox|pu|battery|water|oil|cooling)\b.{0,20}\b(?:overheat\w*|over-heat\w*)|\b(?:overheat\w*|over-heat\w*)\b.{0,20}\b(?:engine|brakes?|gearbox|pu|battery|water|oil)\b|\bwater temp\w* (?:high|rising)|\bcooling (?:problem|issue)"),
        ("leak_fire", 0.3, r"\boverheat\w*|over-heat\w*"),
        ("leak_fire", 0.6, r"\bsmell (?:something|burning|smoke)\b"),
        # ---- electrical / hydraulics / steering
        ("electrical", 0.7, r"\b(?:losing|lost|lose|no|failed) (?:the )?(?:power steering|steering|hydraulics?|dash|display|radio|electrics?)\b"),
        ("electrical", 0.6, r"\b(?:battery|ers|mgu-?k|harvest\w*|deployment|hybrid)\b.{0,25}\b(?:problem|issue|failure|failing|dead|not working|lost|gone|empty|joke|broken)\b"),
        ("electrical", 0.6, r"\b(?:steering|hydraulics?|electrics?|dash(?:board)?|differential|diff) (?:problem|issue|failure|failing|fault|not working|lost|gone)\b"),
        # ---- retirement language
        ("retire", 0.95, r"\b(?:retire the car|stop the car|park the car|park it|box this lap.{0,25}retire|this is to retire|we(?:'ll| will| are| 're) retir\w+|i'm out|i am out|pull over|switch (?:the )?(?:car|engine) off|shut (?:it|the car|the engine) down)\b"),
        ("retire", 0.8, r"\b(?:retire|retirement)\b.{0,12}\bcar\b|\bcar\b.{0,12}\bretire\w*"),
        # ---- generic
        ("generic", 0.6, r"\bsomething(?:'s| is| has)? (?:wrong|broken|off|not right|not good|failing|happening)\b"),
        ("generic", 0.5, r"\b(?:i've|i have|i got|we have|we've got|have got|got) (?:a |an |another |some |big |serious )?(?:problem|issue|failure)\b"),
        ("generic", 0.5, r"\b(?:there is|there's|we have|we see|we can see|we've got|we are monitoring|monitoring) (?:a |an |another |the )?(?:problem|issue|failure|fault)\b"),
        ("generic", 0.45, r"\b(?:problem|issue) (?:with|on|in) (?:the |my )?(?:car|rear|front|left|right|engine|pu|brakes?|gearbox|floor|steering|wing)\b"),
        ("generic", 0.7, r"\bbox,? box,?.{0,40}\b(?:problem|issue|damage|reliab\w+|puncture|broken|vibration|power|brake)\w*"),
        ("generic", 0.7, r"\b(?:problem|issue|damage|reliab\w+|puncture|broken|vibration)\w*.{0,40}\bbox,? box\b"),
        ("generic", 0.5, r"\b(?:we need to|we have to|we must|let's|you need to) box this lap\b.{0,40}\b(?:problem|issue|reliab\w+|damage)"),
        ("generic", 0.6, r"\bit(?:'s| is) (?:completely |totally )?broken\b"),
        ("generic", 0.4, r"\b(?:the )?(?:car|it) (?:is|feels|has been) (?:not |really |so )?(?:undrivable|undriveable|impossible to drive|not drivable)\b"),
    ]
]

# a negation or hypothetical in the words before the hit voids it (the pattern's own "no"/"lost"
# is inside the match, so only the text *before* it is scanned)
_NEG_BEFORE = re.compile(
    r"(?:\bno (?:longer )?(?:more )?\w{0,12}\s*$)|(?:\bnot (?:a |an |any |the )?\s*$)|(?:\bn't\s*\w*\s*$)|(?:\bnothing\s*$)|(?:\bnever\s*$)"
    r"|(?:\bwithout\s*$)|(?:\bif (?:there(?:'s| is| was| are)?|you|we|it|i|the)?\s*(?:have |has |had |get |got |see |saw )?\s*(?:a |an |any |some )?\s*$)"
    r"|(?:\bin case (?:of )?\s*$)|(?:\bany\s*$)|(?:\bwhat if\s*$)|(?:\bavoid(?:ing)?\s*$)|(?:\bwithout any\s*$)"
)
# the hit is itself a "no ... problem": the whole phrase is denied
_NEG_AFTER = re.compile(r"^\s*(?:anymore|any more|now|at the moment|here|for now|so far|today|at all)\b")
_DENIED = re.compile(
    r"\b(?:no|not a|not an|nothing|isn't a|isn't an|aren't|wasn't a|never a|without a|zero|any) (?:big |major |real |serious )?(?:problem|issue|damage|failure|fault|vibration|smoke|fire|leak)s?\b"
    r"|\bnothing (?:wrong|broken|to worry)\b|\bnot (?:broken|damaged)\b|\bis (?:fine|ok|okay|good)\b|\bare (?:fine|ok|okay|good)\b|\bno issues?\b|\ball good\b"
)
_FIXED = re.compile(r"\b(?:we (?:think |have |'ve )?fixed|fixed|resolved|sorted|solved|back to normal|all fixed)\b")
_QUESTION_START = re.compile(r"^(?:do you|did you|is there|is the|are you|are there|any |can you|could you|have you|how(?:'s| is| are)|what(?:'s| is)|why|how big)")
_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def classify(text: str | None) -> tuple[str, float, str]:
    """(issue, score, quote) for one transcript. issue "" and score 0 when nothing credible."""
    if not text:
        return "", 0.0, ""
    best: dict[str, tuple[float, str]] = {}
    for sent in _SPLIT.split(text.strip()):
        low = sent.lower().replace("’", "'")
        if len(low) < 4:
            continue
        question = low.rstrip().endswith("?") or bool(_QUESTION_START.match(low.strip()))
        fixed = bool(_FIXED.search(low))
        for issue, w, pat in _PATTERNS:
            for m in pat.finditer(low):
                before = low[max(0, m.start() - 30):m.start()]
                if _NEG_BEFORE.search(before):
                    continue
                span = low[max(0, m.start() - 12):m.end() + 14]
                if _DENIED.search(span) and not re.search(r"\b(?:lost|losing|can't|cannot|no (?:power|brakes?|drive|gears?))\b", m.group(0)):
                    continue
                s = w * (0.4 if question else 1.0) * (0.5 if fixed else 1.0)
                if s > best.get(issue, (0.0, ""))[0]:
                    best[issue] = (s, sent.strip())
    if not best:
        return "", 0.0, ""
    top = max(best, key=lambda k: best[k][0])
    prod = 1.0
    for s, _ in best.values():
        prod *= 1.0 - s
    score = min(0.97, max(best[top][0], 1.0 - prod))
    quote = best[top][1]
    return top, round(score, 3), quote[:160]


class MechanicRadio(Engineer):
    name = "mechanic_radio"
    requires = ()
    features = ()
    in_bench = False

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self._cache: dict[str, tuple[str, float, str]] = {}

    def _scored(self, state, car: str) -> list[tuple[float, float, str, float, str, str]]:
        """Problem messages of one car: (t, known_at, issue, score, quote, path)."""
        out = []
        for m in state.feeds.radio.messages(car):
            if m.text is None:
                continue
            c = self._cache.get(m.path)
            if c is None:
                c = self._cache[m.path] = classify(m.text)
            if c[1] >= 0.3:
                out.append((m.t, m.known_at, c[0], c[1], c[2], m.path))
        return out

    def _lap_at(self, car: str, t: float) -> int | None:
        laps = self.memory.index.by_driver.get(car, {})
        for lap in sorted(laps):
            rec = laps[lap]
            if rec.t_end >= t:
                return lap
        return (max(laps) + 1) if laps else None

    def car(self, state, number, view):
        out = {"radio_issue": "", "radio_risk": 0.0, "radio_score": 0.0, "radio_quote": "", "radio_t": None,
               "radio_lap": None, "radio_n": 0, "radio_since": None}
        if state.feeds.radio.now <= 0:
            return out
        hits = self._scored(state, number)
        out["radio_n"] = len(hits)
        best = None
        for t, known, issue, score, quote, _ in hits:
            risk = score * math.exp(-math.log(2) * max(0.0, state.t - t) / HALF_LIFE_S)
            if best is None or risk > best[0]:
                best = (risk, t, known, issue, score, quote)
        if best is not None:
            risk, t, known, issue, score, quote = best
            out.update(radio_risk=round(risk, 3), radio_score=score, radio_quote=quote, radio_t=round(t, 1),
                       radio_lap=self._lap_at(number, t), radio_since=round(known, 1),
                       radio_issue=issue if risk >= 0.3 else "")
        return out

    def alerts(self, state, view):
        out = []
        for n in sorted(state.drivers, key=lambda x: int(x) if x.isdigit() else 999):
            v = view.car(self.name, n)
            if v["radio_risk"] >= ALERT_RISK and v["radio_issue"]:
                d = state.drivers[n]
                lap = v["radio_lap"]
                out.append(Alert(
                    state.t, self.name, f"radio_{v['radio_issue']}", "critical" if v["radio_risk"] >= 0.85 else "warn",
                    f"Car {n}{f' ({d.tla})' if d.tla else ''}: driver radio reports {v['radio_issue'].replace('_', ' ')}"
                    f" - \"{v['radio_quote']}\"" + (f" (lap {lap})" if lap else ""),
                    car=n, since=v["radio_since"], data={"risk": v["radio_risk"], "issue": v["radio_issue"]}))
        return out
