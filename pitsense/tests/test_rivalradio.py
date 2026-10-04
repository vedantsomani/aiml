"""Rival radio on synthetic transcripts (no timing data or audio in the repo)."""

from __future__ import annotations

import pytest

from pitsense.feeds import RadioStore
from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, PitWall
from pitsense.pitwall.engineers.rivalradio import RivalRadio, classify
from pitsense.pitwall.types import TeamConfig
from pitsense.state import RaceState

from .conftest import make_log


def top(text: str, intent: str) -> float:
    return classify(text).get(intent, (0.0, ""))[0]


@pytest.mark.parametrize("text,intent", [
    ("Box box box.", "box"),
    ("Box this lap for mediums.", "box"),
    ("We're going to box next lap.", "box"),
    ("Stay out, stay out.", "extend"),
    ("We're going to extend.", "extend"),
    ("Don't box, we stay out.", "extend"),
    ("Plan B, Plan B.", "planchange"),
    ("Box opposite, box opposite.", "planchange"),
    ("Strategy C is on.", "planchange"),
    ("My tyres are gone.", "tyres_gone"),
    ("No grip at the rear, graining.", "tyres_gone"),
    ("Lift and coast into turn 1.", "save"),
    ("Fuel save from here.", "save"),
    ("Max, push now.", "push"),
    ("They've pitted, we need to cover.", "cover"),
    ("Undercut is possible.", "cover"),
    ("Rain in five minutes, drops at turn 4.", "weather"),
    ("I'm losing power.", "problem"),
])
def test_intents(text, intent):
    assert top(text, intent) >= 0.6, classify(text)


def test_negation_flips_or_voids():
    assert top("Don't box.", "box") == 0.0 and top("Don't box.", "extend") >= 0.6
    assert top("Do not box this lap.", "box") == 0.0
    assert top("The tyres are not gone yet.", "tyres_gone") == 0.0
    assert top("No need to push.", "push") == 0.0 and top("No need to push.", "save") > 0
    assert top("We can't stay out.", "extend") == 0.0


def test_questions_and_hypotheticals_are_weak():
    assert top("Box this lap.", "box") >= 0.9
    assert top("Should we box?", "box") < 0.4
    assert top("Do you want to box this lap?", "box") < 0.4
    assert top("Why didn't we box?", "box") == 0.0
    assert top("If we box now we lose track position.", "box") < 0.4
    assert top("In case of a safety car we would box.", "box") < 0.4
    assert top("A lot of this over is red box.", "box") == 0.0
    assert classify("") == {} and classify(None) == {}


def _state(texts: dict[float, tuple[str, str]], now: float, base: RaceState | None = None):
    """A state whose radio store holds transcripts (car, text) at the given publish times."""
    st = base or RaceState({})
    st.feeds.radio = RadioStore(None, latency_s=15.0)
    for t, (car, _) in texts.items():
        st.feeds.radio.add(t, {"Captures": [{"Utc": "2025-01-01T00:00:00Z", "RacingNumber": car, "Path": f"TeamRadio/{car}_{int(t)}.mp3"}]})
    by_path = {f"TeamRadio/{c}_{int(t)}.mp3": tx for t, (c, tx) in texts.items()}
    st.feeds.radio.transcripts.get = by_path.get
    st.feeds.radio.now = now
    st.t = now
    return st


def _eng():
    ctx = Context(prior=PitLossPrior())
    return RivalRadio(ctx, PitWall(ctx, engineers=[]).memory)


def test_as_of_transcript_known_after_latency():
    texts = {500.0: ("16", "Box box box.")}
    eng = _eng()
    early = eng.car(_state(texts, 510.0), "16", None)
    assert early == RivalRadio.sentinels()
    v = eng.car(_state(texts, 520.0), "16", None)
    assert v["rr_box_intent"] > 0.7 and v["rr_last_t"] == 500.0 and v["rr_last_quote"] == "Box box box."
    assert v["rr_n"] == 1 and 0 <= v["rr_age_s"] <= 30
    # decays, and is gone long after
    mid = eng.car(_state(texts, 700.0), "16", None)["rr_box_intent"]
    assert 0 < mid < v["rr_box_intent"] / 2
    assert eng.car(_state(texts, 500.0 + 2000), "16", None)["rr_box_intent"] == 0.0
    # another car's clip is not this car's
    assert eng.car(_state(texts, 520.0), "44", None) == RivalRadio.sentinels()


def test_later_message_supersedes_and_stay_out_cancels_box():
    texts = {500.0: ("16", "Box this lap."), 560.0: ("16", "No, stay out, stay out.")}
    v = _eng().car(_state(texts, 600.0), "16", None)
    assert v["rr_box_intent"] == 0.0 and v["rr_extend"] > 0.7
    texts = {500.0: ("16", "Stay out."), 560.0: ("16", "Box box.")}
    v = _eng().car(_state(texts, 600.0), "16", None)
    assert v["rr_box_intent"] > 0.7 and v["rr_extend"] == 0.0


def test_values_are_scalars_and_features_are_numeric():
    v = _eng().car(_state({500.0: ("16", "Plan B. Tyres are gone.")}, 600.0), "16", None)
    assert v["rr_planchange"] > 0.5 and v["rr_tyres_gone"] > 0.7
    for k in RivalRadio.features:
        assert isinstance(v[k], (int, float)) and v[k] == v[k], k  # never NaN
    assert set(RivalRadio.features) <= set(v)
    # no radio feed at all: every key is its sentinel
    assert _eng().car(RaceState({}), "16", None) == RivalRadio.sentinels()


def test_alert_for_the_focus_team_and_spent_by_a_pit_stop():
    log = make_log()
    st = RaceState(log.meta)
    for e in log.events:
        if e.t <= 190.0:
            st.apply(e)
    ctx = Context(prior=PitLossPrior(), team=TeamConfig(cars=("11",)))
    wall = PitWall(ctx, engineers=[RivalRadio])
    wall.observe(st)
    t = st.t
    _state({t - 40: ("33", "Box box, box this lap.")}, t, base=st)
    alerts = wall.engineer("rivalradio").alerts(st, wall.view(st))
    assert [a.car for a in alerts] == ["33"] and alerts[0].code == "rival_box"
    assert "CCC's engineer" in alerts[0].message and "box this lap" in alerts[0].message.lower() and "you" in alerts[0].message
    assert alerts[0].since == t - 40 + 15.0
    # the focus car's own radio is not an alert; a car that already pitted (22 pitted in lap 3) is spent
    wall2 = PitWall(ctx, engineers=[RivalRadio])
    st2 = RaceState(log.meta)
    for e in log.events:
        if e.t <= 600.0:
            st2.apply(e)
    wall2.observe(st2)
    if any(p.driver == "22" for p in st2.pit_events):
        first = min(p.in_t for p in st2.pit_events if p.driver == "22")
        _state({first - 30: ("22", "Box box."), st2.t - 30: ("11", "Box box.")}, st2.t, base=st2)
        assert wall2.engineer("rivalradio").alerts(st2, wall2.view(st2)) == []
        assert wall2.view(st2).car("rivalradio", "22")["rr_box_intent"] == 0.0
