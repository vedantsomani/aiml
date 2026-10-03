"""Per-season regulation facts known before the race starts.

Everything here is published before lights out (sporting regulations, event
notes, Pirelli bulletins), so an engineer may use it as-of any moment of the
race. Never add a rule because of what happened in a race.

``RegRule`` fields
    min_stops        minimum number of pit stops in a dry race (None = only the
                     two-compound rule)
    max_stint_laps   maximum laps on one set of tyres (None = no limit)
    two_compounds    two different dry compounds must be used in a dry race
    source           where the rule is written down
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegRule:
    year: int
    meeting: str  # lower-case fragment of the meeting name ("" = every race of the year)
    min_stops: int | None = None
    max_stint_laps: int | None = None
    two_compounds: bool = True
    source: str = ""


_SPORTING = "FIA Formula 1 Sporting Regulations, tyre article: two different dry compounds in a dry race, waived once wet tyres are used"

REGS: tuple[RegRule, ...] = (
    RegRule(2025, "", source=_SPORTING),
    RegRule(2026, "", source=_SPORTING),
    RegRule(
        2025, "monaco", min_stops=2,
        source="FIA World Motor Sport Council, Monaco GP 2025: at least two pit stops, three sets of tyres "
               "(https://www.formula1.com/en/latest/article/fia-world-motor-sport-council-confirms-mandatory-two-stop-strategies-for.UUe16nwpOcqBe9f6PYgkq; "
               "https://press.pirelli.com/two-pit-stops-mandatory-in-monaco/)",
    ),
    RegRule(
        2025, "qatar", max_stint_laps=25,
        source="Pirelli / FIA, Qatar GP 2025: at most 25 laps on a set of tyres, so two stops over 57 laps "
               "(https://www.racefans.net/2025/11/17/f1-limits-drivers-to-just-25-laps-per-set-of-tyres-for-qatar-grand-prix/)",
    ),
    # 2026 Monaco: the two-stop rule was dropped (https://www.motorsportweek.com/2026/02/28/mandatory-monaco-two-stop-rule-scrapped-for-2026-following-backlash/),
    # so 2026 Monaco falls back to the default row.
)


def rule_for(meta: dict | None) -> RegRule | None:
    """The merged rule for a race from its session info (year, meeting_name); None if the year is unknown."""
    meta = meta or {}
    try:
        year = int(meta.get("year"))
    except (TypeError, ValueError):
        return None
    name = str(meta.get("meeting_name", "")).lower()
    rows = [r for r in REGS if r.year == year and (not r.meeting or r.meeting in name)]
    if not rows:
        return None
    out = RegRule(year, "")
    for r in rows:  # specific rows override the year default
        out = RegRule(
            year, r.meeting or out.meeting,
            min_stops=r.min_stops if r.min_stops is not None else out.min_stops,
            max_stint_laps=r.max_stint_laps if r.max_stint_laps is not None else out.max_stint_laps,
            two_compounds=r.two_compounds and out.two_compounds,
            source="; ".join(x for x in (out.source, r.source) if x),
        )
    return out
