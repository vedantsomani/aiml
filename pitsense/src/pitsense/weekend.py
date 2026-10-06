"""A race weekend's other sessions: which exist, and how to download them.

Practice, sprint qualifying, sprint and qualifying all start before the race of the same
meeting, so their streams are known when the race starts (see ``before``).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import archive
from .archive import SessionRef

PRACTICE = ("Practice 1", "Practice 2", "Practice 3")
QUALI = ("Qualifying", "Sprint Qualifying")


def meeting_sessions(ref: SessionRef) -> list[SessionRef]:
    """Every session of ref's meeting, in start order."""
    out = [r for r in archive.fetch_index(ref.year) if r.meeting_key == ref.meeting_key]
    return sorted(out, key=lambda r: r.start_utc)


def before(ref: SessionRef) -> list[SessionRef]:
    """Sessions of the same meeting that started before ``ref`` did (known at its start)."""
    return [r for r in meeting_sessions(ref) if r.start_utc < ref.start_utc]


def quali_sessions(year: int) -> list[SessionRef]:
    """Qualifying and Sprint Qualifying sessions of a season that have started."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return [r for r in archive.fetch_index(year) if r.session_name in QUALI and r.start_utc <= now]


def download_weekend(race: SessionRef, topics: tuple[str, ...], *, force: bool = False,
                     feed_topics: tuple[str, ...] = ()) -> list[SessionRef]:
    """Download every non-race session of the race's meeting (Position.z etc. only for quali)."""
    got = []
    for ref in meeting_sessions(race):
        if ref.session_key == race.session_key:
            continue
        extra = feed_topics if ref.session_name in QUALI else ()
        archive.download_session(ref, topics + extra, force=force)
        got.append(ref)
    return got


# --------------------------------------------------------------------------- tyre sets
# Dry sets each driver gets for a weekend, by compound (Pirelli's usual split). Assumption, not
# a feed value: the nominated mix changes by race, and the feed never says which sets a team
# returned. Override per call (``allocation=``) or per race (``ctx.meta["tyre_allocation"]``).
ALLOCATION = {"SOFT": 8, "MEDIUM": 3, "HARD": 2}  # 13 dry sets
ALLOCATION_SPRINT = {"SOFT": 7, "MEDIUM": 3, "HARD": 2}  # 12 dry sets at a sprint weekend
DRY = ("SOFT", "MEDIUM", "HARD")
_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H"}


def allocation(sprint: bool = False, returned: dict[str, int] | None = None,
               override: dict[str, int] | None = None) -> dict[str, int]:
    """Dry sets per compound at the start of the weekend, minus ``returned`` sets."""
    base = dict(override or (ALLOCATION_SPRINT if sprint else ALLOCATION))
    for c, n in (returned or {}).items():
        base[c] = max(0, base.get(c, 0) - int(n))
    return base


@dataclass
class TyreSet:
    compound: str
    laps: int  # laps run on it so far (the feed's TotalLaps)
    first: str = ""  # session where it first ran ("" = not seen new: origin unknown)
    mounted: bool = False  # on the car now / already used in the race itself

    def to_list(self) -> list:
        return [self.compound, self.laps, self.first]


def _real_stints(stints: list) -> list[dict]:
    """Stint records that mean a tyre was fitted (see ``RaceState._real_stints``)."""
    from .state import RaceState

    if isinstance(stints, dict):
        stints = [stints[k] for k in sorted(stints, key=lambda k: int(k) if str(k).isdigit() else 0)]
    return [stints[i] for i in RaceState._real_stints(stints)]


def add_stints(sets: list[TyreSet], stints: list, session: str, meta: dict | None = None) -> None:
    """Fold one car's stint records into its set list (dry compounds only).

    ``New`` "true" is a set nobody had run: a new set. A used one (``New`` "false") is matched
    to the car's set of that compound whose laps equal the stint's ``StartLaps``; with no match
    it is a used set of unknown origin. Wet compounds are not dry-set allocation and are skipped.
    """
    from .state import relative_compound

    for s in _real_stints(stints):
        compound = relative_compound(str(s.get("Compound", "")).upper(), meta or {})
        if compound not in DRY:
            continue
        start, total = _int(s.get("StartLaps")), _int(s.get("TotalLaps"))
        if str(s.get("New", "")).lower() == "true":
            sets.append(TyreSet(compound, max(total, 0), session))
            continue
        cands = [t for t in sets if t.compound == compound and t.laps <= start and not t.mounted]
        if cands:  # the closest earlier age: the set this stint continues
            best = max(cands, key=lambda t: t.laps)
            best.laps = max(total, best.laps)
        else:
            sets.append(TyreSet(compound, max(total, start), ""))


def _int(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def session_stints(session_dir) -> dict[str, list]:
    """Final TimingAppData stints per car of a downloaded session ({} if the stream is missing)."""
    from pathlib import Path

    from .events import load_archive_session
    from .merge import deep_merge

    d = Path(session_dir)
    if not (d / "TimingAppData.jsonStream").exists():
        return {}
    topic = None
    for e in load_archive_session(d, ("TimingAppData",)).events:
        topic = deep_merge(topic, e.data) if e.kind != "snapshot" else e.data
    lines = (topic or {}).get("Lines") or {}
    return {n: (v.get("Stints") or []) for n, v in lines.items() if isinstance(v, dict)}


def weekend_sets(ref: SessionRef) -> dict[str, list[TyreSet]]:
    """Every set each car ran in this meeting's sessions that started before ``ref``.

    Read from those sessions' own streams, all published before ``ref`` starts. A session that
    is not downloaded, or whose stream cannot be read, contributes nothing (see ``pitsense
    fetch --weekend`` and ``sets_basis``).
    """
    out: dict[str, list[TyreSet]] = {}
    meta = {"year": ref.year, "meeting_name": ref.meeting_name}
    for s in before(ref):
        try:
            stints = session_stints(s.local_dir)
        except Exception:  # a broken file loses that session, not the weekend
            continue
        for car, st in stints.items():
            add_stints(out.setdefault(car, []), st, s.session_name, meta)
    return out


def have_sessions(ref: SessionRef) -> list[str]:
    """Names of the earlier sessions of the weekend that are on disk."""
    return [s.session_name for s in before(ref) if (s.local_dir / "TimingAppData.jsonStream").exists()]


def sets_basis(ref: SessionRef) -> tuple[str, int, int]:
    """How much of the weekend the set book rests on: ("weekend" | "partial" | "allocation", on disk, expected).

    "allocation": no earlier session is on disk, so only the standard allocation is known.
    "partial": some are missing, so sets run there are not counted (new sets left is an upper bound).
    """
    want, got = len(before(ref)), len(have_sessions(ref))
    return ("allocation" if got == 0 else "weekend" if got >= want else "partial"), got, want


def is_sprint_weekend(ref: SessionRef) -> bool:
    return any(s.session_name == "Sprint" for s in meeting_sessions(ref))


def remaining(sets: list[TyreSet], alloc: dict[str, int]) -> dict[str, int]:
    """New sets left per compound: allocation minus the sets already run."""
    out = {}
    for c in DRY:
        used_new = sum(1 for t in sets if t.compound == c and t.first)
        out[c] = max(0, alloc.get(c, 0) - used_new)
    return out


def used_str(sets: list[TyreSet]) -> str:
    """Used sets still free to fit, e.g. "S12 S5 M20": compound letter + laps on it."""
    free = [t for t in sets if not t.mounted]
    free.sort(key=lambda t: (DRY.index(t.compound), t.laps))
    return " ".join(f"{_LETTER[t.compound]}{t.laps}" for t in free)


# --------------------------------------------------------------------------- pre-race tool
def cmd_tyresets(a) -> None:
    ref = archive.find_session(a.year, a.race, "Sprint" if a.sprint else "Race")
    sessions = have_sessions(ref)
    if not sessions:
        raise SystemExit(f"no earlier session of {ref.meeting_name} on disk: run `pitsense fetch --weekend`")
    ret = {}
    for item in a.returned or []:
        c, _, n = item.partition("=")
        ret[{"S": "SOFT", "M": "MEDIUM", "H": "HARD"}[c.upper()]] = int(n)
    alloc = allocation(is_sprint_weekend(ref), ret)
    sets = weekend_sets(ref)
    names = {}
    from .events import load_archive_session

    for e in load_archive_session(ref.local_dir, ("DriverList",)).events:
        for n, d in (e.data or {}).items():
            if isinstance(d, dict) and d.get("Tla"):
                names[n] = d["Tla"]
    print(f"{ref.slug}: sessions {', '.join(sessions)}; allocation {alloc} (assumed)")
    print(f"{'car':<8}{'new S':>6}{'new M':>6}{'new H':>6}  used sets (laps)")
    for car in sorted(sets, key=lambda c: int(c) if c.isdigit() else 999):
        left = remaining(sets[car], alloc)
        print(f"{car + ' ' + names.get(car, ''):<8}{left['SOFT']:>6}{left['MEDIUM']:>6}{left['HARD']:>6}  {used_str(sets[car])}")


def add_commands(sub) -> None:
    s = sub.add_parser("tyresets", help="pre-race: new and used dry sets each car has left")
    s.add_argument("--year", type=int, default=2026)
    s.add_argument("--race", required=True)
    s.add_argument("--sprint", action="store_true")
    s.add_argument("--returned", nargs="+", help="sets handed back, e.g. S=1 H=0 (not in the feed)")
    s.set_defaults(fn=cmd_tyresets)


def availability(sets: list[TyreSet], alloc: dict[str, int]) -> dict[str, dict]:
    """Per compound what the car can still fit: new sets left, free used sets and the laps on the freshest one."""
    left = remaining(sets, alloc)
    out = {}
    for c in DRY:
        free = [t.laps for t in sets if t.compound == c and not t.mounted]
        out[c] = {"new": left[c], "used": len(free), "used_laps": min(free) if free else None}
    return out


# --------------------------------------------------------------------------- used vs new pace
OFFSET_CAP = 15  # laps on a set beyond which the offset stops growing
PRIOR_SD = {"used": 0.10, "per_lap": 0.03}  # s: fitted terms shrink toward zero in proportion to their noise


def used_offset(model: dict | None, laps: float | int | None) -> float:
    """Seconds a used set with ``laps`` laps on it is slower than a new one (0 for a new set or no model)."""
    if not model or not laps or laps <= 0:
        return 0.0
    return max(0.0, float(model["used"]) + float(model["per_lap"]) * min(float(laps), OFFSET_CAP))


def _shrink(b: float, se: float, sd: float) -> float:
    return b * sd * sd / (sd * sd + se * se)


def fit_used_offset(year: int, boot: int = 30, seed: int = 0) -> dict:
    """Learn the used-set pace term from the dry green-flag laps of the races of ``year``.

    Lap time ~ driver-race + compound-race + race x lap (fuel) + compound x laps into the stint
    + [used] + per_lap x min(laps on the set at fitting, OFFSET_CAP). The last two terms are shrunk by
    their race-bootstrap noise. Returns {"used", "per_lap", "se_used", "se_per_lap", "n_laps", "races", "year"}.
    """
    import numpy as np
    import pandas as pd

    from .archive import downloaded_sessions
    from .events import load_archive_session
    from .state import RaceState, relative_compound

    rows = []
    for ref in (r for r in downloaded_sessions() if r.year == year and r.session_name == "Race"):
        log = load_archive_session(ref.local_dir)
        st = RaceState(log.meta)
        for e in log.events:
            st.apply(e)
        lines = (st.topics.get("TimingAppData") or {}).get("Lines") or {}
        for rec in st.laps:
            if rec.lap_time is None or rec.is_in_lap or rec.is_out_lap or rec.lap < 2 or rec.track_status != "1":
                continue
            real = _real_stints((lines.get(rec.driver) or {}).get("Stints") or [])
            if not 1 <= rec.stint <= len(real):
                continue
            s = real[rec.stint - 1]
            comp = relative_compound(str(s.get("Compound", "")).upper(), log.meta)
            if comp in DRY:
                rows.append((ref.slug, rec.driver, comp, rec.lap, rec.lap_time, _int(s.get("StartLaps")), rec.tyre_age or 0))
    out = {"year": year, "used": 0.0, "per_lap": 0.0, "se_used": None, "se_per_lap": None, "n_laps": len(rows), "races": 0}
    if len(rows) < 2000:
        return out
    df0 = pd.DataFrame(rows, columns=["race", "drv", "comp", "lap", "t", "start", "age"])
    out["races"] = int(df0.race.nunique())

    def fit(df):
        med = df.groupby(["race", "drv"]).t.transform("median")
        df = df[(df.t < med * 1.05) & (df.t > med * 0.93)]
        sl = (df.age - df.start).clip(lower=0).to_numpy()
        d = lambda k: pd.get_dummies(k, dtype=float).to_numpy()  # noqa: E731
        X = [d(df.race + "|" + df.drv), d(df.race + "|" + df.comp)[:, 1:], d(df.race) * df.lap.to_numpy()[:, None]]
        X += [((df.comp == c).to_numpy() * sl)[:, None] for c in DRY]
        X += [(df.start > 0).to_numpy(float)[:, None], np.minimum(df.start, OFFSET_CAP).to_numpy(float)[:, None]]
        return np.linalg.lstsq(np.hstack(X), df.t.to_numpy(), rcond=None)[0][-2:]

    b = fit(df0)
    rng = np.random.default_rng(seed)
    races = df0.race.unique()
    bs = []
    for _ in range(boot):
        parts = [df0[df0.race == r].assign(race=f"{r}#{j}") for j, r in enumerate(rng.choice(races, len(races)))]
        bs.append(fit(pd.concat(parts)))
    se = np.array(bs).std(0)
    out.update(used=round(_shrink(float(b[0]), float(se[0]), PRIOR_SD["used"]), 4),
               per_lap=round(_shrink(float(b[1]), float(se[1]), PRIOR_SD["per_lap"]), 4),
               se_used=round(float(se[0]), 4), se_per_lap=round(float(se[1]), 4))
    return out


def _cache_path(year: int):
    from .config import data_dir

    return data_dir() / "scratch" / f"used_offset_{year}.json"


def load_used_offset(race_year: int) -> dict | None:
    """The used-set term learned from the season before ``race_year`` (None until the fit has been run for it)."""
    import json

    try:
        return json.loads(_cache_path(race_year - 1).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


if __name__ == "__main__":  # python -m pitsense.weekend fit 2025
    import json
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "fit":
        res = fit_used_offset(int(sys.argv[2]))
        p = _cache_path(res["year"])
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(res), encoding="utf-8")
        tmp.replace(p)
        print(res)
