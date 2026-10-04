"""Command line: `pitsense <command>`. Run `pitsense -h` for the list."""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DEFAULT_JOBS = 1 if os.name == "nt" else 4  # Windows: run in-process (simple and safe)


def _map(fn, items, jobs: int):
    """map() in parallel processes, or in-process when jobs <= 1."""
    items = list(items)
    if jobs <= 1 or len(items) <= 1:
        yield from map(fn, items)
        return
    with ProcessPoolExecutor(jobs) as ex:
        yield from ex.map(fn, items)


def _races(years: list[int], query: str | None, sprints: bool = False):
    from . import archive

    if query:
        return [archive.find_session(years[0], query, "Sprint" if sprints else "Race")]
    refs = []
    for y in years:
        refs += archive.races(y, include_sprints=sprints)
    return refs


def cmd_list(a) -> None:
    from . import archive

    for ref in archive.fetch_index(a.year, refresh=True):
        if ref.session_type != "Race":
            continue
        have = "✓" if (ref.local_dir / "TimingData.jsonStream").exists() else " "
        print(f"{have} R{ref.round_index:02d}  {ref.start_utc:%Y-%m-%d}  {ref.session_name:<6}  {ref.meeting_name}  ({ref.circuit})")


def cmd_fetch(a) -> None:
    from . import archive
    from .config import DEFAULT_TOPICS, RADIO_TOPICS, TELEMETRY_TOPICS, raw_dir

    refs = _races(a.year, a.race, a.sprints)
    t0 = time.time()
    topics = DEFAULT_TOPICS
    if a.telemetry:
        topics += TELEMETRY_TOPICS
    if a.radio:
        topics += RADIO_TOPICS
    for ref in refs:
        archive.download_session(ref, topics, force=a.force)
        extra = ""
        if a.weekend:
            from . import weekend

            got = weekend.download_weekend(ref, DEFAULT_TOPICS, force=a.force,
                                           feed_topics=("Position.z",))
            extra += "  weekend +" + ",".join(r.session_name for r in got)
        if a.radio:
            n, miss = archive.download_radio(ref)
            extra = f"  radio +{n} mp3 ({miss} missing)"
        print(f"  {ref.slug}{extra}")
    print(f"{len(refs)} session(s) in {raw_dir()} ({time.time() - t0:.0f} s)")


def _prior_for(ref):
    from .bench import history as H
    from .config import bench_dir
    from .pitloss import PitLossPrior

    path = bench_dir() / "history.json"
    if not path.exists() or ref is None:
        return PitLossPrior()
    return H.prior_for(H.load(path), ref.circuit_key, ref.start_utc)


def cmd_replay(a) -> None:
    from .bench.features import FeatureBuilder
    from .events import load_archive_session
    from .state import RaceState
    from .views import timing_tower

    ref = None
    if a.file:
        from .live import load_recording

        log = load_recording(Path(a.file))
    elif a.fastf1_file:
        from .live import load_fastf1_recording

        log = load_fastf1_recording(Path(a.fastf1_file))
    else:
        from . import archive

        ref = archive.find_session(a.year, a.race, "Sprint" if a.sprint else "Race")
        if not (ref.local_dir / "TimingData.jsonStream").exists():
            archive.download_session(ref)
        log = load_archive_session(ref.local_dir)

    # stop the moment the leader starts lap N+1 (i.e. completes lap N): nothing later is used
    cut = None
    if a.lap is not None:
        for e in log.events:
            if e.topic == "LapCount" and isinstance(e.data, dict) and (e.data.get("CurrentLap") or 0) > a.lap:
                cut = e.t
                break
    if a.at is not None:
        cut = a.at
    past = log.until(cut) if cut is not None else log
    state, fb = RaceState(log.meta), FeatureBuilder(_prior_for(ref), log.meta)
    for e in past.events:
        state.apply(e)
        fb.observe(state)
    title = ref.slug if ref else Path(a.file or a.fastf1_file).name
    print(f"{title}   as of t={state.t:.1f}s (events used: {len(past)}/{len(log)})\n")
    print(timing_tower(state, fb))


def _validate_one(args):
    ref, cache = args
    from .validate import compare

    warnings.filterwarnings("ignore")
    try:
        rep, _ = compare(ref, cache)
    except Exception as exc:  # e.g. FastF1 can't load one race; report it, keep going
        return ref.slug, f"{type(exc).__name__}: {exc}"
    return ref.slug, rep


def cmd_validate(a) -> None:
    from .config import data_dir

    refs = _races(a.year, a.race)
    cache = str(data_dir() / "fastf1_cache")
    os.makedirs(cache, exist_ok=True)
    reports, failed = [], []
    for slug, rep in _map(_validate_one, [(r, cache) for r in refs], a.jobs):
        if isinstance(rep, str):
            print(f"{slug:<44} not compared: {rep[:150]}", flush=True)
            failed.append(slug)
            continue
        print(rep.line(), flush=True)
        reports.append(rep)
    n = sum(r.n_both for r in reports)
    if n:

        def w(k: str) -> float:
            """Lap-weighted mean over races where the check applies (Spa 2021 has no lap times)."""
            rs = [r for r in reports if getattr(r, k) == getattr(r, k)]  # skip NaN
            n_k = sum(r.n_both for r in rs)
            return sum(getattr(r, k) * r.n_both for r in rs) / n_k if n_k else float("nan")

        fed = [r for r in reports if r.n_feed_age]
        n_fed = sum(r.n_feed_age for r in fed)
        feed = sum(r.feed_age_match * r.n_feed_age for r in fed) / n_fed if n_fed else float("nan")
        print(
            f"\nPooled over {len(reports)} sessions, {n:,} laps: lap time {w('lap_time_exact'):.3%}, "
            f"position {w('position_match'):.3%}, compound {w('compound_match'):.3%}, "
            f"tyre age {w('tyre_age_match'):.3%} (vs feed counter {feed:.3%} of {n_fed:,} laps), "
            f"in-lap {w('in_lap_match'):.3%}, out-lap {w('out_lap_match'):.3%}"
        )
    if failed:
        print(f"Not compared: {', '.join(failed)}")


def cmd_leakcheck(a) -> None:
    from . import archive
    from .bench import leakcheck
    from .events import load_archive_session

    ref = archive.find_session(a.year, a.race)
    log = load_archive_session(ref.local_dir)
    prior = _prior_for(ref)
    modes = ["truncate", "scramble"] if a.mode == "both" else [a.mode]
    failed = False
    for cut in leakcheck.random_cuts(log, a.cuts, seed=a.seed):
        for mode in modes:
            rep = leakcheck.check(log, cut, prior, mode=mode, seed=a.seed)
            status = "PASS" if rep.ok else "FAIL"
            failed |= not rep.ok
            print(f"{status}  {mode:<8} cut t={cut:8.1f}s  rows checked {rep.rows_checked:5d}  mismatches {len(rep.mismatches)}")
            for m in rep.mismatches[:10]:
                print(f"        driver {m[0]} lap {m[1]} column {m[2]}")
    sys.exit(1 if failed else 0)


def _summarize(ref):
    from .bench.history import summarize

    return summarize(ref)


def _build(ref):
    from .bench import history as H
    from .bench.dataset import build_race
    from .config import bench_dir

    return str(build_race(ref, H.load(bench_dir() / "history.json")))


def cmd_bench_build(a) -> None:
    from . import archive
    from .bench import history as H
    from .bench.dataset import write_manifest
    from .bench.features import FEATURE_VERSION
    from .config import bench_dir

    refs = _races(a.year, None)
    for ref in refs:
        archive.download_session(ref)
    t0 = time.time()
    summaries = list(_map(_summarize, refs, a.jobs))
    H.save(summaries, bench_dir() / "history.json")
    print(f"history: {len(summaries)} races ({time.time() - t0:.0f} s)")
    t0 = time.time()
    for path in _map(_build, refs, a.jobs):
        print(f"  {Path(path).name}")
    # the benchmark is exactly these races; files left over from other builds are ignored
    manifest = write_manifest([r.slug for r in refs], years=a.year, feature_version=FEATURE_VERSION)
    print(f"benchmark: {len(refs)} races -> {manifest.parent} ({time.time() - t0:.0f} s)")


def cmd_bench_run(a) -> None:
    from . import registry
    from .bench.dataset import load_bench
    from .bench.evaluate import evaluate_task, write_report
    from .bench.features import FEATURE_VERSION

    warnings.filterwarnings("ignore")
    tasks = registry.tasks(horizons=tuple(a.horizons))
    if a.tasks:
        unknown = set(a.tasks) - {t.name for t in tasks}
        if unknown:
            sys.exit(f"unknown task(s) {sorted(unknown)}; available: {[t.name for t in tasks]}")
        tasks = [t for t in tasks if t.name in a.tasks]
    df = load_bench()
    results = [evaluate_task(t, df, test_year=a.test_year, min_train_races=a.min_train) for t in tasks]
    test_races = df[df.year == a.test_year].race_id.nunique()
    meta = {
        "test_year": a.test_year,
        "test_races": test_races,
        "races": df.race_id.nunique(),
        "decision_points": len(df),
        "feature_version": FEATURE_VERSION,
    }
    path = write_report(results, Path(a.out), meta)
    for r in results:
        print(f"\n== {r.task}")
        print(r.leaderboard.round(4).to_string(index=False))
    print(f"\nReport: {path}")


def cmd_record(a) -> None:
    import logging

    from .live import record

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(a.out)
    print(f"Recording to {out} for up to {a.minutes:.0f} min. Ctrl+C to stop.")
    if not a.no_auth:
        print("You'll be asked to sign in with your F1 TV account in the browser the first time.")
    n = record(out, minutes=a.minutes, no_auth=a.no_auth)
    print(f"Saved {n} messages. Replay it with: pitsense replay --file {out}")


def cmd_import_live(a) -> None:
    from .live import load_fastf1_recording, load_recording

    src = Path(a.file)
    log = load_fastf1_recording(src) if a.fastf1 else load_recording(src)
    out = Path(a.out) if a.out else src.with_suffix(".events.jsonl.gz")
    log.save_jsonl(out)
    print(f"{len(log)} events, topics: {', '.join(sorted(log.topics()))}\nWrote {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pitsense", description="Leakage-safe F1 strategy engine and benchmark.")
    p.add_argument("--data", help="data folder (default ./data or $PITSENSE_DATA)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="list a season's races (✓ = downloaded)")
    s.add_argument("--year", type=int, default=2026)
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("fetch", help="download raw timing streams")
    s.add_argument("--year", type=int, nargs="+", default=[2026])
    s.add_argument("--race", help="one race, e.g. 'hungary' (default: all races of the year)")
    s.add_argument("--sprints", action="store_true", help="sprint sessions instead of Grands Prix")
    s.add_argument("--force", action="store_true", help="re-download")
    s.add_argument("--telemetry", action="store_true", help="also CarData.z and Position.z (~17 MB per race)")
    s.add_argument("--radio", action="store_true", help="also TeamRadio and its mp3 files")
    s.add_argument("--weekend", action="store_true",
                   help="also the meeting's other sessions (practice, qualifying, sprint); Position.z for the qualifying ones")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("replay", help="timing tower + 'pit now' projection as of a lap, using only the past")
    s.add_argument("--year", type=int, default=2026)
    s.add_argument("--race", default="hungary")
    s.add_argument("--sprint", action="store_true")
    s.add_argument("--lap", type=int, help="state when the leader completes this lap")
    s.add_argument("--at", type=float, help="state at this session-clock second")
    s.add_argument("--file", help="replay a recording made with `pitsense record`")
    s.add_argument("--fastf1-file", help="replay a file saved by FastF1's own recorder")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("validate", help="cross-check our lap reconstruction against FastF1 (needs pitsense[dev])")
    s.add_argument("--year", type=int, nargs="+", default=[2026])
    s.add_argument("--race")
    s.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("leakcheck", help="corrupted-future test on one race")
    s.add_argument("--year", type=int, default=2026)
    s.add_argument("--race", default="hungary")
    s.add_argument("--cuts", type=int, default=3)
    s.add_argument("--mode", choices=["truncate", "scramble", "both"], default="both")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_leakcheck)

    b = sub.add_parser("bench", help="build or score the benchmark")
    bsub = b.add_subparsers(dest="bench_cmd", required=True)
    s = bsub.add_parser("build", help="frozen decision points + labels for every race")
    s.add_argument("--year", type=int, nargs="+", default=[2025, 2026])
    s.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    s.set_defaults(fn=cmd_bench_build)
    s = bsub.add_parser("run", help="expanding-window evaluation and leaderboard")
    s.add_argument("--test-year", type=int, default=2026)
    s.add_argument("--horizons", type=int, nargs="+", default=[1, 3])
    s.add_argument("--tasks", nargs="+", help="only these tasks (default: every registered task)")
    s.add_argument("--min-train", type=int, default=10, help="minimum earlier races before a race is scored")
    s.add_argument("--out", default="reports")
    s.set_defaults(fn=cmd_bench_run)

    s = sub.add_parser("record", help="record a live session (needs pitsense[live])")
    s.add_argument("--out", required=True)
    s.add_argument("--minutes", type=float, default=180)
    s.add_argument("--no-auth", action="store_true", help="skip F1 TV sign-in (partial data)")
    s.set_defaults(fn=cmd_record)

    s = sub.add_parser("import-live", help="convert a recording into a PitSense event log")
    s.add_argument("file")
    s.add_argument("--fastf1", action="store_true", help="input was saved by `python -m fastf1.livetiming save`")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_import_live)

    from . import registry

    for add_commands in registry.command_adders():  # commands contributed by engineers (registry.COMMANDS)
        add_commands(sub)

    a = p.parse_args(argv)
    # Windows: keep unicode output (✓, →) from crashing when redirected to a file
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if a.data:
        os.environ["PITSENSE_DATA"] = a.data
    a.fn(a)


if __name__ == "__main__":
    main()
