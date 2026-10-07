"""The pit wall's command line: ``pitsense pitwall``, ``pitsense doctor`` and ``pitsense shadow-score``."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from .. import events as _events  # the feeder: reads the log; exempt by name in tests/test_pitwall.py
from .types import TeamConfig
from .runtime import ASK_WAIT_S, PUBLISH_EVERY_S, PitWallRuntime  # noqa: F401
from .shadow import load_call_log, shadow_score
from .sources import FollowSource, LiveSource, ReplaySource, Source


# ----------------------------------------------------------------------------- CLI
def _team_config(text: str | None, risk: str = "expected") -> TeamConfig:
    if not text:
        return TeamConfig(risk=risk)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if parts and all(p.isdigit() for p in parts):
        return TeamConfig(cars=tuple(parts), risk=risk)
    return TeamConfig(team=text.strip(), risk=risk)


SESSION_NAMES = {None: None, "race": "Race", "sprint": "Sprint", "qualifying": "Qualifying",
                 "sprint-qualifying": "Sprint Qualifying", "practice1": "Practice 1",
                 "practice2": "Practice 2", "practice3": "Practice 3"}


def build_source(a) -> Source:
    if a.live:
        out = Path(a.out) if a.out else Path(os.environ.get("PITSENSE_DATA", "data")) / "live" / (
            datetime.now().strftime("%Y%m%dT%H%M%S") + "-live.jsonl")
        return LiveSource(out, minutes=a.minutes, no_auth=a.no_auth)
    if a.file and a.follow:
        return FollowSource(Path(a.file), follow=True)
    if a.file:  # a finished recording: replay it at --speed
        from ..live import load_recording

        return ReplaySource(load_recording(Path(a.file), feeds=True), a.speed, title=Path(a.file).name)
    if not a.race:
        raise SystemExit("give --race (archive replay), --file (a recording) or --live")
    from .. import archive
    ref = archive.find_session(a.year, a.race, SESSION_NAMES.get(a.session) or ("Sprint" if a.sprint else "Race"))
    if not (ref.local_dir / "TimingData.jsonStream").exists():
        archive.download_session(ref)
    return ReplaySource(load_with_feeds(ref.local_dir), a.speed, ref=ref)


def load_with_feeds(session_dir: Path) -> _events.EventLog:
    """Timing and every feed downloaded: positions, radio and CarData (driver card, speed trace, coach)."""
    return _events.load_archive_session(session_dir, feeds=True)


def _lan_ip() -> str:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))  # no packet is sent; picks the outgoing interface
            return sk.getsockname()[0]
    except OSError:
        return "<this-computer-ip>"


def cmd_doctor(a) -> None:
    """Pre-race check: model bundle, Whisper, Piper, disk space, smoke test."""
    import shutil
    from .. import archive
    from ..state import replay

    print("PitSense pre-race check")
    print("-" * 40)

    checks = []

    # 1. Model bundle available
    try:
        from ..modelstore import latest_bundle_for
        from datetime import datetime, timezone
        bundle = latest_bundle_for(datetime.now(timezone.utc))
        checks.append(("Model bundle", "OK" if bundle else "FAIL", ""))
    except Exception as e:
        checks.append(("Model bundle", "FAIL", str(e)))

    # 2. Whisper importable
    try:
        import faster_whisper  # noqa: F401  (is it installed?)
        checks.append(("Whisper (faster-whisper)", "OK", ""))
    except ImportError:
        checks.append(("Whisper (faster-whisper)", "FAIL", "pip install faster-whisper"))

    # 3. Piper voice files present
    try:
        from ..voice import tts as voice_tts
        if voice_tts.available():
            voice_path = voice_tts.voice_dir() / (voice_tts.DEFAULT_VOICE + ".onnx")
            if voice_path.exists():
                checks.append(("Piper voice files", "OK", f"{voice_tts.DEFAULT_VOICE}"))
            else:
                checks.append(("Piper voice files", "WARN", f"run: python -m piper.download_voices {voice_tts.DEFAULT_VOICE}"))
        else:
            checks.append(("Piper voice files", "FAIL", "pip install piper-tts"))
    except Exception as e:
        checks.append(("Piper voice files", "FAIL", str(e)))

    # 4. Disk space
    try:
        data_dir = Path(os.environ.get("PITSENSE_DATA", "data"))
        stat = shutil.disk_usage(data_dir)
        free_gb = stat.free / (1024 ** 3)
        if free_gb > 1:
            checks.append(("Disk space", "OK", f"{free_gb:.1f} GB free"))
        else:
            checks.append(("Disk space", "WARN", f"only {free_gb:.1f} GB free"))
    except Exception as e:
        checks.append(("Disk space", "FAIL", str(e)))

    # 5. Smoke test: replay a short race
    try:
        ref = archive.find_session(2026, "hungary", "Race")
        if not (ref.local_dir / "TimingData.jsonStream").exists():
            checks.append(("Smoke test (replay)", "SKIP", "Hungary 2026 not downloaded"))
        else:
            log = load_with_feeds(ref.local_dir)
            state = replay(log)
            if state.current_lap >= 3 and len(state.laps) > 0:
                checks.append(("Smoke test (replay)", "OK", f"{len(state.laps)} laps"))
            else:
                checks.append(("Smoke test (replay)", "FAIL", "replay did not produce laps"))
    except Exception as e:
        checks.append(("Smoke test (replay)", "FAIL", str(e)[:60]))

    # Print results
    max_name = max(len(name) for name, _, _ in checks)
    for name, status, detail in checks:
        icon = "✓" if status == "OK" else "✗" if status == "FAIL" else "⚠"
        print(f"{icon} {name:<{max_name}} {status:<4} {detail}")

    # Overall status
    failed = [name for name, status, _ in checks if status == "FAIL"]
    if failed:
        print(f"\nFAIL: {len(failed)} check(s) failed")
        import sys
        sys.exit(1)
    else:
        print("\nOK: Ready for race day")


def cmd_pitwall(a) -> None:
    from ..config import data_dir
    from ..web.server import serve

    src = build_source(a)
    log_dir = Path(a.log_dir) if a.log_dir else data_dir() / "pitwall"
    rt = PitWallRuntime(src, team=_team_config(a.team, a.risk), log_dir=log_dir, models=not a.no_models, coarse=a.coarse,
                        index=getattr(src, "seekable", False))  # a replay: prepare every lap in the background so jumps are instant
    rt.start()
    server = serve(rt, host=a.host, port=a.port, token=a.token)
    port = server.server_address[1]
    q = f"?token={a.token}" if a.token else ""
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else server.server_address[0]}:{port}/{q}"
    print(f"Pit wall on {url}   ({src.mode}: {src.title})   call log: {rt.log_path}")
    if a.host in ("0.0.0.0", ""):
        print(f"On the same Wi-Fi open  http://{_lan_ip()}:{port}/{q}" + ("" if a.token else "   (no --token: anyone on the network can watch and ask)"))
    if not a.no_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        while True:
            time.sleep(0.5)
            if a.no_browser and rt.status in ("finished", "error"):  # headless: exit once the replay is done
                break
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.shutdown()
        rt.stop()


def cmd_shadow_score(a) -> None:
    from .. import archive
    from ..state import replay

    ref = archive.find_session(a.year, a.race, "Sprint" if a.sprint else "Race")
    final = replay(_events.load_archive_session(ref.local_dir))
    calls = load_call_log(Path(a.log))
    res = shadow_score(calls, final, a.k, set(a.cars) if a.cars else None)
    res["race"] = ref.slug
    if a.json:
        print(json.dumps(res, indent=1))
        return
    print(f"{ref.slug}  calls from {a.log}  (+-{a.k} laps)")
    for key, v in res.items():
        if key not in ("race", "k"):
            print(f"  {key:<18} {v}")


def add_commands(sub) -> None:
    s = sub.add_parser("pitwall", aliases=["serve"], help="follow a race and show the strategy screen in the browser (alias: serve)")
    s.add_argument("--year", type=int, default=2026)
    s.add_argument("--race", help="archive replay, e.g. 'hungary'")
    s.add_argument("--sprint", action="store_true")
    s.add_argument("--session", choices=[k for k in SESSION_NAMES if k],
                   help="session of the meeting (default race): qualifying and sprint-qualifying get the qualifying panel")
    s.add_argument("--speed", type=float, default=1.0, help="replay speed (x real time; 0 = as fast as possible)")
    s.add_argument("--coarse", action="store_true",
                   help="replay: decide only when the dashboard updates (faster at high speed; calls can differ from live)")
    s.add_argument("--file", help="a recording from `pitsense record`")
    s.add_argument("--follow", action="store_true", help="with --file: follow it as it grows instead of replaying")
    s.add_argument("--live", action="store_true", help="start the recorder and follow its file (needs pitsense[live])")
    s.add_argument("--out", help="with --live: recording file (default data/live/<time>-live.jsonl)")
    s.add_argument("--minutes", type=float, default=180, help="with --live: stop recording after this long")
    s.add_argument("--no-auth", action="store_true", help="with --live: skip F1 TV sign-in (no positions)")
    s.add_argument("--team", help="your team, e.g. 'ferrari', or car numbers '16,44'")
    s.add_argument("--risk", choices=["expected", "protect", "aggressive"], default="expected",
                   help="how plans are ranked: best average result, smallest downside, or biggest upside")
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 lets phones on the same Wi-Fi connect (use --token)")
    s.add_argument("--token", default=os.environ.get("PITSENSE_TOKEN") or None,
                   help="require ?token=<this> on every request (first page load sets a cookie); default: the "
                        "PITSENSE_TOKEN environment variable, which keeps it out of the process list")
    s.add_argument("--port", type=int, default=8765, help="0 = any free port")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--no-models", action="store_true", help="don't load a trained model bundle")
    s.add_argument("--log-dir", help="where the calls log goes (default data/pitwall)")
    s.set_defaults(fn=cmd_pitwall)

    s = sub.add_parser("shadow-score", help="compare a pit wall calls log with the stops that happened")
    s.add_argument("--log", required=True, help="calls-*.jsonl written by `pitsense pitwall`")
    s.add_argument("--year", type=int, default=2026)
    s.add_argument("--race", required=True)
    s.add_argument("--sprint", action="store_true")
    s.add_argument("--k", type=int, default=2, help="a call is right if the stop is within +-k laps")
    s.add_argument("--cars", nargs="+", help="only these car numbers")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_shadow_score)

    s = sub.add_parser("doctor", help="pre-race checks: model, Whisper, Piper, disk space, smoke test")
    s.set_defaults(fn=cmd_doctor)
