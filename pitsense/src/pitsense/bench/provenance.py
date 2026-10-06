"""Where a number came from: data version, configuration, training cutoff and git commit.

Every result file written for the versioned report (``pitsense results``) carries ``stamp(...)``, and every
headline number in ``reports/results.json`` repeats the stamp of the run that produced it, so numbers from
different runs cannot be mixed silently.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..config import bench_dir

REPO = Path(__file__).resolve().parents[3]


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def git_commit() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    head = run("rev-parse", "--short=10", "HEAD") or "unknown"
    dirty = [l[3:] for l in run("status", "--porcelain", "--untracked-files=no").splitlines() if l.strip()]
    return {"commit": head, "dirty": bool(dirty), "dirty_files": dirty[:20]}


def data_version() -> dict:
    from .features import FEATURE_VERSION

    man = bench_dir() / FEATURE_VERSION / "manifest.json"
    races = None
    by_year: dict[str, int] = {}
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        races = len(m.get("races", []))
        for r in m.get("races", []):
            by_year[str(r)[:4]] = by_year.get(str(r)[:4], 0) + 1
    return {"feature_version": FEATURE_VERSION, "races": races, "races_by_year": by_year,
            "manifest_sha256": _sha(man), "history_sha256": _sha(bench_dir() / "history.json")}


def config() -> dict:
    from ..pitwall.engineers.strategy.analysis import SETTINGS
    from ..pitwall.engineers.strategy.sim import PARAMS

    blob = {"SETTINGS": SETTINGS, "sim.PARAMS": PARAMS}
    h = hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return {"hash": h, **blob}


def stamp(training_cutoff: str, **extra) -> dict:
    """The provenance block of one run. ``training_cutoff`` says what the models of that run were allowed to see."""
    return {"data": data_version(), "config": config(), "training_cutoff": training_cutoff, "git": git_commit(),
            "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **extra}


def short(p: dict) -> str:
    """One line for tables: ``data 189 races #abc123 | config #def456 | cutoff ... | git 1234abcd``."""
    d, g = p.get("data", {}), p.get("git", {})
    return (f"data {d.get('races')} races #{d.get('manifest_sha256')} | config #{p.get('config', {}).get('hash')} | "
            f"cutoff: {p.get('training_cutoff')} | git {g.get('commit')}{'+dirty' if g.get('dirty') else ''}")
