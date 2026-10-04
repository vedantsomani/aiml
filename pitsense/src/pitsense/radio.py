"""Offline transcription of team radio with faster-whisper (own copy of the model, no web service).

Each ``TeamRadio/<name>.mp3`` gets ``TeamRadio/<name>.json`` next to it::

    {"file": "<name>.mp3", "model": "small.en", "duration_s": 6.4, "text": "...", "segments": [{"start", "end", "text"}]}

The cache key is the file: an existing JSON is never recomputed (use ``--force``). The output is
deterministic (greedy decoding, temperature 0). Transcripts are *known* in replay only
``feeds.RADIO_LATENCY_S`` after the message was published (see ``feeds.RadioStore``).

    pitsense radio transcribe --year 2025 2026 --model small.en
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

DEFAULT_MODEL = "small.en"
# Biases decoding toward pit-wall vocabulary; it names no race facts.
PROMPT = "Formula 1 team radio. Box, box. Plan A, undercut, overcut, safety car, DRS, delta, tyres, mediums, hards, softs, pit."


def _add_cuda_dlls() -> None:
    """Windows: ctranslate2 wants CUDA 12 cuBLAS/cuDNN DLLs (pip: nvidia-cublas-cu12, nvidia-cudnn-cu12)."""
    if os.name != "nt":
        return
    import site

    for root in site.getsitepackages():
        for lib in sorted((Path(root) / "nvidia").glob("*/bin")):
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = f"{lib}{os.pathsep}{os.environ.get('PATH', '')}"


def load_model(name: str = DEFAULT_MODEL, device: str = "cuda"):
    from faster_whisper import WhisperModel  # pip install faster-whisper

    if device == "cuda":
        _add_cuda_dlls()
    return WhisperModel(name, device=device, compute_type="float16" if device == "cuda" else "int8")


def transcript_path(mp3: Path) -> Path:
    return mp3.with_suffix(".json")


def decode_audio(mp3: Path):
    """mp3 -> 16 kHz mono float32. faster-whisper's own decoder passes an option PyAV 19 dropped."""
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(str(mp3)) as container:
        for frame in container.decode(audio=0):
            for r in resampler.resample(frame):
                chunks.append(r.to_ndarray().reshape(-1))
        for r in resampler.resample(None):
            chunks.append(r.to_ndarray().reshape(-1))
    return (np.concatenate(chunks) if chunks else np.zeros(0, np.int16)).astype(np.float32) / 32768.0


def transcribe_file(model, mp3: Path, model_name: str = DEFAULT_MODEL) -> dict:
    segments, info = model.transcribe(
        decode_audio(mp3), language="en", beam_size=5, temperature=0.0, vad_filter=True,
        condition_on_previous_text=False, initial_prompt=PROMPT,
    )
    segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
    return {
        "file": mp3.name,
        "model": model_name,
        "duration_s": round(info.duration, 2),
        "text": " ".join(s["text"] for s in segs).strip(),
        "segments": segs,
    }


def transcribe_session(session_dir: Path, model, model_name: str = DEFAULT_MODEL, *, force: bool = False) -> tuple[int, float]:
    """Transcribe every uncached mp3 of a session. Returns (clips done, audio seconds)."""
    done, audio_s = 0, 0.0
    for mp3 in sorted((Path(session_dir) / "TeamRadio").glob("*.mp3")):
        out = transcript_path(mp3)
        if out.exists() and not force:
            continue
        try:
            result = transcribe_file(model, mp3, model_name)
        except Exception as exc:  # a corrupt clip must not stop the batch
            result = {"file": mp3.name, "model": model_name, "duration_s": 0.0, "text": "", "segments": [], "error": str(exc)[:200]}
        tmp = out.with_suffix(".part")
        tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
        tmp.replace(out)
        done += 1
        audio_s += result["duration_s"]
    return done, audio_s


def add_commands(sub) -> None:
    def cmd(a) -> None:
        from . import archive

        model = load_model(a.model, a.device)
        t0, n, audio = time.time(), 0, 0.0
        for y in a.year:
            for ref in archive.races(y):
                if not (ref.local_dir / "TeamRadio").exists():
                    continue
                k, s = transcribe_session(ref.local_dir, model, a.model, force=a.force)
                n, audio = n + k, audio + s
                print(f"  {ref.slug}: {k} clip(s)", flush=True)
        dt = time.time() - t0
        print(f"{n} clip(s), {audio / 60:.1f} min of audio in {dt:.0f} s ({audio / max(dt, 1e-9):.0f}x real time)")

    s = sub.add_parser("radio", help="transcribe downloaded team radio offline (faster-whisper)")
    s.add_argument("action", choices=["transcribe"])
    s.add_argument("--year", type=int, nargs="+", default=[2025, 2026])
    s.add_argument("--model", default=DEFAULT_MODEL, help="small.en (default) or base.en")
    s.add_argument("--device", default="cuda")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd)
