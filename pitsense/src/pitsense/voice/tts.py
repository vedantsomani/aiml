"""Spoken audio: `speak(text) -> wav path`, offline, with Piper (pretrained open TTS, onnx).

    pip install -e .[tts]
    python -m piper.download_voices en_US-lessac-medium --data-dir data/models/voice/tts

The voice file (`<name>.onnx` + `.onnx.json`) is looked up in data/models/voice/tts/ or
the folder in PITSENSE_PIPER_DIR. Nothing is downloaded by this module.
"""

from __future__ import annotations

import hashlib
import os
import wave
from pathlib import Path

from ..config import data_dir

DEFAULT_VOICE = "en_US-lessac-medium"
_CACHE: dict = {}


def available() -> bool:
    try:
        import piper  # noqa: F401
    except ImportError:
        return False
    return True


def voice_dir() -> Path:
    return Path(os.environ.get("PITSENSE_PIPER_DIR") or data_dir() / "models" / "voice" / "tts")


def _load(name: str):
    if name not in _CACHE:
        from piper import PiperVoice

        f = voice_dir() / f"{name}.onnx"
        if not f.exists():
            raise FileNotFoundError(f"Piper voice {f} not found; run: python -m piper.download_voices {name} --data-dir {voice_dir()}")
        _CACHE[name] = PiperVoice.load(str(f))
    return _CACHE[name]


def speak(text: str, out: str | Path | None = None, voice: str = DEFAULT_VOICE) -> Path:
    """Say ``text`` aloud into a wav file (default: data/voice_audio/<hash>.wav) and return its path."""
    if not available():
        raise RuntimeError("speech needs the optional package piper-tts: pip install -e .[tts]")
    v = _load(voice)
    if out is None:
        out = data_dir() / "voice_audio" / (hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16] + ".wav")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        v.synthesize_wav(text, w)
    return out
