"""Speech to text for the pit-wall microphone: browser audio (webm/opus, ogg, wav, mp4) -> text.

PyAV decodes whatever the browser's MediaRecorder produced (no ffmpeg binary needed), then the same
faster-whisper loader the team-radio transcriber uses (``radio.load_model``) turns it into text.
The model is loaded on the first question and kept (GPU, falling back to CPU).
"""

from __future__ import annotations

import io
import logging
import threading

log = logging.getLogger("pitsense.asr")
MODEL = "small.en"
PROMPT = ("Questions to a Formula 1 race strategist. Box now, stay out, safety car, VSC, undercut, plan B, "
          "mediums, hards, softs, gap to the car ahead, tyre age, pit window.")
MAX_BYTES = 8_000_000
MAX_SECONDS = 30.0
_LOCK = threading.Lock()
_MODEL = None


def decode(data: bytes):
    """Container bytes (webm/opus, ogg, wav, mp4, mp3) -> 16 kHz mono float32."""
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.audio:
            raise ValueError("no audio stream in the upload")
        for frame in container.decode(audio=0):
            for r in resampler.resample(frame):
                chunks.append(r.to_ndarray().reshape(-1))
        for r in resampler.resample(None):
            chunks.append(r.to_ndarray().reshape(-1))
    audio = (np.concatenate(chunks) if chunks else np.zeros(0, np.int16)).astype(np.float32) / 32768.0
    return audio[: int(16000 * MAX_SECONDS)]


def _model():
    global _MODEL
    if _MODEL is None:
        from . import radio

        try:
            _MODEL = radio.load_model(MODEL, "cuda")
        except Exception as exc:  # no GPU / cuda libraries
            log.warning("whisper on cuda failed (%s); using the CPU", exc)
            _MODEL = radio.load_model(MODEL, "cpu")
    return _MODEL


def transcribe(data: bytes) -> str:
    """The words in an audio upload ('' if it holds only silence)."""
    audio = decode(data)
    if len(audio) < 1600:  # under 0.1 s
        return ""
    with _LOCK:
        segments, _ = _model().transcribe(audio, language="en", beam_size=5, temperature=0.0, vad_filter=True,
                                          condition_on_previous_text=False, initial_prompt=PROMPT)
        return " ".join(s.text.strip() for s in segments).strip()
