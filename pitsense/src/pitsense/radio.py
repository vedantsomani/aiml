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
import logging
import os
import queue
import re
import threading
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("pitsense.radio")
STATIC_URL = "https://livetiming.formula1.com/static/"
CLIP_RE = re.compile(r"^TeamRadio/[A-Za-z0-9_.-]+\.mp3$")  # the only capture paths we ever fetch

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


def write_transcript(mp3: Path, result: dict) -> None:
    out = transcript_path(mp3)
    tmp = out.with_suffix(".part")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(out)  # atomic: a reader sees all of it or nothing


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
        write_transcript(mp3, result)
        done += 1
        audio_s += result["duration_s"]
    return done, audio_s


def http_download(url: str, dest: Path, timeout: float = 10.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "pitsense"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    if not body:
        raise OSError("empty response")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(dest)


class LiveRadio:
    """Live team radio: download each new clip, then transcribe it, both off the caller's thread.

    ``submit(path)`` (a capture path like ``TeamRadio/X.mp3``) only queues. One thread downloads
    ``base_url + session_path + path`` into ``session_dir/<path>`` (a few retries, then it gives up
    quietly: offline means no audio, never a crash); a second thread runs the transcriber and writes
    ``X.json`` next to the mp3 once it is really done. The race state shows the text when that file
    exists, so live latency is real latency. Clips already on disk (mp3 or transcript) are reused.
    ``downloader(url, dest)`` and ``transcriber(mp3) -> dict`` can be replaced (tests).
    """

    def __init__(self, session_dir: Path, *, base_url: str = STATIC_URL, downloader=None, transcriber=None,
                 model_name: str = DEFAULT_MODEL, transcribe: bool = True, retries: int = 3, backoff_s: float = 2.0) -> None:
        self.dir = Path(session_dir)
        self.base_url, self.model_name, self.retries, self.backoff_s = base_url, model_name, retries, backoff_s
        self.downloader = downloader or http_download
        self.transcriber = transcriber
        self.do_transcribe = transcribe
        self.session_path: str | None = None
        self.stats = {"queued": 0, "downloaded": 0, "reused": 0, "download_failed": 0, "transcribed": 0, "transcribe_failed": 0}
        self.error: str | None = None  # last problem, for /api/health
        self._seen: set[str] = set()
        self._pending: list[str] = []  # submitted before the session path was known
        self._lock = threading.Lock()
        self._dl: queue.Queue = queue.Queue()
        self._tr: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def set_session_path(self, path: str | None) -> None:
        if not path or self.session_path == path:
            return
        with self._lock:
            self.session_path = path
            pending, self._pending = self._pending, []
        for p in pending:
            self._dl.put(p)

    def submit(self, path: str) -> None:
        if not isinstance(path, str) or not CLIP_RE.match(path):
            return
        with self._lock:
            if path in self._seen:
                return
            self._seen.add(path)
            self.stats["queued"] += 1
            known = self.session_path is not None
            if not known:
                self._pending.append(path)
        self._ensure_threads()
        if known:
            self._dl.put(path)

    def _ensure_threads(self) -> None:
        with self._lock:
            if self._threads:
                return
            self._threads = [threading.Thread(target=self._download_loop, name="pitsense-radio-dl", daemon=True),
                             threading.Thread(target=self._transcribe_loop, name="pitsense-radio-asr", daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 10.0) -> bool:
        """Wait until everything queued so far is downloaded and transcribed (tests)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                waiting = bool(self._pending)
            if not waiting and self._dl.unfinished_tasks == 0 and self._tr.unfinished_tasks == 0:
                return True
            time.sleep(0.02)
        return False

    def _download_loop(self) -> None:
        while not self._stop.is_set():
            try:
                path = self._dl.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._fetch(path)
            except Exception as exc:  # never kill the worker
                self.stats["download_failed"] += 1
                self.error = f"download: {type(exc).__name__}: {str(exc)[:120]}"
            finally:
                self._dl.task_done()

    def _fetch(self, path: str) -> None:
        dest = self.dir / path
        if dest.is_file():
            self.stats["reused"] += 1
        else:
            url = f"{self.base_url}{self.session_path}{path}"
            for attempt in range(self.retries):
                try:
                    self.downloader(url, dest)
                    self.stats["downloaded"] += 1
                    break
                except Exception as exc:
                    self.error = f"download: {type(exc).__name__}: {str(exc)[:120]}"
                    if attempt + 1 < self.retries and not self._stop.wait(self.backoff_s * (attempt + 1)):
                        continue
                    self.stats["download_failed"] += 1
                    log.warning("radio clip %s not saved (%s)", path, self.error)
                    return
        if self.do_transcribe and not transcript_path(dest).exists():
            self._tr.put(dest)

    def _transcribe_loop(self) -> None:
        while not self._stop.is_set():
            try:
                mp3 = self._tr.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self.transcriber is None:
                    self.transcriber = self._default_transcriber()
                write_transcript(mp3, self.transcriber(mp3))
                self.stats["transcribed"] += 1
            except Exception as exc:
                self.stats["transcribe_failed"] += 1
                self.error = f"transcribe: {type(exc).__name__}: {str(exc)[:120]}"
                if self.transcriber is None:  # model could not load: keep the audio, stop transcribing
                    self.do_transcribe = False
            finally:
                self._tr.task_done()

    def _default_transcriber(self):
        try:
            model = load_model(self.model_name, "cuda")
        except Exception:
            model = load_model(self.model_name, "cpu")
        return lambda mp3: transcribe_file(model, mp3, self.model_name)


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
