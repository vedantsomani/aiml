# Feeds: telemetry, positions, team radio, track outlines

Three live-timing topics that the timing reducer does not use, and what is built on them.
Code: `feeds.py` (stores), `trackmap.py` (outlines), `radio.py` (transcription), decoding in
`events.py` / `live.py`.

## Fetching

```
pitsense fetch --year 2025 2026 --telemetry      # + CarData.z, Position.z
pitsense fetch --year 2025 2026 --radio          # + TeamRadio and every mp3 it names
pitsense radio transcribe --year 2025 2026       # Whisper, offline, GPU
pitsense tracks --year 2025 2026                 # track outlines from Position.z
```

Opt-in because of size. Everything lands in `data/raw/<session>/` next to the timing streams
(mp3 in `TeamRadio/`, transcripts as `TeamRadio/<name>.json`). None of it is committed.
`load_archive_session` ignores these topics unless `feeds=True`, so the benchmark, the leak
check and every existing replay are byte-for-byte unchanged (verified: all 39 race parquet files
and `history.json` for 2025-26 are identical to the base commit; build time 43 s vs 42 s).

## Formats

| Topic | Payload (after `decode_z`) |
|---|---|
| `CarData.z` | `{"Entries": [{"Utc": ..., "Cars": {"<n>": {"Channels": {"0": rpm, "2": km/h, "3": gear, "4": throttle %, "5": brake, "45": drs}}}}]}` |
| `Position.z` | `{"Position": [{"Timestamp": ..., "Entries": {"<n>": {"Status": "OnTrack"/"OffTrack", "X", "Y", "Z"}}}]}` |
| `TeamRadio` | `{"Captures": [{"Utc", "RacingNumber", "Path"}]}`; later messages use `{"Captures": {"<i>": {...}}}` |

`.z` payloads are a JSON string of base64 of raw deflate (`zlib.decompress(b64decode(s), -15)`),
in the archive (`*.z.jsonStream`) and live (`msg[1]`). `events.decode_z` handles both; archive and
recording loaders emit events whose `data` is already decoded. X/Y/Z are 1/10 m (a lap of
Melbourne measures 52 000). Brake is 0 or 100+; DRS codes >= 10 mean the flap is open.

## As-of rules

* A message carries ~4 entries (~0.25 s apart) whose own `Utc` is earlier. Every entry is stored with
  `t` = the message's publish time and is invisible before it: `TelemetryStore` only ever holds what
  `RaceState.apply` has been given, and every accessor takes `t=` (<= newest message, else `LeakageError`)
  and filters on the published `t` column.
* In the 2025 archive the newest entry of a message is a constant 2.03 s older than the message's
  clock time (5th-95th percentile 2.0317-2.0325 s, Hungary 2025). "Last N seconds" is measured on the
  car's own `utc` clock, from its newest published sample.
* Radio: a message is available from its publish time. Its transcript is **known at
  message time + `feeds.RADIO_LATENCY_S` = 15 s**, in replay and live alike. Budget: mp3 fetch ~1 s,
  Whisper 0.36 s per clip on the GPU (see below), the rest is margin for queueing. Before that
  `RadioMessage.text` is `None`. Change it per run: `Feeds(meta, latency_s=...)`.
* Track outlines come only from races that **ended before** the one asked about (`track_asof`).

## API

```python
state.feeds.telemetry.latest_position()             # {car: {x, y, z, on_track, t, utc}}
state.feeds.telemetry.position_history("44", 60.0)  # arrays: t, utc, x, y, z, on_track
state.feeds.telemetry.telemetry("44", last_s=30.0)  # arrays: t, utc, rpm, speed, gear, throttle, brake, drs
state.feeds.telemetry.latest_telemetry()            # newest sample per car
state.feeds.radio.messages("44", last_s=300.0)      # [RadioMessage(car, t, utc, path, audio, text, known_at)]

from pitsense.trackmap import track_asof
track_asof(ref.circuit_key, ref.start_utc)          # {"x": [...400], "y": [...], "start": {x, y, heading_deg}, "pit": {"x","y"} | None, ...}
```

`state.feeds` is not part of `view()` or `fingerprint()`. Feed events are applied by
`RaceState.apply` only when the log was loaded with `feeds=True` (archive) or `feeds=True` in
`load_recording` / `RecordingTail` (live). Memory is bounded: a numpy ring per car and feed,
3000 samples each (about 10 minutes), 7 MB for a 20-car grid however long the race.

## Track outlines

`trackmap.build_outline` takes the fastest clean lap (green, no pit) of a finished race, keeps the
OnTrack samples of that car between the lap's start and end, requires no hole over 2 s and a closed
loop, and resamples to 400 evenly spaced points. The pit lane is the car path of the quickest
green-flag stop between pit entry and exit (60 points). Start/finish is the first point of the lap
(the timing feed's line-crossing time, so within about a second of the true line). Cached as
`data/feeds/tracks/<circuit_key>/<race slug>.json` (~8 KB each, 38 races, 25 circuits).

## Measured (2025 + 2026 races, 39 sessions)

| What | Number |
|---|---|
| CarData.z | 289 MB total, ~7.4 MB per race |
| Position.z | 309 MB total, ~7.9 MB per race |
| TeamRadio mp3 | 1108 files, 183 MB (4 not published) |
| Decode + parse, both topics, one race (Hungary 2025) | 2.4 s |
| Full replay with feeds (86 k events, 2.6 h race) | 1.4 s, ~64 k events/s, ~6900x real time |
| `latest_position()` | 0.9 ms |
| Track outlines for all 39 races | about 1 min; 38 built |

## Transcription

`faster-whisper` 1.2.1 (CTranslate2) on the RTX 5070, float16, greedy decoding, English, VAD on, a
short F1-vocabulary prompt. Setup notes: `pip install faster-whisper nvidia-cublas-cu12
nvidia-cudnn-cu12` (ctranslate2 needs CUDA 12 DLLs; `radio._add_cuda_dlls` finds them), and
`radio.decode_audio` decodes the mp3 itself because faster-whisper 1.2.1 passes an option PyAV 19 removed.

Model choice on 40 random 2025 clips (415 s of audio): `base.en` 240 ms/clip, `small.en` 355 ms/clip. They
agree on 75 % of words; `small.en` is right where they differ ("brake pedal" vs "pre-pedals", "Leclerc",
"Verstappen"), so **`small.en`** is the default. All 1108 clips (197 min of audio): 405 s, 29x real time.
81 clips (7 %) transcribe to nothing (squelch, noise). Transcripts are cached per file and never recomputed.

## Open issues

* 2026 Monaco: `Position.z` holds only 8 087 samples (about 32 min, X <= 0), so no outline; 2026 Hungary's
  outline is 3.67 km with no pit lane (2025: 4.32 km), treat both as suspect. Later outlines of those
  circuits come from other years, as-of.
* Start/finish is the published crossing time, +/- 1 s of track (about 60 m).
* Telemetry has no tyre or engine temperatures: the feed's channels are RPM, speed, gear, throttle, brake, DRS only.
* Live: `load_recording` / `RecordingTail` skip feed topics unless `feeds=True`; the pit-wall runtime must
  pass it. The live feed's real publish delay was not measured here (no live session).
