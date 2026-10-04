# The live pit wall: a day-one guide for a team

One command follows a race and shows a strategy screen in your browser. It runs every engineer
(tyre, pit stop, rivals, rules, weather, models, strategy, head of strategy), serves a JSON API for
your own tools, and logs every call so you can score it afterwards.

## 1. Install

```
pip install -e .            # from the repo; Python 3.10+
pip install -e .[live]      # only if you will record live sessions (FastF1)
```

Race data for replays: `pitsense fetch --year 2026 --race hungary` (downloads the public archive).
Optional trained models: `pitsense bench build` then `pitsense train` (see `docs/models.md`).
The wall loads the newest bundle that was trained before the race started; with none it runs
without models (the screen says so in the top bar).

## 2. Replay demo

```
pitsense pitwall --year 2026 --race hungary --speed 20 --team ferrari
```

Opens http://127.0.0.1:8765/. `--speed 1` is real time, `--speed 0` is as fast as possible. The
grid wait before the start is skipped (pacing begins one minute before the lights). `--no-browser`
only prints the address. `--port 0` picks a free port.

## 3. Record, then follow live

```
pitsense record --out data/live/race.jsonl --minutes 180      # in one terminal
pitsense pitwall --file data/live/race.jsonl --follow --team ferrari   # in another
```

or in one go: `pitsense pitwall --live --team ferrari` (starts the recorder and follows its file,
default `data/live/<time>-live.jsonl`; `--no-auth` skips the F1 TV sign-in, `--minutes` limits it).
`--file` without `--follow` replays a finished recording at `--speed`. The follower reads the backlog
at once, then polls for new lines; a half-written last line is held back until complete.

**No-auth feeds have no car positions.** The runtime then orders the cars itself (most laps, then
smallest gap to the leader) and the screen shows an amber "ORDER INFERRED" banner. Treat positions
as approximate around pit stops. `state.py` is untouched: the order is set by the runtime only.

## 4. Choose your team

`--team ferrari` matches the team name in the feed (exact, or the only name containing the text).
`--team 16,44` takes car numbers. With no team, click any car in the tower to pin it; pins are
remembered in the browser.

Each browser can also choose its own team: the selector in the header (saved in that browser's
localStorage, "no team" = pins only). It only changes what that screen highlights and asks about; the
server's `--team` is just the default and nothing on the server changes, so a driver coach, a strategist
and a phone can each follow a different car.

## 4b. Phone, tablet and several viewers

- Phone (under 700 px): one section at a time, switched by the tab bar at the bottom (Tower, Map,
  Radio, Team, Alerts); a sticky banner under the header shows the current call of your cars on every
  tab; buttons are 44 px or more; hold the big microphone button to talk (works with touch, no long-press
  menu). Less important tower columns are hidden; there is no horizontal scroll at 360 px.
- Tablet (700-1100 px): two columns. Desktop: as before.
- Open it on a phone: start with `pitsense pitwall --race hungary --speed 20 --team ferrari --host 0.0.0.0 --token pick-a-word`.
  It prints `On the same Wi-Fi open http://192.168.x.y:8765/?token=pick-a-word`; open that on the phone (same
  Wi-Fi). The first load stores the token in a cookie. Without the token every page and API call (and the
  POSTs) answers 401. The microphone needs https or localhost on most phones, so push-to-talk from a phone
  over plain LAN http is blocked by the browser; typing a question works. Allow the port in the firewall.
- Concurrency: every browser has its own server thread and a bounded queue (8); a slow one loses old
  updates, a stalled one is cut after 8 s, more than 64 viewers get 503. The publishing loop never waits
  for a browser. `/api/health` shows `"viewers": N`.
- The token is a shared secret over plain http: fine for a home or garage network, not for the internet.

## 5. The screen

Dark, high contrast, one page.

- **Top bar.** Race, lap n / total, session clock, a track-status banner (green, yellow, SAFETY CAR
  in orange, VSC in purple, RED FLAG in red, with seconds since it began), air and track
  temperature, humidity, rain, the current pit loss in seconds, the mode (REPLAY 20x, FOLLOW, LIVE)
  with "models" or "no models", and the connection state. Below it, warning banners (inferred order,
  red flag).
- **Timing tower (left).** One row per running car: position, team colour, three-letter name and
  number, gap to leader, interval, tyre chip (S red, M yellow, H white, I green, W blue) with age in
  laps, stops, last lap, **box now** (the position the car would rejoin in if it pitted this lap,
  green if it is no worse, red if worse), **pit %** (probability of a stop within 1 / 3 laps, orange
  above 25%, red above 50%), and the car's call. Your cars have a teal marker. Retired cars are listed
  under the table.
- **Our cars (top right).** One card per team car: position, tyre, the call as a large badge (BOX,
  PREPARE BOX, BOX IF SC, STAY OUT; NO CALL is greyed), tyre to fit, confidence, the reasons in
  order, Plan A and Plan B (stops as lap and compound, expected position, trigger), and a line of
  tyre and rival numbers (degradation, cliff risk, undercut threat and chance).
- **Track map (right, top).** SVG, offline. The circuit outline (`trackmap.track_asof`: built only from
  races that finished before this one), pit lane (dashed blue), start/finish bar, and one dot per car in
  its team colour. Your cars are larger, ringed in white, labelled with the TLA and drawn on top; hover any
  car for its position, tyre and the gap to the car ahead and behind. Dots glide between updates. The map
  is rotated so the long axis of the circuit is horizontal and fitted to the panel. With no outline
  (new circuit, or a recording) it draws a provisional outline built from the cars' own positions once two
  laps are done (`provisional: true`), and the trails until then. Cars in the pits or off track fade.
- **Pit wall radio (right, under the map).** A chat feed for the focus cars, oldest first: real driver
  radio on the left (transcript, or "transcript pending" for the first 15 s, and a play button for the real
  mp3), our pit wall on the right (the head-of-strategy call in the voice's words, with a play button for the
  Piper audio) and engineer alerts for those cars (plus any critical alert). Driver speech is only ever real
  radio. "radio on" plays new driver clips and pit-wall clips for your cars one after another, never
  overlapping (a backlog of more than 6 drops the oldest at high replay speed); the connect-time backlog
  is never read out.
- **Ask box (under the radio).** Type a question or hold the microphone; see section 6b.
- **Alerts and race control (right).** Active alerts sorted critical, warn, info, each with the
  engineer that raised it; below, the latest race-control messages.

Today the head of strategy returns NO_CALL, so cards show NO CALL until it ships. Full `Call` records
(action, compound, confidence, reasons, plans) render as soon as they arrive.

## 6. API (JSON; read-only apart from the two ask endpoints)

| Endpoint | Content |
|---|---|
| `GET /api/snapshot` | latest `Snapshot` (`pitwall/types.py`) plus an `extra` block: mode, speed, inferred_order, model bundle, weather, team colours, recent race control |
| `GET /api/calls` | `{"current": [...], "log": [...]}`: calls now, and every change since start (last 500) |
| `GET /api/alerts` | `{"active": [...], "log": [...]}` |
| `GET /api/health` | status, mode, events, snapshots, last-event age, snapshot latency (p50, max), errors, call-log path |
| `GET /api/stream` | server-sent events: `snapshot` (the snapshot JSON), `pos` (car positions, about 3 Hz on the wall clock at any speed) and `status` |
| `GET /api/teamradio?car=N&i=K` | the K-th real driver radio mp3 of car N (`audio/mpeg`); only published clips inside the session's `TeamRadio/` folder, anything else 404 |
| `GET /api/radio.wav?car=N[&i=ID]` | our pit wall's voice message for car N (latest, or by `id` from `extra.wall_msgs`), spoken with Piper |
| `POST /api/ask` | `{"car": "16", "text": "what if we box now?"}`: ask the pit wall (section 6b) |
| `POST /api/ask_audio?car=N` | body = recorded audio (webm/opus, ogg, wav, mp4): transcribed on the server, then as `/api/ask` |

Snapshot `extra` also carries: `positions` `{t, cars: {car: [x, y, on_track]}}` (1/10 m, newest published
sample, refreshed at most every 0.5 session s), `track` `{x, y, start, pit, key}` or null (the outline,
sent with every snapshot so a new browser has it; the page rebuilds only when `key` changes),
`team_radio` (last 80 driver clips of the session: `id, car, tla, t, lap, text, audio`; `text` is null
until 15 s after the message in an archive replay; live and followed recordings show it when the
transcript file really exists, and `audio` is null until the mp3 is on disk), and `wall_msgs` (last 120 pit-wall entries: voice calls and alerts with
`id, kind, car, t, lap, text`). A `pos` event is `{t, speed, cars}` plus `team_radio` when it changed.
Feed events (positions, radio) never run the engineers or publish snapshots, so calls are unchanged.
All sources load feeds (archive: Position.z and TeamRadio; CarData is skipped).

**Live and followed recordings.** The runtime keeps the merged `SessionInfo` (`Meeting.Circuit.Key`,
`StartDate` + `GmtOffset`, `Path`):

- *Outline*: `trackmap.track_asof(circuit_key, session start)`, looked up as soon as SessionInfo arrives.
  If no earlier race of that circuit is stored, `trackmap.live_outline` builds one from the published
  positions of the quickest clean lap once two laps are done (as-of; no pit lane; `track.provisional`).
- *Radio*: every TeamRadio capture is queued on a `radio.LiveRadio` (two daemon threads, the event loop never
  waits). It downloads `https://livetiming.formula1.com/static/` + SessionInfo `Path` + capture path into
  `<recording>.session/TeamRadio/` (next to the recording), then transcribes with faster-whisper `small.en`
  (GPU, CPU fallback) and writes `<clip>.json` atomically. The text shows in `extra.team_radio` the moment
  that file exists (no fixed 15 s), and `/api/teamradio` serves the mp3 from that folder once it is there.
  `pitsense record` fills the same folder. Following a finished recording reuses any mp3 or transcript already
  in `<recording>.session/` and downloads and transcribes what is missing.
- *Offline*: a failed download is retried 3 times (2 s, 4 s), then skipped with a warning. The pit wall, map
  and calls carry on, the clip has no audio and no text, `/api/teamradio` answers 404, and
  `/api/health` shows `radio: {queued, downloaded, reused, download_failed, transcribed, transcribe_failed,
  error}` and `track: stored | provisional`. If the Whisper model cannot load, audio is still saved.

```
curl -s localhost:8765/api/snapshot | python -m json.tool | head
curl -N localhost:8765/api/stream
```

A snapshot is published every 3 session seconds (scaled by speed) and on every leader lap. Timing is
on the session clock, so the same input gives the same snapshots and call log at any speed. The server
binds to 127.0.0.1 only unless you pass `--host`; with `--token` every request needs it (see 4b),
without it there is no authentication, so keep it on a trusted network. A POST from a page on another origin is refused.

## 6b. Ask the pit wall (text and voice)

Under the radio conversation there is a text box and a microphone button (hold it, speak, release). The
question and the answer join the conversation as a "you" bubble and a pit-wall bubble; the answer is spoken
(Piper, `/api/radio.wav?car=N&i=<reply id>`) even with "radio off". A car selector appears when there are
two or more of our cars. The microphone works on `localhost` (or https) in a browser with MediaRecorder.

```
curl -s localhost:8765/api/ask -d '{"car":"16","text":"what if we box in 3 laps for hards?"}'
curl -s "localhost:8765/api/ask_audio?car=16" -H "Content-Type: audio/webm" --data-binary @question.webm
```

Reply: `{ok, car, tla, answer, source, intent, whatif, audio, you, reply, ms}` (`ask_audio` adds `heard`, `asr_ms`).
`source` is `whatif` (the engine's own words), or `llm` / `slm` / `template` / `facts` for fact questions
(the fine-tuned model if it loaded and passed the guard, else the template). Errors: 400 bad body, 403
cross-origin POST, 413 body over 8 MB, 422 nothing heard, 503 no transcriber.

**Parsing** (`whatif.parse_question`, regexes first, then the voice's free-form router `composer.free_kind`):
what-ifs are *box now / in N laps / on lap L / for SOFT, MEDIUM, HARD*, two options separated by "or" /
"versus", *stay out to the end*, *safety car or VSC now / next lap / in N laps* and any of them *against a
rival* (TLA, number, surname, "the car ahead / behind"). Fact questions are gap (ahead, behind or a car),
tyre age, pit window, plan B, why, chance of a safety car, and everything else goes to the voice's free-form
answer (rejoin position, pit loss, weather, laps to go, penalty ...).

**What-if engine** (`whatif.what_if`, `src/pitsense/whatif.py`). It builds the strategy engineer's field and
the same 288 seeded futures (`analysis.build_field`, `Draws`, `simulate_field`) and scores the asked plans
with `evaluate_plans` on those futures, so every plan sees the same race and the comparison is paired. A
"box now" without a compound tries SOFT / MEDIUM / HARD (and a second stop when one stint is too long) and
reports the best by the head's utility. The comparison is plan A unless two options were given. A safety car
question re-draws the futures with the neutralisation forced to start in N laps, then compares staying on plan
A with the best stop under it. Result: `scenario` and `versus` (`label`, `stops`, `exp_pos`, `sd`, `p_top10`,
`p_podium`, `race_time_s`, `p_ahead_rival`), `p_gain` / `p_loss` / `p_same` (the scenario finishes ahead of /
behind / level with the other option, same futures), `delta_pos`, `delta_time_s`, `reasons`, `notes` (rule
breaches such as "must still fit a second compound"), `ms`. As-of: only the race state at `state.t` is read,
so the answer is identical from the full log and from `log.until(t)` (tested). Deterministic (seeded by race,
not by clock). About 0.1 to 0.25 s on a 2026 field; the answer is refused before lap 2, in the wet and after
the flag. The simulator models dry tyres only and the rival plays its default strategy, so "ahead of VER"
is a probability under that assumption, not a guarantee.

**Threading.** The loop applies each event under `PitWallRuntime.sim_lock`; a question takes the same lock,
so it sees one still moment and the loop waits for the (short) answer. The voice and Whisper run outside it.

**Speech to text** (`asr.py`): PyAV decodes the browser's webm/opus (no ffmpeg binary), then faster-whisper
`small.en` through `radio.load_model` (GPU, CPU fallback; loaded on the first spoken question, about 3 s).
Tests and other front ends can set `rt.transcriber = fn(bytes) -> str`.

## 7. The call log and shadow scoring

Each run writes `data/pitwall/calls-<race>-<time>.jsonl` (`--log-dir` to change). One JSON line per
**change** of a car's call (a first NO_CALL is not logged) and per new alert, with `t` (session
seconds), `wall` (Unix time), `lap` (leader lap) and `car_lap` (the car's own lap):

```
{"kind":"call","t":2711.4,"wall":1790000000.1,"lap":21,"car_lap":21,"car":"16","action":"BOX","compound":"HARD","confidence":0.8,"reasons":[...],"plan_a":{...}}
```

After the race:

```
pitsense shadow-score --log data/pitwall/calls-....jsonl --year 2026 --race hungary --k 2
```

It replays the finished race for the real stops (red-flag stops excluded) and reports: box-call
precision (a BOX, PREPARE_BOX or BOX_IF_SC call from lap L is right if that car's real in-lap is within
L-k..L+k), STAY_OUT accuracy (no stop in L..L+k), and stop recall (real stops that had a box call
within +-k laps). NO_CALL is not scored. `--cars 16 44` limits it, `--json` for machines.

## 8. Performance (this machine, 2026 Hungarian GP, 73,835 events, 70 laps)

- No models: 18,500 events/s, whole race in 4 s (2,350x real time), 322 snapshots, snapshot p50 3.8 ms,
  p95 8.9 ms, max 18 ms.
- With the model bundle: snapshot p50 450 ms, max 860 ms (the `models` engineer builds feature rows for
  every car), whole race in 121 s (78x). Fine for live; at high replay speeds the tick is longer
  (`--speed 20` publishes every 60 session seconds).

## 9. Troubleshooting

- "live recording needs the optional dependency": `pip install -e .[live]`.
- Blank page, "connecting": the browser needs the stream; open `/api/health` to see the loop's status.
- Nothing moves in follow mode: check `last_event_age_s` in `/api/health`; the recorder may be waiting
  for the session to start.
- Model bundles are pickle: load only ones you trained yourself.
