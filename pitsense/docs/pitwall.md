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
- **Alerts and race control (right).** Active alerts sorted critical, warn, info, each with the
  engineer that raised it; below, the latest race-control messages.

Today the head of strategy returns NO_CALL, so cards show NO CALL until it ships. Full `Call` records
(action, compound, confidence, reasons, plans) render as soon as they arrive.

## 6. API (read-only, JSON)

| Endpoint | Content |
|---|---|
| `GET /api/snapshot` | latest `Snapshot` (`pitwall/types.py`) plus an `extra` block: mode, speed, inferred_order, model bundle, weather, team colours, recent race control |
| `GET /api/calls` | `{"current": [...], "log": [...]}`: calls now, and every change since start (last 500) |
| `GET /api/alerts` | `{"active": [...], "log": [...]}` |
| `GET /api/health` | status, mode, events, snapshots, last-event age, snapshot latency (p50, max), errors, call-log path |
| `GET /api/stream` | server-sent events: `snapshot` (the snapshot JSON) and `status` |

```
curl -s localhost:8765/api/snapshot | python -m json.tool | head
curl -N localhost:8765/api/stream
```

A snapshot is published every 3 session seconds (scaled by speed) and on every leader lap. Timing is
on the session clock, so the same input gives the same snapshots and call log at any speed. The server
binds to 127.0.0.1 only unless you pass `--host`; it has no authentication, so keep it on a trusted
network.

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
