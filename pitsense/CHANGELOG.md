# Changelog

## Unreleased (branch `realworld`)

Decisions for the viewer spec (simplest option that keeps replay == live and leak safety):

- **No FastAPI / WebSocket / pydantic rewrite, no React + Vite app.** The existing stdlib server (`web/server.py`:
  JSON API + server-sent events, token, replay controls with lap checkpoints) and the dashboard (`web/`, no build
  step) already deliver the backend and screens the spec asks for. Rewriting them would duplicate tested code and
  add a Node toolchain; new panels go into the existing dashboard. The snapshot stays a plain dict (versioned, see Phase 2).

### Phase 1: recorder streams
- Already in place: the recorder subscribes to `WeatherData`, `CarData.z` and `Position.z` by default (`live.TOPICS`),
  the .z topics are decoded on the way in and kept raw in the log, `RaceState.weather` holds the latest weather,
  `feeds.TelemetryStore` keeps per-car ring buffers of CarData (speed, rpm, gear, throttle, brake, DRS) and Position
  (x, y, z) with the publication time of every sample, and `pitsense fetch --telemetry` pulls them for past races.
- New: `pitsense record --topics T1 T2 ...` restricts the recording; the timing core (`live.CORE_TOPICS`) is always
  kept and unknown names are refused.
- New test: corrupting every weather, CarData and Position message after t leaves the state and every feed query
  up to t bit-identical; asking past the newest message raises `LeakageError`.

### Phase 2: viewer backend
- Already in place: `pitsense pitwall` serves the dashboard, a JSON API and server-sent events; replay play / pause /
  speed / seek by lap or bookmark, rebuilt from lap checkpoints and never reading past the seek point; track outlines
  derived from Position data and cached per circuit (`trackmap.py`); end state of a followed recording equals replay.
- New: `pitsense serve` (alias of `pitwall`). Snapshots carry `schema` (`SNAPSHOT_SCHEMA = 2`) and a `predictions`
  block (`as_of`, `input_t`, model bundle, per car pit probabilities, rejoin position, undercut threat), and
  `extra.telemetry` (speed, gear, throttle, brake, DRS, rpm); the ~3 Hz `pos` events carry telemetry too.
- New test: a recording followed as live and the same recording replayed make identical calls.
- Kept as is (simplest option): server-sent events instead of a WebSocket (controls are POSTs), snapshots every 3 s
  of session time plus every leader lap with positions and telemetry at ~3 Hz, instead of a fixed 4 Hz snapshot.

### Phase 3: viewer screens
- Already in place (dashboard, `web/`): track map with team-coloured cars, driver codes, interpolation and the
  selected car highlighted; timing tower (position, gap, interval, compound and age, last lap, pit); header with lap,
  clock and coloured flag; weather; replay bar (play / pause, speed, lap jumps, live / replay); dark, responsive.
- New: **Championship** panel (Season tab): drivers' standings before this race, the race's points if it finished
  in the current order, projected total and places moved. Points are data-driven by era (`standings.POINTS`: no
  fastest-lap point from 2025, sprint points by year). Results come from each finished session's own feed
  (`pitsense standings --year Y` -> `data/bench/results_<year>.json`); a race sees only sessions that ended before
  it started. Simplification: the feed's final order, so penalties given after the session are not reflected.
- New: **Driver** card (Team tab): speed, gear, throttle and brake bars and DRS for our cars, from the newest CarData
  sample, updated about 3 times a second.

### Phase 4: PitSense overlays
- Already in place: pit probability within 1 / 3 laps in the tower with heat colouring (cross-race models, else the
  rivals engineer); rejoin position if boxing now; every call change logged to a timestamped JSONL with session and
  wall-clock time (pre-registered, scored by `shadow-score` and `bench/callscore.py`).
- New on the track map: a **rejoin ghost** (dashed ring at the car it would come out behind, labelled with the
  predicted position) and **undercut threat lines** (car behind -> threatened car when `rivals__undercut_threat`
  >= 0.35), for our cars or, with "all cars", the whole field.
- The snapshot's `predictions` block (Phase 2) carries the numbers behind them with `as_of` and the model bundle.

### Phase 5: analysis panels (Analysis tab, `GET /api/analysis`)
- `insights.py`, on any live or recorded race state, as of now. Clean laps: a lap time, not lap 1, no in / out lap,
  green flag the whole lap, at least 1.0 s to the car ahead; fuel-corrected with a linear 0.055 s/lap term.
- **Tyre degradation**: per car and stint, lap time against tyre age with the fitted line, its 80 % band (residual
  10th-90th percentiles) and the rate in s/lap. The fit is behind `insights.DegFit` (`LinearDeg` now) so a Kalman
  state estimator can replace it.
- **Track evolution**: median clean lap per lap and its trend (s/lap), also published as the tyre engineer's
  `track_evo_s_per_lap` race value (a feature for the pace model).
- **Two drivers**: lap-time overlay, the gap at each line crossing, per-stint pace delta.
- **Speed trace**: speed against distance for a lap, CarData aligned to distance through Position.
  Simplifications: no sector markers (sector times are not kept in the lap record); only laps still in the feeds'
  window (about ten minutes of telemetry per car) can be traced.
- The rival-cover model (in the first commit of this branch) is now **off by default**: on the 2026 races it did not
  improve the calls (box precision 0.536 -> 0.533, stop recall 0.448 -> 0.434, stay-out 0.932 -> 0.935).

### Phase 6: driving coach (Analysis tab, "driving coach")
- `coach.py`: two laps (a driver against another driver's lap, or against their own best clean lap) resampled onto
  a 5 m distance grid; cumulative time delta; corners from the reference lap's speed minima; per corner the brake
  point, minimum speed and full-throttle differences and the time lost; the top three as sentences ("Turn 1: brake
  30 m earlier, -12 km/h min speed, potential gain 0.21 s") and the reference line coloured by where time goes.
- Telemetry through an adapter (`coach.LapSource`); only `F1Adapter` is implemented. Corners are numbered in the
  order driven, not by the circuit's official turn numbers.
- Checked on Suzuka 2026 (NOR's best lap against PIA's): -0.89 s on the distance grid against -0.62 s by lap times
  (telemetry is sampled at about 4 Hz).
