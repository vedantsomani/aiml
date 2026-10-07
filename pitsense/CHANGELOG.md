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
