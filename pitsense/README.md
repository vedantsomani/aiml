# PitSense

A virtual F1 pit wall that runs on the public timing feed. It has one engineer per role on a race-strategy team, and every one of them is scored against what actually happened, without ever seeing the future.

| Engineer | Job | Doc |
|---|---|---|
| Data | rebuilds the race from the raw feed; every Grand Prix 2018–2026, validated against FastF1 | [data](docs/data.md) |
| Tyres & pace | fuel-corrected pace, wear, tyre-cliff risk, pace on fresh tyres | [tyre](docs/engineers/tyre.md) |
| Pit stops | what a stop costs now (green / SC / VSC), where the car rejoins | [pitstop](docs/engineers/pitstop.md) |
| Rivals | who pits when, undercut threats and chances | [rivals](docs/engineers/rivals.md) |
| Rules | race control as structured facts: penalties, SC phases, the compound rule | [rules](docs/engineers/rules.md) |
| Weather | rain in the next 10 minutes, slick / intermediate crossover | [weather](docs/engineers/weather.md) |
| Models | cross-race models served live, identical to the benchmark | [models](docs/models.md) |
| Strategy & head | Monte Carlo race simulation, Plan A / B, BOX / STAY OUT calls with reasons | [strategy](docs/engineers/strategy.md) |
| Voice | fine-tuned SmolLM2 plus our own small model: radio calls and briefs, fact-checked, spoken by Piper | [voice](docs/engineers/voice.md) |
| Mechanics | telemetry, radio and chief mechanic: power loss, slowdowns, problem reports | [mechanics](docs/engineers/mechanics.md) |
| Safety car | calibrated SC / VSC probability for the next 2 laps, "safety car likely" alert | [safetycar](docs/engineers/safetycar.md) |
| Rival radio | other teams' radio as alerts: "box box", "Plan B", "tyres are gone" | [rivalradio](docs/engineers/rivalradio.md) |
| Weekend | tyre sets left from FP/quali; qualifying pit wall (cut line, send-now calls) | [weekend](docs/engineers/weekend.md) |

```bash
pitsense pitwall --year 2026 --race hungary --speed 20 --team ferrari   # replay with the dashboard
pitsense pitwall --live --team ferrari                                  # live race (F1 TV sign-in)
pitsense pitwall --live --team ferrari --host 0.0.0.0 --token secret    # also open it on a phone on your Wi-Fi
pitsense pitwall --year 2026 --race hungary --session qualifying        # qualifying pit wall
pitsense briefing --year 2026 --race bahrain --team ferrari             # pre-race strategy briefing (HTML)
pitsense report --year 2026 --race bahrain --team ferrari               # post-race report: every call graded
pitsense pitwall --year 2026 --race hungary --team ferrari --risk protect   # rank plans by their downside
pitsense calibrate                                                       # refit calibrated call confidence (as-of)
```

The dashboard shows the timing tower, a live track map, your team's calls with Plan A/B, and a radio conversation panel. In that panel, real driver radio is transcribed by Whisper, and the pit wall's calls are written by the fine-tuned voice model and spoken by Piper. Type or hold-to-talk questions such as "what if we box now?", "what if the safety car comes next lap?" or "box now for inters?"; the race simulator answers them (the wet simulator when it is wet). Each call can be accepted or rejected with a reason, shows what it changed from and why, and carries a confidence calibrated on earlier races; `pitsense shadow-score` grades the calls and the operator's answers after the race.

How a team uses it from day one: [docs/pitwall.md](docs/pitwall.md). How it is built, and the rules every engineer follows: [docs/ENGINEERING.md](docs/ENGINEERING.md).

**Honest status.** Every number below is a backtest on the 2026 races, each scored with models trained only on races that finished before it. Box-call timing is still the weakest part (see [Results](#results)). Run it in shadow mode (`pitsense shadow-score`) before trusting it.

---

## Setup

Requires Python 3.10+ (3.12 recommended).

**Windows (PowerShell)**

```powershell
cd $HOME\Downloads\aiml\pitsense
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once: lets PowerShell run the activate script
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1                            # prompt now starts with (.venv)
python -m pip install -e ".[dev]"
python -m pytest -q                                   # the real-race tests skip until data is downloaded
```

If you can't change the execution policy, skip activation and put `.venv\Scripts\` in front of commands. For example: `.venv\Scripts\python -m pip install -e ".[dev]"`, then `.venv\Scripts\pitsense bench run`. Without either, `pip` installs into your global Python instead of the project's environment.

**macOS / Linux**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pytest -q
```

## Quick start

```bash
pitsense bench build          # downloads 2025+2026 races (~250 MB) and builds the benchmark (~2 min)
pitsense bench run            # leaderboard -> reports/leaderboard.md
pitsense replay --year 2026 --race hungary --lap 16
pitsense leakcheck --year 2026 --race italian --cuts 3
pitsense validate --year 2026 # compare our lap reconstruction with FastF1 (slow: downloads via FastF1)
```

`replay` prints the timing tower using only data up to the moment the leader finished that lap. It adds a **pit-now projection** for every car: where it would rejoin if it pitted now.

```
Lap 17/70   track: GREEN   pit loss now ≈ 19.9 s (prior circuit:2025-14-hungarian-grand-prix-race, 3 stops measured this race)

 P  CAR  GAP        INT       TYRE   STOPS  LAST       PIT NOW →
 1  NOR  +0.000     --        M 15     0    1:25.613   P1
 3  LEC  +7.386     +6.426    S 18     0    1:26.066   P8
 6  VER  +24.438    +5.616    H  1     1    1:43.352   P13
```

### Recording a live session (Sepang, Sun 4 Oct, 15:00 local = 12:30 IST)

```powershell
pitsense record --out data\live\2026-sepang-race.jsonl --minutes 150
```

- Start it about 15 minutes before lights out. It prints a status line every minute and stops after `--minutes`, or press Ctrl+C.
- **Dry run first:** test it during a practice or qualifying session.
- **F1 TV sign-in:** the first time, FastF1 prints a link to its sign-in helper page. Log in with an F1 account that has F1 TV Access, Pro or Premium. FastF1 stores the token on your computer, so you won't be asked again until it expires.
- **Without F1 TV:** add `--no-auth`. Since the 2025 Dutch GP, car positions and pit-stop times need the login, so those fields will be missing.
- **Replay the recording:** `pitsense replay --file data\live\2026-sepang-race.jsonl --lap 20`.
- **Fallback:** if `record` fails, use FastF1's own recorder (`python -m fastf1.livetiming save race.txt`). Then run `pitsense replay --fastf1-file race.txt`.
- **Not yet tested live.** The recorder hasn't been run against a live session: my build environment couldn't reach the live feed (HTTP 403). The file formats are covered by tests, but Sepang is its first real run.
- **After the race:** `pitsense fetch --year 2026` downloads the complete archived version, and `pitsense bench build` adds it to the benchmark.

---

## How it works

```
archive .jsonStream / live recording
        │  events.py   one time-ordered EventLog (t = when the message was published / received)
        ▼
state.py   RaceState.apply(event)   ← the only way state changes (deterministic)
        │  laps, pit events, stints, track status, race control — each stamped with when it was known
        ▼
bench/features.py   decision rows from the state at time t (+ a pre-race prior)
bench/labels.py     what happened next (reads the future; never imported by features)
bench/history.py    cross-race priors, UTC cutoff: only races that ended before this one started
bench/models.py     baselines + models, trained per test race on earlier races only
bench/evaluate.py   expanding-window scoring → reports/leaderboard.md
```

**Rules that keep it honest**

1. **The reducer only sees the past.** `EventLog.until(t)` is the past at `t`. The reducer only moves forward, and a decision row is written only after every message with that timestamp has been applied.
2. **No end-of-session snapshots.** The archive's `Topic.json` files hold the final state of the session (the future), so PitSense never reads them.
3. **Raw stream, not FastF1's tables.** FastF1 builds its `Laps` table after the session, using lap deletions from later messages, laps it fills in itself, and a cross-lap accuracy check. PitSense rebuilds the race from the raw stream instead.
4. **Training cutoff.** For race R, priors and models use only races that finished before R started (`HistoryStore.asof`, `assert_trained_before`).
5. **Corrupted-future test** (`pitsense leakcheck`). Rebuild every row with the future deleted (`truncate`) or shuffled (`scramble`). Rows before the cut must not change, including a hash of the live state (timing tower, tyres, track status and record counts). A unit test plants a classic leak (a row keeping a live reference to a growing list) and checks that the test catches it.

**Quirks of the feed that the reducer handles** (found while validating):

- **Repeated values are omitted.** A lap time identical to the previous one isn't re-sent, so PitSense assumes it unchanged and overrides it if a value arrives.
- **Stint records can be placeholders.** Records flagged "tyres not changed" (pit-lane starts, stops without a tyre change) don't count as a new set. A record whose compound differs from the previous set does count.
- **Tyre changes are sometimes confirmed late.** The feed can confirm a change 1–8 laps afterwards. The current stint is then re-anchored to the stop where it happened. Laps recorded earlier keep what was known at the time.
- **Tyre age is derived.** It's computed from the stint start, because the feed's own lap counter arrives about 2 s after the line and sometimes freezes mid-race.
- **Red-flag pit-lane entries aren't stops.** Everyone goes in and tyres are changed for free, so they're excluded from the labels.

## Validation against FastF1

All races 2018–2026 are validated per season in [docs/data.md](docs/data.md). The 2025–26 detail is below.

### 2025–26 (39 races, 43,856 laps)

| Check | Match |
|---|---|
| Lap time identical to the millisecond | 99.998% |
| In-lap / out-lap flags | 100% / 100% |
| Position at the line | 99.46% |
| Compound | 99.66% |
| Tyre age | 97.64% |

The remaining differences come from timing, not parsing.

- **Position:** we record the tower *at the moment of crossing*, when a car in the pit lane may already be shown behind. FastF1 re-orders by crossing times after the session.
- **Tyre age:** FastF1 uses the final, corrected stint data. We use what had been published by then. In one race (Miami 2025) FastF1 itself mislabels stints because the feed had a gap. Our derived age matches the feed's own lap counter there.

Full per-race output: `reports/validation.txt`.

## Results

Every number in this section is generated: `pitsense results --readme` writes `reports/results.md` / `results.json` (each table tagged with data version, config, training cutoff and git commit) and replaces the block below. Method: [docs/evaluation.md](docs/evaluation.md).

<!-- results:begin -->
<!-- results:end -->

Every engineer value goes into the benchmark rows, so the corrupted-future check (`pitsense leakcheck`) covers all of them. The simulator and calls have their own check (`pitsense strategy-leakcheck`). Full tables are in each engineer's doc.

## Add your own model

Implement `fit(train_df, target)` and `predict(test_df)` (see `src/pitsense/bench/models.py`), add the class to `PIT_MODELS` or `REJOIN_MODELS`, then run `pitsense bench run`. Feature columns are listed in `NUMERIC_FEATURES` in `bench/features.py`. Labels start with `y_`.

## Known limitations

- **Live: one real race so far.** The recorder ran through the 2026 Bahrain GP with an F1 TV login (74k messages, about 37 reconnects) and the race rebuilds fully from it. The pit wall made no calls that day because of a wet-start bug, since fixed. Call quality on races like that is being worked on.
- **Box calls are the weakest part.** Precision and recall differ a lot between races (per-circuit table in `reports/results.md`), and most stops under a safety car can't be foreseen a lap ahead. Matching what the team did is scored separately from decision value, which is only a simulator estimate (see [Results](#results)).
- **Wet strategy is weak.** The wet engine makes tyre-class calls (BOX for INTERS / SLICKS), but most of them are early or late (wet races are too few in 2026 to report here; see docs/engineers/strategy.md for the 2025-26 study). It also ignores rivals and traffic.
- **Public timing only:** no fuel loads, tyre temperatures or car telemetry. PitSense is strongest on rivals, whom teams also see only through timing.
- **Voice.** The fine-tuned SmolLM2-360M writes the radio calls: fact-checked by a guard (held-out 2026 scores in [Results](#results)). Piper (British voice) speaks them offline. Each message takes about 1 s on the GPU.
- **The simulator is simplified:** at most one SC and one VSC per simulated future, no red-flag restarts, no lapped traffic or blue flags, a fixed 6 s added to every stop for warm-up and traffic, and rivals that don't react to our stops.

## Roadmap

1. **Shadow mode at real races.** Record live, run `pitsense pitwall --follow`, then publish `pitsense shadow-score` after each race.
2. **Better box calls.** Fix the simulator's stop-timing bias and calibrate opponents' stops with the models engineer.
3. **A wet-weather strategy model.**
4. **Open the benchmark.** Publish a data card and score other open strategy engines.

## Layout

```
src/pitsense/
  archive, events, merge, state      raw feed -> deterministic race state
  asof, pitloss, validate, live      as-of stores, pit loss, FastF1 check, recorder
  pitwall/                           engineers (tyre, pitstop, rivals, rules, weather, models,
                                     strategy, head), PitWall orchestrator, runtime (live / replay)
  web/                               dashboard (standard library, offline)
  bench/                             features, labels, tasks, models, evaluation, leak check
  voice/                             radio / brief composer, own SLM, fine-tune, guard, TTS
  modelstore.py, registry.py, cli.py
docs/                                ENGINEERING.md, data.md, models.md, pitwall.md, engineers/*.md
tests/                               synthetic-feed tests, leak tests, parity tests
```

## Data and terms

- **Unofficial.** PitSense isn't associated with Formula 1. F1, FORMULA ONE and related marks belong to Formula One Licensing B.V.
- **No timing data in the repo.** Timing data comes from F1's live-timing service and stays under its terms, so the repository ships none (`data/` is gitignored).
- **Publish predictions, not the raw feed.** Check the F1 TV and OpenF1 terms before any commercial use.
