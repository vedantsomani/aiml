# PitSense

A leakage-safe F1 race-strategy engine, and an open benchmark to score strategy engines against what actually happened.

**v0.1 (Oct 2026)** contains the foundation:

- **One code path for replay and live.** Raw timing messages go into one deterministic state machine, whether they come from the public archive or from a live recording.
- **"As of" guarantees.** Every value carries the time it became known. Two checks enforce this:
  - a corrupted-future test: rebuild every decision with the future deleted or shuffled, and require bit-identical results;
  - a unit test that plants a leak to prove the check catches one.
- **PitSense-Bench v0.1.** 43,123 recorded decision points from 39 races (all of 2025 and 2026 so far), with what happened next. Two tasks so far:
  - will this car pit within *k* laps?
  - where will a car that just entered the pits rejoin?
- **Baselines and first models**, scored race by race on 2026 with models trained only on earlier races.
- **A live recorder** for this weekend's race at Sepang.

Strategy simulation, backup plans and reacting rivals come next (see [Roadmap](#roadmap)).

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
python -m pytest -q                                   # 27 tests; the real-race test skips until data is downloaded
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

## Validation against FastF1 (39 races, 43,856 laps)

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

## Benchmark results (v0.1)

**How it was scored:**

- **Test set:** the 15 races of 2026, each scored with models trained on every race that finished before it started.
- **Model settings:** chosen on 2025 races only.

**Will this car pit within k laps?** (16,177 lap-end decisions, excluding cars that retire within 3 laps)

| k | model | log loss ↓ | Brier ↓ | AUC ↑ | Brier skill vs base rate |
|---|---|---|---|---|---|
| 1 | `gbm_hazard` | **0.1182** | **0.0281** | **0.798** | **+5.7%** |
| 1 | `tyre_age_logit` | 0.1311 | 0.0294 | 0.699 | +1.4% |
| 1 | `base_rate` | 0.1375 | 0.0298 | 0.483 | 0 |
| 3 | `gbm_hazard` | **0.2703** | **0.0758** | **0.759** | **+10.3%** |
| 3 | `tyre_age_logit` | 0.2936 | 0.0810 | 0.697 | +4.2% |
| 3 | `base_rate` | 0.3102 | 0.0845 | 0.483 | 0 |

The probabilities are honest: when `gbm_hazard` says 10–20% for "pits next lap", it happened 13% of the time; at 2–5% it happened 4%. Calibration tables are in `reports/leaderboard.md`.

History matters. When I trained on 2026 races alone (`pitsense bench build --year 2026`), `gbm_hazard` did *worse* than the base rate on k=3. Adding older seasons (the archive goes back to 2018) is the cheapest improvement available.

**Where does a car rejoin after a stop?** (533 pit entries)

| model | exact position | within one place | mean error |
|---|---|---|---|
| `rejoin_gbm` | **53.9%** | **86.3%** | **0.74** |
| `gap_minus_pitloss` (broadcast-style pit window) | 49.9% | 79.2% | 0.99 |
| `no_change` | 32.7% | 52.0% | 1.97 |

- **Failure case:** Azerbaijan 2026, where the whole field pitted under the safety car. The gap rule assumes nobody else stops, so it predicts big losses and scores 22%. Keeping the same position scores 86%.
- **Next fix:** model the cars that pit together.

**Pit loss:** each stop's time loss is measured against cars that stayed out on the same laps. It's combined with a per-circuit prior from the previous visit, with no future data used.

## Add your own model

Implement `fit(train_df, target)` and `predict(test_df)` (see `src/pitsense/bench/models.py`), add the class to `PIT_MODELS` or `REJOIN_MODELS`, then run `pitsense bench run`. Feature columns are listed in `NUMERIC_FEATURES` in `bench/features.py`. Labels start with `y_`.

## Known limitations (v0.1)

- **Pit loss under SC/VSC is noisy.** Laps are compared by lap number, but cars reach the same lap number at different times. The safety-car prior is too high.
- **Lap 1 has no lap time live**, because the feed doesn't publish one.
- **Penalties, damage and red-flag tyre strategy** aren't modelled yet.
- **Data covers 2025–2026 only.** The archive goes back to 2018. `pitsense bench build --year 2018 ...` should work, but older seasons haven't been validated.
- **The live recorder is untested against a real session** (see above).

## Roadmap

1. **Score each remaining 2026 race** as it happens: `pitsense fetch --year 2026` → `bench build` → `bench run`.
2. **Live shadow mode.** Run the models on the recording during a race and post calls with timestamps before the outcome is known.
3. **Strategy layer:**
   - pace and tyre-wear tracking that updates every lap;
   - a lap-by-lap race simulator that tests every plan against the same simulated futures;
   - Plan A/B with triggers, plus the conditions under which the call flips.
4. **New features:**
   - rivals that react (each team's habit of covering an undercut);
   - wet-to-dry and dry-to-wet calls (test races: Sepang, Singapore, Brazil);
   - an explanation of why one plan beats another.
5. **Open the benchmark.**
   - Publish a data card.
   - Run other open engines through it and publish the leaderboard.

## Layout

```
src/pitsense/
  archive.py   download raw streams          events.py   EventLog, parsing, JSONL
  merge.py     feed merge semantics          state.py    deterministic reducer
  asof.py      as-of stores, leak guards     pitloss.py  pit-loss measurement, priors
  live.py      recorder + recording loaders  views.py    timing tower text view
  validate.py  FastF1 cross-check            cli.py      `pitsense` command
  bench/       features, labels, history, dataset, models, metrics, evaluate, leakcheck
tests/         synthetic race + unit tests, planted-leak test, integration test
reports/       leaderboard.md, validation.txt (generated)
```

## Data and terms

- **Unofficial.** PitSense isn't associated with Formula 1. F1, FORMULA ONE and related marks belong to Formula One Licensing B.V.
- **No timing data in the repo.** Timing data comes from F1's live-timing service and stays under its terms, so the repository ships none (`data/` is gitignored).
- **Publish predictions, not the raw feed.** Check the F1 TV and OpenF1 terms before any commercial use.
