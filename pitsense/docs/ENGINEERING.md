# Engineering handbook

How PitSense is built and the rules every change follows. Written for the
engineers building it (people or agents) and for teams adopting it.

## The idea

PitSense is a **virtual pit wall**: one engineer per role on a race-strategy
team, all reading the same live timing feed every team receives.

- **Our own code and models.** Every engineer is code and models we train
  ourselves. No outside AI services; the race-engineer voice is our own small
  language model.
- **Only what is known now.** An engineer sees what had been published by that
  moment, never later.
- **Scored before it is trusted.** Every engineer is measured against what
  actually happened (PitSense-Bench) before its output reaches a pit wall.

## Architecture

```
archive streams / live recording
        │  events.py: one time-ordered EventLog
        ▼
RaceState.apply(event)            the only way race state changes (state.py)
        │  after every event
        ▼
PitWall.observe(state)            RaceMemory + every engineer's observe()   (pitwall/)
        │  when asked: at each lap end / pit entry (benchmark), or every update (live)
        ├──► benchmark row: base features + "<engineer>__<key>" values  ──► labels ──► tasks ──► leaderboard
        └──► Snapshot: tower, engineer values, alerts, calls ──► dashboard / calls log / voice (own SLM)
```

Engineers, in dependency order:

| Engineer | Role on a real team | Runs per benchmark row |
|---|---|---|
| `tyre` | tyre & performance engineer: pace, wear, tyre cliff | yes |
| `pitstop` | pit-stop analyst: time a stop costs, rejoin position | yes |
| `rivals` | rival strategist: who pits when, undercut threats | yes |
| `rules` | sporting director: regulations, penalties, SC/VSC phases | yes |
| `weather` | weather: rain risk, slick/intermediate crossover | yes |
| `models` | live predictions from the cross-race models | no (pit wall only) |
| `strategy` | strategy engineer: race simulation, Plan A/B | no (too slow) |
| `head` | head of strategy: one call per car, with reasons | no |

## Rules (non-negotiable)

1. **As of now, never later.** An engineer reads only `state`, `self.ctx`,
   `self.memory` and `view`. Never the event log; never `bench.labels`,
   `evaluate`, `dataset` or `leakcheck` (`tests/test_pitwall.py` checks imports).
2. **Scalars out.** `car()` and `race()` return flat dicts of `float | int |
   str | bool | None` (enforced). Never a reference to a live object.
3. **Pre-race knowledge comes from `ctx`.** `ctx.prior`; `ctx.past_races`, which
   only ever holds races that ended before this one started; `ctx.models`,
   checked to be trained before the race.
4. **Train before you test.** Cross-race models are trained only on races that
   ended before the test race started. Use `bench.evaluate.evaluate_task` /
   `fit_predict`, which enforce the cutoff.
5. **Prove it.** `python -m pytest -q` and `pitsense leakcheck` must pass. The
   leak check rebuilds every row with the future deleted or shuffled. It
   compares every engineer value, so it covers your engineer automatically.
6. **No timing data in the repo.** F1's terms: tests use synthetic feed
   snippets (see `tests/conftest.py`).
7. **Deterministic.** The same input gives the same output. Seed every random
   draw, e.g. from `(race_id, t, car)`.
8. **Measured or it doesn't ship.** Every engineer comes with a benchmark result
   against a baseline.

## Contracts

### Engineer (`src/pitsense/pitwall/engineer.py`)

```python
class TyreEngineer(Engineer):
    name = "tyre"                      # values appear as "tyre__<key>"
    requires = ()                      # engineers whose values you read via view
    features = ("deg_s_per_lap",)      # keys benchmark models may use as inputs (numeric/bool)
    in_bench = True                    # False if too slow to run per decision row

    @classmethod
    def summarize_race(cls, final, meta) -> dict: ...   # learn from a finished race (JSON values)
    def observe(self, state): ...                        # after every event; keep it cheap
    def car(self, state, number, view) -> dict: ...      # {key: scalar} about one car
    def race(self, state, view) -> dict: ...             # {key: scalar} about the race
    def alerts(self, state, view) -> list[Alert]: ...    # conditions active right now
    def calls(self, state, view) -> list[Call]: ...      # head of strategy only
```

- `view.car("pitstop", n)` / `view.race("tyre")` give another engineer's values
  at this same moment. That engineer must be listed in `requires`.
- `self.memory` (`RaceMemory`): laps by driver and lap number
  (`memory.index`), the order at each pit entry, pit losses measured so far.
- `summarize_race` runs on the complete race when the benchmark history is
  built, and is stored in `RaceSummary.extra[name]`. Read it back from
  `self.ctx.past_races`.
- Alerts are level-triggered: report what is true now (with `since`), so the
  result never depends on how often you are asked.

### Values engineers exchange

Keep these names and units. Other engineers build on them. Add more keys freely.

| Engineer | Scope | Key | Meaning | Unit |
|---|---|---|---|---|
| tyre | car | `pace_s` | expected time of the car's next clean lap on its current tyres, fuel included | s |
| tyre | car | `deg_s_per_lap` | lap-time increase per extra lap on the current set, fuel effect removed | s/lap |
| tyre | car | `fresh_soft_s`, `fresh_medium_s`, `fresh_hard_s` | expected first clean lap on a new set of that compound fitted now; None if unknown | s |
| tyre | car | `cliff_risk` | probability the tyres fall off (sudden loss ≥ 1 s/lap) within 3 laps | 0–1 |
| tyre | race | `fuel_s_per_lap` | lap time gained per lap from fuel burn | s/lap |
| pitstop | race | `loss_green`, `loss_sc`, `loss_vsc`, `loss_now` | time a stop costs vs staying out, by track status / now | s |
| pitstop | car | `rejoin_if_box_now` | position the car would rejoin in if it boxed this lap | position |
| pitstop | car | `cars_within_loss` | cars behind within one pit loss | count |
| rivals | car | `ahead`, `behind` | car numbers directly ahead / behind | str |
| rivals | car | `gap_ahead`, `gap_behind` | gaps to them | s |
| rivals | car | `pit_prob_1`, `pit_prob_3` | probability this car pits within 1 / 3 laps (in-race + pre-race knowledge) | 0–1 |
| rivals | car | `undercut_threat` | probability the car behind gets ahead by pitting first | 0–1 |
| rivals | car | `undercut_chance` | probability this car gets ahead of the car in front by pitting first | 0–1 |
| rules | race | `race_dry` | only dry compounds used so far | bool |
| rules | race | `sc_phase` | `none`, `sc`, `sc_ending`, `vsc`, `vsc_ending` or `red` | str |
| rules | race | `pit_lane_open` | pit entry open | bool |
| rules | car | `must_stop` | still has to fit a second dry compound | bool |
| rules | car | `penalty_s_pending` | time penalty still to serve (at the next stop or after the flag) | s |
| rules | car | `drive_through_pending` | a drive-through or stop-go is pending | bool |
| weather | race | `rainfall`, `track_temp`, `air_temp`, `humidity` | latest feed readings | feed units |
| weather | race | `wet_running` | any running car on intermediates or wets | bool |
| weather | race | `rain_prob_10min` | probability of rain within 10 minutes | 0–1 |
| weather | race | `crossover` | `none`, `to_inters` or `to_slicks`: which tyre has become faster | str |
| models | car | `pit_prob_1`, `pit_prob_3`, `rejoin_pred` | cross-race model predictions (pit wall only) | 0–1, position |
| strategy | car | defined by its owner | plans for the focus cars | |
| head | — | `Call` records (`pitwall/types.py`) | the call, reasons, Plan A / B | |

The stubs already return every key, with None where not yet built. Read other
engineers' keys with `.get()` and fall back sensibly when a value is None.

### Records (`src/pitsense/pitwall/types.py`)

`Alert`, `Reason`, `PlanStop`, `Plan`, `Call` (action is one of `BOX`,
`STAY_OUT`, `PREPARE_BOX`, `BOX_IF_SC`, `NO_CALL`), `TeamConfig` ("my team")
and `Snapshot`. All are immutable; `Snapshot.to_dict()` is strict JSON.

### Benchmark (`src/pitsense/bench/`)

- **Rows.** One per lap end (`kind="lap_end"`) and per pit entry
  (`kind="pit_entry"`). Base columns are in `features.py`; engineer columns are
  `"<engineer>__<key>"`. Labels start with `y_`.
- **Labelers** read the finished race: `f(rows, final_state)` adds `y_*`
  columns. They live in `bench/`, never in `pitwall/`.
- **Tasks** (`tasks.py`): `Task(name, kind, target, select, models, keep)`.
  `kind` is `binary`, `position` or `regression`; the first model is the
  reference baseline. A task can provide its own `evaluate` (simulations).
- **Models**: a class with `name`, `fit(train_df, target) -> self` and
  `predict(test_df) -> array`. Better models for an existing task go in
  `registry.TASK_MODELS`.
- **Protocol.** The test year is 2026, and each race is scored by models trained
  on races that ended before it started. Choose settings on 2025
  (`--test-year 2025`), then score 2026.

### Registry (`src/pitsense/registry.py`)

One line per engineer, task factory, extra model, labeler or CLI command
(`add_commands(subparsers)` in your module). Lines are `"module:attribute"`,
imported on first use.

## Ownership

Edit only your own files. If a change is needed elsewhere, ask the lead in your report.

| Area | Owner | Files |
|---|---|---|
| Data layer | data | `archive.py`, `events.py`, `merge.py`, `state.py`, `validate.py`, `config.py`, `docs/data.md` |
| Tyres & pace | tyre | `pitwall/engineers/tyre.py` (may become a package), `bench/tyre.py`, `tests/test_tyre*.py`, `docs/engineers/tyre.md` |
| Pit stops | pitstop | `pitloss.py`, `pitwall/engineers/pitstop.py`, `bench/pitstop.py`, `tests/test_pitstop*.py`, `docs/engineers/pitstop.md` |
| Rivals | rivals | `pitwall/engineers/rivals.py`, `bench/rivals.py`, `tests/test_rivals*.py`, `docs/engineers/rivals.md` |
| Rules | rules | `pitwall/engineers/rules.py`, `bench/rules.py`, `tests/test_rules*.py`, `docs/engineers/rules.md` |
| Weather | weather | `pitwall/engineers/weather.py`, `bench/weather.py`, `tests/test_weather*.py`, `docs/engineers/weather.md` |
| Strategy & head | strategy | `pitwall/engineers/strategy.py`, `pitwall/engineers/head.py`, `bench/strategy.py`, `tests/test_strategy*.py`, `docs/engineers/strategy.md` |
| Models live | ml | `pitwall/engineers/models.py`, `modelstore.py`, `bench/features.py` (refactor only; outputs identical), `tests/test_models*.py`, `docs/models.md` |
| Live pit wall | live | `pitwall/runtime.py`, `web/`, `live.py`, `views.py`, `tests/test_runtime*.py`, `docs/pitwall.md` |
| Voice (own SLM) | voice | `voice/`, `tests/test_voice*.py`, `docs/engineers/voice.md` |
| Core | lead | everything else: `registry.py`, `cli.py`, `pyproject.toml`, `README.md`, `asof.py`, other `bench/*.py`, `pitwall/{engineer,memory,types,wall}.py`, `tests/conftest.py` |

Exceptions:
- Anyone may add their own lines to `registry.py`.
- The voice and live owners may add their own optional-dependency extra to `pyproject.toml`.
- Any engineer module may become a package of the same name.

## Working in a git worktree

The global Python (3.14) has pitsense installed in editable mode, pointing at
the main checkout. Put your worktree first, and give yourself a private data
folder that shares the raw race data.

```powershell
# once (PowerShell)
$W = (git rev-parse --show-toplevel)
New-Item -ItemType Directory -Force "$W\data" | Out-Null
New-Item -ItemType Junction -Path "$W\data\raw" -Target "C:\Users\vedan\Downloads\aiml\pitsense\data\raw"
```

```bash
# every Bash command (shell state doesn't persist between calls)
export PYTHONPATH="$(git rev-parse --show-toplevel)/src" PITSENSE_DATA="$(git rev-parse --show-toplevel)/data"
python -c "import pitsense; print(pitsense.__file__)"          # must be inside your worktree
python -m pitsense bench build --year 2025 2026 --jobs 3        # ~30 s, into your private data folder
python -m pitsense bench run --tasks <yours> --out "$PITSENSE_DATA/reports"
python -m pitsense leakcheck --year 2026 --race hungary --cuts 3
```

- Never write to the main checkout's `data/` or `reports/`.
- Only the data engineer downloads new seasons. Everyone else uses
  `--year 2025 2026` until 2018–2024 is merged.
- Use `--jobs 3` at most: several engineers share 16 cores.
- Don't install or upgrade packages, except the voice engineer's PyTorch.
  scikit-learn, pandas, numpy and pyarrow are available.
- Only the voice engineer uses the GPU.

## Definition of done

1. Your engineer fills its keys from the table above (plus anything useful),
   documented in `docs/engineers/<role>.md` with meanings and units.
2. **Measured.** At least one benchmark task (your own, or a better model for a
   core task) is scored on 2026 against a baseline, and the numbers are in your
   doc.
3. **Tests** on synthetic data, with `python -m pytest -q` passing at every
   commit. `pitsense leakcheck` passes on at least two real 2026 races.
4. **Leaderboard.** No unintended change to the core leaderboard; report
   intended changes with numbers.
5. **Speed.** `bench build --year 2025 2026` stays under about twice its current
   time. Pit-wall-only engineers stay under 2 s per snapshot for the focus cars.

**Commits.** Small and logical, and the message says why. End every message with:

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

Don't push, and don't touch other branches.

**Final report to the lead:**
- branch and last commit;
- what you built;
- which keys you fill;
- benchmark table: yours against the baseline, on 2026;
- leakcheck results and core-leaderboard changes;
- open issues, and anything you need from other engineers.

## Working economically (agents)

Token and compute use is a real cost. Every engineer works like this:

- **Read narrowly.** grep/rg for the lines you need, then read small ranges; never dump whole large files or re-read what you already read.
- **Print little.** Pipe through `| tail -n 20` or a grep; scripts write results to `data/scratch/*.json` and print one-line summaries.
- **Run expensive jobs once.** Cache replays, traces and simulator results in `data/scratch`; tune by re-scoring cached results; score the test year once at the end.
- **No sub-agents.** Plan briefly, then act.
- **Short reports.** At most about 40 lines, with tables only for the key numbers.
