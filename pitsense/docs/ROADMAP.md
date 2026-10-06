# Roadmap: from replay-good to race-day-real

What stands between PitSense and a tool a real F1 strategy team would trust, ordered by impact.
Each item names the gap, why it matters on a real pit wall, and how we would measure it.

## 1. Data a real team has and we don't (biggest gap)

| Gap | Why it matters | What to do |
|---|---|---|
| **Tyre sets left** (new/used, per compound) | Real plans are limited by sets in the garage; we assume any compound is available | Parse the TyreStintSeries `New` flag plus quali/practice usage into a per-car set inventory; plans may only use sets that exist |
| **Fuel / energy and car damage** | Teams box for damage and manage energy; we only see lap times | Detect damage from sudden pace loss plus RC "car damage" messages, and add an unscheduled-stop hazard |
| **Practice long runs** | Teams know each compound's degradation before the race; we learn it during the race | Fit tyre deg on FP2 long runs and use it as the race-start prior |
| **Pit-crew speed per team** | Stationary time varies by 0.5–1 s between teams | Use per-team stationary-time priors from history (pitstop engineer) |

## 2. Simulator realism

- **More than one SC / VSC per race,** red flags with free tyre changes, and race restarts.
- **Traffic after the stop:** where a car rejoins and how much time it loses stuck behind slower cars. A stop's real cost depends on this.
- **Overtaking difficulty per circuit** (Monaco vs Bahrain), learnt from position changes in history.
- **Rivals that react:** when we stop, the car behind covers. Today rivals follow fixed plans.
- **Teammates planned together:** stop order, and avoiding double-stack losses.
- **Wet v2:** crossover laps per circuit and a drying-line model.

Measure: simulated vs actual position change per stop, on 2026 races the model has never seen.

## 3. Decision quality (the question a team actually asks: "did the call help?")

- Decision value is now in place (simulator counterfactual). Next, run it on every 2026 race and publish per-call value in `reports/results.md`.
- Calibrate confidence: "80% sure" should be right 80% of the time (reliability plot).
- Fix the known failure modes from the live Bahrain test:
  - calls 2–7 laps early;
  - compound flip-flops;
  - neutralisations nobody could have predicted, which should become instant reactions.

## 4. Race-day operations

- **Feed failover:** a second live source, plus graceful degradation when the timing feed drops.
- **Latency budget:** a call within 1 s of the lap finishing (measure p95 on the live recording).
- **Audit log:** every call with its inputs, so the team can review after the race.
- **A pit-wall operator role:** accept or reject calls, with the reason fed back for learning.
- **Multi-screen setup:** strategist, race engineer and mechanics views.

## 5. Frontend

Done:
- replay lab (pause, speed, lap jumps, bookmarks);
- alarm banner;
- strategy comparison tab (ranked plans, outcome spread, SC backup plan, first-stop timing).

Next, by value:
1. **Gap chart:** gap to the cars around us over the last 15 laps, with the pit-loss window drawn as a band (shows the undercut at a glance).
2. **Stint chart:** every car's stints as coloured bars by compound. It shows who has stopped and who is due.
3. **"What if" box:** pick a lap and compound, then see the simulated finishing position before calling it.
4. **Call history:** our calls vs what the team did, colour-coded right or wrong after the fact.
5. **Pace and degradation chart** per stint for our cars.
6. **Keyboard shortcuts** and a big-type pit-wall mode for distance reading.
7. **Phone layout polish:** the strategy tab on small screens.
