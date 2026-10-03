# Voice engineer

Turns a `Call` plus engineer values into radio messages (at most 20 words), strategist briefs
(2-4 sentences), answers to fixed questions (gap to a car, tyre age, pit window, Plan B trigger,
why this call) and free-form questions, and can speak them. Code: `src/pitsense/voice/`.

```
Call + Snapshot --facts.py--> one-line input --backend--> text --guard.py--> served text
                                              (llm, scratch)      fail -> composer.py (template)
```

| File | Role |
|---|---|
| `facts.py` | `Facts` from a Call and a Snapshot (or a benchmark row); the compact input line |
| `composer.py` | template grammar (6 style digits pick paraphrases); reference and fallback |
| `synth.py` | corpus: real 2025-26 situations from benchmark rows + sampled calls, targets from the grammar |
| `tokenizer.py`, `slm.py`, `train.py` | our own byte-level BPE and ~26M-parameter decoder-only transformer, from scratch |
| `llm.py` | LoRA fine-tune of a pretrained small LLM (SmolLM2-360M-Instruct) on the same corpus |
| `guard.py` | every number, car number, TLA, compound and lap in the output must be in the input; length; action |
| `api.py`, `cli.py` | `say`, `brief`, `answer`, `ask`; `pitsense voice train | eval | say` |
| `tts.py` | `speak(text) -> wav` with Piper (optional) |

## Facts and the guard

The input is one deterministic line, e.g.
`radio S511140 | ACT BOX | CAR 16 LEC | LAP 21 57 36 | POS 3 | TY MEDIUM 18 1 | FIT HARD | AH 1 VER 1.2 | LOSS 21.5 | REJ 5 | CLF 60 | PA 1 22 HARD | TRG if sc comes before lap 30 | R tyre_cliff 60`.
The guard rejects an output with any number, three-letter name or compound the line does not
hold, spelled-out numbers, a length over the limit, or a stated action that is not the call
(`composer.stated_action`). A rejected output is replaced by the template text; the fallback rate
is reported. `say()` tries the backends best-first (fine-tuned LLM, then scratch model), uses the
first output that passes the guard, else the template. `Voice(model="llm" | "scratch" | "template")`
selects one; `PITSENSE_VOICE=template` forces the template.

Calls in the corpus are **sampled** (the head of strategy was a stub): an action that follows
from the values (cliff, SC, must-stop, undercut threat, rain), reasons, Plan A/B, and sometimes a
penalty or rain the row did not have. Situations are real; the calls are not a measure of strategy.
Splits are by race: train = 2025 (all but the last two), validation = last two 2025, test = all 15
2026 races. Training also invents three-letter names (70% of samples) so the model learns to copy
names, because the 2026 grid differs.

## Results, held-out 2026 (5,024 samples, 15 races)

Scratch model (25.9 M parameters, 4 epochs, 14 min on the RTX 5070). Template baseline: 0 fact
errors, 100% action, 100% length by construction.

| Task | Fact errors before guard | Fallback rate | Action ok before / after guard | Length ok before / after |
|---|---|---|---|---|
| radio | 0.16% | 0.16% | 100% / 100% | 100% / 100% |
| brief | 14.3% | 14.3% | 100% / 100% | 100% / 100% |
| ask_why | 0.4% | 0.4% | 100% / 100% | 100% / 100% |
| ask_gap | 20.5% | 20.5% | - | 100% / 100% |
| ask_tyre_age, pit_window, plan_b | 0.4%, 0%, 0.2% | same | - | 100% / 100% |

Target was <= 1% fact errors before the guard: met for radio and four of the five question
types; **not met for briefs (14%) or gap questions (20%)**; the guard makes served output 0%
for all of them. Overall fallback 5.8% (281 number, 31 name errors). Exact match with the
template: radio 99.4%, brief 83.7%. Diversity (served): radio 99.4% unique messages,
distinct-2-gram 0.086 (template 0.086); brief 100% unique, 0.049 (template 0.049): the model
reproduces the grammar's variety, no more. Latency (single message): template 5 us; scratch
model GPU about 170 ms radio and 650 ms brief; CPU about 150 ms radio, 830 ms brief. Size: 25.9 M
parameters, 104 MB weights.

Earlier run without invented names: radio 14.5%, brief 33% fact errors, almost all invented
names for 2026-only drivers: the fix above cut them.

### Fine-tuned LLM and speech: not yet measured

The LLM backend, free-form task, comparison harness and TTS are written and unit-tested (a LoRA
run on a tiny random local model, end to end), but **SmolLM2-360M-Instruct and a Piper voice have
not been downloaded**: the permission system denied the download, so no fine-tuned numbers exist
and the comparison table has no LLM column yet. Fill it by running the commands below.
Data cutoff: SmolLM2's model card gives no cutoff date; this must be read from the card
(HuggingFaceTB/SmolLM2-360M-Instruct) and recorded here after the download. Do not claim
"before 2025" until then. Qwen2.5-0.5B-Instruct is the fallback (`--base`).

## Train and run

```
pip install -e .[slm]            # torch is already installed; adds transformers, peft, accelerate
pip install -e .[tts]            # optional: piper-tts
pitsense bench build --year 2025 2026 --jobs 3
pitsense voice train                       # scratch model -> data/models/voice/ (~14 min)
pitsense voice train --backend llm         # LoRA on SmolLM2-360M -> data/models/voice/llm/
pitsense voice eval --out data/reports/voice_eval.json     # template vs scratch vs llm
pitsense voice say --race hungary --lap 30 --car 16 --task brief [--backend llm] [--audio]
python -m piper.download_voices en_US-lessac-medium --data-dir data/models/voice/tts
```

```python
from pitsense.voice import say, brief, answer, ask, speak
say(call, snapshot); brief(call, snapshot)
answer(call, snapshot, "gap", target="44")      # gap | tyre_age | pit_window | plan_b | why
ask(call, snapshot, "how far is the car ahead?")
speak(say(call, snapshot))                      # wav path
```

Weights, adapters and audio live under `data/` and are never committed. Tests: `tests/test_voice.py`,
`tests/test_voice_llm.py` (model and speech tests skip without torch, peft, piper).

## Open issues

- Briefs and gap questions: 14-20% fact errors before the guard; more data or epochs, or a
  copy-friendlier input for gaps, would help. The guard hides it at the cost of fallbacks.
- Calls are sampled; once the head of strategy ships, regenerate the corpus from real calls and retrain.
- Free-form answers are routed by keywords in the template; the LLM learns that mapping, not new reasoning.
