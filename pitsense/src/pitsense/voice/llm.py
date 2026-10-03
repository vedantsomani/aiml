"""A pretrained small LLM, fine-tuned with LoRA to be the pit-wall voice (needs torch, transformers, peft).

Base: SmolLM2-360M-Instruct (HuggingFaceTB). Its pretraining data predates the 2025-26
seasons we backtest on, so it cannot know their results; the fine-tune only teaches the
message format, on the same corpus and race splits as the from-scratch model. Adapters go
under data/models/voice/llm/ and the base weights live in the Hugging Face cache; neither is
committed. The same guard applies to its output.

    pitsense voice train --backend llm
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch

from .train import SEED, model_dir

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
FALLBACK_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
SUBDIR = "llm"


def llm_dir(path=None) -> Path:
    return model_dir(path) / SUBDIR


def _prompt(inp: str) -> str:
    return inp + "\n=>"


def _lm(base: str, dtype, local_only: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base, local_files_only=local_only)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, AutoModelForCausalLM.from_pretrained(base, dtype=dtype, local_files_only=local_only)


def _encode(tok, samples, max_len):
    out = []
    for s in samples:
        p = tok(_prompt(s.input), add_special_tokens=False)["input_ids"]
        t = tok(" " + s.target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(p) + len(t) <= max_len:
            out.append((p + t, len(p)))
    return out


def train_llm(out=None, base: str = BASE, n_samples: int = 30000, epochs: int = 1, lr: float = 2e-4, rank: int = 32,
              tokens_per_batch: int = 12000, per_race: int = 1000, calls: int = 3, log=print, model=None, tok=None) -> Path:
    """LoRA fine-tune on the training split of the voice corpus (2025 races; 2026 stays held out)."""
    from peft import LoraConfig, get_peft_model

    from . import synth
    from .train import batches, collate, loss_on  # noqa: F401  (loss is computed here, below)

    random.seed(SEED)
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = llm_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sit = synth.load_situations((2025, 2026), None, SEED)
    tr, _, _ = synth.split_races(sit)
    samples = synth.build((2025, 2026), per_race, calls, SEED, races=set(tr), alias=0.7, free=True)
    random.Random(SEED).shuffle(samples)
    samples = samples[:n_samples]
    if tok is None:
        tok, model = _lm(base, torch.bfloat16 if dev.type == "cuda" else torch.float32)
    seqs = _encode(tok, samples, 512)
    log(f"{base}: {len(seqs)} training sequences, mean {sum(len(s[0]) for s in seqs) / len(seqs):.0f} tokens ({time.time() - t0:.0f} s)")
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model.to(dev), cfg)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # adapters in fp32, base in bf16
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(SEED)
    pad = tok.pad_token_id
    steps = epochs * len(batches(seqs, tokens_per_batch, random.Random(0)))
    step = 0
    model.train()
    for ep in range(epochs):
        for idx in batches(seqs, tokens_per_batch, rng):
            L = max(len(seqs[i][0]) for i in idx)
            x = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            lab = torch.full((len(idx), L), -100, dtype=torch.long)
            for r, i in enumerate(idx):
                ids, start = seqs[i]
                x[r, : len(ids)] = torch.tensor(ids)
                att[r, : len(ids)] = 1
                lab[r, start : len(ids)] = torch.tensor(ids[start:])
            cur = lr * min(1.0, (step + 1) / 30) * (0.5 * (1 + math.cos(math.pi * step / steps)) * 0.9 + 0.1)
            for g in opt.param_groups:
                g["lr"] = cur
            with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
                loss = model(input_ids=x.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 25 == 0:
                log(f"step {step}/{steps} loss {float(loss):.4f} ({time.time() - t0:.0f} s)")
    model.save_pretrained(out)
    (out / "meta.json").write_text(json.dumps({"base": base, "rank": rank, "samples": len(seqs), "epochs": epochs, "seed": SEED,
                                               "train_seconds": round(time.time() - t0)}, indent=1), encoding="utf-8")
    log(f"saved adapter to {out} ({time.time() - t0:.0f} s)")
    return out


class LLMBackend:
    """Greedy generation from the fine-tuned model (adapter merged into the base for speed)."""

    name = "llm"

    def __init__(self, path=None, device: str | None = None, base: str | None = None, local_only: bool = True):
        from peft import PeftModel

        d = llm_dir(path)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok, m = _lm(base or meta["base"], dtype, local_only)
        self.model = PeftModel.from_pretrained(m, d).merge_and_unload().to(self.device).eval()
        self.meta = meta

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def to(self, device: str) -> None:
        self.device = device
        self.model.to(device).to(torch.bfloat16 if device == "cuda" else torch.float32)

    @torch.no_grad()
    def decode(self, inputs: list[str], batch: int = 32, max_new: int = 160) -> list[str]:
        outs = [""] * len(inputs)
        order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            enc = self.tok([_prompt(inputs[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id,
                                      eos_token_id=self.tok.eos_token_id)
            for i, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
                outs[i] = self.tok.decode(g, skip_special_tokens=True).strip()
        return outs
