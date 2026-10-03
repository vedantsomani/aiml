"""Train the voice model from scratch on the template corpus (needs torch).

    pitsense voice train [--epochs 3] [--out data/models/voice]

Fixed seeds; bf16 autocast on the GPU. Weights go under data/models/voice/, never in git.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import data_dir
from . import synth
from .slm import Config, VoiceLM, pack
from .tokenizer import PAD, Tokenizer

SEED = 1234
FILES = ("model.pt", "tokenizer.json", "meta.json")


def model_dir(path: str | Path | None = None) -> Path:
    return Path(path) if path else data_dir() / "models" / "voice"


def corpus(per_race=1000, calls=3, seed=SEED):
    """(train, val, test) sample lists. Test = every 2026 race, validation = the last two 2025 races."""
    sit = synth.load_situations((2025, 2026), None, seed)
    tr, va, te = synth.split_races(sit)
    mk = lambda races, n, c: synth.build((2025, 2026), n, c, seed, races=set(races))  # noqa: E731
    return mk(tr, per_race, calls), mk(va, 300, 1), mk(te, 150, 1)


def test_set(per_race=150, seed=SEED):
    """The held-out samples: every 2026 race (same as `corpus()[2]`), without building the training set."""
    sit = synth.load_situations((2025, 2026), None, seed)
    return synth.build((2025, 2026), per_race, 1, seed, races=set(synth.split_races(sit)[2]))


def encode(tok: Tokenizer, samples, ctx: int):
    seqs = []
    for s in samples:
        ids, start = pack(tok.encode(s.input), tok.encode(s.target))
        if len(ids) <= ctx:
            seqs.append((ids, start))
    return seqs


def batches(seqs, tokens_per_batch: int, rng: random.Random):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + rng.random() * 8)
    groups, cur, longest = [], [], 0
    for i in order:
        longest = max(longest, len(seqs[i][0]))
        if cur and longest * (len(cur) + 1) > tokens_per_batch:
            groups.append(cur)
            cur, longest = [], len(seqs[i][0])
        cur.append(i)
    if cur:
        groups.append(cur)
    rng.shuffle(groups)
    return groups


def collate(seqs, idx, dev):
    L = max(len(seqs[i][0]) for i in idx)
    x = torch.full((len(idx), L), PAD, dtype=torch.long)
    w = torch.zeros((len(idx), L), dtype=torch.bool)  # target positions (what is predicted)
    for r, i in enumerate(idx):
        ids, start = seqs[i]
        x[r, : len(ids)] = torch.tensor(ids)
        w[r, start - 1 : len(ids) - 1] = True  # logits at t predict token t+1
    return x.to(dev), w.to(dev)


def loss_on(model, x, w):
    logits = model(x[:, :-1])
    tgt = x[:, 1:]
    return F.cross_entropy(logits[w[:, :-1]].float(), tgt[w[:, :-1]])


@torch.no_grad()
def val_loss(model, seqs, dev, tpb=24000):
    model.eval()
    tot, n = 0.0, 0
    for idx in batches(seqs, tpb, random.Random(0)):
        x, w = collate(seqs, idx, dev)
        with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
            l = loss_on(model, x, w)
        k = int(w[:, :-1].sum())
        tot, n = tot + float(l) * k, n + k
    model.train()
    return tot / max(1, n)


def train(out=None, epochs=3, per_race=1000, calls=3, vocab=1024, d=512, layers=8, heads=8, lr=6e-4,
          tokens_per_batch=24000, log=print) -> Path:
    random.seed(SEED)
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = model_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, va, te = corpus(per_race, calls)
    log(f"corpus: train {len(tr)}  val {len(va)}  test {len(te)}  ({time.time() - t0:.0f} s)")
    tok = Tokenizer.train([s.input for s in tr[::4]] + [s.target for s in tr], vocab)
    cfg = Config(vocab=tok.vocab_size, d=d, layers=layers, heads=heads, ff=4 * d)
    seq_tr, seq_va = encode(tok, tr, cfg.ctx), encode(tok, va, cfg.ctx)
    log(f"tokens/seq mean {sum(len(s[0]) for s in seq_tr) / len(seq_tr):.0f}  max {max(len(s[0]) for s in seq_tr)}  dropped {len(tr) - len(seq_tr)}")
    model = VoiceLM(cfg).to(dev)
    log(f"model: {model.n_params() / 1e6:.1f} M parameters on {dev}")
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": nodecay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
    rng = random.Random(SEED)
    steps_per_epoch = len(batches(seq_tr, tokens_per_batch, random.Random(0)))
    total, warm, step = epochs * steps_per_epoch, 200, 0
    best = float("inf")
    for ep in range(epochs):
        for idx in batches(seq_tr, tokens_per_batch, rng):
            cur = lr * min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * step / total)) * 0.95 + 0.05)
            for g in opt.param_groups:
                g["lr"] = cur
            x, w = collate(seq_tr, idx, dev)
            with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
                loss = loss_on(model, x, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                log(f"ep {ep} step {step}/{total} loss {float(loss):.4f} lr {cur:.1e} ({time.time() - t0:.0f} s)")
        v = val_loss(model, seq_va, dev)
        log(f"epoch {ep} val loss {v:.4f}")
        if v < best:
            best = v
            torch.save(model.state_dict(), out / "model.pt")
    tok.save(out / "tokenizer.json")
    meta = {"config": cfg.__dict__, "params": model.n_params(), "val_loss": best, "epochs": epochs, "seed": SEED,
            "train_samples": len(tr), "train_seconds": round(time.time() - t0), "corpus": {"per_race": per_race, "calls": calls}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"saved {out} (best val loss {best:.4f}, {time.time() - t0:.0f} s)")
    return out
