"""A small decoder-only transformer, written from scratch (needs torch).

Pre-norm blocks, learned positions, tied embeddings, causal attention through
`scaled_dot_product_attention`, a key/value cache for generation. Generation is
greedy (deterministic). The model reads `input <sep>` and writes the message.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import BOS, EOS, PAD, SEP


@dataclass
class Config:
    vocab: int = 1024
    ctx: int = 384
    d: int = 512
    layers: int = 8
    heads: int = 8
    ff: int = 2048
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.h = c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.proj = nn.Linear(c.d, c.d, bias=False)
        self.fc1, self.fc2 = nn.Linear(c.d, c.ff, bias=False), nn.Linear(c.ff, c.d, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, mask=None, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        if mask is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class VoiceLM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln = nn.LayerNorm(c.d)
        self.apply(self._init)
        for b in self.blocks:  # scaled residual init
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * c.layers))
            nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * c.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, pos=None, mask=None, caches=None):
        B, T = ids.shape
        if pos is None:
            pos = torch.arange(T, device=ids.device).expand(B, T)
        x = self.tok(ids) + self.pos(pos)
        for i, b in enumerate(self.blocks):
            x = b(x, mask, None if caches is None else caches[i])
        return self.ln(x) @ self.tok.weight.T  # tied output

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new: int = 160) -> list[list[int]]:
        """Greedy decoding for a batch of prompts (each ends with <sep>). Returns the new tokens, without <eos>."""
        dev = next(self.parameters()).device
        B = len(prompts)
        L = max(len(p) for p in prompts)
        ids = torch.full((B, L), PAD, dtype=torch.long)
        keep = torch.zeros((B, L), dtype=torch.bool)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.tensor(p)
            keep[i, L - len(p):] = True
        ids, keep = ids.to(dev), keep.to(dev)
        pos = (keep.long().cumsum(1) - 1).clamp(min=0)
        causal = torch.ones(L, L, dtype=torch.bool, device=dev).tril()
        mask = causal[None, None] & keep[:, None, None, :]
        mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]  # pad rows attend to themselves
        caches = [dict() for _ in self.blocks]
        logits = self(ids, pos, mask, caches)[:, -1]
        out = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        nxt_pos = pos[:, -1] + 1
        for step in range(max_new):
            tok = logits.argmax(-1)
            for i in range(B):
                if not done[i]:
                    t = int(tok[i])
                    if t == EOS:
                        done[i] = True
                    else:
                        out[i].append(t)
            if bool(done.all()) or L + step + 1 >= self.c.ctx:
                break
            keep = torch.cat([keep, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
            m = keep[:, None, None, :]
            logits = self(tok[:, None], nxt_pos[:, None].clamp(max=self.c.ctx - 1), m, caches)[:, -1]
            nxt_pos = nxt_pos + 1
        return out


def make(cfg: Config) -> VoiceLM:
    return VoiceLM(cfg)


def config_dict(cfg: Config) -> dict:
    return asdict(cfg)


def pack(inp_ids: list[int], tgt_ids: list[int]) -> tuple[list[int], int]:
    """[bos] input [sep] target [eos]; returns the ids and the index where the target starts."""
    prompt = [BOS, *inp_ids, SEP]
    return prompt + tgt_ids + [EOS], len(prompt)
