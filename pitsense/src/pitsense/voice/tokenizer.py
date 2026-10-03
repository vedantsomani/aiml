"""Our own tokenizer: byte-level BPE trained on the voice corpus. No outside vocabulary.

Text is split into words, single digits (so every number is spelled digit by digit)
and punctuation; each piece is encoded as UTF-8 bytes and merged by the learned BPE
rules. Special tokens come first.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SPECIAL = ("<pad>", "<bos>", "<sep>", "<eos>")
PAD, BOS, SEP, EOS = range(4)
_PRE = re.compile(r" ?[A-Za-z]+| ?\d| ?[^\sA-Za-z\d]|\s+")


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.vocab_size = len(SPECIAL) + 256 + len(self.merges)
        self._cache: dict[str, tuple[int, ...]] = {}
        self._bytes: list[bytes] = [b""] * len(SPECIAL) + [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._bytes.append(self._bytes[a] + self._bytes[b])

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts, vocab_size: int = 1024) -> "Tokenizer":
        words = Counter(w for t in texts for w in _PRE.findall(t))
        seqs = {w: [len(SPECIAL) + b for b in w.encode()] for w in words}
        merges: list[tuple[int, int]] = []
        next_id = len(SPECIAL) + 256
        while next_id < vocab_size:
            pairs: Counter = Counter()
            for w, s in seqs.items():
                n = words[w]
                for x, y in zip(s, s[1:]):
                    pairs[(x, y)] += n
            if not pairs:
                break
            best = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            merges.append(best)
            for w, s in seqs.items():
                if len(s) < 2:
                    continue
                out, i = [], 0
                while i < len(s):
                    if i + 1 < len(s) and (s[i], s[i + 1]) == best:
                        out.append(next_id)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                seqs[w] = out
            next_id += 1
        return cls(merges)

    # ------------------------------------------------------------------ coding
    def _word(self, w: str) -> tuple[int, ...]:
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        s = [len(SPECIAL) + b for b in w.encode()]
        while len(s) > 1:
            cand = [(self.rank[(a, b)], i) for i, (a, b) in enumerate(zip(s, s[1:])) if (a, b) in self.rank]
            if not cand:
                break
            r, i = min(cand)
            s[i : i + 2] = [len(SPECIAL) + 256 + r]
        self._cache[w] = tuple(s)
        return self._cache[w]

    def encode(self, text: str) -> list[int]:
        return [t for w in _PRE.findall(text) for t in self._word(w)]

    def decode(self, ids) -> str:
        return b"".join(self._bytes[i] for i in ids if i >= len(SPECIAL)).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ files
    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["merges"])
