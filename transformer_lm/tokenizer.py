"""Byte-level BPE tokenizer (no regex pre-tokenization)."""

from __future__ import annotations

import re
from collections import Counter


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """Train a byte-level BPE tokenizer.

    Merge order: at each step, merge the most frequent adjacent pair.
    Break ties by selecting the pair with the smallest ``(id1, id2)``
    in lexicographic (tuple) order.

    IDs 0–255 are single bytes. Merge tokens get IDs starting from 256.
    Special tokens get the highest IDs in the vocab.

    Args:
        input_path: Path to a UTF-8 text file.
        vocab_size: Target vocabulary size (>= 256 + len(special_tokens)).
        special_tokens: Optional special token strings.

    Returns:
        vocab: ``dict[int, bytes]`` mapping token ID to byte string.
        merges: ``list[tuple[int, int]]`` merge pairs in order.
    """
    with open(input_path, "rb") as f:
        corpus = f.read()

    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    special_tokens = special_tokens or []
    if special_tokens:
        ordered = sorted((t.encode("utf-8") for t in special_tokens), key=len, reverse=True)
        pattern = b"|".join(re.escape(s) for s in ordered)
        segments = [list(chunk) for chunk in re.split(pattern, corpus) if chunk]
    else:
        segments = [list(corpus)]

    merge_target = vocab_size - len(special_tokens)
    while len(vocab) < merge_target:
        c = Counter()
        for seg in segments:
            for pair in zip(seg[:-1], seg[1:]):
                c[pair] += 1

        if not c:
            break

        most_freq = min(c.items(), key=lambda x:(-x[1], x[0]))
        new_pair = most_freq[0]

        new_entry_id = len(vocab)
        vocab[new_entry_id] = vocab[new_pair[0]] + vocab[new_pair[1]]
        merges.append(new_pair)

        new_segments = []
        for seg in segments:
            i, n = 0, len(seg)
            new_seg = []
            while i < n:
                if i < n-1 and (seg[i], seg[i+1]) == new_pair:
                    new_seg.append(new_entry_id)
                    i += 2
                else:
                    new_seg.append(seg[i])
                    i += 1
            new_segments.append(new_seg)
        segments = new_segments

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    return vocab, merges


class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ) -> None:
        raise NotImplementedError("TODO: Implement BPETokenizer.__init__()")

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        raise NotImplementedError("TODO: Implement encode()")

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        raise NotImplementedError("TODO: Implement decode()")
