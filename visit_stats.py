"""Raw, count-system-agnostic visit sampling.

This module intentionally stores raw card counts only.  Count-system labels and
true-count neighborhoods are derived later by evaluate.py.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


class RawVisitSampler:
    def __init__(
        self,
        sample_rate: float,
        capacity: int,
        num_decks: int,
        max_hands: int,
        seed: int = 0,
    ):
        self.sample_rate = float(sample_rate)
        self.capacity = int(capacity)
        self.num_decks = int(num_decks)
        self.max_hands = int(max_hands)
        self.rng = np.random.default_rng(seed)
        self.total_visits = 0
        self.size = 0
        self.total = np.empty(capacity, np.uint8)
        self.upcard = np.empty(capacity, np.uint8)
        self.soft = np.empty(capacity, bool)
        self.pair = np.empty(capacity, np.uint8)
        self.hands_remaining = np.empty(capacity, np.uint8)
        self.counts = np.empty((capacity, 10), np.uint16)

    def observe(self, states: np.ndarray, raw_counts: np.ndarray) -> None:
        n = len(states)
        self.total_visits += n
        if self.size >= self.capacity or n == 0:
            return
        chosen = np.flatnonzero(self.rng.random(n) < self.sample_rate)
        room = self.capacity - self.size
        chosen = chosen[:room]
        if not len(chosen):
            return
        sl = slice(self.size, self.size + len(chosen))
        self.total[sl] = np.rint(states[chosen, 0] * 21).astype(np.uint8)
        self.upcard[sl] = np.rint(states[chosen, 1] * 10).astype(np.uint8)
        self.soft[sl] = states[chosen, 2] > 0.5
        self.pair[sl] = np.rint(states[chosen, 3] * 10).astype(np.uint8)
        self.hands_remaining[sl] = np.rint(states[chosen, 4] * self.max_hands).astype(np.uint8)
        self.counts[sl] = raw_counts[chosen]
        self.size += len(chosen)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sl = slice(0, self.size)
        np.savez_compressed(
            path,
            total=self.total[sl],
            upcard=self.upcard[sl],
            soft=self.soft[sl],
            pair=self.pair[sl],
            hands_remaining=self.hands_remaining[sl],
            counts=self.counts[sl],
            sample_rate=np.float64(self.sample_rate),
            total_visits=np.int64(self.total_visits),
            num_decks=np.int64(self.num_decks),
            max_hands=np.int64(self.max_hands),
            capacity_reached=np.bool_(self.size >= self.capacity),
        )
