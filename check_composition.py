"""check_composition.py — composition-dependent play on razor-thin cells.

The strategy tables use canonical hands like hard 12 = [10, 2]; those states
literally remove a ten from the deck, and for ten-rich player hands in 1-2
decks the composition-dependent correct play on several stiff cells is HIT
even though the total-based chart says STAND.  This script enumerates the same
cells with a ten-free hand to test whether the agent's "chart errors" are in
fact composition-dependent play.

Run from the project directory after Phase 1:
    python check_composition.py
"""
import numpy as np

from agents import PlayAgent
from config import ProjectConfig
from env import DOUBLE, SPLIT
from evaluate import construct_remaining_counts

ACTION = ("S", "H", "D", "P")

CASES = [
    # (label, hand_with_ten, hand_without_ten, upcard)
    ("hard 12 v 4", [10, 2], [9, 3], 4),
    ("hard 12 v 5", [10, 2], [9, 3], 5),
    ("hard 12 v 6", [10, 2], [9, 3], 6),
    ("hard 13 v 2", [10, 3], [9, 4], 2),
    ("hard 13 v 3", [10, 3], [9, 4], 3),
    ("hard 14 v 2", [10, 4], [9, 5], 2),
    ("hard 16 v 10", [10, 6], [9, 7], 10),
]

cfg = ProjectConfig()
play = PlayAgent(cfg.training.play, cfg.device, cfg.training.seed)
play.load("outputs/play_final.pt", load_optimizer=False)
play.set_eval_frozen()


def act_for(hand: list[int], upcard: int) -> str:
    remaining = construct_remaining_counts(cfg, [*hand, upcard], 0.0)
    total = sum(hand)
    state = np.zeros(16, dtype=np.float32)
    state[0] = total / 21.0
    state[1] = upcard / 10.0
    state[2] = 0.0
    state[3] = 0.0
    state[4] = 1.0 / cfg.rules.max_hands
    state[5:15] = remaining / remaining.sum()
    state[15] = remaining.sum() / cfg.rules.shoe_size
    mask = np.ones(4, dtype=bool)
    mask[DOUBLE] = len(hand) == 2
    mask[SPLIT] = False
    return ACTION[int(play.act(state[None, :], mask[None, :], 0.0)[0])]


print(f"{'cell':<14} {'ten-rich hand':<16} {'ten-free hand':<16} verdict")
for label, with_ten, without_ten, up in CASES:
    a1 = act_for(with_ten, up)
    a2 = act_for(without_ten, up)
    verdict = "composition-dependent" if a1 != a2 else "same"
    print(
        f"{label:<14} {str(with_ten)+' -> '+a1:<16} "
        f"{str(without_ten)+' -> '+a2:<16} {verdict}"
    )
