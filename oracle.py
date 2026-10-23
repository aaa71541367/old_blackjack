"""Evaluation-only Monte Carlo oracle using a fixed basic strategy.

Nothing in this module may be imported by env.py, agents.py, or train.py.
"""

from __future__ import annotations

import numpy as np

from config import RulesConfig
from env import DOUBLE, HIT, SPLIT, STAND


def scalar_hand_value(cards: list[int]) -> tuple[int, bool]:
    total = sum(cards)
    soft = 1 in cards and total + 10 <= 21
    return total + (10 if soft else 0), soft


def basic_strategy(cards: list[int], upcard: int, can_double: bool, can_split: bool) -> int:
    """Two-deck S17, DAS strategy adapted to the project's liberal ace rules."""
    total, soft = scalar_hand_value(cards)
    pair = cards[0] if len(cards) == 2 and cards[0] == cards[1] else 0
    if pair and can_split:
        split = (
            pair in (1, 8)
            or (pair in (2, 3, 7) and 2 <= upcard <= 7)
            or (pair == 4 and 5 <= upcard <= 6)
            or (pair == 6 and 2 <= upcard <= 6)
            or (pair == 9 and upcard in (2, 3, 4, 5, 6, 8, 9))
        )
        if split:
            return SPLIT
    if soft:
        if total <= 17:
            double_ranges = {
                13: (5, 6), 14: (5, 6), 15: (4, 5, 6),
                16: (4, 5, 6), 17: (3, 4, 5, 6),
            }
            if can_double and upcard in double_ranges.get(total, ()):
                return DOUBLE
            return HIT
        if total == 18:
            if can_double and 3 <= upcard <= 6:
                return DOUBLE
            return STAND if upcard in (2, 7, 8) else HIT
        return STAND
    if total <= 8:
        return HIT
    if total == 9:
        return DOUBLE if can_double and 3 <= upcard <= 6 else HIT
    if total == 10:
        return DOUBLE if can_double and 2 <= upcard <= 9 else HIT
    if total == 11:
        return DOUBLE if can_double else HIT
    if total == 12:
        return STAND if 4 <= upcard <= 6 else HIT
    if 13 <= total <= 16:
        return STAND if 2 <= upcard <= 6 else HIT
    return STAND


def _deck_from_counts(counts: np.ndarray, rng: np.random.Generator) -> list[int]:
    deck = np.repeat(np.arange(1, 11), counts.astype(int)).tolist()
    rng.shuffle(deck)
    return deck


def rollout_once(shoe_counts: np.ndarray, rules: RulesConfig, rng: np.random.Generator) -> float:
    deck = _deck_from_counts(shoe_counts, rng)
    pos = 0

    def draw() -> int:
        nonlocal pos, deck
        if pos >= len(deck):
            raise RuntimeError("Oracle rollout exhausted supplied shoe")
        card = int(deck[pos])
        pos += 1
        return card

    player = [draw()]
    dealer = [draw()]
    player.append(draw())
    dealer.append(draw())
    upcard = dealer[0]
    player_total, _ = scalar_hand_value(player)
    dealer_total, _ = scalar_hand_value(dealer)
    dealer_bj = upcard in (1, 10) and dealer_total == 21
    player_bj = player_total == 21
    if dealer_bj:
        return 0.0 if player_bj else -1.0
    if player_bj:
        return rules.blackjack_payout

    hands: list[dict] = [{"cards": player, "bet": 1, "lazy": False}]
    active = 0
    while active < len(hands):
        hand = hands[active]
        if hand["lazy"]:
            hand["cards"].append(draw())
            hand["lazy"] = False
        while True:
            cards = hand["cards"]
            total, _ = scalar_hand_value(cards)
            if total > 21:
                break
            pair = len(cards) == 2 and cards[0] == cards[1]
            action = basic_strategy(
                cards,
                upcard,
                can_double=len(cards) == 2 and (rules.double_after_split or len(hands) == 1),
                can_split=pair and len(hands) < rules.max_hands,
            )
            if action == STAND:
                break
            if action == HIT:
                cards.append(draw())
                continue
            if action == DOUBLE:
                hand["bet"] = 2
                cards.append(draw())
                break
            if action == SPLIT:
                pair_card = cards[0]
                hand["cards"] = [pair_card, draw()]
                hands.insert(active + 1, {"cards": [pair_card], "bet": 1, "lazy": True})
                continue
        active += 1

    live = any(scalar_hand_value(h["cards"])[0] <= 21 for h in hands)
    if live:
        while True:
            dealer_total, dealer_soft = scalar_hand_value(dealer)
            hit = dealer_total < 17 or (
                dealer_total == 17 and dealer_soft and not rules.dealer_stands_soft_17
            )
            if not hit:
                break
            dealer.append(draw())
    dealer_total, _ = scalar_hand_value(dealer)
    result = 0.0
    for hand in hands:
        total, _ = scalar_hand_value(hand["cards"])
        stake = float(hand["bet"])
        if total > 21:
            result -= stake
        elif dealer_total > 21 or total > dealer_total:
            result += stake
        elif total < dealer_total:
            result -= stake
    return result


def estimate_ev(
    shoe_counts: np.ndarray,
    rules: RulesConfig,
    rollouts: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray([rollout_once(shoe_counts, rules, rng) for _ in range(rollouts)])
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(max(1, rollouts)))
