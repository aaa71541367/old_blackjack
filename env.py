"""Vectorized two-deck blackjack environment.

The environment exposes raw remaining-card composition only.  It contains no
hand-crafted card-counting or strategy feature.

Important timing details:
* boundary reshuffles happen in ``prepare_round`` only;
* an emergency refill can happen only after the current shoe is literally
  exhausted, and it reshuffles the discard pile while excluding current-round
  cards;
* the dealer hole card is not removed from the observable composition until it
  is revealed;
* a split child receives its second card lazily when that hand becomes active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from config import RulesConfig


STAND, HIT, DOUBLE, SPLIT = range(4)
ACTION_NAMES = ("stand", "hit", "double", "split")


@dataclass(slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray
    reward_hand: int = -1


@dataclass(slots=True)
class RoundResult:
    units: np.ndarray
    transitions: list[Transition]
    emergency_refills: int


def hand_value(card_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return best total and whether an ace is currently valued as 11."""
    values = np.arange(1, 11, dtype=np.int16)
    hard = card_counts.astype(np.int16) @ values
    has_ace = card_counts[:, 0] > 0
    soft = has_ace & (hard + 10 <= 21)
    return hard + soft.astype(np.int16) * 10, soft


class BlackjackEnv:
    """Synchronous vector batch with independently persistent shoes."""

    def __init__(self, num_envs: int, rules: RulesConfig, seed: int = 0):
        self.num_envs = int(num_envs)
        self.rules = rules
        self.rng = np.random.default_rng(seed)
        self.full_counts = np.asarray(rules.full_counts, dtype=np.int16)
        self.max_cards = rules.shoe_size

        self.shoe_order = np.zeros((num_envs, self.max_cards), dtype=np.int8)
        self.shoe_size = np.zeros(num_envs, dtype=np.int16)
        self.shoe_pos = np.zeros(num_envs, dtype=np.int16)
        self.shoe_counts = np.zeros((num_envs, 10), dtype=np.int16)
        # Counts known to the agent: dealer hole remains here until reveal.
        self.observable_counts = np.zeros((num_envs, 10), dtype=np.int16)

        h = rules.max_hands
        self.hands = np.zeros((num_envs, h, 10), dtype=np.int8)
        self.hand_size = np.zeros((num_envs, h), dtype=np.int8)
        self.hand_bet = np.ones((num_envs, h), dtype=np.int8)
        self.from_split = np.zeros((num_envs, h), dtype=bool)
        self.hand_finished = np.zeros((num_envs, h), dtype=bool)
        self.needs_card = np.zeros((num_envs, h), dtype=bool)
        self.num_hands = np.ones(num_envs, dtype=np.int8)
        self.active_hand = np.zeros(num_envs, dtype=np.int8)
        self.player_active = np.zeros(num_envs, dtype=bool)

        self.dealer = np.zeros((num_envs, 10), dtype=np.int8)
        self.dealer_upcard = np.zeros(num_envs, dtype=np.int8)
        self.dealer_hole = np.zeros(num_envs, dtype=np.int8)
        self.hole_hidden = np.zeros(num_envs, dtype=bool)
        self.round_dealt = np.zeros((num_envs, 10), dtype=np.int16)
        self.round_units = np.zeros(num_envs, dtype=np.float32)
        self.bet_actions = np.zeros(num_envs, dtype=np.int8)
        self.immediate_done = np.zeros(num_envs, dtype=bool)
        self._prepared = False
        self._trajectories: list[list[Transition]] = [[] for _ in range(num_envs)]
        self.total_emergency_refills = 0

        self._reset_shoes(np.arange(num_envs))

    def _cards_from_counts(self, counts: np.ndarray) -> np.ndarray:
        return np.repeat(np.arange(1, 11, dtype=np.int8), counts.astype(int))

    def _reset_shoes(self, indices: np.ndarray) -> None:
        for i in np.asarray(indices, dtype=int):
            deck = self._cards_from_counts(self.full_counts)
            self.rng.shuffle(deck)
            self.shoe_order[i].fill(0)
            self.shoe_order[i, : len(deck)] = deck
            self.shoe_size[i] = len(deck)
            self.shoe_pos[i] = 0
            self.shoe_counts[i] = self.full_counts
            self.observable_counts[i] = self.full_counts

    def _emergency_refill(self, env_index: int) -> None:
        """Reshuffle discards, excluding every card in the current round.

        This is an explicit, measurable fallback for the astronomically rare
        case where a round that started with > cut-card threshold cards consumes
        the entire shoe.  It is not the normal cut-card reshuffle.
        """
        available = self.full_counts - self.round_dealt[env_index]
        if np.any(available < 0) or int(available.sum()) == 0:
            raise RuntimeError("Current round exhausted every physically available card")
        deck = self._cards_from_counts(available)
        self.rng.shuffle(deck)
        self.shoe_order[env_index].fill(0)
        self.shoe_order[env_index, : len(deck)] = deck
        self.shoe_size[env_index] = len(deck)
        self.shoe_pos[env_index] = 0
        self.shoe_counts[env_index] = available
        hidden = np.zeros(10, dtype=np.int16)
        if self.hole_hidden[env_index]:
            hidden[self.dealer_hole[env_index] - 1] = 1
        self.observable_counts[env_index] = available + hidden
        self.total_emergency_refills += 1

    def _draw(self, indices: np.ndarray, visible: bool = True) -> np.ndarray:
        indices = np.asarray(indices, dtype=int)
        for i in indices[self.shoe_pos[indices] >= self.shoe_size[indices]]:
            self._emergency_refill(int(i))
        cards = self.shoe_order[indices, self.shoe_pos[indices]]
        self.shoe_pos[indices] += 1
        cols = cards.astype(int) - 1
        self.shoe_counts[indices, cols] -= 1
        self.round_dealt[indices, cols] += 1
        if visible:
            self.observable_counts[indices, cols] -= 1
        return cards

    @staticmethod
    def _add_cards(target: np.ndarray, indices: np.ndarray, cards: np.ndarray) -> None:
        target[indices, cards.astype(int) - 1] += 1

    def prepare_round(self) -> np.ndarray:
        """Apply cut card at the round boundary and return the 11-D bet state."""
        if self._prepared or np.any(self.player_active):
            raise RuntimeError("Cannot prepare a new round while another round is active")
        remaining = self.shoe_size - self.shoe_pos
        reshuffle = np.flatnonzero(remaining <= self.rules.reshuffle_threshold)
        if reshuffle.size:
            self._reset_shoes(reshuffle)
        self._prepared = True
        return self.bet_states()

    def bet_states(self) -> np.ndarray:
        remaining = self.observable_counts.sum(axis=1).astype(np.float32)
        denom = np.maximum(remaining, 1.0)
        state = np.empty((self.num_envs, 11), dtype=np.float32)
        state[:, :10] = self.observable_counts / denom[:, None]
        state[:, 10] = remaining / float(self.rules.shoe_size)
        return state

    def start_round(self, bet_actions: np.ndarray | None = None) -> None:
        if not self._prepared:
            raise RuntimeError("Call prepare_round before start_round")
        self._prepared = False
        if bet_actions is None:
            bet_actions = np.zeros(self.num_envs, dtype=np.int64)
        bet_actions = np.asarray(bet_actions, dtype=np.int64)
        if bet_actions.shape != (self.num_envs,) or np.any((bet_actions < 0) | (bet_actions > 2)):
            raise ValueError("bet_actions must contain one action index (0..2) per env")
        self.bet_actions[:] = bet_actions

        self.hands.fill(0)
        self.hand_size.fill(0)
        self.hand_bet.fill(1)
        self.from_split.fill(False)
        self.hand_finished.fill(False)
        self.needs_card.fill(False)
        self.num_hands.fill(1)
        self.active_hand.fill(0)
        self.player_active.fill(False)
        self.dealer.fill(0)
        self.dealer_upcard.fill(0)
        self.dealer_hole.fill(0)
        self.hole_hidden.fill(False)
        self.round_dealt.fill(0)
        self.round_units.fill(0)
        self.immediate_done.fill(False)
        self._trajectories = [[] for _ in range(self.num_envs)]

        idx = np.arange(self.num_envs)
        p1 = self._draw(idx, visible=True)
        self._add_cards(self.hands[:, 0], idx, p1)
        self.hand_size[:, 0] += 1
        up = self._draw(idx, visible=True)
        self._add_cards(self.dealer, idx, up)
        self.dealer_upcard[:] = up
        p2 = self._draw(idx, visible=True)
        self._add_cards(self.hands[:, 0], idx, p2)
        self.hand_size[:, 0] += 1
        hole = self._draw(idx, visible=False)
        self._add_cards(self.dealer, idx, hole)
        self.dealer_hole[:] = hole
        self.hole_hidden.fill(True)

        player_total, _ = hand_value(self.hands[:, 0])
        dealer_total, _ = hand_value(self.dealer)
        player_bj = player_total == 21
        peek_window = np.isin(self.dealer_upcard, (1, 10))
        dealer_bj = peek_window & (dealer_total == 21)

        # American peek is resolved before any double or split decision.
        immediate = dealer_bj | player_bj
        both = dealer_bj & player_bj
        self.round_units[dealer_bj & ~player_bj] = -1.0
        self.round_units[player_bj & ~dealer_bj] = self.rules.blackjack_payout
        self.round_units[both] = 0.0
        self.immediate_done[:] = immediate
        self.player_active[:] = ~immediate
        if np.any(immediate):
            self._reveal_holes(np.flatnonzero(immediate))

    def _reveal_holes(self, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=int)
        indices = indices[self.hole_hidden[indices]]
        if not len(indices):
            return
        cols = self.dealer_hole[indices].astype(int) - 1
        self.observable_counts[indices, cols] -= 1
        self.hole_hidden[indices] = False

    def _active_hand_arrays(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hand_idx = self.active_hand[indices].astype(int)
        return self.hands[indices, hand_idx], hand_idx

    def decision_batch(
        self, include_raw: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.flatnonzero(self.player_active)
        if not len(indices):
            empty = (indices, np.empty((0, 16), np.float32), np.empty((0, 4), bool))
            return (*empty, np.empty((0, 10), np.int16)) if include_raw else empty
        cards, hand_idx = self._active_hand_arrays(indices)
        total, soft = hand_value(cards)
        pair = np.zeros(len(indices), dtype=np.int8)
        two = self.hand_size[indices, hand_idx] == 2
        pair_rows = np.flatnonzero(two & (np.count_nonzero(cards, axis=1) == 1))
        if len(pair_rows):
            pair[pair_rows] = np.argmax(cards[pair_rows], axis=1) + 1

        remaining = self.observable_counts[indices].sum(axis=1).astype(np.float32)
        denom = np.maximum(remaining, 1.0)
        state = np.empty((len(indices), 16), dtype=np.float32)
        state[:, 0] = total / 21.0
        state[:, 1] = self.dealer_upcard[indices] / 10.0
        state[:, 2] = soft.astype(np.float32)
        state[:, 3] = pair / 10.0
        state[:, 4] = (self.num_hands[indices] - hand_idx) / float(self.rules.max_hands)
        state[:, 5:15] = self.observable_counts[indices] / denom[:, None]
        state[:, 15] = remaining / float(self.rules.shoe_size)

        mask = np.ones((len(indices), 4), dtype=bool)
        mask[:, DOUBLE] = two
        mask[:, SPLIT] = (pair > 0) & (self.num_hands[indices] < self.rules.max_hands)
        if not self.rules.double_after_split:
            mask[:, DOUBLE] &= ~self.from_split[indices, hand_idx]
        if not self.rules.resplit_aces:
            mask[:, SPLIT] &= ~((pair == 1) & self.from_split[indices, hand_idx])
        raw = self.observable_counts[indices].copy()
        return (indices, state, mask, raw) if include_raw else (indices, state, mask)

    def _append_transition(
        self,
        env_index: int,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
        reward_hand: int = -1,
    ) -> None:
        self._trajectories[env_index].append(
            Transition(state.copy(), int(action), 0.0, next_state.copy(), done, next_mask.copy(), reward_hand)
        )

    def _split_one(self, env_index: int) -> None:
        h = int(self.active_hand[env_index])
        n = int(self.num_hands[env_index])
        pair_col = int(np.argmax(self.hands[env_index, h]))
        # Shift queued hands right so both children are consecutive.
        for pos in range(n, h + 1, -1):
            self.hands[env_index, pos] = self.hands[env_index, pos - 1]
            self.hand_size[env_index, pos] = self.hand_size[env_index, pos - 1]
            self.hand_bet[env_index, pos] = self.hand_bet[env_index, pos - 1]
            self.from_split[env_index, pos] = self.from_split[env_index, pos - 1]
            self.hand_finished[env_index, pos] = self.hand_finished[env_index, pos - 1]
            self.needs_card[env_index, pos] = self.needs_card[env_index, pos - 1]
        self.hands[env_index, h].fill(0)
        self.hands[env_index, h, pair_col] = 1
        self.hand_size[env_index, h] = 1
        self.hand_bet[env_index, h] = 1
        self.from_split[env_index, h] = True
        self.hand_finished[env_index, h] = False
        self.needs_card[env_index, h] = False
        self.hands[env_index, h + 1].fill(0)
        self.hands[env_index, h + 1, pair_col] = 1
        self.hand_size[env_index, h + 1] = 1
        self.hand_bet[env_index, h + 1] = 1
        self.from_split[env_index, h + 1] = True
        self.hand_finished[env_index, h + 1] = False
        self.needs_card[env_index, h + 1] = True
        self.num_hands[env_index] += 1
        # Lazy split deal: only the active (first) child sees a new card now.
        card = self._draw(np.array([env_index]), visible=True)[0]
        self.hands[env_index, h, card - 1] += 1
        self.hand_size[env_index, h] += 1

    def step(self, actions: np.ndarray) -> None:
        indices, states, masks = self.decision_batch()
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (len(indices),):
            raise ValueError("actions must match the current decision batch")
        if len(indices) and np.any(~masks[np.arange(len(indices)), actions]):
            raise ValueError("Policy selected an illegal action")
        if not len(indices):
            return

        old_hand = self.active_hand[indices].astype(int).copy()
        terminal = actions == STAND
        reward_hand = np.full(len(indices), -1, dtype=np.int8)
        has_next = np.zeros(len(indices), dtype=bool)

        # Splits require a short queue shift, but only the rare split subset uses
        # a Python loop. Hit/double draws and all ordinary state updates are
        # batched across the vector environments.
        for env_index in indices[actions == SPLIT]:
            self._split_one(int(env_index))

        for action in (HIT, DOUBLE):
            rows = np.flatnonzero(actions == action)
            if not len(rows):
                continue
            env_idx = indices[rows]
            hand_idx = old_hand[rows]
            cards = self._draw(env_idx, visible=True)
            self.hands[env_idx, hand_idx, cards.astype(int) - 1] += 1
            self.hand_size[env_idx, hand_idx] += 1
            if action == DOUBLE:
                self.hand_bet[env_idx, hand_idx] = 2
                terminal[rows] = True
            else:
                totals, _ = hand_value(self.hands[env_idx, hand_idx])
                terminal[rows] = totals > 21

        terminal_rows = np.flatnonzero(terminal)
        if len(terminal_rows):
            env_idx = indices[terminal_rows]
            hand_idx = old_hand[terminal_rows]
            reward_hand[terminal_rows] = hand_idx
            self.hand_finished[env_idx, hand_idx] = True
            next_hand = hand_idx + 1
            next_exists = next_hand < self.num_hands[env_idx]
            has_next[terminal_rows] = next_exists
            continuing_env = env_idx[next_exists]
            continuing_hand = next_hand[next_exists]
            self.active_hand[continuing_env] = continuing_hand
            ending_env = env_idx[~next_exists]
            self.player_active[ending_env] = False

            lazy = self.needs_card[continuing_env, continuing_hand]
            lazy_env = continuing_env[lazy]
            lazy_hand = continuing_hand[lazy]
            if len(lazy_env):
                cards = self._draw(lazy_env, visible=True)
                self.hands[lazy_env, lazy_hand, cards.astype(int) - 1] += 1
                self.hand_size[lazy_env, lazy_hand] += 1
                self.needs_card[lazy_env, lazy_hand] = False

        # Compose transitions only after every environment update, so all next
        # states are observed from one consistent vector snapshot.
        next_indices, next_states, next_masks = self.decision_batch()
        next_lookup = {int(env_index): row for row, env_index in enumerate(next_indices)}
        for row, (env_index, action) in enumerate(zip(indices.tolist(), actions.tolist())):
            done = bool(terminal[row] and not has_next[row])
            if done:
                next_state = np.zeros(16, np.float32)
                next_mask = np.zeros(4, bool)
            else:
                next_row = next_lookup[env_index]
                next_state = next_states[next_row]
                next_mask = next_masks[next_row]
            self._append_transition(
                env_index, states[row], action, next_state, done, next_mask, int(reward_hand[row])
            )

    @property
    def round_complete(self) -> bool:
        return not np.any(self.player_active)

    def settle_round(self) -> RoundResult:
        if not self.round_complete:
            raise RuntimeError("Cannot settle while player decisions remain")
        pending = np.flatnonzero(~self.immediate_done)
        self._reveal_holes(pending)

        # The dealer need not draw when every player hand in an env has busted.
        dealer_needed = np.zeros(self.num_envs, dtype=bool)
        for i in pending:
            n = int(self.num_hands[i])
            totals, _ = hand_value(self.hands[i, :n])
            dealer_needed[i] = np.any(totals <= 21)
        active_dealer = pending[dealer_needed[pending]]
        while len(active_dealer):
            totals, soft = hand_value(self.dealer[active_dealer])
            hit = (totals < 17) | (
                (totals == 17) & soft & (not self.rules.dealer_stands_soft_17)
            )
            draw_idx = active_dealer[hit]
            if not len(draw_idx):
                break
            cards = self._draw(draw_idx, visible=True)
            self._add_cards(self.dealer, draw_idx, cards)

        dealer_total, _ = hand_value(self.dealer)
        hand_rewards = np.zeros((self.num_envs, self.rules.max_hands), dtype=np.float32)
        for i in pending:
            n = int(self.num_hands[i])
            totals, _ = hand_value(self.hands[i, :n])
            for h in range(n):
                stake = float(self.hand_bet[i, h])
                if totals[h] > 21:
                    reward = -stake
                elif dealer_total[i] > 21 or totals[h] > dealer_total[i]:
                    reward = stake
                elif totals[h] < dealer_total[i]:
                    reward = -stake
                else:
                    reward = 0.0
                hand_rewards[i, h] = reward
            self.round_units[i] = hand_rewards[i, :n].sum()

        transitions: list[Transition] = []
        for i, trajectory in enumerate(self._trajectories):
            for tr in trajectory:
                if tr.reward_hand >= 0:
                    tr.reward = float(hand_rewards[i, tr.reward_hand])
                transitions.append(tr)

        # At the next boundary observable and physical counts must agree.
        if np.any(self.hole_hidden):
            self._reveal_holes(np.flatnonzero(self.hole_hidden))
        if not np.array_equal(self.observable_counts, self.shoe_counts):
            raise RuntimeError("Observable composition did not reconcile at round end")
        result = RoundResult(
            units=self.round_units.copy(),
            transitions=transitions,
            emergency_refills=self.total_emergency_refills,
        )
        self.immediate_done.fill(False)
        return result

    def force_shoe_for_test(self, env_index: int, draw_order: Iterable[int]) -> None:
        """Install a deterministic remaining shoe. Intended for unit tests only."""
        cards = np.asarray(list(draw_order), dtype=np.int8)
        if np.any((cards < 1) | (cards > 10)) or len(cards) > self.max_cards:
            raise ValueError("Invalid test shoe")
        self.shoe_order[env_index].fill(0)
        self.shoe_order[env_index, : len(cards)] = cards
        self.shoe_size[env_index] = len(cards)
        self.shoe_pos[env_index] = 0
        counts = np.bincount(cards, minlength=11)[1:].astype(np.int16)
        self.shoe_counts[env_index] = counts
        self.observable_counts[env_index] = counts
