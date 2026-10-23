from __future__ import annotations

import numpy as np

from config import RulesConfig
from env import HIT, SPLIT, STAND, BlackjackEnv


def test_state_dimensions_and_no_predeal_hand_features() -> None:
    env = BlackjackEnv(8, RulesConfig(), seed=1)
    bet = env.prepare_round()
    assert bet.shape == (8, 11)
    assert np.allclose(bet[:, :10].sum(axis=1), 1.0)
    env.start_round(np.zeros(8, dtype=np.int64))
    _, play, mask = env.decision_batch()
    assert play.shape[1] == 16
    assert mask.shape[1] == 4


def test_cut_card_is_checked_only_at_round_boundary() -> None:
    rules = RulesConfig(reshuffle_threshold=15)
    env = BlackjackEnv(1, rules, seed=2)
    # Sixteen cards survives the boundary check, then immediately falls below
    # the threshold during the initial deal. It must not reshuffle mid-round.
    env.force_shoe_for_test(0, [10, 6, 10, 10, 5, 4, 3, 2, 7, 8, 9, 1, 2, 3, 4, 5])
    env.prepare_round()
    assert env.shoe_size[0] == 16
    env.start_round(np.array([0]))
    assert env.shoe_size[0] == 16
    assert env.shoe_pos[0] == 4
    env.step(np.array([STAND]))
    env.settle_round()
    env.prepare_round()
    assert env.shoe_size[0] == rules.shoe_size
    assert env.shoe_pos[0] == 0


def test_american_peek_happens_before_actions() -> None:
    env = BlackjackEnv(1, RulesConfig(reshuffle_threshold=0), seed=3)
    env.force_shoe_for_test(0, [10, 1, 9, 10] + [2] * 16)
    env.prepare_round()
    env.start_round(np.array([0]))
    assert env.round_complete
    result = env.settle_round()
    assert result.units[0] == -1.0
    assert result.transitions == []


def test_player_blackjack_peek_order_push_and_payout() -> None:
    env = BlackjackEnv(2, RulesConfig(reshuffle_threshold=0), seed=4)
    env.force_shoe_for_test(0, [1, 10, 10, 1] + [2] * 16)  # both BJ
    env.force_shoe_for_test(1, [1, 10, 10, 9] + [2] * 16)  # player BJ only
    env.prepare_round()
    env.start_round(np.zeros(2, dtype=np.int64))
    result = env.settle_round()
    assert result.units.tolist() == [0.0, 1.5]


def test_split_second_child_is_dealt_lazily_and_credit_is_chained() -> None:
    env = BlackjackEnv(1, RulesConfig(reshuffle_threshold=0), seed=5)
    # P:8,8; D:10,6. First child receives 2, second receives 3 only
    # after the first stands; dealer then draws 5 to 21.
    env.force_shoe_for_test(0, [8, 10, 8, 6, 2, 3, 5] + [4] * 13)
    env.prepare_round()
    env.start_round(np.array([0]))
    env.step(np.array([SPLIT]))
    assert env.shoe_pos[0] == 5
    assert env.needs_card[0, 1]
    assert env.observable_counts[0, 2] == 1  # the future 3 remains unseen
    env.step(np.array([STAND]))
    assert env.shoe_pos[0] == 6
    assert not env.needs_card[0, 1]
    env.step(np.array([STAND]))
    result = env.settle_round()
    assert [t.reward for t in result.transitions] == [0.0, -1.0, -1.0]
    assert [t.done for t in result.transitions] == [False, False, True]
    assert sum(t.reward for t in result.transitions) == result.units[0] == -2.0


def test_dealer_hole_does_not_leak_into_play_composition() -> None:
    env = BlackjackEnv(1, RulesConfig(reshuffle_threshold=0), seed=6)
    cards = [10, 6, 6, 9] + [2] * 16
    initial_nines = cards.count(9)
    env.force_shoe_for_test(0, cards)
    env.prepare_round()
    env.start_round(np.array([0]))
    _, _, _, raw = env.decision_batch(include_raw=True)
    assert raw[0, 8] == initial_nines  # hidden 9 remains in unseen composition


def test_exhausted_shoe_uses_explicit_discard_fallback() -> None:
    env = BlackjackEnv(1, RulesConfig(reshuffle_threshold=0), seed=9)
    env.force_shoe_for_test(0, [2, 10, 2, 6, 2])
    env.prepare_round()
    env.start_round(np.array([0]))
    env.step(np.array([HIT]))  # consumes the last physical shoe card
    assert env.shoe_pos[0] == env.shoe_size[0]
    env.step(np.array([HIT]))  # now and only now may discards be reshuffled
    assert env.total_emergency_refills == 1
    assert np.all(env.shoe_counts[0] <= np.asarray(env.rules.full_counts))
