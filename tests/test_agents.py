from __future__ import annotations

import numpy as np

from agents import PlayAgent
from config import DQNConfig
from env import Transition
from train import normalize_bet_rewards


def test_terminal_all_false_next_mask_is_safe() -> None:
    cfg = DQNConfig(replay_capacity=8, batch_size=1, warmup_transitions=1)
    agent = PlayAgent(cfg, device="cpu", seed=7)
    transition = Transition(
        np.zeros(16, np.float32), 0, 1.0, np.zeros(16, np.float32), True, np.zeros(4, bool)
    )
    agent.add_transitions([transition])
    loss = agent.train_step()
    assert loss is not None and np.isfinite(loss)


def test_illegal_actions_are_never_selected() -> None:
    cfg = DQNConfig(replay_capacity=8, batch_size=1, warmup_transitions=1)
    agent = PlayAgent(cfg, device="cpu", seed=8)
    states = np.zeros((100, 16), np.float32)
    masks = np.tile(np.array([True, False, False, False]), (100, 1))
    assert np.all(agent.act(states, masks, epsilon=1.0) == 0)


def test_bet_reward_unit_scaling() -> None:
    units = np.asarray([-8.0, 1.5, 8.0])
    bets = np.asarray([1000.0, 500.0, 1000.0])
    assert np.allclose(normalize_bet_rewards(units, bets), [-8.0, 0.75, 8.0])

