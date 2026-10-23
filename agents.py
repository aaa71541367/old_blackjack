"""Replay buffers and the two independent DQN agents."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from config import DQNConfig


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, state_dim), np.float32)
        self.next_states = np.empty((capacity, state_dim), np.float32)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.dones = np.empty(capacity, bool)
        self.next_masks = np.empty((capacity, action_dim), bool)
        self.size = 0
        self.pos = 0

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        next_masks: np.ndarray,
    ) -> None:
        n = len(states)
        if n == 0:
            return
        if n >= self.capacity:
            states, actions, rewards = states[-self.capacity :], actions[-self.capacity :], rewards[-self.capacity :]
            next_states, dones, next_masks = (
                next_states[-self.capacity :], dones[-self.capacity :], next_masks[-self.capacity :]
            )
            n = self.capacity
        idx = (np.arange(n) + self.pos) % self.capacity
        self.states[idx] = states
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_states[idx] = next_states
        self.dones[idx] = dones
        self.next_masks[idx] = next_masks
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_states[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
            torch.as_tensor(self.next_masks[idx], device=device),
        )


class MLP(nn.Module):
    def __init__(self, dims: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers.extend((nn.Linear(a, b), nn.ReLU()))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: tuple[int, ...],
        gamma: float,
        config: DQNConfig,
        device: str = "auto",
        seed: int = 0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = float(gamma)
        self.config = config
        self.device = resolve_device(device)
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        dims = (state_dim, *hidden, action_dim)
        self.online = MLP(dims).to(self.device)
        self.target = MLP(dims).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.replay_capacity, state_dim, action_dim, seed)
        self.gradient_steps = 0

    def epsilon(self, environment_steps: int) -> float:
        fraction = min(1.0, environment_steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    @torch.no_grad()
    def q_values(self, states: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return self.online(x).cpu().numpy()

    @torch.no_grad()
    def act(self, states: np.ndarray, masks: np.ndarray | None = None, epsilon: float = 0.0) -> np.ndarray:
        if len(states) == 0:
            return np.empty(0, dtype=np.int64)
        if masks is None:
            masks = np.ones((len(states), self.action_dim), dtype=bool)
        if np.any(~masks.any(axis=1)):
            raise ValueError("Every nonterminal state must have a legal action")
        q = self.q_values(states)
        q[~masks] = -np.inf
        actions = q.argmax(axis=1)
        explore_rows = np.flatnonzero(self.rng.random(len(states)) < epsilon)
        if len(explore_rows):
            # Uniform over legal actions, vectorized: the argmax of iid U(0,1)
            # noise restricted to legal entries is uniform on those entries.
            noise = self.rng.random((len(explore_rows), self.action_dim)) * masks[explore_rows]
            actions[explore_rows] = noise.argmax(axis=1)
        return actions.astype(np.int64)

    def add_transitions(self, transitions: list) -> None:
        if not transitions:
            return
        self.buffer.add_batch(
            np.stack([t.state for t in transitions]),
            np.asarray([t.action for t in transitions]),
            np.asarray([t.reward for t in transitions]),
            np.stack([t.next_state for t in transitions]),
            np.asarray([t.done for t in transitions]),
            np.stack([t.next_mask for t in transitions]),
        )

    def train_step(self) -> float | None:
        if self.buffer.size < max(self.config.warmup_transitions, self.config.batch_size):
            return None
        states, actions, rewards, next_states, dones, next_masks = self.buffer.sample(
            self.config.batch_size, self.device
        )
        q = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            # Double DQN: online chooses among legal actions; target evaluates it.
            online_next = self.online(next_states).masked_fill(~next_masks, -torch.inf)
            next_actions = online_next.argmax(dim=1)
            target_next = self.target(next_states).gather(1, next_actions[:, None]).squeeze(1)
            target = rewards + (~dones).float() * self.gamma * target_next
        if self.config.loss_fn == "mse":
            loss = F.mse_loss(q, target)
        else:
            loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def set_eval_frozen(self) -> None:
        self.online.eval()
        self.target.eval()
        for parameter in self.online.parameters():
            parameter.requires_grad_(False)

    def save(self, path: str | Path, environment_steps: int = 0) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "gradient_steps": self.gradient_steps,
                "environment_steps": environment_steps,
                "gamma": self.gamma,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "config": asdict(self.config),
            },
            path,
        )

    def load(self, path: str | Path, load_optimizer: bool = True) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
        return int(checkpoint.get("environment_steps", 0))


class PlayAgent(DQNAgent):
    def __init__(self, config: DQNConfig, device: str = "auto", seed: int = 0):
        super().__init__(16, 4, (256, 256), 1.0, config, device, seed)


class BetAgent(DQNAgent):
    """γ=0 contextual bandit represented with the same DQN machinery.

    Bets cannot affect the next shoe state.  Consequently the Bellman equation
    reduces exactly to Q(s,a)=E[r|s,a].  The target network is retained only so
    both project agents use one auditable DQN implementation.
    """

    def __init__(self, config: DQNConfig, device: str = "auto", seed: int = 0):
        super().__init__(11, 3, (128, 128), 0.0, config, device, seed)

