"""Project configuration.

All casino rules and training hyperparameters live in dataclasses so an
experiment can be reproduced from a single JSON snapshot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass(slots=True)
class RulesConfig:
    num_decks: int = 2
    reshuffle_threshold: int = 15
    max_hands: int = 4
    dealer_stands_soft_17: bool = True
    dealer_peek: bool = True
    double_after_split: bool = True
    hit_split_aces: bool = True
    resplit_aces: bool = True
    blackjack_payout: float = 1.5
    bet_sizes: tuple[int, int, int] = (100, 500, 1000)

    def __post_init__(self) -> None:
        if self.num_decks < 1:
            raise ValueError("num_decks must be positive")
        if self.reshuffle_threshold < 0:
            raise ValueError("reshuffle_threshold cannot be negative")
        if self.max_hands < 1:
            raise ValueError("max_hands must be positive")
        if not self.dealer_peek:
            raise ValueError("This project specification requires American peek rules")

    @property
    def shoe_size(self) -> int:
        return 52 * self.num_decks

    @property
    def full_counts(self) -> tuple[int, ...]:
        return (4 * self.num_decks,) * 9 + (16 * self.num_decks,)


@dataclass(slots=True)
class DQNConfig:
    replay_capacity: int = 1_000_000
    batch_size: int = 512
    learning_rate: float = 1e-4
    target_sync_steps: int = 1_000
    warmup_transitions: int = 50_000
    gradient_updates_per_vector_step: int = 16
    max_grad_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 15_000_000
    # "huber" is robust for bootstrapped play targets.  The bet head must use
    # "mse": its reward scale differs by arm (x0.1/x0.5/x1.0), and Huber's L1
    # region is a median-pulling M-estimator, so the bias becomes arm-dependent
    # (measured ~-0.083 Q-units on the x1.0 arm vs ~0 on x0.1), which
    # structurally suppresses the max-bet arm.
    loss_fn: str = "huber"


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 20260716
    num_envs: int = 2_048
    phase1_rounds: int = 20_000_000
    phase2_rounds: int = 100_000_000
    log_interval_rounds: int = 100_000
    checkpoint_interval_rounds: int = 1_000_000
    visit_sample_rate: float = 0.02
    visit_sample_capacity: int = 2_000_000
    play: DQNConfig = field(
        default_factory=lambda: DQNConfig(gradient_updates_per_vector_step=32)
    )
    bet: DQNConfig = field(
        default_factory=lambda: DQNConfig(
            learning_rate=5e-5,
            epsilon_decay_steps=80_000_000,
            warmup_transitions=100_000,
            loss_fn="mse",
        )
    )


@dataclass(slots=True)
class EvaluationConfig:
    strategy_remaining_fraction: float = 0.75
    neutral_true_count: float = 0.0
    high_true_count: float = 5.0
    tc_half_width: float = 0.5
    fallback_tc_half_width: float = 1.0
    penetration_half_width: float = 0.10
    low_confidence_visits: int = 1_000
    auto_widen_min_visits_per_cell: int = 1_000
    bet_state_samples: int = 100_000
    oracle_states: int = 200
    oracle_rollouts_per_state: int = 1_300_000  # SE ~= 1.15/sqrt(n) ~= 0.001 per unit
    final_profit_rounds: int = 5_000_000


@dataclass(slots=True)
class ProjectConfig:
    rules: RulesConfig = field(default_factory=RulesConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "outputs"
    device: str = "auto"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def smoke_config(output_dir: str = "outputs/smoke") -> ProjectConfig:
    """Tiny configuration for integration testing; not a scientific run."""
    cfg = ProjectConfig(output_dir=output_dir)
    cfg.training.num_envs = 32
    cfg.training.phase1_rounds = 512
    cfg.training.phase2_rounds = 512
    cfg.training.log_interval_rounds = 256
    cfg.training.checkpoint_interval_rounds = 512
    cfg.training.play.warmup_transitions = 64
    cfg.training.bet.warmup_transitions = 64
    cfg.training.play.batch_size = 32
    cfg.training.bet.batch_size = 32
    cfg.training.play.gradient_updates_per_vector_step = 1
    cfg.training.bet.gradient_updates_per_vector_step = 1
    cfg.evaluation.bet_state_samples = 128
    cfg.evaluation.oracle_states = 4
    cfg.evaluation.oracle_rollouts_per_state = 20
    cfg.evaluation.final_profit_rounds = 128
    return cfg
