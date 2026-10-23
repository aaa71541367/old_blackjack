"""Strictly sequential Phase 1 / Phase 2 training loops."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents import BetAgent, PlayAgent
from config import ProjectConfig
from env import BlackjackEnv, Transition
from visit_stats import RawVisitSampler


def seed_everything(seed: int) -> None:
    np.random.seed(seed)


def run_player_policy(
    env: BlackjackEnv,
    play_agent: PlayAgent,
    epsilon: float,
    visit_sampler: RawVisitSampler | None = None,
) -> tuple[np.ndarray, list[Transition]]:
    while not env.round_complete:
        indices, states, masks, raw = env.decision_batch(include_raw=True)
        if visit_sampler is not None:
            visit_sampler.observe(states, raw)
        actions = play_agent.act(states, masks, epsilon)
        env.step(actions)
    result = env.settle_round()
    return result.units, result.transitions


def _append_log(path: Path, row: dict[str, float | int]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _plot_training_log(csv_path: Path, output_path: Path, title: str) -> None:
    if not csv_path.exists():
        return
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return
    data = np.atleast_1d(data)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(data["rounds"], data["mean_loss"])
    axes[0].set_ylabel("Huber loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(data["rounds"], data["mean_ev"])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Mean reward / round")
    axes[1].set_xlabel("Rounds")
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_play(cfg: ProjectConfig, resume: str | None = None) -> Path:
    """Phase 1: flat betting, update only the play Double DQN."""
    seed_everything(cfg.training.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    env = BlackjackEnv(cfg.training.num_envs, cfg.rules, cfg.training.seed)
    agent = PlayAgent(cfg.training.play, cfg.device, cfg.training.seed)
    rounds = agent.load(resume) if resume else 0
    visits = RawVisitSampler(
        cfg.training.visit_sample_rate,
        cfg.training.visit_sample_capacity,
        cfg.rules.num_decks,
        cfg.rules.max_hands,
        cfg.training.seed + 1,
    )
    log_path = out / "phase1_log.csv"
    last_log = rounds
    last_checkpoint = rounds
    recent_units: list[np.ndarray] = []
    recent_losses: list[float] = []
    started = time.time()

    while rounds < cfg.training.phase1_rounds:
        env.prepare_round()
        env.start_round(np.zeros(env.num_envs, dtype=np.int64))
        epsilon = agent.epsilon(rounds)
        units, transitions = run_player_policy(env, agent, epsilon, visits)
        agent.add_transitions(transitions)
        for _ in range(cfg.training.play.gradient_updates_per_vector_step):
            loss = agent.train_step()
            if loss is not None:
                recent_losses.append(loss)
        actual = min(env.num_envs, cfg.training.phase1_rounds - rounds)
        # The final vector batch can overshoot only in the displayed count.
        rounds += actual
        recent_units.append(units[:actual])

        if rounds - last_log >= cfg.training.log_interval_rounds or rounds >= cfg.training.phase1_rounds:
            row = {
                "rounds": rounds,
                "epsilon": epsilon,
                "mean_loss": float(np.mean(recent_losses)) if recent_losses else float("nan"),
                "mean_ev": float(np.concatenate(recent_units).mean()),
                "buffer_size": agent.buffer.size,
                "gradient_steps": agent.gradient_steps,
                "emergency_refills": env.total_emergency_refills,
                "elapsed_seconds": round(time.time() - started, 2),
            }
            _append_log(log_path, row)
            print(json.dumps(row), flush=True)
            recent_units.clear()
            recent_losses.clear()
            last_log = rounds
        if rounds - last_checkpoint >= cfg.training.checkpoint_interval_rounds:
            agent.save(out / f"play_{rounds}.pt", rounds)
            agent.save(out / "play_latest.pt", rounds)
            visits.save(out / "play_visits.npz")
            last_checkpoint = rounds

    checkpoint = out / "play_final.pt"
    agent.save(checkpoint, rounds)
    visits.save(out / "play_visits.npz")
    _plot_training_log(log_path, out / "phase1_training.png", "Phase 1: play agent")
    return checkpoint


def _bet_transitions(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> list[Transition]:
    zero_state = np.zeros(11, dtype=np.float32)
    zero_mask = np.zeros(3, dtype=bool)
    return [
        Transition(s, int(a), float(r), zero_state, True, zero_mask)
        for s, a, r in zip(states, actions, rewards)
    ]


def normalize_bet_rewards(units: np.ndarray, bet_chips: np.ndarray) -> np.ndarray:
    """Change chip units to stable DQN targets without changing the argmax."""
    return np.asarray(units, dtype=np.float32) * np.asarray(bet_chips, dtype=np.float32) / 1000.0


def train_bet(
    cfg: ProjectConfig,
    play_checkpoint: str,
    resume: str | None = None,
) -> Path:
    """Phase 2: freeze play policy and update only the γ=0 bet DQN.

    The chip result is divided by 1000 before regression.  The action values
    therefore correspond to bet scales {0.1, 0.5, 1.0}; this unit change leaves
    argmax decisions unchanged and prevents thousand-scale Q targets.
    """
    seed_everything(cfg.training.seed + 10)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = BlackjackEnv(cfg.training.num_envs, cfg.rules, cfg.training.seed + 10)
    play = PlayAgent(cfg.training.play, cfg.device, cfg.training.seed)
    play.load(play_checkpoint, load_optimizer=False)
    play.set_eval_frozen()
    bet = BetAgent(cfg.training.bet, cfg.device, cfg.training.seed + 10)
    rounds = bet.load(resume) if resume else 0
    log_path = out / "phase2_log.csv"
    last_log = rounds
    last_checkpoint = rounds
    recent_chip: list[np.ndarray] = []
    recent_losses: list[float] = []
    started = time.time()
    bet_sizes = np.asarray(cfg.rules.bet_sizes, dtype=np.float32)

    while rounds < cfg.training.phase2_rounds:
        bet_states = env.prepare_round()
        epsilon = bet.epsilon(rounds)
        bet_actions = bet.act(bet_states, epsilon=epsilon)
        env.start_round(bet_actions)
        units, _ = run_player_policy(env, play, epsilon=0.0)
        chips = units * bet_sizes[bet_actions]
        normalized_rewards = normalize_bet_rewards(units, bet_sizes[bet_actions])
        bet.add_transitions(_bet_transitions(bet_states, bet_actions, normalized_rewards))
        for _ in range(cfg.training.bet.gradient_updates_per_vector_step):
            loss = bet.train_step()
            if loss is not None:
                recent_losses.append(loss)
        actual = min(env.num_envs, cfg.training.phase2_rounds - rounds)
        rounds += actual
        recent_chip.append(chips[:actual])

        if rounds - last_log >= cfg.training.log_interval_rounds or rounds >= cfg.training.phase2_rounds:
            row = {
                "rounds": rounds,
                "epsilon": epsilon,
                "mean_loss": float(np.mean(recent_losses)) if recent_losses else float("nan"),
                "mean_ev": float(np.concatenate(recent_chip).mean()),
                "buffer_size": bet.buffer.size,
                "gradient_steps": bet.gradient_steps,
                "emergency_refills": env.total_emergency_refills,
                "elapsed_seconds": round(time.time() - started, 2),
            }
            _append_log(log_path, row)
            print(json.dumps(row), flush=True)
            recent_chip.clear()
            recent_losses.clear()
            last_log = rounds
        if rounds - last_checkpoint >= cfg.training.checkpoint_interval_rounds:
            bet.save(out / f"bet_{rounds}.pt", rounds)
            bet.save(out / "bet_latest.pt", rounds)
            last_checkpoint = rounds

    checkpoint = out / "bet_final.pt"
    bet.save(checkpoint, rounds)
    _plot_training_log(log_path, out / "phase2_training.png", "Phase 2: bet agent")
    return checkpoint
