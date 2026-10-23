"""Evaluation, plotting, neighborhood confidence, and count-concept analysis.

Hi-Lo exists only here (plus tests that police module boundaries).  It is never
part of an environment state, replay item, reward, policy target, or trainer.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import BetAgent, PlayAgent
from config import ProjectConfig
from env import ACTION_NAMES, DOUBLE, HIT, SPLIT, STAND, BlackjackEnv
from oracle import estimate_ev
from train import run_player_policy


HILO_WEIGHTS = np.asarray((-1, 1, 1, 1, 1, 1, 0, 0, 0, -1), dtype=np.float64)
UPCARDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
ACTION_LABELS = np.asarray(("S", "H", "D", "P"))
ILLUSTRIOUS_COMPARISON_CELLS = ((16, 10), (12, 3), (13, 2), (12, 2))


def true_count(remaining_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(remaining_counts, dtype=np.float64)
    decks_remaining = np.maximum(counts.sum(axis=-1) / 52.0, 1e-9)
    # A balanced full shoe has dot(full_counts, HILO_WEIGHTS) == 0.
    running_count = -(counts @ HILO_WEIGHTS)
    return running_count / decks_remaining


def canonical_hand(kind: str, value: int) -> list[int]:
    if kind == "soft":
        if not 13 <= value <= 20:
            raise ValueError("soft total must be 13..20")
        return [1, value - 11]
    if kind == "pair":
        if not 1 <= value <= 10:
            raise ValueError("pair value must be 1..10")
        return [value, value]
    hard = {
        5: [2, 3], 6: [2, 4], 7: [3, 4], 8: [3, 5], 9: [4, 5],
        10: [4, 6], 11: [5, 6], 12: [10, 2], 13: [10, 3],
        14: [10, 4], 15: [10, 5], 16: [10, 6], 17: [10, 7],
        18: [10, 8], 19: [10, 9], 20: [10, 5, 5],
    }
    if value not in hard:
        raise ValueError("hard total must be 5..20")
    return hard[value]


def _cards_to_counts(cards: list[int]) -> np.ndarray:
    return np.bincount(np.asarray(cards), minlength=11)[1:].astype(np.int16)


def _adjust_running_count(
    counts: np.ndarray,
    caps: np.ndarray,
    desired_rc: int,
) -> np.ndarray:
    counts = counts.copy()
    for _ in range(1_000):
        current = int(round(-counts @ HILO_WEIGHTS))
        delta = desired_rc - current
        if delta == 0:
            return counts
        # Preserve the remaining-card total by moving one physical card between
        # Hi-Lo categories.  One category step changes RC by exactly one.
        if delta > 0:
            donors = np.flatnonzero((counts > 0) & (HILO_WEIGHTS >= 0))
            receivers = np.flatnonzero((counts < caps) & (HILO_WEIGHTS < HILO_WEIGHTS[donors].max(initial=-2)))
        else:
            donors = np.flatnonzero((counts > 0) & (HILO_WEIGHTS <= 0))
            receivers = np.flatnonzero((counts < caps) & (HILO_WEIGHTS > HILO_WEIGHTS[donors].min(initial=2)))
        best: tuple[int, int, int] | None = None
        for donor in donors:
            for receiver in receivers:
                change = int(HILO_WEIGHTS[donor] - HILO_WEIGHTS[receiver])
                if change == 0 or np.sign(change) != np.sign(delta):
                    continue
                if best is None or abs(delta - change) < abs(delta - best[2]):
                    best = (int(donor), int(receiver), change)
        if best is None:
            break
        counts[best[0]] -= 1
        counts[best[1]] += 1
    raise RuntimeError("Could not construct a physical shoe at the requested true count")


def construct_remaining_counts(
    cfg: ProjectConfig,
    visible_cards: list[int],
    target_tc: float,
) -> np.ndarray:
    """Construct a physical unseen composition after removing visible cards.

    The canonical player cards and dealer upcard are removed first.  Composition
    adjustments are then performed, and TC is checked only after that removal.
    """
    full = np.asarray(cfg.rules.full_counts, dtype=np.int16)
    visible = _cards_to_counts(visible_cards)
    pre_visible = np.rint(full * cfg.evaluation.strategy_remaining_fraction).astype(np.int16)
    pre_visible = np.maximum(pre_visible, visible)
    remaining = pre_visible - visible
    caps = full - visible
    desired_rc = int(round(target_tc * remaining.sum() / 52.0))
    remaining = _adjust_running_count(remaining, caps, desired_rc)
    achieved = float(true_count(remaining[None, :])[0])
    tolerance = 0.5 if target_tc else 0.5
    if abs(achieved - target_tc) > tolerance:
        raise RuntimeError(f"Constructed TC {achieved:.3f}, outside target {target_tc:+.1f}")
    if np.any(remaining < 0) or np.any(remaining > caps):
        raise RuntimeError("Canonical strategy state is not physically possible")
    return remaining


def canonical_play_state(
    cfg: ProjectConfig,
    kind: str,
    value: int,
    upcard: int,
    target_tc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    hand = canonical_hand(kind, value)
    visible = [*hand, upcard]
    remaining = construct_remaining_counts(cfg, visible, target_tc)
    total = sum(hand)
    soft = 1 in hand and total + 10 <= 21
    total += 10 if soft else 0
    pair = hand[0] if len(hand) == 2 and hand[0] == hand[1] else 0
    if kind != "pair":
        pair = 0
    state = np.zeros(16, dtype=np.float32)
    state[0] = total / 21.0
    state[1] = upcard / 10.0
    state[2] = float(soft)
    state[3] = pair / 10.0
    state[4] = 1.0 / cfg.rules.max_hands
    state[5:15] = remaining / remaining.sum()
    state[15] = remaining.sum() / cfg.rules.shoe_size
    mask = np.ones(4, dtype=bool)
    mask[DOUBLE] = len(hand) == 2
    mask[SPLIT] = kind == "pair"
    return state, mask, remaining, hand


class VisitNeighborhoods:
    def __init__(self, path: str | Path, cfg: ProjectConfig):
        data = np.load(path)
        self.total = data["total"]
        self.upcard = data["upcard"]
        self.soft = data["soft"]
        self.pair = data["pair"]
        self.counts = data["counts"]
        self.sample_rate = float(data["sample_rate"])
        self.capacity_reached = bool(data["capacity_reached"])
        self.tc = true_count(self.counts)
        self.penetration = self.counts.sum(axis=1) / cfg.rules.shoe_size

    def count(
        self,
        kind: str,
        value: int,
        upcard: int,
        target_tc: float,
        target_penetration: float,
        tc_half_width: float,
        penetration_half_width: float,
    ) -> int:
        semantic = (self.upcard == upcard)
        if kind == "hard":
            semantic &= (self.total == value) & (~self.soft) & (self.pair == 0)
        elif kind == "soft":
            semantic &= (self.total == value) & self.soft & (self.pair == 0)
        else:
            semantic &= self.pair == value
        nearby = (
            semantic
            & (np.abs(self.tc - target_tc) <= tc_half_width)
            & (np.abs(self.penetration - target_penetration) <= penetration_half_width)
        )
        return int(round(np.count_nonzero(nearby) / max(self.sample_rate, 1e-12)))


def _strategy_values(kind: str) -> list[int]:
    return list(range(5, 21)) if kind == "hard" else list(range(13, 21)) if kind == "soft" else list(range(1, 11))


def choose_tc_window(visits: VisitNeighborhoods, cfg: ProjectConfig) -> float:
    counts: list[int] = []
    for kind in ("hard", "soft", "pair"):
        for value in _strategy_values(kind):
            for upcard in UPCARDS:
                _, _, rem, _ = canonical_play_state(cfg, kind, value, upcard, cfg.evaluation.high_true_count)
                counts.append(
                    visits.count(
                        kind, value, upcard, cfg.evaluation.high_true_count,
                        rem.sum() / cfg.rules.shoe_size,
                        cfg.evaluation.tc_half_width,
                        cfg.evaluation.penetration_half_width,
                    )
                )
    if np.median(counts) < cfg.evaluation.auto_widen_min_visits_per_cell:
        return cfg.evaluation.fallback_tc_half_width
    return cfg.evaluation.tc_half_width


def enumerate_strategy(
    agent: PlayAgent,
    cfg: ProjectConfig,
    visits: VisitNeighborhoods,
    kind: str,
    target_tc: float,
    tc_window: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    values = _strategy_values(kind)
    actions = np.empty((len(values), len(UPCARDS)), dtype=object)
    visit_grid = np.zeros_like(actions, dtype=int)
    achieved: list[float] = []
    for r, value in enumerate(values):
        for c, upcard in enumerate(UPCARDS):
            state, mask, remaining, _ = canonical_play_state(cfg, kind, value, upcard, target_tc)
            action = int(agent.act(state[None, :], mask[None, :], 0.0)[0])
            actions[r, c] = ACTION_LABELS[action]
            achieved.append(float(true_count(remaining[None, :])[0]))
            visit_grid[r, c] = visits.count(
                kind, value, upcard, target_tc,
                remaining.sum() / cfg.rules.shoe_size,
                tc_window, cfg.evaluation.penetration_half_width,
            )
    labels = ["A" if x == 1 else str(x) for x in UPCARDS]
    index = ["A" if kind == "pair" and x == 1 else str(x) for x in values]
    return (
        pd.DataFrame(actions, index=index, columns=labels),
        pd.DataFrame(visit_grid, index=index, columns=labels),
        {"min_tc": min(achieved), "max_tc": max(achieved)},
    )


def plot_strategy_heatmap(
    table: pd.DataFrame,
    visits: pd.DataFrame,
    path: Path,
    title: str,
    low_threshold: int,
) -> None:
    action_to_int = {"S": 0, "H": 1, "D": 2, "P": 3}
    numeric = np.vectorize(action_to_int.get)(table.to_numpy())
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(table) * 0.42)))
    ax.imshow(numeric, cmap=ListedColormap(("#4daf4a", "#377eb8", "#ff7f00", "#984ea3")), vmin=0, vmax=3)
    low = visits.to_numpy() < low_threshold
    overlay = np.where(low, 1.0, np.nan)
    ax.imshow(overlay, cmap=ListedColormap(("#bdbdbd",)), alpha=0.72, vmin=0, vmax=1)
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            suffix = "\nLOW" if low[r, c] else f"\nn={visits.iat[r, c]}"
            ax.text(c, r, f"{table.iat[r, c]}{suffix}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("Dealer upcard")
    ax.set_ylabel("Player total / pair")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_diff(neutral: pd.DataFrame, high: pd.DataFrame, path: Path, title: str) -> pd.DataFrame:
    diff = neutral != high
    labels = neutral + "→" + high
    labels = labels.where(diff, "·")
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(neutral) * 0.4)))
    ax.imshow(diff.to_numpy(dtype=int), cmap=ListedColormap(("#f7f7f7", "#e41a1c")), vmin=0, vmax=1)
    for r in range(diff.shape[0]):
        for c in range(diff.shape[1]):
            ax.text(c, r, labels.iat[r, c], ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(diff.columns)), diff.columns)
    ax.set_yticks(range(len(diff.index)), diff.index)
    ax.set_xlabel("Dealer upcard")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return labels


def collect_bet_states(
    cfg: ProjectConfig,
    play: PlayAgent,
    rounds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    env = BlackjackEnv(min(cfg.training.num_envs, rounds), cfg.rules, seed)
    states: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    collected = 0
    while collected < rounds:
        state = env.prepare_round()
        states.append(state.copy())
        counts.append(env.observable_counts.copy())
        env.start_round(np.zeros(env.num_envs, dtype=np.int64))
        run_player_policy(env, play, 0.0)
        collected += env.num_envs
    return np.concatenate(states)[:rounds], np.concatenate(counts)[:rounds]


def analyze_count_concept(
    bet: BetAgent,
    states: np.ndarray,
    counts: np.ndarray,
    out: Path,
) -> dict[str, float | list[float]]:
    q = bet.q_values(states)
    q_diff = q[:, 2] - q[:, 0]
    # The ten proportions sum to one, so their coefficients are identifiable
    # only up to an additive constant; penetration enters as a control so its
    # effect is not absorbed into the composition weights.  Ridge resolves the
    # rank deficiency, and the composition block is re-centered afterwards to
    # report the unique zero-mean representative.  corr(w, Hi-Lo) is invariant
    # to that additive shift.
    x = states[:, :11].astype(np.float64)
    x_centered = x - x.mean(axis=0)
    y_centered = q_diff - q_diff.mean()
    ridge = 1e-6
    full = np.linalg.solve(x_centered.T @ x_centered + ridge * np.eye(11), x_centered.T @ y_centered)
    weights = full[:10] - full[:10].mean()
    penetration_coefficient = float(full[10])
    corr_hilo = float(np.corrcoef(weights, HILO_WEIGHTS)[0, 1])
    corr_inverse = float(np.corrcoef(weights, -HILO_WEIGHTS)[0, 1])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(HILO_WEIGHTS, weights)
    for i, label in enumerate(("A", "2", "3", "4", "5", "6", "7", "8", "9", "10")):
        ax.annotate(label, (HILO_WEIGHTS[i], weights[i]), xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Hi-Lo weight")
    ax.set_ylabel("Linear coefficient for Q(1000)-Q(100)")
    ax.set_title(f"Learned composition weights (r={corr_hilo:.3f})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "weights_vs_hilo.png", dpi=170)
    plt.close(fig)

    tc = true_count(counts)
    actions = q.argmax(axis=1)
    bets = np.asarray((100, 500, 1000))[actions]
    bins = np.arange(np.floor(tc.min()), np.ceil(tc.max()) + 1.01, 1.0)
    centers = (bins[:-1] + bins[1:]) / 2
    mean_bet = np.asarray([bets[(tc >= a) & (tc < b)].mean() if np.any((tc >= a) & (tc < b)) else np.nan for a, b in zip(bins[:-1], bins[1:])])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, mean_bet, marker="o")
    ax.set_xlabel("Hi-Lo true count (evaluation only)")
    ax.set_ylabel("Mean greedy bet")
    ax.set_ylim(50, 1050)
    ax.set_title("Bet ramp learned from raw composition")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "bet_ramp.png", dpi=170)
    plt.close(fig)
    return {
        "linear_weights": weights.tolist(),
        "identifiability_note": "composition weights reported as the zero-mean representative; only differences between card weights are identified",
        "penetration_coefficient": penetration_coefficient,
        "correlation_with_hilo": corr_hilo,
        "correlation_with_inverse_hilo": corr_inverse,
    }


def evaluate_oracle_and_regret(
    bet: BetAgent,
    states: np.ndarray,
    counts: np.ndarray,
    cfg: ProjectConfig,
    out: Path,
) -> dict:
    rng = np.random.default_rng(cfg.training.seed + 500)
    eligible = np.flatnonzero(counts.sum(axis=1) >= 30)
    n = min(cfg.evaluation.oracle_states, len(eligible))
    chosen = rng.choice(eligible, size=n, replace=False)
    q = bet.q_values(states[chosen])
    agent_action = q.argmax(axis=1)
    bets = np.asarray(cfg.rules.bet_sizes)
    ev = np.empty(n)
    ev_se = np.empty(n)
    for row, idx in enumerate(chosen):
        ev[row], ev_se[row] = estimate_ev(
            counts[idx], cfg.rules, cfg.evaluation.oracle_rollouts_per_state,
            cfg.training.seed + 10_000 + row,
        )
    oracle_action = np.where(ev > 0, 2, 0)
    regret = ev * (bets[oracle_action] - bets[agent_action])
    cumulative = np.cumsum(regret)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, n + 1), cumulative)
    ax.set_xlabel("Evaluated shoe states")
    ax.set_ylabel("Cumulative expected-chip regret")
    ax.set_title("Bet agent regret vs Monte Carlo oracle")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "cumulative_regret.png", dpi=170)
    plt.close(fig)

    q_sign = np.sign(q[:, 2] - q[:, 0])
    ev_sign = np.sign(ev)
    # Only score states where the oracle itself distinguishes the EV sign; a
    # coin-flip label would contaminate the accuracy statistic.
    confident = np.abs(ev) > 2.0 * ev_se
    abs_tc = np.abs(true_count(counts[chosen]))
    edges = (0, 1, 2, 3, 5, np.inf)
    accuracy: dict[str, dict[str, float | int]] = {}
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (abs_tc >= a) & (abs_tc < b) & confident
        label = f"[{a},{b})"
        accuracy[label] = {
            "n": int(mask.sum()),
            "accuracy": float(np.mean(q_sign[mask] == ev_sign[mask])) if np.any(mask) else float("nan"),
        }
    pd.DataFrame(
        {
            "sample_index": chosen,
            "true_count": true_count(counts[chosen]),
            "oracle_ev_per_unit": ev,
            "oracle_ev_se": ev_se,
            "agent_bet": bets[agent_action],
            "oracle_bet": bets[oracle_action],
            "regret": regret,
            "oracle_sign_confident": confident,
        }
    ).to_csv(out / "oracle_samples.csv", index=False)
    return {
        "oracle_states": n,
        "oracle_sign_confident_fraction": float(confident.mean()) if n else float("nan"),
        "mean_regret": float(regret.mean()),
        "cumulative_regret": float(cumulative[-1]) if n else 0.0,
        "decision_accuracy_by_abs_tc": accuracy,
    }


def final_profit_test(
    play: PlayAgent,
    bet: BetAgent,
    cfg: ProjectConfig,
) -> dict[str, float | int]:
    target = cfg.evaluation.final_profit_rounds
    env = BlackjackEnv(min(cfg.training.num_envs, target), cfg.rules, cfg.training.seed + 900)
    profits: list[np.ndarray] = []
    done = 0
    bet_sizes = np.asarray(cfg.rules.bet_sizes)
    while done < target:
        states = env.prepare_round()
        actions = bet.act(states, epsilon=0.0)
        env.start_round(actions)
        units, _ = run_player_policy(env, play, 0.0)
        profits.append(units * bet_sizes[actions])
        done += env.num_envs
    x = np.concatenate(profits)[:target].astype(np.float64)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return {
        "rounds": len(x), "mean_chips_per_round": mean,
        "standard_error": se, "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se, "total_profit": float(x.sum()),
    }


def evaluate_all(
    cfg: ProjectConfig,
    play_checkpoint: str,
    bet_checkpoint: str,
    visits_path: str,
) -> Path:
    out = Path(cfg.output_dir) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    play = PlayAgent(cfg.training.play, cfg.device, cfg.training.seed)
    play.load(play_checkpoint, load_optimizer=False)
    play.set_eval_frozen()
    bet = BetAgent(cfg.training.bet, cfg.device, cfg.training.seed)
    bet.load(bet_checkpoint, load_optimizer=False)
    bet.set_eval_frozen()
    visits = VisitNeighborhoods(visits_path, cfg)
    tc_window = choose_tc_window(visits, cfg)
    strategy_report: dict[str, dict] = {}
    policy_tables: dict[tuple[str, str], pd.DataFrame] = {}

    for kind in ("hard", "soft", "pair"):
        neutral, neutral_visits, neutral_range = enumerate_strategy(
            play, cfg, visits, kind, cfg.evaluation.neutral_true_count, tc_window
        )
        high, high_visits, high_range = enumerate_strategy(
            play, cfg, visits, kind, cfg.evaluation.high_true_count, tc_window
        )
        policy_tables[(kind, "neutral")] = neutral
        policy_tables[(kind, "high")] = high
        neutral.to_csv(out / f"{kind}_neutral.csv")
        neutral_visits.to_csv(out / f"{kind}_neutral_visits.csv")
        high.to_csv(out / f"{kind}_tc_plus5.csv")
        high_visits.to_csv(out / f"{kind}_tc_plus5_visits.csv")
        plot_strategy_heatmap(neutral, neutral_visits, out / f"{kind}_neutral.png", f"{kind.title()} strategy: TC≈0", cfg.evaluation.low_confidence_visits)
        plot_strategy_heatmap(high, high_visits, out / f"{kind}_tc_plus5.png", f"{kind.title()} strategy: TC≈+5", cfg.evaluation.low_confidence_visits)
        diff = plot_diff(neutral, high, out / f"{kind}_diff.png", f"{kind.title()} changes: neutral → TC +5")
        diff.to_csv(out / f"{kind}_diff.csv")
        strategy_report[kind] = {
            "neutral_constructed_tc": neutral_range,
            "high_constructed_tc": high_range,
            "changed_cells": int((neutral != high).to_numpy().sum()),
        }

    illustrious_rows = []
    for total, upcard in ILLUSTRIOUS_COMPARISON_CELLS:
        column = "A" if upcard == 1 else str(upcard)
        neutral_action = policy_tables[("hard", "neutral")].loc[str(total), column]
        high_action = policy_tables[("hard", "high")].loc[str(total), column]
        illustrious_rows.append(
            {
                "cell": f"{total}v{column}",
                "neutral_action": neutral_action,
                "tc_plus5_action": high_action,
                "changed": bool(neutral_action != high_action),
            }
        )
    pd.DataFrame(illustrious_rows).to_csv(out / "illustrious18_comparison.csv", index=False)
    strategy_report["illustrious18_named_cells"] = illustrious_rows

    bet_states, bet_counts = collect_bet_states(
        cfg, play, cfg.evaluation.bet_state_samples, cfg.training.seed + 300
    )
    count_report = analyze_count_concept(bet, bet_states, bet_counts, out)
    oracle_report = evaluate_oracle_and_regret(bet, bet_states, bet_counts, cfg, out)
    profit_report = final_profit_test(play, bet, cfg)
    report = {
        "rules": asdict(cfg.rules),
        "contextual_bandit_explanation": "Bet size does not affect shoe transition; gamma=0 and Q(s,a)=E[r|s,a]. Training rewards are chips/1000.",
        "visit_count": {
            "definition": "estimated visits in (strategy cell × TC interval × penetration interval)",
            "sample_rate": visits.sample_rate,
            "sample_capacity_reached": visits.capacity_reached,
            "tc_half_width_used": tc_window,
            "penetration_half_width": cfg.evaluation.penetration_half_width,
        },
        "strategy": strategy_report,
        "count_concept": count_report,
        "oracle_and_regret": oracle_report,
        "final_profit": profit_report,
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path
