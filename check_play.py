"""check_play.py — measure the frozen play agent's greedy (epsilon=0) EV.

Flat 1-unit bets, 2M rounds.  Run from the project directory:
    python check_play.py
"""
import numpy as np

from agents import PlayAgent
from config import ProjectConfig
from env import BlackjackEnv
from train import run_player_policy

cfg = ProjectConfig()
env = BlackjackEnv(cfg.training.num_envs, cfg.rules, seed=777)
play = PlayAgent(cfg.training.play, cfg.device, cfg.training.seed)
play.load("outputs/play_final.pt", load_optimizer=False)
play.set_eval_frozen()

target = 2_000_000
chunks: list[np.ndarray] = []
done = 0
while done < target:
    env.prepare_round()
    env.start_round(np.zeros(env.num_envs, dtype=np.int64))
    units, _ = run_player_policy(env, play, epsilon=0.0)
    chunks.append(units)
    done += env.num_envs

x = np.concatenate(chunks)[:target].astype(np.float64)
mean = x.mean()
half = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
print(f"greedy EV per round: {mean:+.5f} +/- {half:.5f} (95% CI), n={len(x):,}")
