"""simulate_bankroll.py — bankroll trajectories with the real trained model.

2,000 independent sessions x 500 rounds, each session a continuing shoe,
starting bankroll 100,000.  Run from the project directory (~2-3 min):
    python simulate_bankroll.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents import BetAgent, PlayAgent
from config import ProjectConfig
from env import BlackjackEnv
from train import run_player_policy

SESSIONS = 2_000
ROUNDS = 500
START = 100_000.0

cfg = ProjectConfig()
env = BlackjackEnv(SESSIONS, cfg.rules, seed=2026)
play = PlayAgent(cfg.training.play, cfg.device, 0)
play.load("outputs/play_final.pt", load_optimizer=False)
play.set_eval_frozen()
bet = BetAgent(cfg.training.bet, cfg.device, 1)
bet.load("outputs_v2/bet_final.pt", load_optimizer=False)
bet.set_eval_frozen()

bet_sizes = np.asarray(cfg.rules.bet_sizes, dtype=np.float64)
bank = np.full(SESSIONS, START)
traj = np.empty((ROUNDS + 1, SESSIONS))
traj[0] = bank
bet_sum = 0.0
big = 0

for t in range(ROUNDS):
    states = env.prepare_round()
    actions = bet.act(states, epsilon=0.0)
    env.start_round(actions)
    units, _ = run_player_policy(env, play, 0.0)
    chips = units * bet_sizes[actions]
    bank += chips
    traj[t + 1] = bank
    bet_sum += bet_sizes[actions].sum()
    big += int((actions == 2).sum())

fin = traj[-1]
lows = traj.min(axis=0)
per = np.diff(traj, axis=0)
n = ROUNDS * SESSIONS
print(f"平均下注        : {bet_sum / n:,.0f}  (押1000比例 {big / n:.3f})")
print(f"每手損益        : mean {per.mean():+,.2f}   std {per.std():,.0f}")
print(f"500局終點       : mean {fin.mean():+,.0f}   median {np.median(fin):+,.0f}")
print(f"  p5 / p95      : {np.percentile(fin, 5):,.0f} / {np.percentile(fin, 95):,.0f}")
print(f"  P(收在虧損)   : {np.mean(fin < START):.3f}")
print(f"路徑最低點      : median {np.median(lows):,.0f}   p5 {np.percentile(lows, 5):,.0f}   worst {lows.min():,.0f}")
print(f"破產(<100)      : {(lows < 100).sum()} / {SESSIONS}")

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
x = np.arange(ROUNDS + 1)
for i in range(60):
    ax.plot(x, traj[:, i], color="steelblue", alpha=0.18, lw=0.8)
ax.fill_between(
    x,
    np.percentile(traj, 5, axis=1),
    np.percentile(traj, 95, axis=1),
    color="steelblue",
    alpha=0.15,
    label="5-95%",
)
ax.plot(x, traj.mean(axis=1), color="crimson", lw=2.2, label=f"mean (final {fin.mean():+,.0f})")
ax.axhline(START, color="black", ls="--", lw=1)
ax.set_xlabel("Rounds")
ax.set_ylabel("Bankroll (start 100,000)")
ax.set_title("500-round bankroll paths - trained model (2,000 sessions)")
ax.legend()
fig.tight_layout()
fig.savefig("outputs_v2/bankroll_sim.png")
print("figure -> outputs_v2/bankroll_sim.png")
