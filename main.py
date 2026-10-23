"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import ProjectConfig, smoke_config
from evaluate import evaluate_all
from train import train_bet, train_play


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Two-level blackjack Double DQN")
    p.add_argument("--output", default="outputs")
    p.add_argument("--device", default="auto")
    sub = p.add_subparsers(dest="command", required=True)
    play = sub.add_parser("train-play")
    play.add_argument("--resume")
    bet = sub.add_parser("train-bet")
    bet.add_argument("--play", required=True)
    bet.add_argument("--resume")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--play", required=True)
    ev.add_argument("--bet", required=True)
    ev.add_argument("--visits", required=True)
    sub.add_parser("smoke")
    return p


def main() -> None:
    args = parser().parse_args()
    cfg = smoke_config(args.output) if args.command == "smoke" else ProjectConfig(output_dir=args.output)
    cfg.device = args.device
    if args.command == "train-play":
        print(train_play(cfg, args.resume))
    elif args.command == "train-bet":
        print(train_bet(cfg, args.play, args.resume))
    elif args.command == "evaluate":
        print(evaluate_all(cfg, args.play, args.bet, args.visits))
    else:
        play = train_play(cfg)
        bet = train_bet(cfg, str(play))
        print(evaluate_all(cfg, str(play), str(bet), str(Path(cfg.output_dir) / "play_visits.npz")))


if __name__ == "__main__":
    main()
