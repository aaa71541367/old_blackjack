from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from config import ProjectConfig
from evaluate import canonical_play_state, true_count


def test_canonical_hard_16_is_self_consistent() -> None:
    cfg = ProjectConfig()
    state, mask, remaining, hand = canonical_play_state(cfg, "hard", 16, 10, 5.0)
    assert hand == [10, 6]
    assert np.isclose(state[0], 16 / 21)
    assert state[1] == 1.0
    assert state[2] == 0
    assert state[3] == 0
    assert state[4] == 0.25
    assert mask.tolist() == [True, True, True, False]
    assert 4.5 <= true_count(remaining[None, :])[0] <= 5.5
    full = np.asarray(cfg.rules.full_counts)
    visible = np.bincount([10, 6, 10], minlength=11)[1:]
    assert np.all(remaining <= full - visible)


def test_canonical_soft_and_pair_fields_are_consistent() -> None:
    cfg = ProjectConfig()
    soft, _, _, hand = canonical_play_state(cfg, "soft", 18, 6, 0.0)
    assert hand == [1, 7]
    assert np.isclose(soft[0], 18 / 21) and soft[2] == 1 and soft[3] == 0
    pair, mask, _, hand = canonical_play_state(cfg, "pair", 8, 10, 0.0)
    assert hand == [8, 8]
    assert np.isclose(pair[0], 16 / 21) and pair[2] == 0 and pair[3] == 0.8
    assert mask.tolist() == [True, True, True, True]


def test_training_modules_cannot_import_count_or_oracle_code() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden_modules = {"evaluate", "oracle"}
    for filename in ("env.py", "agents.py", "train.py"):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not ({alias.name.split(".")[0] for alias in node.names} & forbidden_modules)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in forbidden_modules
        text = (root / filename).read_text(encoding="utf-8")
        assert "HILO_WEIGHTS" not in text

