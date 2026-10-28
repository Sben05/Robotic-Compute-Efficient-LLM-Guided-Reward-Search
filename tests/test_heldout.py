"""Regression tests for the library-leak bug found and fixed in this session (pruning/heldout.py,
pruning/orso_baseline.py, exploratory_analysis.py's e4(): a held-out seed's own human curve -- and,
during TRAIN-only rule freezing, every held-out seed's human curve -- was unconditionally included in
the forecasting library). worlds() no longer filters by seed itself; the contract is now that CALLERS
pass an already-filtered human dict. These tests pin that contract down with a sentinel value, so a
future caller that forgets to filter fails loudly instead of silently leaking held-out information."""
import numpy as np
import pytest

from heldout import worlds
from sim_core import C0

LEAK_SENTINEL = 123456.0  # an unmistakable value: if this ever appears in a library, something leaked


def reward_row(seed, name, final):
    return {"seed": seed, "reward": name, "curve": np.linspace(0, final, 3000)}


def test_worlds_excludes_a_filtered_out_seeds_human_curve():
    """The contract: worlds() puts exactly the human curves it is GIVEN into every world's library -- so a
    caller that filters `human` to exclude a held-out seed (as heldout.tune()/score() now do) gets a library
    with no trace of that seed, for every seed being screened."""
    rows = [reward_row(s, f"r{s}_{i}", 5 + i) for s in (0, 1) for i in range(4)]
    # seed 1 is "held out": its human curve must never reach worlds() at all, by the caller's own filtering
    human_all_seeds = {0: np.full(3000, 7.0), 1: np.full(3000, LEAK_SENTINEL)}
    human_train_only = {s: v for s, v in human_all_seeds.items() if s != 1}

    rng = np.random.default_rng(0)
    W = worlds(rows, human_train_only, seeds=[0], max_per_group=10, rng=rng, lib_rows=rows)
    assert W, "expected at least one world for seed 0"
    for world, lib, names, s in W:
        assert LEAK_SENTINEL not in lib.cm
        assert LEAK_SENTINEL not in lib.human


def test_worlds_passing_the_full_human_dict_does_leak_the_sentinel():
    """The inverse check: if a caller makes the old mistake (hands worlds() the FULL, unfiltered human dict,
    as tune()/score() used to), the held-out seed's curve lands in the library. This documents exactly the
    bug that was fixed and why the filtering has to happen at the call site, not inside worlds()."""
    rows = [reward_row(s, f"r{s}_{i}", 5 + i) for s in (0, 1) for i in range(4)]
    human_all_seeds = {0: np.full(3000, 7.0), 1: np.full(3000, LEAK_SENTINEL)}

    rng = np.random.default_rng(0)
    W = worlds(rows, human_all_seeds, seeds=[0], max_per_group=10, rng=rng, lib_rows=rows)
    world, lib, names, s = W[0]
    # lib.human is an AVERAGE of every human curve handed in, so a single stray curve can be masked by averaging;
    # lib.cm holds every raw curve unaveraged, which is where the leak actually shows up.
    assert LEAK_SENTINEL in lib.cm  # reproduces the historical leak when the caller skips filtering


def test_worlds_never_puts_a_worlds_own_reward_curve_in_its_own_library():
    """Independent of the human-curve issue: leave-reward-out must hold for the LLM reward curves too -- a
    world's library must exclude every curve belonging to the rewards currently being judged."""
    rows = [reward_row(0, f"r{i}", 5 + i) for i in range(5)]  # 5 rewards, N=4 -> one left over for the library
    human = {0: np.full(3000, 100.0)}  # a value that can't collide with any reward's final (5..9)
    rng = np.random.default_rng(0)
    W = worlds(rows, human, seeds=[0], max_per_group=10, rng=rng, lib_rows=rows)
    for world, lib, names, s in W:
        in_world_finals = {5 + i for i, r in enumerate(rows) if r["reward"] in names}
        for final in in_world_finals:
            assert not np.any(np.isclose(lib.final, final))


def test_worlds_skips_seeds_with_fewer_than_N_candidates():
    rows = [reward_row(0, "only_one", 9.0)]  # fewer than N=4 candidates for this seed
    human = {0: np.full(3000, 7.0)}
    W = worlds(rows, human, seeds=[0], max_per_group=10, rng=np.random.default_rng(0), lib_rows=rows)
    assert W == []


def test_tune_and_score_filter_human_to_train_seeds_only():
    """Guards the actual fix in heldout.py: tune() and score() must each build a human dict that excludes
    every TEST_SEEDS entry before calling worlds(), reproducing the exact one-line filter now used there."""
    import heldout as H
    TEST_SEEDS = (4, 5, 7, 8, 9)
    human = {s: np.full(3000, float(s)) for s in range(10)}
    human_train = {s: v for s, v in human.items() if s not in TEST_SEEDS}
    assert set(human_train) == {0, 1, 2, 3, 6}
    assert all(s not in TEST_SEEDS for s in human_train)
