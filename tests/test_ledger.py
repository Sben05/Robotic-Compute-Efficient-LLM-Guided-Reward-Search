"""Regression test for the ledger accounting bug from AUDIT_REPORT.md finding 6: environment-step totals
used to assume every task ran 4,096 envs at horizon 16, which is wrong for Cartpole (512 envs) and Humanoid
(horizon 32). Locks in the corrected per-task table so a future edit can't silently reintroduce the old
one-size-fits-all assumption."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vast_ledger", ROOT / "vast" / "ledger.py")
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


def test_steps_per_iteration_matches_each_tasks_real_envs_and_horizon():
    # Ant / BallBalance: 4,096 envs, horizon_length 16 (AntPPO.yaml-style config)
    assert ledger.STEPS_PER_ITERATION["ant"] == 4096 * 16
    assert ledger.STEPS_PER_ITERATION["ballbalance"] == 4096 * 16
    # Cartpole: only 512 envs
    assert ledger.STEPS_PER_ITERATION["cartpole"] == 512 * 16
    # Humanoid: 4,096 envs but horizon_length 32
    assert ledger.STEPS_PER_ITERATION["humanoid"] == 4096 * 32


def test_cartpole_and_humanoid_are_not_accidentally_equal_to_the_ant_default():
    # the bug this guards against: every task silently getting Ant's 4096*16 figure
    assert ledger.STEPS_PER_ITERATION["cartpole"] != ledger.STEPS_PER_ITERATION["ant"]
    assert ledger.STEPS_PER_ITERATION["humanoid"] != ledger.STEPS_PER_ITERATION["ant"]
