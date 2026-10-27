"""Shared loaders: finished PPO curves, generation metadata. Reads results/ only; nothing is written outside results/llm_efficiency."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "llm_efficiency"
T = 3000

# name -> (results subdir, kind). fresh = round-1-style independent sample (task description + env code only);
# refined = round-2 sample written after seeing training feedback of the best round-1 candidate.
LLM = {
    "r1c0": ("exp_b_minimax/round1/cand0", "fresh"), "r1c1": ("exp_b_minimax/round1/cand1", "fresh"),
    "r1c2": ("exp_b_minimax/round1/cand2", "fresh"),
    "p0": ("exp_d_pool_minimax/round1/cand0", "fresh"), "p1": ("exp_d_pool_minimax/round1/cand1", "fresh"),
    "p2": ("exp_d_pool_minimax/round1/cand2", "fresh"), "p3": ("exp_d_pool_minimax/round1/cand3", "fresh"),
    "p4": ("exp_d_pool_minimax/round1/cand4", "fresh"), "p5": ("exp_d_pool_minimax/round1/cand5", "fresh"),
    "r2c0": ("exp_b_minimax/round2/cand0", "refined"), "r2c1": ("exp_b_minimax/round2/cand1", "refined"),
}
HUMAN = {"human0": "exp_a_human_3000", "human1": "exp_a_human_3000_seed1"}
SEC_PER_ITER_SOLO = 704.0 / 3000  # REPORT.md: one 3,000-it run alone on an A100


def curve(rel):
    d = json.loads((ROOT / "results" / rel / "curves.json").read_text())
    y = np.asarray(d["consecutive_successes"], float)
    assert len(y) == T
    return y


def load_curves():
    llm = {k: curve(v[0]) for k, v in LLM.items()}
    human = {k: curve(v) for k, v in HUMAN.items()}
    return llm, human


def generation(name):
    return json.loads((ROOT / "results" / LLM[name][0] / "generation.json").read_text())
