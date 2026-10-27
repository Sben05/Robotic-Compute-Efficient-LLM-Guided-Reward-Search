"""Load every finished PPO learning curve (consecutive_successes per iteration, 3,000 its) from results/.

Pool (actual runs, all seed 0 unless stated):
  old11  : 3+2 Mini-Eureka (exp_b) + 6 extra round-1 samples (exp_d), trained straight to 3,000.
  live16 : live trials 1-2 (exp_e). Each was trained 500 its, stopped, and later resumed to 3,000 (curves_full.json).
           The resume splice is validated separately (human 500+2,500 control in exp_f); until then these 3,000-it
           values are labelled "audit, resumed".
  seeds  : exp_f_seeds/seed{S}/<reward> (500 + resume to 3,000) for the 7 strongest rewards + human, seeds 1-5 when present.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T = 3000

OLD = {
    "r1c0": "exp_b_minimax/round1/cand0", "r1c1": "exp_b_minimax/round1/cand1", "r1c2": "exp_b_minimax/round1/cand2",
    "r2c0": "exp_b_minimax/round2/cand0", "r2c1": "exp_b_minimax/round2/cand1",
    "p0": "exp_d_pool_minimax/round1/cand0", "p1": "exp_d_pool_minimax/round1/cand1", "p2": "exp_d_pool_minimax/round1/cand2",
    "p3": "exp_d_pool_minimax/round1/cand3", "p4": "exp_d_pool_minimax/round1/cand4", "p5": "exp_d_pool_minimax/round1/cand5",
}
LIVE = {f"t{t}r{r}c{c}": f"exp_e_live/trial{t}/round{r}/cand{c}" for t in (1, 2) for r in (1, 2) for c in range(4)}
HUMAN = {"human0": "exp_a_human_3000", "human1": "exp_a_human_3000_seed1"}
# real search rounds (candidate sets that were actually screened together)
ROUNDS = {"exp_b r1": ["r1c0", "r1c1", "r1c2"], "exp_b r2": ["r2c0", "r2c1"],
          **{f"live t{t} r{r}": [f"t{t}r{r}c{c}" for c in range(4)] for t in (1, 2) for r in (1, 2)}}
# exp_f reward keys -> pool names (seed 0 curve)
SEED_KEYS = {"A_round2c0_10.93": "r2c0", "B_round2c1_10.75": "r2c1", "C_pool_c3_10.73": "p3",
             "D_live2_r2c2_10.43": "t2r2c2", "E_live2_r1c3_9.62": "t2r1c3", "F_live2_r2c3_9.61_winner": "t2r2c3",
             "G_live1_r2c0_9.29_winner": "t1r2c0", "human": "human0"}


def _load(rel, fname):
    d = json.loads((ROOT / "results" / rel / fname).read_text())
    y = np.asarray(d["consecutive_successes"], float)
    assert len(y) == T, (rel, len(y))
    return y


def load_all(live=True):
    llm = {k: _load(v, "curves.json") for k, v in OLD.items()}
    if live:
        llm.update({k: _load(v, "curves_full.json") for k, v in LIVE.items()})
    human = {k: _load(v, "curves.json") for k, v in HUMAN.items()}
    return llm, human


def load_seeds():
    """{pool name: {seed: curve}} from exp_f (500 + resume) plus the seed-0 curve from the pool."""
    llm, human = load_all()
    out = {}
    for key, name in SEED_KEYS.items():
        out[name] = {0: (human if name.startswith("human") else llm)[name]}
        for d in sorted((ROOT / "results/exp_f_seeds").glob("seed[0-9]*")):
            f = d / key / "curves_full.json"
            if f.exists():
                y = np.asarray(json.loads(f.read_text())["consecutive_successes"], float)
                if len(y) == T:
                    out[name][int(d.name[4:])] = y
    return out


if __name__ == "__main__":
    llm, human = load_all()
    print(len(llm), "LLM curves,", len(human), "human")
    for k, y in {**llm, **human}.items():
        cm = np.maximum.accumulate(y)
        print(f"{k:8s} peak={y.max():6.2f} final={y[-1]:6.2f} best@500={cm[499]:6.2f}")
