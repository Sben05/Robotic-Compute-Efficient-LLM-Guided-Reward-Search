"""Novelty probe 2 (exploratory, offline): spend screening compute on longer training or on more seeds?

Eureka and ORSO select the winning reward from ONE training seed. Our data show large seed noise (the seed-0 winner r2c0
falls from 10.93 to ~9.2 on other seeds). Question: at equal screening compute, which picks the reward that is truly best
(best mean peak over seeds it was NOT screened on)?

Data: the 7 strong LLM rewards with 9 seeds each (seed 0 + exp_f seeds 1-5, 7-9). Candidate sets: all 35 four-reward subsets.
For each set and 300 random seed draws: screening seeds are drawn per policy; TRUE quality of a reward = mean 3,000-it peak
over the seeds not used for screening by any policy in that draw (5 seeds held out). Selection quality = true(chosen) / true(best).
Policies: k seeds x t iterations each, score = mean over the k seeds of the best-so-far at t; compute = 4 * k * t (screening only;
deploying the winner costs the same for every policy).  Usage: python seed_allocation.py
"""
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
KEYS = {"A_round2c0_10.93": "exp_b_minimax/round2/cand0", "B_round2c1_10.75": "exp_b_minimax/round2/cand1",
        "C_pool_c3_10.73": "exp_d_pool_minimax/round1/cand3", "D_live2_r2c2_10.43": "exp_e_live/trial2/round2/cand2",
        "E_live2_r1c3_9.62": "exp_e_live/trial2/round1/cand3", "F_live2_r2c3_9.61_winner": "exp_e_live/trial2/round2/cand3",
        "G_live1_r2c0_9.29_winner": "exp_e_live/trial1/round2/cand0"}
SEEDS = [0, 1, 2, 3, 4, 5, 7, 8, 9]


def load():
    C = {}
    for k, src in KEYS.items():
        C[k] = {}
        f0 = R / src / ("curves.json" if src.startswith(("exp_b", "exp_d")) else "curves_full.json")
        C[k][0] = np.maximum.accumulate(np.asarray(json.loads(f0.read_text())["consecutive_successes"], float))
        for s in SEEDS[1:]:
            f = R / f"exp_f_seeds/seed{s}/{k}/curves_full.json"
            C[k][s] = np.maximum.accumulate(np.asarray(json.loads(f.read_text())["consecutive_successes"], float))
    return C


def main():
    C = load()
    names = list(C)
    POL = [(1, 3000), (1, 500), (1, 1000), (2, 500), (3, 500), (4, 500), (2, 1000), (1, 1500), (2, 1500), (4, 750), (3, 1000), (2, 250), (4, 250)]
    rng = np.random.default_rng(0)
    res = {p: [] for p in POL}
    for sub in itertools.combinations(names, 4):
        for _ in range(300):
            perm = list(rng.permutation(SEEDS))
            screen, held = perm[:4], perm[4:]
            true = {n: np.mean([C[n][s][-1] for s in held]) for n in sub}
            best = max(true.values())
            for k, t in POL:
                ss = screen[:k]
                score = {n: np.mean([C[n][s][t - 1] for s in ss]) for n in sub}
                pick = max(score, key=score.get)
                res[(k, t)].append((true[pick] / best, float(true[pick] == best)))
    out = {}
    print(f"{len(names)} rewards x {len(SEEDS)} seeds; 35 sets x 300 seed draws; truth = mean peak on 5 held-out seeds")
    print(f"{'policy':22s} {'screen compute':>15s} {'retained':>9s} {'winner':>7s}")
    for (k, t), v in sorted(res.items(), key=lambda kv: 4 * kv[0][0] * kv[0][1]):
        a = np.array(v)
        out[f"{k} seed(s) x {t} its"] = {"screen_iterations": 4 * k * t, "retained": float(a[:, 0].mean()), "winner": float(a[:, 1].mean())}
        print(f"{k} seed(s) x {t:4d} its {4 * k * t:15d} {a[:, 0].mean():9.4f} {a[:, 1].mean():7.3f}")
    (R / "pruning/seed_allocation.json").write_text(json.dumps({"label": "exploratory offline (overlapping sets; same 7 rewards)", "policies": out}, indent=1))


if __name__ == "__main__":
    main()
