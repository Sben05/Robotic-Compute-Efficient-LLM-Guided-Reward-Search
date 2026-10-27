"""Experiment A: simulate allocation policies on the 11 finished LLM reward curves (+2 human seeds as library only).
Usage: python run_alloc.py [N_ARMS] [SIGMA_MULT] [REPS]"""
import json
import sys
import time

import numpy as np

from configs import build
from data import load_all
from sim_core import calibrate_warp, evaluate, make_worlds

N_ARMS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SIGMA_MULT = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
LIB_MODE = sys.argv[4] if len(sys.argv) > 4 else "rest"
POOL = int(sys.argv[5]) if len(sys.argv) > 5 else 0      # >0: that many random with-replacement pools (for N > 11)
MAX_SUBSETS = 200

llm, human = load_all()
s_star, err = calibrate_warp(human["human0"], human["human1"])
SIGMA = abs(np.log(s_star)) / np.sqrt(2) * SIGMA_MULT
print(f"human seed pair: time-scale s*={s_star:.3f} (mean abs err {err:.3f}); sigma_s={SIGMA:.3f} (x{SIGMA_MULT})")


def worlds():
    rng = np.random.default_rng(123)
    return list(make_worlds(llm, human, SIGMA, N_ARMS, REPS if SIGMA > 0 else 1, rng, MAX_SUBSETS, LIB_MODE,
                            POOL or None))


W = worlds()
print(f"{len(W)} worlds, {N_ARMS} arms each")

CFG = build()

res = []
t_start = time.time()
for fam, label, pol, kw in CFG:
    r = evaluate(pol, W, seed=1, **kw)
    r.update(family=fam, label=label)
    res.append(r)
    print(f"{fam:26s} {label:26s} compute {r['compute']:.2f} retained {r['retained']:.3f} hit {r['hit']:.2f} ovh {r['overhead_its_per_arm']:.0f} "
          f"p05 {r['ret_p05']:.3f} <95%: {r['frac_below_95']:.2f}", flush=True)
print(f"{time.time() - t_start:.0f}s")
json.dump({"n_arms": N_ARMS, "sigma_s": SIGMA, "s_star": s_star, "n_worlds": len(W), "results": res},
          open(f"../results/controller_feasibility/alloc_N{N_ARMS}_sig{SIGMA_MULT}_{LIB_MODE}{'_pool' if POOL else ''}.json", "w"), indent=1)
