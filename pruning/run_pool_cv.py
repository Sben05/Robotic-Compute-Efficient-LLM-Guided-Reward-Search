"""Pruning comparison on the enlarged pool (retrospective simulation on actual curves).

1) Held-out CV: each split puts half the 27 LLM curves in TRAIN and half in TEST. Every policy family picks its parameters on
   worlds built from TRAIN curves only (min compute subject to mean retained >= THR, else max retained) and is scored on worlds
   built from TEST curves, whose library holds only TRAIN curves + the 2 human runs. Fixed checkpoints are scored untuned.
2) Leave-one-round-out on the 6 real search rounds: tune on worlds from all curves outside the round, then run once on the
   actual candidate set of that round (no warp).
Worlds: N-subsets of curves, each curve time-warped by exp(sigma*z) (seed-noise model calibrated on the human seed pair).

  python run_pool_cv.py [N=4] [NSPLIT=20] [SIGMA_MULT=1.0] [THR=0.98]
"""
import json
import multiprocessing as mp
import sys
import time

import numpy as np

import policies as P
from configs import build
from data import ROOT, ROUNDS, load_all
from sim_core import Library, World, calibrate_warp, evaluate, make_worlds, run_policy

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
NSPLIT = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SIGMA_MULT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
THR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.98
OUT = ROOT / "results/pruning"
OUT.mkdir(parents=True, exist_ok=True)
llm, human = load_all()
s_star, _ = calibrate_warp(human["human0"], human["human1"])
SIGMA = abs(np.log(s_star)) / np.sqrt(2) * SIGMA_MULT
CFG = build()
FAMS = sorted({c[0] for c in CFG if c[0] != "baseline"})
REF = ["fixed@100 keep-1", "fixed@250 keep-1", "fixed@500 keep-1", "fixed@750 keep-1", "fixed@1000 keep-1",
       "fixed@1500 keep-1", "fixed@500 keep-2", "SH t0=250 eta=2", "random pick, train 1"]
HUM_CM = [np.maximum.accumulate(h) for h in human.values()]


def tune(train_worlds):
    res = {}
    for fam, label, pol, kw in CFG:
        res[label] = (fam, pol, kw, evaluate(pol, train_worlds, seed=1, **kw))
    chosen = {}
    for fam in FAMS:
        rows = [(lab, v) for lab, v in res.items() if v[0] == fam]
        ok = [(lab, v) for lab, v in rows if v[3]["retained"] >= THR]
        lab, v = min(ok, key=lambda x: x[1][3]["compute"]) if ok else max(rows, key=lambda x: x[1][3]["retained"])
        chosen[fam] = (lab, v)
    return res, chosen


def one_split(sp):
    rng = np.random.default_rng(1000 + sp)
    names = list(llm)
    perm = rng.permutation(len(names))
    half = len(names) // 2
    train = {names[k]: llm[names[k]] for k in perm[:half]}
    test = {names[k]: llm[names[k]] for k in perm[half:]}
    Wtr = list(make_worlds(train, human, SIGMA, N, 2, rng, max_subsets=120))
    lib = Library([np.maximum.accumulate(v) for v in train.values()] + HUM_CM, HUM_CM)
    Wte = [(w, lib, ids) for w, _, ids in make_worlds(test, human, SIGMA, N, 2, rng, max_subsets=150, lib_mode="human")]
    res, chosen = tune(Wtr)
    rec = {}
    for fam, (lab, v) in chosen.items():
        rec[fam] = {"chosen": lab, "train_retained": v[3]["retained"], "train_compute": v[3]["compute"],
                    **evaluate(v[1], Wte, seed=2, **v[2])}
    for lab in REF:
        f, pol, kw, _ = res[lab]
        rec[lab] = {"chosen": lab, **evaluate(pol, Wte, seed=2, **kw)}
    return rec


def one_round(rname):
    """Leave-one-round-out: tune on the other curves, then run on the real round's candidates (no warp)."""
    rng = np.random.default_rng(7)
    members = ROUNDS[rname]
    rest = {k: v for k, v in llm.items() if k not in members}
    n = len(members)
    Wtr = list(make_worlds(rest, human, SIGMA, n, 2, rng, max_subsets=150))
    res, chosen = tune(Wtr)
    lib = Library([np.maximum.accumulate(v) for v in rest.values()] + HUM_CM, HUM_CM)
    raw = np.stack([llm[m] for m in members])
    out = {"members": members, "truth": {m: float(llm[m].max()) for m in members}}
    for fam, (lab, v) in chosen.items():
        r = run_policy(v[1], World(raw), lib, np.random.default_rng(3), **v[2])
        out[fam] = {"chosen": lab, **r}
    for lab in REF:
        f, pol, kw, _ = res[lab]
        out[lab] = {"chosen": lab, **run_policy(pol, World(raw), lib, np.random.default_rng(3), **kw)}
    return rname, out


def summarize(recs):
    s = {}
    for key in recs[0]:
        c = np.array([r[key]["compute"] for r in recs]); rt = np.array([r[key]["retained"] for r in recs])
        h = np.array([r[key]["hit"] for r in recs]); p5 = np.array([r[key]["ret_p05"] for r in recs])
        s[key] = {"compute": c.mean(), "compute_sd": c.std(), "saved": 1 - c.mean(), "retained": rt.mean(), "retained_sd": rt.std(),
                  "hit": h.mean(), "ret_p05": p5.mean(),
                  "chosen": sorted({r[key]["chosen"] for r in recs}, key=lambda x: -sum(r[key]["chosen"] == x for r in recs))[:3]}
    return s


if __name__ == "__main__":
    t0 = time.time()
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        recs = pool.map(one_split, range(NSPLIT))
        rounds = dict(pool.map(one_round, list(ROUNDS)))
    summ = summarize(recs)
    print(f"held-out CV: {NSPLIT} splits, N={N}, sigma={SIGMA:.3f}, THR={THR}, pool={len(llm)} curves ({time.time()-t0:.0f}s)")
    for k, v in sorted(summ.items(), key=lambda kv: kv[1]["compute"]):
        print(f"  {k:30s} compute {v['compute']:.3f}±{v['compute_sd']:.3f} retained {v['retained']:.3f}±{v['retained_sd']:.3f} "
              f"hit {v['hit']:.2f} p05 {v['ret_p05']:.3f}  [{', '.join(v['chosen'])}]")
    print("leave-one-round-out on the 6 real rounds (retained / compute):")
    keys = FAMS + REF
    for k in keys:
        rr = [rounds[r][k] for r in rounds]
        print(f"  {k:30s} retained {np.mean([x['retained'] for x in rr]):.3f} hit {np.mean([x['hit'] for x in rr]):.2f} "
              f"compute {np.mean([x['cost'] for x in rr]):.3f}  per-round {[round(x['retained'], 2) for x in rr]}")
    json.dump({"label": "retrospective simulation on actual curves", "N": N, "n_splits": NSPLIT, "sigma": SIGMA, "thr": THR,
               "pool": list(llm), "summary": summ, "splits": recs, "rounds": rounds},
              open(OUT / f"cv_N{N}_sig{SIGMA_MULT}_thr{THR}.json", "w"), indent=1, default=float)
