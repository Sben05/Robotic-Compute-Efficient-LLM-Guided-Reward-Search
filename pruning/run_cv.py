"""Experiment A2: out-of-sample comparison. For each random split of the 11 LLM curves into 6 train / 5 test, every policy
family picks its parameters on worlds built from the train curves only (min compute subject to mean retained >= 0.98, else
max retained), then is scored on worlds built from the held-out curves, with a library of train curves + human runs.
N=4 candidates per world, seed-noise replicates (sigma_s from the human seed pair)."""
import json
import sys

import numpy as np

import policies as P
from configs import build
from data import load_all
from sim_core import calibrate_warp, evaluate, make_worlds

NSPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SIGMA_MULT = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
THR = 0.98
llm, human = load_all()
s_star, _ = calibrate_warp(human["human0"], human["human1"])
SIGMA = abs(np.log(s_star)) / np.sqrt(2) * SIGMA_MULT
CFG = build()
families = sorted({c[0] for c in CFG if c[0] != "baseline"})
rng = np.random.default_rng(7)
names = list(llm)
records = {f: [] for f in families + ["fixed@500 keep-1", "fixed@750 keep-1", "SH t0=250 eta=2"]}
fixed_cfgs = {c[1]: c for c in CFG}

for sp in range(NSPLIT):
    perm = rng.permutation(len(names))
    train = {names[k]: llm[names[k]] for k in perm[:6]}
    test = {names[k]: llm[names[k]] for k in perm[6:]}
    Wtr = list(make_worlds(train, human, SIGMA, 4, 6, rng))
    # test worlds: library may use only train curves + human (test curves never appear in a library)
    Wte = []
    for w, _, ids in make_worlds(test, human, SIGMA, 4, 10, rng, lib_mode="human"):
        from sim_core import Library
        import numpy as _np
        hum = [_np.maximum.accumulate(h) for h in human.values()]
        lib = Library([_np.maximum.accumulate(v) for v in train.values()] + hum, hum)
        Wte.append((w, lib, ids))
    cache = {}
    for fam, label, pol, kw in CFG:
        if fam == "baseline":
            continue
        cache[label] = (fam, pol, kw, evaluate(pol, Wtr, seed=1, **kw))
    for fam in families:
        rows = [(lab, v) for lab, v in cache.items() if v[0] == fam]
        ok = [(lab, v) for lab, v in rows if v[3]["retained"] >= THR]
        lab, v = min(ok, key=lambda x: x[1][3]["compute"]) if ok else max(rows, key=lambda x: x[1][3]["retained"])
        te = evaluate(v[1], Wte, seed=2, **v[2])
        records[fam].append({"split": sp, "chosen": lab, "train_compute": v[3]["compute"], "train_ret": v[3]["retained"], **te})
    for lab in ("fixed@500 keep-1", "fixed@750 keep-1", "SH t0=250 eta=2"):
        f, pol, kw, _ = cache[lab]
        records[lab].append({"split": sp, "chosen": lab, **evaluate(pol, Wte, seed=2, **kw)})
    print("split", sp, flush=True)

print(f"\nout-of-sample (mean over {NSPLIT} splits; +/- = sd across splits), N=4, sigma_s={SIGMA:.2f}")
summary = {}
for fam, recs in records.items():
    c = np.array([r["compute"] for r in recs]); rt = np.array([r["retained"] for r in recs]); h = np.array([r["hit"] for r in recs])
    p05 = np.array([r["ret_p05"] for r in recs]); ov = np.array([r["overhead_its_per_arm"] for r in recs])
    tc = np.array([r.get("train_compute", np.nan) for r in recs]); tr = np.array([r.get("train_ret", np.nan) for r in recs])
    summary[fam] = dict(compute=c.mean(), compute_sd=c.std(), retained=rt.mean(), retained_sd=rt.std(), hit=h.mean(), p05=p05.mean(),
                        overhead=ov.mean(), train_compute=float(np.nanmean(tc)), train_retained=float(np.nanmean(tr)))
    print(f"{fam:26s} test compute {c.mean():.3f}±{c.std():.3f}  retained {rt.mean():.3f}±{rt.std():.3f}  hit {h.mean():.2f}  "
          f"p05 {p05.mean():.3f}  overhead {ov.mean():.0f} its/arm   (in-sample train: compute {np.nanmean(tc):.3f} ret {np.nanmean(tr):.3f})")
json.dump({"n_splits": NSPLIT, "sigma_s": SIGMA, "summary": summary, "records": records},
          open(f"../results/controller_feasibility/cv_sig{SIGMA_MULT}.json", "w"), indent=1, default=float)
