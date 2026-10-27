"""Post-hoc ORSO baseline (added after the held-out results were known; tuned on TRAIN seeds 0-3 only, same criterion as
every frozen rule: min compute s.t. mean retained >= 0.98, else max retained). Scored on held-out seeds 4,5,7,8,9 and on the
clean seeds 7-9 next to the frozen VoI, PID and fixed@500.  Usage: python orso_baseline.py"""
import json

import numpy as np

import heldout as H
from configs import build

rows, human = H.load_curves()
train = [r for r in rows if r["seed"] not in (4, 5, 7, 8, 9)]
cfg = {lab: (fam, pol, kw) for fam, lab, pol, kw in build()}
Wtr = H.worlds(train, human, sorted({r["seed"] for r in train}), 250, np.random.default_rng(0), train)
chosen = {}
for algo in ("d3rb", "exp3"):
    fam = f"ORSO ({algo}, post-hoc baseline)"
    cands = []
    for lab, (f, pol, kw) in cfg.items():
        if f == fam:
            s = H.summary(H.run(pol, kw, Wtr))
            cands.append((lab, s))
            print(f"TRAIN {lab:24s} compute {s['compute']:.3f} retained {s['retained']:.3f} hit {s['hit']:.2f}")
    ok = [c for c in cands if c[1]["retained"] >= 0.98]
    chosen[algo] = (min(ok, key=lambda c: c[1]["compute"]) if ok else max(cands, key=lambda c: c[1]["retained"]))[0]
print("chosen on TRAIN:", chosen)
out = {"label": "post-hoc baseline, tuned on TRAIN only; retrospective held-out", "chosen": chosen, "test": {}}
for name, seeds in (("seeds 4,5,7,8,9", (4, 5, 7, 8, 9)), ("clean seeds 7-9", (7, 8, 9))):
    test = [r for r in rows if r["seed"] in seeds]
    W = H.worlds(test, human, seeds, 10 ** 6, np.random.default_rng(1), train)
    out["test"][name] = {}
    for lab in [chosen["d3rb"], chosen["exp3"], "VoI lam=1.0 t0=250", "PID ki=1 kd=1 th=0.4", "fixed@500 keep-1"]:
        f, pol, kw = cfg[lab]
        recs = H.run(pol, kw, W)
        s = H.summary(recs)
        s["retained_ci"] = H.boot(recs, "retained")
        out["test"][name][lab] = s
        print(f"{name:16s} {lab:24s} compute {s['compute']:.3f} retained {s['retained']:.3f} {np.round(s['retained_ci'], 3)} hit {s['hit']:.2f}")
json.dump(out, open(H.ROOT / "results/pruning/orso_baseline.json", "w"), indent=1, default=float)
