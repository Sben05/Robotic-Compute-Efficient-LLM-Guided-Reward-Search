"""Matched-cost gap between family frontiers (upper envelope over each family's configs), per c_gen and noise level.
Optimistic for every family (config picked on the same worlds); the *difference* between families is what to read."""
import json, sys
import numpy as np
from common import OUT

def env(pts, grid):
    pts = sorted(pts, key=lambda r: r["cost"])
    xs, ys, top = [], [], -1
    for r in pts:
        top = max(top, r["score"]); xs.append(r["cost"]); ys.append(top)
    return np.interp(grid, xs, ys, left=np.nan, right=ys[-1])

out = {}
for sig in ("0.0", "0.35", "0.7"):
    rows = json.loads((OUT / f"joint_alloc_sig{sig}.json").read_text())
    print(f"\nsigma={sig}: mean score gap over matched budgets 4k-12k PPO-it equivalents (family minus fixed-N frontier)")
    for cg in (0, 40, 250, 1000):
        grid = np.linspace(4000 + 0 * cg, 12000, 41)
        f = {fam: env([r for r in rows if r["cg"] == cg and r["family"] == fam], grid) for fam in ("fixed", "target", "joint")}
        ok = ~np.isnan(f["fixed"]) & ~np.isnan(f["target"]) & ~np.isnan(f["joint"])
        g = {k: float(np.mean(f[k][ok] - f["fixed"][ok])) for k in ("target", "joint")}
        mx = {k: float(np.max(f[k][ok] - f["fixed"][ok])) for k in ("target", "joint")}
        out[f"{sig}_{cg}"] = dict(gap=g, max_gap=mx, fixed_mean=float(np.mean(f["fixed"][ok])), n_budgets=int(ok.sum()))
        print(f"  c_gen={cg:5d}: target {g['target']:+.2f} (max {mx['target']:+.2f})  joint {g['joint']:+.2f} (max {mx['joint']:+.2f})   fixed level {np.mean(f['fixed'][ok]):.2f}  [{ok.sum()} budgets]")
(OUT / "frontier_gap.json").write_text(json.dumps(out, indent=1))
