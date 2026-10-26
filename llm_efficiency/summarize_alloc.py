"""Frontiers and figure for joint_alloc_sim output."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT

SIG = sys.argv[1] if len(sys.argv) > 1 else "0.35"
rows = json.loads((OUT / f"joint_alloc_sig{SIG}.json").read_text())
CG = sorted({r["cg"] for r in rows})
FAM = ["fixed", "target", "joint"]
LAB = {"fixed": "fixed N, screen, finish best", "target": "adaptive N: stop at human-relative target", "joint": "joint EI-per-cost (generate or train)"}
COL = {"fixed": "#4c78a8", "target": "#59a14f", "joint": "#e15759"}


def best_under(pts, budget):
    ok = [p for p in pts if p["cost"] <= budget]
    return max(ok, key=lambda p: p["score"]) if ok else None


summary = {}
fig, axs = plt.subplots(1, len(CG), figsize=(4.2 * len(CG), 3.6), sharey=True)
for ax, cg in zip(axs, CG):
    print(f"\n=== c_gen = {cg} PPO-iteration equivalents per LLM call (sigma={SIG}) ===")
    base = next(r for r in rows if r["cg"] == cg and r["family"] == "fixed" and r["cfg"] == {"N": 4, "s": 500})
    print(f"thesis default fixed(N=4,s=500): score {base['score']:.2f}  cost {base['cost']:.0f}  (its {base['its']:.0f}, calls {base['calls']:.1f})")
    summary[cg] = {"default": base, "budgets": {}}
    budgets = [3000, 4000, 5000, 6000, 8000, 10000, 14000, 20000]
    print(f"{'budget':>7s} " + " ".join(f"{f:>22s}" for f in FAM))
    for B in budgets:
        cells = []
        for f in FAM:
            pts = [r for r in rows if r["cg"] == cg and r["family"] == f]
            b = best_under(pts, B)
            cells.append(f"{b['score']:.2f} ({b['cost']:.0f})" if b else "-")
            summary[cg]["budgets"].setdefault(B, {})[f] = b
        print(f"{B:7d} " + " ".join(f"{c:>22s}" for c in cells))
    for f in FAM:
        pts = sorted([r for r in rows if r["cg"] == cg and r["family"] == f], key=lambda r: r["cost"])
        front, top = [], -1
        for r in pts:
            if r["score"] > top: front.append(r); top = r["score"]
        ax.plot([r["cost"] for r in front], [r["score"] for r in front], "-o", ms=3, color=COL[f], label=LAB[f], lw=1.4)
    ax.plot(base["cost"], base["score"], "k*", ms=11, label="thesis default (N=4, 500 its)")
    ax.set_xscale("log"); ax.set_title(f"1 LLM call = {cg} PPO iterations", fontsize=10); ax.set_xlabel("total cost (PPO iterations + calls)")
    ax.grid(alpha=.3)
axs[0].set_ylabel("expected 3000-it peak of selected reward")
axs[-1].legend(fontsize=7, loc="lower right")
fig.suptitle(f"Offline simulation on real curves (9 fresh MiniMax rewards, 84 library/test splits, seed-noise sigma={SIG}); proxies, not PPO evidence", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / f"fig_joint_alloc_sig{SIG}.png", dpi=140)
(OUT / f"joint_alloc_summary_sig{SIG}.json").write_text(json.dumps(summary, indent=1, default=float))
