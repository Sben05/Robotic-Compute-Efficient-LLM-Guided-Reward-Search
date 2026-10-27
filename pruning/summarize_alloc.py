"""Tables + Pareto figure from results/controller_feasibility/alloc_*.json."""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results/controller_feasibility"
FAM_ORDER = ["fixed checkpoint", "successive halving", "hyperband", "human-reference threshold", "ratio-bound racing",
             "PID (P)", "PID (PD)", "PID (PI)", "PID (PID)", "Bayesian pruning", "MPC (receding horizon)",
             "Thompson BAI (TTTS)"]


def best_for(rows, fam, thr):
    ok = [r for r in rows if r["family"] == fam and r["retained"] >= thr]
    return min(ok, key=lambda r: r["compute"]) if ok else None


def table(path, thr):
    d = json.load(open(path))
    rows = d["results"]
    base = {r["label"]: r for r in rows}
    lines = [f"{path.name}: N={d['n_arms']} sigma_s={d['sigma_s']:.2f} worlds={d['n_worlds']}  (min compute s.t. mean retained >= {thr})"]
    for k in ("random pick, train 1", "fixed@500 keep-1", "fixed@500 keep-2", "SH t0=250 eta=2"):
        r = base[k]
        lines.append(f"   ref {k:24s} compute {r['compute']:.3f} retained {r['retained']:.3f} hit {r['hit']:.2f} p05 {r['ret_p05']:.3f} ovh {r['overhead_its_per_arm']:.0f}")
    for fam in FAM_ORDER:
        r = best_for(rows, fam, thr)
        lines.append(f"   {fam:26s} " + (f"{r['label']:28s} compute {r['compute']:.3f} retained {r['retained']:.3f} hit {r['hit']:.2f} "
                                         f"p05 {r['ret_p05']:.3f} ovh {r['overhead_its_per_arm']:.0f}" if r else "-- none reaches threshold"))
    return "\n".join(lines)


def pareto(ax, rows, ykey, title):
    colors = plt.cm.tab20.colors
    for k, fam in enumerate(FAM_ORDER):
        pts = sorted([(r["compute"], r[ykey]) for r in rows if r["family"] == fam])
        if not pts:
            continue
        front, best = [], -1
        for x, y in pts:
            if y > best:
                front.append((x, y)); best = y
        ax.plot(*zip(*front), "-o", ms=3, lw=1.2, color=colors[k], label=fam)
    ax.set_xlabel("PPO compute / train-everything compute"); ax.set_ylabel(ykey); ax.set_title(title, fontsize=9)
    ax.grid(alpha=.3)


if __name__ == "__main__":
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.98
    for p in sorted(OUT.glob("alloc_*.json")):
        print(table(p, thr)); print()
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (f, t) in zip(axs.flat, [("alloc_N6_sig0.0_rest.json", "N=6, observed curves (no seed noise)"),
                                     ("alloc_N6_sig1.0_rest.json", "N=6, seed noise x1"),
                                     ("alloc_N6_sig2.0_rest.json", "N=6, seed noise x2"),
                                     ("alloc_N4_sig1.0_rest.json", "N=4, seed noise x1"),
                                     ("alloc_N8_sig1.0_rest.json", "N=8, seed noise x1"),
                                     ("alloc_N24_sig1.0_human_pool.json", "N=24 (resampled pools), library = human only")]):
        pareto(ax, json.load(open(OUT / f))["results"], "retained", t + " : best-kept score fraction")
        ax.set_ylim(0.85, 1.005)
    axs[0, 0].legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_alloc_pareto.png", dpi=130)
