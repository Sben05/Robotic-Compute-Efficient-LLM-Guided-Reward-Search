"""A1 false-pruning rates from the live trials (audit) and A10 slow-starter table over the pooled curves.

A1 (actual runs): in every live round all four candidates trained 500 its, one was kept, the rest stopped; the stopped ones were
later resumed to 3,000 (audit). A stop is a FALSE PRUNE if the stopped candidate's 3,000-it peak beats the kept one's.
A10 (actual curves): which rewards look weak at 500 its but end strong.
Writes results/pruning/a1_false_pruning.json, a10_slow_starters.json and prints Markdown tables.
"""
import json

import numpy as np

from data import LIVE, ROOT, ROUNDS, load_all

OUT = ROOT / "results/pruning"
OUT.mkdir(parents=True, exist_ok=True)
llm, human = load_all()
cm = {k: np.maximum.accumulate(v) for k, v in {**llm, **human}.items()}


def a1():
    rows, stopped, false = [], 0, 0
    for rname, members in ROUNDS.items():
        if not rname.startswith("live"):
            continue
        kept = max(members, key=lambda m: cm[m][499])
        for m in members:
            if m == kept:
                continue
            stopped += 1
            fp = cm[m][-1] > cm[kept][-1]
            false += fp
            rows.append({"round": rname, "stopped": m, "at500": cm[m][499], "peak3000": cm[m][-1], "kept": kept,
                         "kept_at500": cm[kept][499], "kept_peak3000": cm[kept][-1], "false_prune": bool(fp),
                         "margin_at500": cm[kept][499] - cm[m][499]})
    rounds_with_fp = sorted({r["round"] for r in rows if r["false_prune"]})
    out = {"label": "actual runs; 3,000-it values are resumed audit runs (unvalidated until the human resume control passes)",
           "stopped": stopped, "false_prunes": int(false), "false_prune_rate": false / stopped,
           "rounds": 4, "rounds_with_false_prune": len(rounds_with_fp), "rows": rows,
           "false_prune_margins_at500": [r["margin_at500"] for r in rows if r["false_prune"]]}
    json.dump(out, open(OUT / "a1_false_pruning.json", "w"), indent=1, default=float)
    print(f"A1: {false}/{stopped} stopped candidates were false prunes ({false/stopped:.0%}); rounds affected {len(rounds_with_fp)}/4")
    print("| round | stopped | best@500 | peak@3000 | kept | kept best@500 | kept peak@3000 | false prune |\n|---|---|---:|---:|---|---:|---:|:-:|")
    for r in rows:
        print(f"| {r['round']} | {r['stopped']} | {r['at500']:.2f} | {r['peak3000']:.2f} | {r['kept']} | {r['kept_at500']:.2f} | "
              f"{r['kept_peak3000']:.2f} | {'yes' if r['false_prune'] else ''} |")


def a10():
    names = list(llm)
    at500 = np.array([cm[k][499] for k in names]); fin = np.array([cm[k][-1] for k in names])
    r500 = (-at500).argsort().argsort() + 1; r3k = (-fin).argsort().argsort() + 1
    rows = []
    for j, k in enumerate(names):
        t90 = int(np.argmax(cm[k] >= 0.9 * cm[k][-1])) + 1 if cm[k][-1] > 1 else None
        rows.append({"reward": k, "source": "live (resumed)" if k in LIVE else "straight 3,000", "best@500": at500[j],
                     "peak@3000": fin[j], "gain_ratio": fin[j] / max(at500[j], 1.0), "rank@500": int(r500[j]), "rank@3000": int(r3k[j]),
                     "iter_to_90pct_of_own_peak": t90})
    n = len(names)
    slow = [r for r in rows if r["rank@500"] > n / 2 and r["rank@3000"] <= n / 4]
    rho = np.corrcoef(r500, r3k)[0, 1]
    out = {"label": "actual curves (retrospective reading)", "n": n, "spearman_rank500_vs_rank3000": rho,
           "slow_starters_definition": "rank@500 in the bottom half and rank@3000 in the top quarter", "slow_starters": slow, "rows": rows}
    json.dump(out, open(OUT / "a10_slow_starters.json", "w"), indent=1, default=float)
    print(f"\nA10: n={n}, Spearman(rank@500, rank@3000) = {rho:.2f}; slow starters: {[r['reward'] for r in slow]}")
    print("| reward | best@500 | peak@3000 | x gain | rank@500 | rank@3000 | its to 90% of own peak |\n|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: -r["gain_ratio"])[:10]:
        print(f"| {r['reward']} | {r['best@500']:.2f} | {r['peak@3000']:.2f} | {r['gain_ratio']:.2f} | {r['rank@500']} | {r['rank@3000']} | "
              f"{r['iter_to_90pct_of_own_peak']} |")


if __name__ == "__main__":
    a1()
    a10()
