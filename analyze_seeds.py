"""Experiment F analysis: are reward quality and the 500-iteration selection reproducible across seeds?

Inputs (actual runs): seed 0 = the original straight 3,000-it runs (exp_b, exp_d, human exp_a) or the live audit runs (exp_e,
500 + resume); seeds >= 1 = exp_f_seeds/seed{S}/<reward> (500 + resume to 3,000, same live protocol).
Outputs results/exp_f_seeds/analysis.json, fig_seeds.png.

Resume control: the human reward trains 500 + 2,500 (resumed) in every exp_f seed. Seed 1 is compared with the straight
3,000-it human seed 1 run (exp_a_human_3000_seed1, same seed) and all seeds with the straight seed-0/1 band. PASS = no reset
(min raw score over iterations 501-520 stays above 20% of the score at 500) and peak within 1 point of the straight runs' range.
Caveat: the reward set was chosen for being strongest on seed 0, so seed-0 values are biased upward (winner's curse).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "pruning"))
from data import ROOT, SEED_KEYS, load_seeds  # noqa: E402

OUT = ROOT / "results/exp_f_seeds"
STOP = 500


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    S = load_seeds()  # name -> {seed: curve}
    straight_h1 = np.asarray(json.loads((ROOT / "results/exp_a_human_3000_seed1/curves.json").read_text())["consecutive_successes"])
    seeds = sorted({s for v in S.values() for s in v})
    llm = [n for n in S if not n.startswith("human")]
    per = {}
    for n, cur in S.items():
        pk = {s: float(c.max()) for s, c in cur.items()}
        fn = {s: float(c[-1]) for s, c in cur.items()}
        new = [pk[s] for s in pk if s > 0]
        per[n] = {"peak": pk, "final": fn, "at500": {s: float(np.max(c[:STOP])) for s, c in cur.items()},
                  "mean_peak_all": float(np.mean(list(pk.values()))), "sd_peak_all": float(np.std(list(pk.values()), ddof=1)) if len(pk) > 1 else None,
                  "mean_peak_new_seeds": float(np.mean(new)) if new else None,
                  "sd_peak_new_seeds": float(np.std(new, ddof=1)) if len(new) > 1 else None,
                  "seed0_minus_new_mean": pk[0] - float(np.mean(new)) if new else None, "n_seeds": len(pk)}
    # per-seed selection
    sel = {}
    for s in seeds:
        have = [n for n in llm if s in S[n]]
        if len(have) < 2:
            continue
        at500 = {n: per[n]["at500"][s] for n in have}
        peak = {n: per[n]["peak"][s] for n in have}
        kept = max(at500, key=at500.get)
        best = max(peak, key=peak.get)
        scr = OUT / f"seed{s}/screen.json"
        sel[s] = {"kept_by_500": kept, "best_at_3000": best, "pick_correct": kept == best, "retained": peak[kept] / peak[best],
                  "kept_peak": peak[kept], "human_peak": per["human0"]["peak"].get(s),
                  "kept_vs_human": peak[kept] / per["human0"]["peak"][s] if s in per["human0"]["peak"] else None,
                  "best_llm_vs_human": peak[best] / per["human0"]["peak"][s] if s in per["human0"]["peak"] else None,
                  "spearman_500_vs_3000": spearman([at500[n] for n in have], [peak[n] for n in have]),
                  "live_screen_json_kept": json.loads(scr.read_text())["kept"] if scr.exists() else "seed 0: n/a (selection recomputed)"}
    # rank stability across seeds (LLM rewards present in every seed)
    common = [n for n in llm if all(s in S[n] for s in seeds)]
    pairs = {}
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            pairs[f"{a}-{b}"] = spearman([per[n]["peak"][a] for n in common], [per[n]["peak"][b] for n in common]) if len(common) > 2 else None
    winners = {s: max(common, key=lambda n: per[n]["peak"][s]) for s in seeds} if common else {}
    # resume control
    rc = {}
    for s, c in S["human0"].items():
        if s == 0:
            continue
        dip = float(np.min(c[STOP:STOP + 20])); at = float(c[STOP - 1])
        rc[s] = {"score_at_500": at, "min_501_520": dip, "reset_to_zero": dip < 0.2 * at, "peak": float(c.max()), "final": float(c[-1])}
    band = [10.477, 10.334]
    ok_peaks = all(min(band) - 1.0 <= v["peak"] <= max(band) + 1.0 for v in rc.values())
    ctrl = {"runs": rc, "straight_human_peaks_seed0_seed1": band,
            "same_seed_comparison_seed1": {"split_peak": rc.get(1, {}).get("peak"), "straight_peak": float(straight_h1.max()),
                                           "diff": rc[1]["peak"] - float(straight_h1.max()) if 1 in rc else None},
            "pass": bool(rc) and ok_peaks and not any(v["reset_to_zero"] for v in rc.values())}
    human_new = [per["human0"]["peak"][s] for s in per["human0"]["peak"] if s > 0]
    out = {"label": "actual runs (seeds >= 1 are 500 + resumed runs)", "seeds": seeds, "rewards": per, "selection": sel,
           "rank_spearman_between_seeds": pairs, "per_seed_winner": winners, "resume_control": ctrl,
           "mean_kept_retained": float(np.mean([v["retained"] for v in sel.values()])) if sel else None,
           "pick_correct_rate": float(np.mean([v["pick_correct"] for v in sel.values()])) if sel else None,
           "caveat": "reward set chosen on seed-0 peaks: seed-0 values are winner's-curse inflated; judge reproducibility on seeds >= 1",
           "human_mean_peak_new_seeds": float(np.mean(human_new)) if human_new else None}
    json.dump(out, open(OUT / "analysis.json", "w"), indent=1, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = sorted(S, key=lambda n: -(per[n]["mean_peak_new_seeds"] or per[n]["peak"][0]))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for x, n in enumerate(order):
        pk = per[n]["peak"]
        ax.scatter([x] * (len(pk) - 1), [pk[s] for s in pk if s > 0], color="#2a78d6", s=26, zorder=3, label="seeds 1+" if x == 0 else None)
        ax.scatter([x], [pk[0]], color="#eb6834", marker="D", s=30, zorder=3, label="seed 0 (selection seed)" if x == 0 else None)
        if per[n]["mean_peak_new_seeds"] is not None:
            ax.hlines(per[n]["mean_peak_new_seeds"], x - 0.3, x + 0.3, color="#0b0b0b", lw=1.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([("human" if n.startswith("human") else n) for n in order], rotation=30, fontsize=8)
    ax.set(ylabel="Peak task score (3,000 its)", title="Reward quality across seeds (bar = mean of seeds 1+)")
    ax.grid(alpha=0.25, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_seeds.png", dpi=160)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rewards",)}, indent=1, default=float))
    for n in order:
        r = per[n]
        print(f"{n:8s} seed0 {r['peak'][0]:.2f}  new-seed mean {r['mean_peak_new_seeds']}  sd {r['sd_peak_new_seeds']}  n={r['n_seeds']}")


if __name__ == "__main__":
    main()
