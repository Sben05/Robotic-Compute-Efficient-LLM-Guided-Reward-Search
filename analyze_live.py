"""Analysis of Experiment E (live early stopping): results/exp_e_live/trial*/ -> analysis.json, fig_live.png.

Live numbers (what the search itself saw and spent) are kept apart from audit numbers (stopped candidates resumed
to 3,000 iterations afterwards, only to measure what stopping cost).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
E = ROOT / "results/exp_e_live"
STOP, FULL = 500, 3000
HUMAN_PEAK = 10.477  # seed 0, 3,000 iterations (results/exp_a_human_3000)


def load(p):
    return json.loads(Path(p).read_text())


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):  # average ranks for ties
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def trial_rows(tdir):
    rows = []
    for rd in (1, 2):
        d = tdir / f"round{rd}"
        screen = load(d / "screen.json")
        for c in sorted(d.glob("cand*")):
            full = load(c / "result_full.json") if (c / "result_full.json").exists() else {}
            valid = load(c / "generation.json")["validation"]["valid"]
            rows.append({"name": f"round{rd}/{c.name}", "round": rd, "valid": valid,
                         "score_at_stop": screen["scores_at_stop"].get(c.name),
                         "kept_in_round": screen["kept"] == c.name,
                         "peak_3000": full.get("peak_task_score"), "final_3000": full.get("final_task_score")})
    return rows


def main():
    out = {"human_peak_seed0": HUMAN_PEAK, "trials": {}}
    pooled = []
    for tdir in sorted(E.glob("trial[0-9]*")):
        if not (tdir / "winner.json").exists() or not (tdir / "round2/screen.json").exists():
            continue
        rows = trial_rows(tdir)
        winner = load(tdir / "winner.json")["winner"]
        trained = [r for r in rows if r["score_at_stop"] is not None]
        have_full = [r for r in trained if r["peak_3000"] is not None]
        live_iters = STOP * len(trained) + (FULL - STOP)
        full_iters = FULL * len(trained)
        best = max(have_full, key=lambda r: r["peak_3000"]) if have_full else None
        win = next(r for r in rows if r["name"] == winner)
        per_round = {}
        for rd in (1, 2):
            g = [r for r in have_full if r["round"] == rd]
            kept = next(r for r in g if r["kept_in_round"])
            top = max(g, key=lambda r: r["peak_3000"])
            per_round[rd] = {"kept": kept["name"], "kept_peak": kept["peak_3000"], "best": top["name"],
                             "best_peak": top["peak_3000"], "top1_correct": kept["name"] == top["name"],
                             "retained": kept["peak_3000"] / top["peak_3000"]}
        out["trials"][tdir.name] = {
            "candidates": rows, "winner": winner, "winner_peak": win["peak_3000"], "winner_final": win["final_3000"],
            "best_candidate": best["name"], "best_peak": best["peak_3000"],
            "winner_retained_of_best": win["peak_3000"] / best["peak_3000"],
            "winner_vs_human": win["peak_3000"] / HUMAN_PEAK,
            "live_iterations": live_iters, "train_all_iterations": full_iters, "saved_fraction": 1 - live_iters / full_iters,
            "live_env_steps": live_iters * 4096 * 16, "per_round": per_round,
            "spearman_stop_vs_peak": spearman([r["score_at_stop"] for r in have_full], [r["peak_3000"] for r in have_full])}
        pooled += [dict(r, trial=tdir.name) for r in have_full]
    if pooled:
        ok = [r for r in pooled if r["score_at_stop"] > 1]
        t = out["trials"].values()
        out["pooled"] = {
            "candidates": len(pooled), "spearman_all": spearman([r["score_at_stop"] for r in pooled], [r["peak_3000"] for r in pooled]),
            "spearman_excluding_collapsed": spearman([r["score_at_stop"] for r in ok], [r["peak_3000"] for r in ok]),
            "rounds": 2 * len(out["trials"]),
            "top1_correct_rounds": sum(p["top1_correct"] for x in t for p in x["per_round"].values()),
            "mean_retained_per_round": sum(p["retained"] for x in t for p in x["per_round"].values()) / (2 * len(out["trials"])),
            "mean_winner_peak": sum(x["winner_peak"] for x in t) / len(out["trials"]),
            "mean_saved_fraction": sum(x["saved_fraction"] for x in t) / len(out["trials"]),
            "mean_peak_round1_ok": sum(r["peak_3000"] for r in ok if r["round"] == 1) / max(1, sum(r["round"] == 1 for r in ok)),
            "mean_peak_round2_ok": sum(r["peak_3000"] for r in ok if r["round"] == 2) / max(1, sum(r["round"] == 2 for r in ok))}
    (E / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {"trial1": "#2a78d6", "trial2": "#eb6834", "trial3": "#1baf7a"}
    for r in pooled:
        ax.scatter(r["score_at_stop"], r["peak_3000"], s=70 if r["kept_in_round"] else 28, color=colors.get(r["trial"], "#555"),
                   marker="o" if r["round"] == 1 else "s", edgecolor="#0b0b0b" if r["kept_in_round"] else "none", lw=1.4, alpha=0.9)
    ax.axhline(HUMAN_PEAK, color="#0b0b0b", lw=1, ls="--")
    ax.text(0.2, HUMAN_PEAK + 0.1, "human reward (10.48)", fontsize=8)
    for tr, c in colors.items():
        if tr in out["trials"]:
            ax.scatter([], [], color=c, label=tr.replace("trial", "trial "))
    ax.scatter([], [], color="#8a8984", marker="o", label="round 1")
    ax.scatter([], [], color="#8a8984", marker="s", label="round 2")
    ax.scatter([], [], color="none", edgecolor="#0b0b0b", s=70, label="kept at 500")
    ax.set(xlabel="Score at the 500-iteration stop", ylabel="Peak score after resuming to 3,000 (audit)",
           title="Live stopping: what the 500-iteration score tells you")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(E / "fig_live.png", dpi=160)
    print(json.dumps({k: v for k, v in out.items() if k != "trials"}, indent=1))
    for k, v in out["trials"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a not in ("candidates", "per_round")})


if __name__ == "__main__":
    main()
