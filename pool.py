"""Experiment D (exploratory): a 16-sample first-round pool, every member trained for 3,000 iterations.

The pool is Experiment B's 3 round-1 candidates plus 13 new samples from the same upstream
initial prompt, so screening rules can be evaluated retrospectively on more than 3 candidates.
Generation, validation and training reuse eureka_ab.py unchanged; only the output directory and
task-name prefix differ. Several runs can share one GPU ("pack") to cut queueing and GPU-hours.

  python pool.py generate N          new samples into results/exp_d_pool_minimax/round1
  python pool.py pack NAME...        train several candidates concurrently on one GPU
  python pool.py train NAME          (used by pack)
  python pool.py analyze             retrospective screening rules over all sub-pools
"""
import itertools
import json
import os
import subprocess
import sys

import eureka_ab as e

POOL_OUT = "exp_d_pool_" + e.LLM
e.OUT = e.ROOT / "results" / POOL_OUT
e.suffix_for = lambda name: "GPT" + e.ARM["tag"] + "D" + name.replace("round", "R").replace("/cand", "C")
FULL = e.ITERATIONS
POOL_DIRS = ["exp_b_" + e.LLM + "/round1", POOL_OUT + "/round1"]


def load_pool():
    """Fully trained pool members: name -> task-score curve (consecutive_successes per PPO iteration)."""
    pool = {}
    for base in POOL_DIRS:
        for c in sorted((e.ROOT / "results" / base).glob("cand*")):
            f = c / "curves.json"
            if f.exists():
                curve = e.load_json(f).get("consecutive_successes", [])
                if len(curve) == FULL:
                    pool["%s/%s" % (base, c.name)] = curve
    return pool


def peak(curve, k=FULL):
    return max(curve[:k])


def keep(frac, pool):
    return max(1, round(frac * len(pool)))


# Each rule maps (sub-pool, ctx) to (chosen candidate, PPO iterations spent). A choice uses only what
# is known at decision time: curves up to the checkpoint, the blind LLM ranking, or probe scores.
def rule_checkpoint(k, m=1):
    def rule(pool, ctx):
        early = sorted(pool, key=lambda c: -peak(pool[c], k))[:m]
        return max(early, key=lambda c: peak(pool[c])), len(pool) * k + m * (FULL - k)
    return rule


def rule_halving(k1, frac, k2):
    def rule(pool, ctx):
        m = keep(frac, pool)
        stage1 = sorted(pool, key=lambda c: -peak(pool[c], k1))[:m]
        best = max(stage1, key=lambda c: peak(pool[c], k2))
        return best, len(pool) * k1 + m * (k2 - k1) + (FULL - k2)
    return rule


def rule_prerank(key, frac):
    def rule(pool, ctx):
        ranked = sorted(pool, key=lambda c: ctx[key].get(c, len(ctx[key])))[:keep(frac, pool)]
        return max(ranked, key=lambda c: peak(pool[c])), len(ranked) * FULL
    return rule


def rule_race(frac, threshold):
    """LLM top fraction trains in parallel; the first to reach the threshold continues alone."""
    def rule(pool, ctx):
        ranked = sorted(pool, key=lambda c: ctx["llm_rank"].get(c, len(ctx["llm_rank"])))[:keep(frac, pool)]
        hit = {c: next((i + 1 for i, v in enumerate(pool[c]) if v >= threshold), None) for c in ranked}
        reached = [c for c in ranked if hit[c] is not None]
        if not reached:
            return max(ranked, key=lambda c: peak(pool[c])), len(ranked) * FULL
        first = min(reached, key=lambda c: hit[c])
        return first, len(ranked) * hit[first] + (FULL - hit[first])
    return rule


def load_context():
    ctx = {"llm_rank": {}, "probe_rank": {}, "probe_score": {}}
    rk = e.OUT / "llm_ranking/ranking.json"
    if rk.exists():
        ctx["llm_rank"] = {c: i for i, c in enumerate(e.load_json(rk)["consensus"])}
    pr = e.ROOT / ("results/probe/probe_pool_%s.json" % e.LLM)
    if pr.exists():
        for f, v in e.load_json(pr)["candidates"].items():
            name = f.replace("results/", "", 1).rsplit("/", 1)[0]
            ctx["probe_score"][name] = v.get("spearman_vs_progress", -1.0) if not v.get("error") else -1.0
        ctx["probe_rank"] = {c: i for i, c in enumerate(sorted(ctx["probe_score"], key=lambda c: -ctx["probe_score"][c]))}
    return ctx


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pool, ctx = load_pool(), load_context()
    names = sorted(pool)
    out = e.OUT
    human = e.load_json(e.HUMAN / "curves.json")["consecutive_successes"]
    rules = {}
    for k in [100, 250, 500, 1000, 1500]:
        rules["checkpoint@%d, keep 1" % k] = rule_checkpoint(k)
        rules["checkpoint@%d, keep 2" % k] = rule_checkpoint(k, 2)
    rules["halving 100 -> 1/4 -> 500"] = rule_halving(100, 0.25, 500)
    rules["halving 250 -> 1/4 -> 1000"] = rule_halving(250, 0.25, 1000)
    if ctx["llm_rank"]:
        rules["LLM pre-rank, train top 1/4"] = rule_prerank("llm_rank", 0.25)
        rules["LLM pre-rank, train top 1/2"] = rule_prerank("llm_rank", 0.5)
        rules["race: LLM top 1/4, first to human@500"] = rule_race(0.25, human[499])
    if ctx["probe_rank"]:
        rules["probe pre-rank, train top 1/4"] = rule_prerank("probe_rank", 0.25)
        rules["probe pre-rank, train top 1/2"] = rule_prerank("probe_rank", 0.5)
    results = {"pool": {c: {"peak": peak(pool[c]), "final": pool[c][-1], "llm_rank": ctx["llm_rank"].get(c),
                            "probe_score": ctx["probe_score"].get(c)} for c in names},
               "human_peak": max(human), "human_at_500": human[499], "sizes": {}}
    for size in sorted({4, 8, len(names)}):
        if size > len(names) or size < 2:
            continue
        subsets = list(itertools.combinations(names, size))
        best = [max(peak(pool[c]) for c in s) for s in subsets]
        table = {"subsets": len(subsets), "full_iterations": size * FULL,
                 "random pick (expected)": {
                     "retained": sum(sum(peak(pool[c]) for c in s) / size / b for s, b in zip(subsets, best)) / len(subsets),
                     "winner_kept": 1 / size, "iterations": FULL, "saved": 1 - 1 / size}}
        for name, rule in rules.items():
            ret, hits, iters = [], 0, []
            for s, b in zip(subsets, best):
                chosen, spent = rule({c: pool[c] for c in s}, ctx)
                ret.append(peak(pool[chosen]) / b)
                hits += peak(pool[chosen]) == b
                iters.append(spent)
            table[name] = {"retained": sum(ret) / len(ret), "min_retained": min(ret),
                           "winner_kept": hits / len(subsets), "iterations": sum(iters) / len(iters),
                           "saved": 1 - sum(iters) / len(iters) / (size * FULL)}
        results["sizes"][size] = table
    e.save_json(out / "analysis.json", results)

    size = max(results["sizes"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, v in results["sizes"][size].items():
        if isinstance(v, dict):
            color = "#8a8984" if name.startswith("random") else "#2a78d6"
            ax.scatter(100 * v["saved"], 100 * v["retained"], s=36, color=color, zorder=3)
            ax.annotate(name, (100 * v["saved"], 100 * v["retained"]), fontsize=7, xytext=(4, 2),
                        textcoords="offset points", color="#52514e")
    ax.scatter(0, 100, s=36, color="#0b0b0b", zorder=3)
    ax.annotate("train all", (0, 100), fontsize=7, xytext=(4, -10), textcoords="offset points")
    ax.set(xlabel="PPO compute saved vs training all %d candidates (%%)" % size,
           ylabel="Best task score retained (%)", title="Screening rules, %d-candidate pool (retrospective)" % size)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_screening_rules.png", dpi=160)
    print(json.dumps(results, indent=1))


def seeds():
    """Does a seed-0 screening decision transfer to a fresh seed? (exploratory)"""
    pool0 = load_pool()
    pool1 = {}
    for name in pool0:
        f = e.ROOT / "results" / (name.split("/")[0] + "_seed1") / "/".join(name.split("/")[1:]) / "curves.json"
        if f.exists():
            c = e.load_json(f).get("consecutive_successes", [])
            if len(c) == FULL:
                pool1[name] = c
    names = sorted(pool1)
    hf = e.ROOT / "results/exp_a_human_3000_seed1/curves.json"
    human = [max(e.load_json(e.HUMAN / "curves.json")["consecutive_successes"]),
             max(e.load_json(hf)["consecutive_successes"]) if hf.exists() else None]
    p0 = {c: peak(pool0[c]) for c in names}
    p1 = {c: peak(pool1[c]) for c in names}
    ranks = lambda d: {c: i for i, c in enumerate(sorted(d, key=lambda c: -d[c]))}
    r0, r1 = ranks(p0), ranks(p1)
    n = len(names)
    mean = lambda d: sum(d.values()) / n
    cov = sum((p0[c] - mean(p0)) * (p1[c] - mean(p1)) for c in names)
    var = (sum((p0[c] - mean(p0)) ** 2 for c in names) * sum((p1[c] - mean(p1)) ** 2 for c in names)) ** 0.5
    out = {"candidates": n, "human_peak_seed0_seed1": human,
           "peak_pearson_seed0_vs_seed1": cov / var if var else None,
           "per_candidate": {c: {"peak_seed0": p0[c], "peak_seed1": p1[c], "rank0": r0[c], "rank1": r1[c]} for c in names},
           "transfer": {}}
    best1 = max(p1.values())
    for k in [100, 250, 500, 1000, 1500, FULL]:
        pick = max(names, key=lambda c: peak(pool0[c], k))  # chosen with seed-0 curves only, through iteration k
        out["transfer"][k] = {"picked": pick, "seed1_peak_of_pick": p1[pick], "best_seed1_peak": best1,
                              "retained_on_seed1": p1[pick] / best1, "pick_is_seed1_best": p1[pick] == best1}
    e.save_json(e.OUT / "seed_transfer.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_candidate"}, indent=1))


def pack(names):
    env = dict(os.environ, PACKED_RUNS=str(len(names)))
    procs = [subprocess.Popen([sys.executable, "-u", __file__, "train", n], env=env) for n in names]
    codes = [p.wait() for p in procs]
    print("pack exit codes:", dict(zip(names, codes)))
    sys.exit(max(codes))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "generate":
        e.generate(1, int(sys.argv[2]))
    elif cmd == "pack":
        pack(sys.argv[2:])
    elif cmd == "train":
        e.train(sys.argv[2])
    elif cmd == "analyze":
        analyze()
    elif cmd == "seeds":
        seeds()
