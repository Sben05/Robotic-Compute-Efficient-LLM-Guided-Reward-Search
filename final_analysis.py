"""Final analysis, exactly as pre-registered in STATS_PLAN.md (+ amendment 1). Writes results/final/*.json and prints a summary.

  python final_analysis.py          (Mac venv: numpy, scipy)

Primary hypotheses H1-H5: one-sided tests, alpha 0.05, Holm correction across the five.
Exploratory items: effect sizes and 95% CIs only. Every output carries a label: actual run / retrospective held-out / simulation.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
OUT = R / "final"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "pruning"))
import heldout as H  # noqa: E402
import policies as P  # noqa: E402
from configs import build  # noqa: E402

RNG = np.random.default_rng(2026)
PRICES = {"minimax": (0.15e-6, 0.60e-6), "gemini": (0.30e-6, 2.50e-6), "qwen": (0.15e-6, 0.60e-6)}


def load(p):
    return json.loads(Path(p).read_text())


def curve(p):
    p = Path(p)
    if not p.exists():
        return None
    y = load(p).get("consecutive_successes")  # absent when the run crashed
    return np.asarray(y, float) if y else None


def peak(p, k=None):
    c = curve(p)
    if c is None or len(c) == 0:
        return None
    return float(np.max(c[:k] if k else c))


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


# ------------------------------------------------------------------ H1
def h1():
    H.TEST_SEEDS = (4, 5, 7, 8, 9)  # STATS_PLAN amendment 1; rules stay frozen
    fr = load(R / "pruning/frozen_rules.json")
    rows, human = H.load_curves()
    train = [r for r in rows if r["seed"] not in H.TEST_SEEDS]
    test = [r for r in rows if r["seed"] in H.TEST_SEEDS]
    W = H.worlds(test, human, H.TEST_SEEDS, 10 ** 6, np.random.default_rng(1), train)
    cfg = {lab: (pol, kw) for fam, lab, pol, kw in build()}
    rule = "VoI lam=1.0 t0=250"  # chosen by best_rule_criterion.md on the interim scores (commit 90e1572)
    rec_a = H.run(*cfg[rule], W)
    rec_f = H.run(*cfg["fixed@500 keep-1"], W)
    comp_a = float(np.mean([r["cost"] for r in rec_a]))
    # equal-compute fixed checkpoint: (N*k + (T-k)) / (N*T) = comp_a  ->  k = (N*T*comp_a - T) / (N-1)
    k_eq = int(round((H.N * 3000 * comp_a - 3000) / (H.N - 1)))
    rec_e = H.run(P.fixed, dict(t_c=k_eq, keep=1), W)
    seeds = [w[3] for w in W]
    d = np.array([a["retained"] - f["retained"] for a, f in zip(rec_a, rec_f)])
    de = np.array([a["retained"] - e["retained"] for a, e in zip(rec_a, rec_e)])

    def boot_p(diff):
        by = {}
        for s, v in zip(seeds, diff):
            by.setdefault(s, []).append(v)
        ss = list(by)
        means = []
        for _ in range(10000):
            pick = RNG.choice(ss, len(ss))
            means.append(np.concatenate([RNG.choice(by[s], len(by[s])) for s in pick]).mean())
        means = np.array(means)
        return float(diff.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], float((means <= 0).mean())
    m, ci, p = boot_p(d)
    me, cie, pe = boot_p(de)
    s = lambda recs: {k: float(np.mean([r[k] for r in recs])) for k in ("cost", "retained", "hit")}
    out = {"label": "retrospective (held-out curves, seeds 4,5,7,8,9; frozen rules)", "n_worlds": len(W), "test_seeds": list(H.TEST_SEEDS),
           "n_test_curves": len(test), "rule": rule, "adaptive": s(rec_a), "fixed500": s(rec_f), f"fixed@{k_eq} (equal compute)": s(rec_e),
           "diff_retained_vs_fixed500": {"mean": m, "ci95": ci, "p_one_sided": p},
           "diff_retained_vs_equal_compute_fixed": {"k": k_eq, "mean": me, "ci95": cie, "p_one_sided": pe},
           "per_seed": {int(sd): {"adaptive": float(np.mean([a["retained"] for a, w in zip(rec_a, W) if w[3] == sd])),
                                  "fixed500": float(np.mean([f["retained"] for f, w in zip(rec_f, W) if w[3] == sd]))} for sd in sorted(set(seeds))},
           "full_table": "results/pruning/heldout_scores.json (every frozen rule on the same held-out seeds)"}
    json.dump(out, open(OUT / "h1.json", "w"), indent=1)
    H.score()  # every frozen rule, untuned, on the final held-out seeds -> results/pruning/heldout_scores.json
    return out, p


# ------------------------------------------------------------------ H2 / H3 data
def ant_minimax_calls():
    """Per-call records for Ant/MiniMax: batch arm B (4 per call) vs independent samples (1 per call)."""
    calls = []
    tp = R / "llm_efficiency/trial_plan"
    for f in sorted(tp.glob("result_B_batch4_*.json")):
        u = load(f)["response"]["usage"]
        i = int(f.stem.rsplit("_", 1)[1])
        valid = sum(load(g)["validation"]["valid"] for g in (R / "exp_h_llm_trial/B_batch4").glob(f"call{i}_f*/generation.json"))
        calls.append({"task": "ant", "arm": "batch4", "tokens": u["completion_tokens"], "prompt": u["prompt_tokens"], "valid": valid})
    indep_dirs = sorted(list((R / "exp_b_minimax/round1").glob("cand*")) + list((R / "exp_d_pool_minimax/round1").glob("cand*")) +
                        list((R / "exp_e_live").glob("trial*/round1/cand*")))
    for c in indep_dirs:
        g = load(c / "generation.json")
        u = g.get("usage") or {}
        if not u:
            continue
        calls.append({"task": "ant", "arm": "indep", "tokens": u.get("completion_tokens", 0), "prompt": u.get("prompt_tokens", 0),
                      "valid": int(g["validation"]["valid"]), "dir": str(c.relative_to(R))})
    return calls


def task_calls(base, task):
    calls = []
    for arm in ("indep", "batch4"):
        for f in sorted((base / task / arm).glob("reply*.json")):
            r = load(f).get("response") or {}
            u = r.get("usage") or {}
            i = int(f.stem[5:])
            valid = sum(load(g)["validation"]["valid"] for g in (base / task / arm).glob(f"call{i}*/generation.json")
                        if g.parent.name == f"call{i}" or g.parent.name.startswith(f"call{i}_"))
            calls.append({"task": task, "arm": arm, "tokens": u.get("completion_tokens", 0), "prompt": u.get("prompt_tokens", 0), "valid": valid})
    return calls


def h2(calls):
    tasks = sorted({c["task"] for c in calls})

    def stat(cs):
        tot = 0.0
        for t in tasks:
            a = [c for c in cs if c["task"] == t]
            ind = [c for c in a if c["arm"] == "indep"]
            bat = [c for c in a if c["arm"] == "batch4"]
            tpv = lambda xs: sum(c["tokens"] for c in xs) / max(1, sum(c["valid"] for c in xs))
            tot += math.log(tpv(ind) / tpv(bat))
        return tot
    obs = stat(calls)
    per_task = {}
    for t in tasks:
        a = [c for c in calls if c["task"] == t]
        tpv = lambda xs: sum(c["tokens"] for c in xs) / max(1, sum(c["valid"] for c in xs))
        ind, bat = [c for c in a if c["arm"] == "indep"], [c for c in a if c["arm"] == "batch4"]
        per_task[t] = {"indep_tokens_per_valid": tpv(ind), "batch4_tokens_per_valid": tpv(bat), "ratio": tpv(ind) / tpv(bat),
                       "indep_calls": len(ind), "batch4_calls": len(bat), "indep_valid": sum(c["valid"] for c in ind),
                       "batch4_valid": sum(c["valid"] for c in bat), "indep_validity": sum(c["valid"] for c in ind) / max(1, len(ind)),
                       "batch4_validity": sum(c["valid"] for c in bat) / max(1, 4 * len(bat))}
    perm = []
    for _ in range(20000):
        cs = []
        for t in tasks:
            a = [c for c in calls if c["task"] == t]
            arms = RNG.permutation([c["arm"] for c in a])
            cs += [dict(c, arm=arm) for c, arm in zip(a, arms)]
        perm.append(stat(cs))
    p_perm = float((np.array(perm) >= obs).mean())
    wins = sum(v["ratio"] > 1 for v in per_task.values())
    p_sign = float(stats.binomtest(wins, len(tasks), 0.5, alternative="greater").pvalue)
    out = {"label": "actual LLM calls", "per_task": per_task, "log_ratio_sum": obs, "p_one_sided_permutation": p_perm,
           "sign_test": {"tasks_cheaper": wins, "of": len(tasks), "p_one_sided": p_sign}}
    json.dump(out, open(OUT / "h2.json", "w"), indent=1)
    return out, p_perm


def h3():
    """Batch vs independent reward quality, normalised by the same task+seed human score. Non-inferiority margin 0.10."""
    vals = []  # (task, arm, normalised score)
    # Ant: 500-it best score, seeds 0 and 1 for arm B; independent round-1 samples at seed 0
    h500 = {0: peak(R / "exp_a_human_3000/curves.json", 500), 1: peak(R / "exp_f_seeds/seed1/human/curves.json", 500)}
    for sd, base in ((0, R / "exp_h_llm_trial/B_batch4"), (1, R / "exp_h_llm_trial_seed1/B_batch4")):
        for c in sorted(base.glob("call*")):
            v = peak(c / "curves.json", 500)
            if v is not None:
                vals.append(("ant", "batch4", v / h500[sd]))
    for c in sorted(list((R / "exp_b_minimax/round1").glob("cand*")) + list((R / "exp_d_pool_minimax/round1").glob("cand*")) +
                    list((R / "exp_e_live").glob("trial*/round1/cand*"))):
        v = peak(c / "curves.json", 500)
        if v is not None:
            vals.append(("ant", "indep", v / h500[0]))
    # other tasks: full-training peak / human peak of the same seed
    G = R / "exp_i_generalize"
    for t in ("cartpole", "ballbalance", "humanoid"):
        hp = {sd: peak(G / t / "human" / f"seed{sd}" / "curves.json") for sd in (0, 1)}
        for arm in ("indep", "batch4"):
            for c in sorted((G / t / arm).glob("call*/seed*")):
                sd = int(c.name[4:])
                v = peak(c / "curves.json")
                if v is not None and hp.get(sd):
                    vals.append((t, arm, v / hp[sd]))
    tasks = sorted({v[0] for v in vals})

    def diff(vs):
        ds = []
        for t in tasks:
            b = [v[2] for v in vs if v[0] == t and v[1] == "batch4"]
            i = [v[2] for v in vs if v[0] == t and v[1] == "indep"]
            if b and i:
                ds.append(np.mean(b) - np.mean(i))
        return float(np.mean(ds))
    obs = diff(vals)
    boots = []
    for _ in range(10000):
        bs = []
        for t in tasks:
            for arm in ("batch4", "indep"):
                g = [v for v in vals if v[0] == t and v[1] == arm]
                if g:
                    bs += [g[k] for k in RNG.integers(len(g), size=len(g))]
        boots.append(diff(bs))
    boots = np.array(boots)
    per_task = {t: {arm: {"n": sum(1 for v in vals if v[0] == t and v[1] == arm),
                          "mean": float(np.mean([v[2] for v in vals if v[0] == t and v[1] == arm] or [np.nan])),
                          "median": float(np.median([v[2] for v in vals if v[0] == t and v[1] == arm] or [np.nan])),
                          "best": float(np.max([v[2] for v in vals if v[0] == t and v[1] == arm] or [np.nan])),
                          "beats_human": int(sum(v[2] > 1 for v in vals if v[0] == t and v[1] == arm))}
                    for arm in ("batch4", "indep")} for t in tasks}
    p = float((boots <= -0.10).mean())
    out = {"label": "actual runs (Ant: 500-it screens; other tasks: full training)", "margin": -0.10, "mean_diff_batch_minus_indep": obs,
           "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "p_one_sided_noninferiority": p, "per_task": per_task}
    json.dump(out, open(OUT / "h3.json", "w"), indent=1)
    return out, p


def h4():
    orig = [peak(R / "exp_b_minimax/round1/cand1/curves.json")] + [peak(R / f"exp_g_ablation/r1c1_orig/seed{s}/curves.json") for s in (1, 2, 3)]
    tau = [peak(R / f"exp_g_ablation/r1c1_tau10/seed{s}/curves.json") for s in (0, 1, 2, 3)]
    orig, tau = [x for x in orig if x is not None], [x for x in tau if x is not None]
    t = stats.ttest_ind(tau, orig, equal_var=False, alternative="greater")
    out = {"label": "actual runs (straight 3,000 its)", "r1c1_orig": orig, "r1c1_tau10": tau, "mean_orig": float(np.mean(orig)),
           "mean_tau10": float(np.mean(tau)), "diff": float(np.mean(tau) - np.mean(orig)), "welch_t": float(t.statistic),
           "p_one_sided": float(t.pvalue), "preregistered_reading": "tau10 mean >= 9.0 -> edit explains most of the gain",
           "reading": "explains most of the gain" if np.mean(tau) >= 9.0 else ("does not" if np.mean(tau) <= 8.0 else "partial"),
           "r1c1_tau6": [peak(R / f"exp_g_ablation/r1c1_tau6/seed{s}/curves.json") for s in (0, 1)],
           "r2c0_tau4": [peak(R / f"exp_g_ablation/r2c0_tau4/seed{s}/curves.json") for s in (0, 1)]}
    json.dump(out, open(OUT / "h4.json", "w"), indent=1)
    return out, float(t.pvalue)


def h5():
    p3 = {0: peak(R / "exp_d_pool_minimax/round1/cand3/curves.json")}
    hu = {0: peak(R / "exp_a_human_3000/curves.json")}
    for s in (1, 2, 3, 4, 5):
        p3[s] = peak(R / f"exp_f_seeds/seed{s}/C_pool_c3_10.73/curves_full.json")
        hu[s] = peak(R / f"exp_f_seeds/seed{s}/human/curves_full.json")
    sd = [s for s in sorted(p3) if p3[s] is not None and hu[s] is not None]
    t = stats.ttest_rel([p3[s] for s in sd], [hu[s] for s in sd], alternative="greater")
    out = {"label": "actual runs (seed 0 straight 3,000; seeds 1-5 500 + resume)", "seeds": sd, "p3": [p3[s] for s in sd], "human": [hu[s] for s in sd],
           "mean_diff": float(np.mean([p3[s] - hu[s] for s in sd])), "paired_t": float(t.statistic), "p_one_sided": float(t.pvalue)}
    json.dump(out, open(OUT / "h5.json", "w"), indent=1)
    return out, float(t.pvalue)


def holm(ps):
    names = list(ps)
    order = sorted(names, key=lambda n: ps[n])
    adj, running = {}, 0.0
    for i, n in enumerate(order):
        running = max(running, min(1.0, (len(names) - i) * ps[n]))
        adj[n] = running
    return adj


def main():
    res, ps = {}, {}
    res["H1"], ps["H1"] = h1()
    calls = ant_minimax_calls()
    for t in ("cartpole", "ballbalance", "humanoid"):
        calls += task_calls(R / "exp_i_generalize", t)
    json.dump(calls, open(OUT / "llm_calls_minimax.json", "w"), indent=1)
    res["H2"], ps["H2"] = h2(calls)
    res["H3"], ps["H3"] = h3()
    res["H4"], ps["H4"] = h4()
    res["H5"], ps["H5"] = h5()
    adj = holm(ps)
    summary = {h: {"p_one_sided": ps[h], "p_holm": adj[h], "significant_at_0.05_holm": adj[h] < 0.05} for h in ps}
    json.dump({"primary": summary}, open(OUT / "primary_hypotheses.json", "w"), indent=1)
    for h, v in summary.items():
        print(f"{h}: p={v['p_one_sided']:.4f} holm={v['p_holm']:.4f} {'SIGNIFICANT' if v['significant_at_0.05_holm'] else 'not significant'}")


if __name__ == "__main__":
    main()
