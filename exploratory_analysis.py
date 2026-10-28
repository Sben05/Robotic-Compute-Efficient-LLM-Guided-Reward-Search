"""Exploratory analyses (STATS_PLAN.md: effect sizes and 95% CIs, no significance claims). Writes results/final/explore_*.json.

  E1  LLM-trial arms on Ant (MiniMax): calls, tokens, $, validity, 500-it scores on seeds 0 and 1, $ per valid reward,
      $ per reward beating the human 500-it score, top-6 resumed to 3,000.                                   [actual runs]
  E2  Second/third LLM (Gemini 2.5 Flash, Qwen-3-32B) on Ant: same table per model and arm, top 4 resumed.     [actual runs]
  E3  Generalisation: per task (Cartpole, BallBalance, Humanoid) one-per-call vs four-per-call rewards at full training,
      vs the human reward on the same seed.                                                                  [actual runs]
  E4  Transfer test: the FROZEN Ant pruning rules on each new task's curves (time rescaled to 3,000 points, score scaled so
      the same-seed human peak = 10.48; library = Ant TRAIN curves).                    [retrospective, held-out task]
  E5  False-prune rate of fixed-500 in the live trials with a Wilson CI; live pruning confirmation summary.   [actual runs]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
OUT = R / "final"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "pruning"))
import heldout as H  # noqa: E402
from configs import build  # noqa: E402
from sim_core import Library, World, run_policy  # noqa: E402

PRICE = {"minimax": (0.15e-6, 0.60e-6), "gemini": (0.30e-6, 2.50e-6), "qwen": (0.15e-6, 0.60e-6)}
RNG = np.random.default_rng(7)


def load(p):
    return json.loads(Path(p).read_text())


def cur(p):
    p = Path(p)
    if not p.exists():
        return None
    y = load(p).get("consecutive_successes")  # absent when the run crashed (e.g. a reward that fails at run time)
    return np.asarray(y, float) if y else None


def pk(p, k=None):
    c = cur(p)
    return None if c is None or not len(c) else float(np.max(c[:k] if k else c))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def boot_ci(x, B=5000):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) < 2:
        return [None, None]
    m = [RNG.choice(x, len(x)).mean() for _ in range(B)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(scores, usage, n_calls, n_funcs, n_valid, model, human500):
    pin, pout = PRICE[model]
    usd = sum(u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout for u in usage)
    tok = sum(u.get("completion_tokens", 0) for u in usage)
    s = [v for v in scores if v is not None]
    beat = sum(v > human500 for v in s)
    return {"calls": n_calls, "functions": n_funcs, "valid": n_valid, "validity": n_valid / max(1, n_funcs),
            "completion_tokens": tok, "usd": usd, "usd_per_valid": usd / max(1, n_valid), "tokens_per_valid": tok / max(1, n_valid),
            "n_screened": len(s), "best_500": max(s) if s else None, "median_500": float(np.median(s)) if s else None,
            "mean_500": float(np.mean(s)) if s else None, "mean_500_ci95": boot_ci(s), "beats_human_500": beat,
            "usd_per_reward_beating_human_500": usd / beat if beat else None}


def e1():
    h500 = {0: pk(R / "exp_a_human_3000/curves.json", 500), 1: pk(R / "exp_f_seeds/seed1/human/curves.json", 500)}
    out = {"label": "actual runs (Ant, MiniMax M3.1 Flash)", "human_500": h500, "arms": {}}
    tp = R / "llm_efficiency/trial_plan"
    for arm in ("B_batch4", "C_batch4_diverse", "D_indep_strategy", "E_indep_short_reasoning"):
        usage = [load(f)["response"]["usage"] for f in sorted(tp.glob(f"result_{arm}_*.json"))]
        gens = [load(g) for g in sorted((R / "exp_h_llm_trial" / arm).glob("call*/generation.json"))]
        row = {}
        for sd, base in ((0, R / "exp_h_llm_trial"), (1, R / "exp_h_llm_trial_seed1")):
            sc = [pk(c / "curves.json", 500) for c in sorted((base / arm).glob("call*"))]
            row[f"seed{sd}"] = summarize(sc, usage, len(usage), len(gens), sum(g["validation"]["valid"] for g in gens), "minimax", h500[sd])
        both = [(pk(R / "exp_h_llm_trial" / arm / c.name / "curves.json", 500), pk(c / "curves.json", 500))
                for c in sorted((R / "exp_h_llm_trial_seed1" / arm).glob("call*"))]
        both = [b for b in both if None not in b]
        row["seed_corr_500"] = float(np.corrcoef(*zip(*both))[0, 1]) if len(both) > 2 else None
        out["arms"][arm] = row
    # baseline arm A: the independent round-1 samples already trained (seed 0 only), their own logged usage
    dirs = sorted(list((R / "exp_b_minimax/round1").glob("cand*")) + list((R / "exp_d_pool_minimax/round1").glob("cand*")) +
                  list((R / "exp_e_live").glob("trial*/round1/cand*")))
    gens = [load(d / "generation.json") for d in dirs]
    usage = [g.get("usage") or {} for g in gens]
    sc = [pk(d / "curves.json", 500) if (d / "curves.json").exists() else None for d in dirs]
    out["arms"]["A_indep_baseline"] = {"seed0": summarize(sc, usage, len(dirs), len(dirs), sum(g["validation"]["valid"] for g in gens), "minimax", h500[0])}
    top = {}
    for d in sorted((R / "exp_h_llm_trial").glob("*/call*")):
        if (d / "result_full.json").exists():
            f = load(d / "result_full.json")
            top[f"{d.parent.name}/{d.name}"] = {"peak_3000": f["peak_task_score"], "final_3000": f["final_task_score"],
                                                 "best_500": pk(d / "curves.json", 500), "label": "500 + resume (audit protocol)"}
    out["top6_resumed_to_3000"] = top
    out["human_peak_3000_seed0"] = pk(R / "exp_a_human_3000/curves.json")
    json.dump(out, open(OUT / "explore_e1_llm_arms_ant.json", "w"), indent=1)
    return out


def e2():
    h500 = pk(R / "exp_a_human_3000/curves.json", 500)
    out = {"label": "actual runs (Ant, seed 0)", "models": {}}
    for model in ("gemini", "qwen"):
        base = R / "exp_k_llm_robustness" / model / "ant"
        out["models"][model] = {}
        for arm in ("indep", "batch4"):
            usage = [(load(f).get("response") or {}).get("usage") or {} for f in sorted((base / arm).glob("reply*.json"))]
            gens = [load(g) for g in sorted((base / arm).glob("call*/generation.json"))]
            sc = [pk(c / "seed0/curves.json", 500) for c in sorted((base / arm).glob("call*")) if (c / "seed0/curves.json").exists()]
            row = summarize(sc, usage, len(usage), len(gens), sum(g["validation"]["valid"] for g in gens), model, h500)
            row["resumed_3000"] = {c.name: load(c / "seed0/result_full.json")["peak_task_score"]
                                   for c in sorted((base / arm).glob("call*")) if (c / "seed0/result_full.json").exists()}
            out["models"][model][arm] = row
    json.dump(out, open(OUT / "explore_e2_other_llms_ant.json", "w"), indent=1)
    return out


def task_curves(t):
    G = R / "exp_i_generalize" / t
    rows = []
    for arm in ("indep", "batch4"):
        for c in sorted((G / arm).glob("call*/seed*")):
            y = cur(c / "curves.json")
            if y is not None and len(y):
                rows.append({"reward": f"{arm}/{c.parent.name}", "arm": arm, "seed": int(c.name[4:]), "curve": y})
    human = {int(c.name[4:]): cur(c / "curves.json") for c in sorted((G / "human").glob("seed*")) if (c / "curves.json").exists()}
    return rows, human


def e3():
    out = {"label": "actual runs (full training, task default length)", "tasks": {}}
    for t in ("cartpole", "ballbalance", "humanoid"):
        rows, human = task_curves(t)
        hp = {s: float(np.max(h)) for s, h in human.items() if h is not None}
        res = {"human_peak": hp}
        for arm in ("indep", "batch4"):
            v = [float(np.max(r["curve"])) / hp[r["seed"]] for r in rows if r["arm"] == arm and r["seed"] in hp]
            res[arm] = {"n_runs": len(v), "mean_norm": float(np.mean(v)) if v else None, "ci95": boot_ci(v), "median_norm": float(np.median(v)) if v else None,
                        "best_norm": max(v) if v else None, "beats_human": int(sum(x > 1 for x in v))}
        out["tasks"][t] = res
    json.dump(out, open(OUT / "explore_e3_generalisation_arms.json", "w"), indent=1)
    return out


def e4():
    fr = load(R / "pruning/frozen_rules.json")
    cfg = {lab: (pol, kw) for fam, lab, pol, kw in build()}
    rows_ant, human_ant = H.load_curves()
    train = [r for r in rows_ant if r["seed"] not in (4, 5, 7, 8, 9)]
    hum = [np.maximum.accumulate(h) for sd, h in human_ant.items() if sd not in (4, 5, 7, 8, 9)]  # match the "TRAIN curves" label above
    lib = Library([np.maximum.accumulate(r["curve"]) for r in train] + hum, hum)
    rules = {f"{fam}: {v['config']}": v["config"] for fam, v in fr["frozen"].items() if "paper" not in fam}
    rules["REFERENCE fixed@500 keep-1"] = "fixed@500 keep-1"
    out = {"label": "retrospective transfer test (frozen Ant rules; time rescaled to 3,000; scores scaled to human peak 10.48)", "tasks": {}}
    import itertools
    for t in ("cartpole", "ballbalance", "humanoid"):
        rows, human = task_curves(t)
        worlds = []
        for sd in sorted({r["seed"] for r in rows}):
            if sd not in human or human[sd] is None:
                continue
            scale = 10.48 / float(np.max(human[sd]))
            g = [r for r in rows if r["seed"] == sd]
            curves = [np.interp(np.linspace(0, len(r["curve"]) - 1, 3000), np.arange(len(r["curve"])), r["curve"]) * scale for r in g]
            subs = list(itertools.combinations(range(len(g)), 4))
            if len(subs) > 400:
                subs = [subs[k] for k in RNG.choice(len(subs), 400, replace=False)]
            worlds += [(np.stack([curves[k] for k in s]), sd) for s in subs]
        tab = {}
        for name, lab in rules.items():
            pol, kw = cfg[lab]
            recs = [run_policy(pol, World(raw), lib, np.random.default_rng(3), **kw) for raw, sd in worlds]
            tab[name] = {"compute": float(np.mean([r["cost"] for r in recs])), "retained": float(np.mean([r["retained"] for r in recs])),
                         "retained_ci95": boot_ci([r["retained"] for r in recs]), "hit": float(np.mean([r["hit"] for r in recs]))}
        out["tasks"][t] = {"n_worlds": len(worlds), "n_curves": len(rows), "rules": tab}
    json.dump(out, open(OUT / "explore_e4_pruning_transfer.json", "w"), indent=1)
    return out


def e5():
    a1 = load(R / "pruning/a1_false_pruning.json")
    lp = load(R / "exp_j_live_prune/seed6/live_prune_result.json")
    out = {"a1_false_prune": {"label": "actual runs + resumed audit", "false_prunes": a1["false_prunes"], "stopped": a1["stopped"],
                              "rate": a1["false_prune_rate"], "wilson95": wilson(a1["false_prunes"], a1["stopped"])},
           "live_pruning_confirmation": {"label": "actual run", **{k: lp[k] for k in ("rule", "seed", "selected_live", "replay_matches_live",
                                         "replay_within_recorded_prefixes", "total_iterations", "train_all_iterations", "compute_fraction", "pick_peak",
                                         "iterations_logged_per_arm", "live_targets")}}}
    json.dump(out, open(OUT / "explore_e5_false_prune_live.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    for f in (e1, e2, e3, e4, e5):
        try:
            r = f()
            print(f.__name__, "ok")
        except Exception as ex:  # report and continue: some data may still be training
            print(f.__name__, "FAILED:", repr(ex)[:300])
