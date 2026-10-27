"""Novelty probe (exploratory, offline, no GPU): do reward-structure signals predict the final winner better than the task score?

Prior work (ORSO, ICLR 2025; successive halving; learning-curve extrapolation) decides from the task-score curve only. A reward
search run also logs the candidate's own reward (gpt_reward) and each of its components every iteration. Question: at an early
checkpoint t, does adding generic features of those logs improve the prediction of which of two same-seed runs ends higher?

Data: every Ant run (seeds 0-9) whose curves include gpt_reward. Unit: same-seed pairs of different rewards.
Split: 5 folds BY REWARD IDENTITY (a reward and all its seeds are in one fold) -> no reward is in both train and test.
Models: logistic regression on pair feature differences. (a) task score only: best-so-far, slope over the last 20% of [0,t];
(b) + structure: proxy (gpt_reward) slope, proxy-task gap (normalised proxy gain minus task gain), proxy-task correlation,
share of components still rising, penalty share. Metric: pair accuracy, on all pairs and on CLOSE pairs (|score@t| diff < 1).
Usage: python structure_probe.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
T = 3000
SKIP = {"gpt_reward", "gt_reward", "consecutive_successes", "episode_lengths"}


def runs():
    out = []
    srcs = [(p, 0, "curves.json") for p in list((R / "exp_b_minimax").glob("round*/cand*")) + list((R / "exp_d_pool_minimax/round1").glob("cand*"))]
    srcs += [(p, 0, "curves_full.json") for p in (R / "exp_e_live").glob("trial*/round*/cand*")]
    for d in (R / "exp_f_seeds").glob("seed*"):
        s = int(d.name[4:])
        srcs += [(p, s, "curves_full.json") for p in d.iterdir() if p.is_dir() and p.name != "human"]
    srcs += [(p, int(p.name[4:]), "curves.json") for p in (R / "exp_g_ablation").glob("*/seed*")]
    for p, s, f in srcs:
        fp = p / f
        if not fp.exists():
            continue
        c = json.loads(fp.read_text())
        y = c.get("consecutive_successes")
        if not y or len(y) != T or "gpt_reward" not in c or len(c["gpt_reward"]) != T:
            continue
        gen = json.loads((p / "generation.json").read_text()) if (p / "generation.json").exists() else {}
        src = (p / "source.txt").read_text().strip() if (p / "source.txt").exists() else None
        rid = (json.loads((p / "generation.json").read_text()).get("response_text") or "")[:400] if gen else str(p)
        comps = {k: np.asarray(v, float) for k, v in c.items() if k not in SKIP and len(v) == T}
        out.append({"rid": hash(rid), "seed": s, "task": np.asarray(y, float), "proxy": np.asarray(c["gpt_reward"], float), "comps": comps,
                    "final": float(np.max(y)), "path": str(p.relative_to(R))})
    return out


def slope(x):
    n = len(x)
    if n < 3:
        return 0.0
    t = np.arange(n)
    return float(np.polyfit(t, x, 1)[0] * n)


def feats(r, t):
    y, p = r["task"][:t], r["proxy"][:t]
    w = max(5, t // 5)
    base = [float(np.max(y)), slope(y[-w:])]
    pn = (p - p[: w].mean()) / (np.std(p) + 1e-6)
    yn = (y - y[: w].mean()) / (np.std(y) + 1e-6)
    gap = float(pn[-w:].mean() - yn[-w:].mean())
    corr = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else 0.0
    rising = np.mean([slope(v[:t][-w:]) > 0 for v in r["comps"].values()]) if r["comps"] else 0.5
    neg = np.mean([v[:t][-w:].mean() < 0 for v in r["comps"].values()]) if r["comps"] else 0.5
    struct = [slope(p[-w:]), gap, corr, float(rising), float(neg)]
    return np.array(base), np.array(base + struct)


def fit_logreg(X, y, l2=1.0, it=500, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w / len(y))
    return lambda Xn: 1 / (1 + np.exp(-((Xn - mu) / sd) @ w))


def main():
    rs = runs()
    rids = sorted({r["rid"] for r in rs})
    rng = np.random.default_rng(0)
    fold = {rid: k % 5 for k, rid in enumerate(rng.permutation(rids))}
    print(f"{len(rs)} runs, {len(rids)} distinct rewards, seeds {sorted({r['seed'] for r in rs})}")
    res = {}
    for t in (250, 500, 750):
        F = [feats(r, t) for r in rs]
        pairs = [(i, j) for i in range(len(rs)) for j in range(i + 1, len(rs))
                 if rs[i]["seed"] == rs[j]["seed"] and rs[i]["rid"] != rs[j]["rid"] and rs[i]["final"] != rs[j]["final"]]
        acc = {"score": [], "struct": [], "score_close": [], "struct_close": [], "fixed": [], "fixed_close": []}
        for k in range(5):
            tr = [(i, j) for i, j in pairs if fold[rs[i]["rid"]] != k and fold[rs[j]["rid"]] != k]
            te = [(i, j) for i, j in pairs if fold[rs[i]["rid"]] == k and fold[rs[j]["rid"]] == k]
            if not te:
                continue
            ytr = np.array([rs[i]["final"] > rs[j]["final"] for i, j in tr], float)
            for name, idx in (("score", 0), ("struct", 1)):
                Xtr = np.array([F[i][idx] - F[j][idx] for i, j in tr])
                m = fit_logreg(Xtr, ytr)
                Xte = np.array([F[i][idx] - F[j][idx] for i, j in te])
                pred = m(Xte) > 0.5
                for (i, j), pr in zip(te, pred):
                    ok = float(pr == (rs[i]["final"] > rs[j]["final"]))
                    acc[name].append(ok)
                    if abs(F[i][0][0] - F[j][0][0]) < 1.0:
                        acc[name + "_close"].append(ok)
            for i, j in te:  # fixed-t rule: higher best-so-far wins
                ok = float((F[i][0][0] > F[j][0][0]) == (rs[i]["final"] > rs[j]["final"]))
                acc["fixed"].append(ok)
                if abs(F[i][0][0] - F[j][0][0]) < 1.0:
                    acc["fixed_close"].append(ok)

        def ci(v):
            v = np.array(v)
            b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            return round(float(v.mean()), 3), [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)], len(v)
        res[t] = {k: ci(v) for k, v in acc.items() if v}
        print(f"t={t}: " + "  ".join(f"{k} {v[0]} {v[1]} n={v[2]}" for k, v in res[t].items()))
    out = {"label": "exploratory offline probe (pairs overlap; CIs are naive bootstraps over pairs)", "n_runs": len(rs), "n_rewards": len(rids),
           "results": {str(k): v for k, v in res.items()}}
    (R / "pruning/structure_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
