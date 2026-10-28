"""Held-out pruning comparison (A-3/A-4).

Interface: every rule is a causal policy  rule(World, Library, rng, **params) -> candidate(s) to finish. A World holds the
candidates' curves but exposes only the prefix each has been trained for (World.cm(i), World.at(i, t)); advancing a candidate
costs iterations. So every rule decides keep/stop per candidate per checkpoint from prefixes and the budget it has spent only.

  python heldout.py tune    -> results/pruning/frozen_rules.json   (TRAIN curves only; commit this BEFORE scoring)
  python heldout.py score   -> results/pruning/heldout_scores.json (frozen rules on the held-out seeds 4-5)

TRAIN = every curve that is not seed 4 or 5: seed 0 (old 11 + live 16), exp_f seeds 1-3, the seed-1 replication of the live rounds,
G1 ablation variants, borderline extra seeds. TEST = exp_f seeds 4 and 5 (7 LLM rewards + human per seed).
Worlds are sets of N candidates trained on the SAME seed (as in a real search round). Libraries never contain any curve of a
reward that is in the world being decided (leave-reward-out), so forecasts cannot use other seeds of the same reward.
Labels: TRAIN tuning and TEST scoring are retrospective simulations on actual curves (no warp).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import policies as P
from configs import build
from data import LIVE, OLD, ROOT, SEED_KEYS, T
from sim_core import Library, World, run_policy

RES = ROOT / "results"
OUT = RES / "pruning"
TEST_SEEDS = (4, 5)
N = 4
THR = 0.98
EXTRA_KEYS = {"R_" + k: k for k in LIVE}  # seed-1 replication of live candidates
EXTRA_KEYS.update({"X_t1r1c2": "t1r1c2", "X_t1r1c3": "t1r1c3", "X_p1": "p1"})


def _curve(f):
    if not f.exists():
        return None
    y = np.asarray(json.loads(f.read_text())["consecutive_successes"], float)
    return y if len(y) == T else None


def load_curves():
    """list of dicts: reward, seed, curve, resumed, source. Human runs are kept separately (reference + library)."""
    rows, human = [], {}
    for k, rel in OLD.items():
        rows.append(dict(reward=k, seed=0, curve=_curve(RES / rel / "curves.json"), resumed=False, source=rel))
    for k, rel in LIVE.items():
        rows.append(dict(reward=k, seed=0, curve=_curve(RES / rel / "curves_full.json"), resumed=True, source=rel))
    human[0] = _curve(RES / "exp_a_human_3000/curves.json")
    for d in sorted((RES / "exp_f_seeds").glob("seed[0-9]*")):
        s = int(d.name[4:])
        for c in sorted(d.iterdir()):
            if not c.is_dir():
                continue
            y = _curve(c / "curves_full.json")
            if y is None:
                continue
            name = SEED_KEYS.get(c.name) or EXTRA_KEYS.get(c.name)
            if name is None:
                continue
            if name.startswith("human"):
                human[s] = y
            else:
                rows.append(dict(reward=name, seed=s, curve=y, resumed=True, source=str(c.relative_to(RES))))
    for c in sorted((RES / "exp_g_ablation").glob("*/seed*")):
        y = _curve(c / "curves.json")
        if y is not None:
            name = "r1c1" if c.parent.name == "r1c1_orig" else c.parent.name
            rows.append(dict(reward=name, seed=int(c.name[4:]), curve=y, resumed=False, source=str(c.relative_to(RES))))
    rows = [r for r in rows if r["curve"] is not None]
    return rows, {s: v for s, v in human.items() if v is not None}


def worlds(rows, human, seeds, max_per_group, rng, lib_rows):
    """Same-seed N-subsets; library = lib_rows minus every curve of the world's rewards, plus the human curves the
    caller passes in `human`. Callers must pre-filter `human` to the seeds that are legitimately visible (e.g.
    TRAIN seeds only for both tuning and scoring): this function no longer decides that itself, since including
    ANY test-seed human curve -- not just the world's own seed -- leaked held-out information into the library
    (audit finding 3; tune() was the worse case, since it ran before seeds 4-5's status as held-out even mattered)."""
    hum = [np.maximum.accumulate(h) for h in human.values()]
    out = []
    for s in seeds:
        g = [r for r in rows if r["seed"] == s]
        if len(g) < N:
            continue
        subs = list(itertools.combinations(range(len(g)), N))
        if len(subs) > max_per_group:
            subs = [subs[k] for k in rng.choice(len(subs), max_per_group, replace=False)]
        for sub in subs:
            names = {g[k]["reward"] for k in sub}
            lib = Library([np.maximum.accumulate(r["curve"]) for r in lib_rows if r["reward"] not in names] + hum, hum)
            out.append((World(np.stack([g[k]["curve"] for k in sub])), lib, [g[k]["reward"] for k in sub], s))
    return out


def run(pol, kw, W):
    rng = np.random.default_rng(11)
    recs = []
    for w, lib, names, s in W:
        r = run_policy(pol, w, lib, rng, **kw)
        tr = w.truth()
        top2 = set(np.argsort(-tr)[:2])
        stopped = [i for i in range(w.N) if w.prog[i] < T]
        r["false_prunes"] = sum(i in top2 for i in stopped)
        r["n_stopped"] = len(stopped)
        r["seed"] = s
        recs.append(r)
    return recs


def summary(recs):
    a = {k: np.array([r[k] for r in recs], float) for k in ("cost", "retained", "hit", "false_prunes", "n_stopped")}
    return {"compute": a["cost"].mean(), "saved": 1 - a["cost"].mean(), "retained": a["retained"].mean(), "hit": a["hit"].mean(),
            "false_prune_rate": a["false_prunes"].sum() / max(1, a["n_stopped"].sum()), "n_worlds": len(recs)}


def tune():
    rows, human = load_curves()
    train = [r for r in rows if r["seed"] not in TEST_SEEDS]
    human_train = {s: v for s, v in human.items() if s not in TEST_SEEDS}  # no test-seed curve, human or LLM, during freezing
    rng = np.random.default_rng(0)
    W = worlds(train, human_train, sorted({r["seed"] for r in train}), 250, rng, train)
    res = {}
    for fam, label, pol, kw in build():
        if fam == "baseline":
            continue
        res[label] = (fam, summary(run(pol, kw, W)))
        print(f"{label:34s} {res[label][1]['compute']:.3f} {res[label][1]['retained']:.3f}", flush=True)
    frozen = {}
    for fam in sorted({v[0] for v in res.values()}):
        cand = [(lab, s) for lab, (f, s) in res.items() if f == fam]
        ok = [(lab, s) for lab, s in cand if s["retained"] >= THR]
        lab, s = min(ok, key=lambda x: x[1]["compute"]) if ok else max(cand, key=lambda x: x[1]["retained"])
        frozen[fam] = {"config": lab, "train": s}
    out = {"label": "parameters frozen on TRAIN curves only, before seeds 4-5 existed", "N": N, "criterion": f"min compute s.t. mean retained >= {THR} (else max retained)",
           "train_curves": len(train), "train_worlds": len(W), "train_seeds": sorted({r["seed"] for r in train}),
           "test_seeds": list(TEST_SEEDS), "frozen": frozen,
           "reference_fixed500_train": res["fixed@500 keep-1"][1]}
    json.dump(out, open(OUT / "frozen_rules.json", "w"), indent=1, default=float)
    print(json.dumps({k: v["config"] for k, v in frozen.items()}, indent=1))


def boot(recs, key, B=2000, rng=None):
    """Bootstrap CI over worlds, resampling seeds first, then worlds within seed."""
    rng = rng or np.random.default_rng(5)
    by = {}
    for r in recs:
        by.setdefault(r["seed"], []).append(r[key])
    seeds = list(by)
    vals = []
    for _ in range(B):
        ss = rng.choice(seeds, len(seeds))
        v = np.concatenate([rng.choice(by[s], len(by[s])) for s in ss])
        vals.append(v.mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def score():
    fr = json.load(open(OUT / "frozen_rules.json"))
    rows, human = load_curves()
    train = [r for r in rows if r["seed"] not in TEST_SEEDS]
    test = [r for r in rows if r["seed"] in TEST_SEEDS]
    assert test, "no held-out curves yet"
    human_train = {s: v for s, v in human.items() if s not in TEST_SEEDS}  # no test-seed human curve in the library
    W = worlds(test, human_train, TEST_SEEDS, 10 ** 6, np.random.default_rng(1), train)
    cfg = {lab: (pol, kw) for fam, lab, pol, kw in build()}
    entries = {f"{fam}: {v['config']}": v["config"] for fam, v in fr["frozen"].items()}
    entries["REFERENCE fixed@500 keep-1"] = "fixed@500 keep-1"
    entries["baseline random pick"] = "random pick, train 1"
    out = {"label": "retrospective, held-out (frozen rules scored on seeds 4-5)", "n_worlds": len(W),
           "test_curves": len(test), "test_seeds": sorted({r["seed"] for r in test}), "rules": {}}
    ref = None
    for name, lab in entries.items():
        pol, kw = cfg[lab]
        recs = run(pol, kw, W)
        for r, (w, lib, names, s) in zip(recs, W):
            r["regret_vs_human"] = float(human[s].max() - (r["retained"] * w.truth().max())) if s in human else None
        s = summary(recs)
        s.update(retained_ci=boot(recs, "retained"), hit_ci=boot(recs, "hit"), compute_ci=boot(recs, "cost"),
                 mean_regret_vs_human=float(np.mean([r["regret_vs_human"] for r in recs if r["regret_vs_human"] is not None])),
                 iterations_per_world=float(np.mean([r["cost"] for r in recs]) * N * T))
        out["rules"][name] = s
        if lab == "fixed@500 keep-1":
            ref = s
    for name, s in out["rules"].items():
        s["beats_fixed500"] = bool(s["retained"] > ref["retained"] and s["retained_ci"][0] > ref["retained"] - 1e-9) if ref else None
    json.dump(out, open(OUT / "heldout_scores.json", "w"), indent=1, default=float)
    for name, s in sorted(out["rules"].items(), key=lambda kv: kv[1]["compute"]):
        print(f"{name:52s} compute {s['compute']:.3f} retained {s['retained']:.3f} {np.round(s['retained_ci'], 3)} hit {s['hit']:.2f} "
              f"false-prune {s['false_prune_rate']:.2f} regret-vs-human {s['mean_regret_vs_human']:+.2f}")


if __name__ == "__main__":
    {"tune": tune, "score": score}[sys.argv[1]]()
