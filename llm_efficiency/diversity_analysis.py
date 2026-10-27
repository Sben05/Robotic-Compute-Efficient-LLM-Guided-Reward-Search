"""E2: how diverse are independent MiniMax samples, and does code-level diversity relate to PPO outcome?
Static analysis only (ast/difflib). The 9 'fresh' samples are independent draws from the same round-1 prompt."""
import ast, difflib, io, itertools, json, re, tokenize

import numpy as np
from scipy.stats import spearmanr

from common import *

FRESH = [k for k, v in LLM.items() if v[1] == "fresh"]
ALL = list(LLM)
llm, _ = load_curves()


def src(k):
    return (ROOT / "results" / LLM[k][0] / "env_rewardonly.py").read_text()


def features(code):
    tree = ast.parse(code)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "compute_reward")
    args = {a.arg for a in fn.args.args}
    consts = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    calls = {ast.unparse(n.func) if hasattr(ast, "unparse") else "" for n in ast.walk(fn) if isinstance(n, ast.Call)}
    comps = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Dict):
            comps |= {k.value for k in n.keys if isinstance(k, ast.Constant)}
    transforms = {c for c in calls if any(t in c for t in ("exp", "tanh", "sigmoid", "clamp", "abs", "sqrt", "where", "relu"))}
    body = ast.get_source_segment(code, fn)
    toks = []
    for t in tokenize.generate_tokens(io.StringIO(body).readline):
        if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.STRING):
            continue
        toks.append("NUM" if t.type == tokenize.NUMBER else t.string)
    return dict(args=args, comps=comps, n_comps=len(comps), n_consts=len(consts), transforms=transforms,
                toks=toks, lines=body.count("\n"), consts=consts)


F = {k: features(src(k)) for k in ALL}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def dist(a, b):
    fa, fb = F[a], F[b]
    return {"args": 1 - jac(fa["args"], fb["args"]), "transforms": 1 - jac(fa["transforms"], fb["transforms"]),
            "code": 1 - difflib.SequenceMatcher(None, fa["toks"], fb["toks"], autojunk=False).ratio(),
            "tokset": 1 - jac(set(fa["toks"]), set(fb["toks"]))}


pairs = list(itertools.combinations(FRESH, 2))
D = {p: dist(*p) for p in pairs}
out = {"n_fresh": len(FRESH), "n_pairs": len(pairs)}
for m in ["args", "transforms", "code", "tokset"]:
    v = np.array([D[p][m] for p in pairs])
    out[f"pairwise_{m}"] = {"mean": v.mean(), "min": v.min(), "max": v.max(), "frac_identical": float((v == 0).mean())}
    print(f"pairwise distance[{m:10s}] mean {v.mean():.2f}  min {v.min():.2f}  max {v.max():.2f}  identical {100*(v==0).mean():.0f}%")

# how many distinct 'recipes' among independent samples? recipe = (sorted input signature, sorted transform set)
rec = {}
for k in FRESH:
    rec.setdefault((tuple(sorted(F[k]["args"])), tuple(sorted(F[k]["transforms"]))), []).append(k)
out["distinct_recipes"] = len(rec)
out["recipe_groups"] = [v for v in rec.values()]
print(f"\ndistinct (inputs, transforms) recipes among 9 independent samples: {len(rec)}  groups {list(rec.values())}")

# does code distance predict outcome distance?
dc = np.array([D[p]["code"] for p in pairs])
do = np.array([abs(llm[a].max() - llm[b].max()) for a, b in pairs])
rho, pv = spearmanr(dc, do)
out["code_dist_vs_outcome_dist"] = {"spearman": rho, "p": pv}
print(f"code distance vs |peak difference| over {len(pairs)} pairs: Spearman {rho:.2f} (p={pv:.2f}; pairs share candidates, so p is optimistic)")

# nearest-neighbour smoothness: is the PPO peak of the nearest code-neighbour closer than a random sample's?
nn_gap, rand_gap = [], []
for k in FRESH:
    others = [o for o in FRESH if o != k]
    nn = min(others, key=lambda o: dist(k, o)["code"])
    nn_gap.append(abs(llm[k].max() - llm[nn].max()))
    rand_gap.append(np.mean([abs(llm[k].max() - llm[o].max()) for o in others]))
out["nn_smoothness"] = {"mean_gap_nearest_code_neighbour": float(np.mean(nn_gap)), "mean_gap_random": float(np.mean(rand_gap))}
print(f"|peak gap| to nearest code neighbour {np.mean(nn_gap):.2f} vs to a random sample {np.mean(rand_gap):.2f}")

# refinement edit distance vs fresh-to-fresh: what does one feedback round change?
parent = "r1c1"
for ch in ["r2c0", "r2c1"]:
    d = dist(parent, ch)
    out[f"refine_{ch}_vs_parent"] = d
    fresh_d = np.mean([dist(parent, o)["code"] for o in FRESH if o != parent])
    print(f"{ch} vs parent {parent}: code dist {d['code']:.2f} (parent to other fresh samples: {fresh_d:.2f}); args dist {d['args']:.2f}")
    out[f"refine_{ch}_vs_fresh_mean_code"] = fresh_d

print("\nper-sample features:")
for k in ALL:
    f = F[k]
    print(f"{k:5s} peak {llm[k].max():6.2f} lines {f['lines']:3d} comps {f['n_comps']} floats {f['n_consts']:2d} args {sorted(f['args'])} transforms {sorted(f['transforms'])}")
    out.setdefault("features", {})[k] = {"peak": float(llm[k].max()), "lines": f["lines"], "n_comps": f["n_comps"], "n_float_consts": f["n_consts"],
                                         "args": sorted(f["args"]), "comps": sorted(f["comps"]), "transforms": sorted(f["transforms"])}
(OUT / "diversity.json").write_text(json.dumps(out, indent=1, default=float))
