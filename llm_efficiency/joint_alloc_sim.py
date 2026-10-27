"""E4: joint 'generate a new reward vs train an existing one' allocation, simulated on the 9 real fresh MiniMax PPO curves.

World: each generated reward is a draw from the TEST curves (time-warped to mimic seed noise). Policies see only prefixes plus a
LIBRARY of other finished runs (the 6 non-test fresh curves + 2 human curves); they never see the curve they are scored on.
All 84 = C(9,3) library/test splits are enumerated, so each curve is test in 28 splits.

Currency: PPO iterations. One LLM call costs c_gen iterations (swept; see report for the dollar / latency conversions).
Score: the true 3,000-it peak of the reward the policy selects (always trained to 3,000; that cost is counted).

Policies (all deterministic given the world):
  fixed(N, s)    : generate N, train each to s, finish the best          (the thesis' current screen; N=4, s=500)
  target(th, s)  : generate one at a time, screen to s; stop once best screened peak >= th * human peak@s (max 12 calls)
  joint(lam, s)  : EI-per-cost. Each step compare  gen: E[(Y_prior - m)+]/(c_gen+s)   vs   advance k: E[(final_k - m)+]/s
                   (m = leader's median forecast); stop when the best ratio < lam.  lam is the points-per-iteration the user will pay.
"""
import itertools, json, sys

import numpy as np

from common import *

SIG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35  # lognormal sd of the time-warp seed-noise model
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
MAXC = 12
GRID = 250
CGEN = [0, 40, 250, 1000]  # LLM-call cost in PPO iterations (0 = free; 40 ~ dollars on rented A100; 1000 ~ serial wall-clock on a slow endpoint)

llm, human = load_curves()
fresh = [k for k, v in LLM.items() if v[1] == "fresh"]
CM = {k: np.maximum.accumulate(v) for k, v in {**llm, **human}.items()}
H500 = np.mean([CM[h][499] for h in human])


def ratios(keys):
    """R[j, g]: library curve j's final peak / its peak after g*GRID its (>=1: peaks never decrease)"""
    out = np.ones((len(keys), T // GRID + 1))
    for j, k in enumerate(keys):
        for g in range(1, T // GRID + 1):
            out[j, g] = max(1.0, CM[k][-1] / max(CM[k][g * GRID - 1], 0.3))
    return out


class World:
    def __init__(self, test_keys, rng):
        self.c = [(test_keys[rng.integers(len(test_keys))], float(np.exp(SIG * rng.standard_normal()))) for _ in range(MAXC)]

    def peak(self, i, t):  # best score of candidate i after t PPO iterations
        k, w = self.c[i]
        return CM[k][min(int(t * w), T) - 1]


def fixed(world, N, s, lib, cg):
    pk = [world.peak(i, s) for i in range(N)]
    i = int(np.argmax(pk))
    return world.peak(i, T), N * s + (T - s), N


def target(world, th, s, lib, cg, minn=1):
    best, bi = -1, 0
    n = 0
    for i in range(MAXC):
        n += 1
        p = world.peak(i, s)
        if p > best:
            best, bi = p, i
        if (best >= th * np.mean([CM[h][s - 1] for h in human]) and n >= minn) or n == MAXC:
            break
    return world.peak(bi, T), n * s + (T - s), n


def joint(world, lam, s, lib, cg):
    R, Y = lib
    cands, calls, its = [], 0, 0  # cands: [index, t]
    while True:
        peaks = [world.peak(i, t) for i, t in cands]
        fc = [p * np.median(R[:, t // GRID]) for p, (i, t) in zip(peaks, cands)]
        L = int(np.argmax(fc)) if cands else -1
        m = fc[L] if cands else 0.0
        best, act = lam, None
        if calls < MAXC:
            r = np.mean(np.maximum(Y - m, 0)) / (cg + s)
            if r > best: best, act = r, ("gen",)
        for j, ((i, t), p) in enumerate(zip(cands, peaks)):
            if j == L or t >= T: continue
            r = np.mean(np.maximum(p * R[:, t // GRID] - m, 0)) / s
            if r > best: best, act = r, ("adv", j)
        if act is None and not cands: act = ("gen",)  # need at least one candidate
        if act is None: break
        if act[0] == "gen":
            cands.append([calls, s]); calls += 1; its += s
        else:
            cands[act[1]][1] += s; its += s
    iL, tL = cands[L]
    return world.peak(iL, T), its + (T - tL), calls


def configs():
    for s in (250, 500, 750):
        for N in range(1, 13): yield ("fixed", dict(N=N, s=s))
        for th in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3): yield ("target", dict(th=th, s=s))
    for lam in np.geomspace(2e-4, 3e-2, 14): yield ("joint", dict(lam=float(lam), s=250))
    for lam in np.geomspace(2e-4, 3e-2, 14): yield ("joint", dict(lam=float(lam), s=500))

FN = dict(fixed=fixed, target=target, joint=joint)


def main():
    cfgs = list(configs())
    acc = {}  # (cg, family, cfgkey) -> list of per-split (score, its, calls)
    splits = list(itertools.combinations(range(len(fresh)), 3))
    for si, test_idx in enumerate(splits):
        test = [fresh[i] for i in test_idx]
        libk = [k for k in fresh if k not in test]
        R = ratios(libk + list(human))
        Y = np.array([CM[k][-1] for k in libk])
        rng = np.random.default_rng(1000 + si)
        worlds = [World(test, rng) for _ in range(REPS)]
        for cg in CGEN:
            for fam, kw in cfgs:
                if fam != "joint" and cg != CGEN[0]:  # fixed/target do not depend on cg; computed once, costed per cg later
                    continue
                res = np.array([FN[fam](w, lib=(R, Y), cg=cg, **kw) for w in worlds])
                key = (cg if fam == "joint" else -1, fam, json.dumps(kw, sort_keys=True))
                acc.setdefault(key, []).append(res.mean(0))

    rows = []
    for (cg, fam, kw), v in acc.items():
        v = np.array(v)  # splits x (score, its, calls)
        cgs = CGEN if fam != "joint" else [cg]
        for c in cgs:
            cost = v[:, 1] + c * v[:, 2]
            rows.append(dict(cg=c, family=fam, cfg=json.loads(kw), score=v[:, 0].mean(), score_se=v[:, 0].std(ddof=1) / np.sqrt(len(v)),
                             cost=cost.mean(), its=v[:, 1].mean(), calls=v[:, 2].mean()))
    (OUT / f"joint_alloc_sig{SIG}.json").write_text(json.dumps(rows, indent=1, default=float))
    print("rows", len(rows), "splits", len(splits), "H500", round(H500, 2))


if __name__ == "__main__":
    main()
