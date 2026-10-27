"""E4b: a few pre-specified policies (no per-world tuning), paired against the thesis default fixed(N=4, s=500).
Paired differences are computed per library/test split (common worlds); SE is across the 84 splits (they overlap, so it is optimistic)."""
import itertools, json, sys
import numpy as np
import joint_alloc_sim as J
from common import OUT

POL = {
    "fixed N=1 (no search)": ("fixed", dict(N=1, s=500)),
    "fixed N=2": ("fixed", dict(N=2, s=500)),
    "fixed N=4 (thesis default)": ("fixed", dict(N=4, s=500)),
    "fixed N=8": ("fixed", dict(N=8, s=500)),
    "adaptive: 2 first, stop at >= human@500": ("target", dict(th=1.0, s=500, minn=2)),
    "adaptive: 1 first, stop at >= human@500": ("target", dict(th=1.0, s=500, minn=1)),
    "adaptive: 2 first, stop at >= 0.8 human@500": ("target", dict(th=0.8, s=500, minn=2)),
    "joint EI/cost lam=1pt/1500its s=250": ("joint", dict(lam=1 / 1500, s=250)),
    "joint EI/cost lam=1pt/1500its s=500": ("joint", dict(lam=1 / 1500, s=500)),
    "joint EI/cost lam=1pt/500its s=250": ("joint", dict(lam=1 / 500, s=250)),
}
BASE = "fixed N=4 (thesis default)"
out = {}
splits = list(itertools.combinations(range(len(J.fresh)), 3))
for cg in J.CGEN:
    per = {k: [] for k in POL}
    for si, test_idx in enumerate(splits):
        test = [J.fresh[i] for i in test_idx]
        libk = [k for k in J.fresh if k not in test]
        R = J.ratios(libk + list(J.human)); Y = np.array([J.CM[k][-1] for k in libk])
        rng = np.random.default_rng(5000 + si)
        worlds = [J.World(test, rng) for _ in range(J.REPS)]
        for name, (fam, kw) in POL.items():
            r = np.array([J.FN[fam](w, lib=(R, Y), cg=cg, **kw) for w in worlds]).mean(0)
            per[name].append([r[0], r[1] + cg * r[2], r[2]])
    b = np.array(per[BASE])
    out[cg] = {}
    print(f"\n=== sigma={J.SIG}, c_gen={cg} its/call ===  {'policy':44s} {'score':>6s} {'d_score':>8s} {'cost':>7s} {'d_cost%':>8s} {'calls':>5s}")
    for name in POL:
        a = np.array(per[name]); d = a[:, 0] - b[:, 0]; dc = (a[:, 1] - b[:, 1]) / b[:, 1].mean()
        se = d.std(ddof=1) / np.sqrt(len(d))
        out[cg][name] = dict(score=a[:, 0].mean(), d_score=d.mean(), d_score_se=se, cost=a[:, 1].mean(), d_cost_frac=dc.mean(), calls=a[:, 2].mean())
        print(f"{'':36s} {name:44s} {a[:,0].mean():6.2f} {d.mean():+6.2f}±{se:.2f} {a[:,1].mean():7.0f} {100*dc.mean():+7.1f}% {a[:,2].mean():5.1f}")
(OUT / f"defaults_compare_sig{J.SIG}.json").write_text(json.dumps(out, indent=1, default=float))
