"""Policy configurations swept by every allocation experiment."""
import policies as P


def build():
    CFG = []  # (family, label, policy, kwargs)
    CFG += [("baseline", "train all to 3000", P.full, {}), ("baseline", "random pick, train 1", P.random_one, {})]
    for tc in (100, 250, 500, 750, 1000, 1500):
        CFG.append(("fixed checkpoint", f"fixed@{tc} keep-1", P.fixed, dict(t_c=tc, keep=1)))
    for tc in (250, 500):
        CFG.append(("fixed checkpoint", f"fixed@{tc} keep-2", P.fixed, dict(t_c=tc, keep=2)))
    for t0 in (50, 100, 250, 500):
        for eta in (2, 3):
            CFG.append(("successive halving", f"SH t0={t0} eta={eta}", P.seq_halving, dict(t0=t0, eta=eta)))
    for t0s in ((50, 150, 450), (100, 300, 900), (100, 250, 500)):
        CFG.append(("hyperband", f"HB {t0s}", P.hyperband, dict(t0s=t0s, eta=3 if t0s[1] == 3 * t0s[0] else 2)))
    for eta in (3, 2):
        CFG.append(("hyperband (paper)", f"HB-paper eta={eta}", P.hyperband_paper, dict(eta=eta)))
    for lam in (0.05, 0.2, 0.5, 1.0, 2.0):
        for t0 in (250, 500):
            CFG.append(("VoI (expected improvement)", f"VoI lam={lam} t0={t0}", P.race, dict(rule=P.rule_voi(lam), t0=t0)))
    for rho in (0.5, 0.6, 0.7, 0.8, 0.9):
        for t0 in (250, 500):
            CFG.append(("human-reference threshold", f"ref rho={rho} t0={t0}", P.race, dict(rule=P.rule_human_ref(rho), t0=t0)))
    for q in (0.75, 0.9, 1.0):
        for t0 in (100, 250, 500):
            CFG.append(("ratio-bound racing", f"bound q={q} t0={t0}", P.race, dict(rule=P.rule_ratio_bound(q), t0=t0)))
    for kd in (0, 1, 3):
        for ki in (0, 1):
            for th in (0.15, 0.25, 0.4, 0.6, 0.9):
                fam = "P" if kd == 0 and ki == 0 else ("PD" if ki == 0 else ("PI" if kd == 0 else "PID"))
                CFG.append((f"PID ({fam})", f"{fam} ki={ki} kd={kd} th={th}", P.race,
                            dict(rule=P.rule_pid(1.0, ki, kd, th), t0=250)))
    for d in (0.01, 0.03, 0.1, 0.2):
        for t0 in (100, 250, 500):
            CFG.append(("Bayesian pruning", f"P(best)<{d} t0={t0}", P.race, dict(rule=P.rule_bayes(d), t0=t0)))
    for lam in (0.2, 0.5, 1.0, 2.0, 4.0):
        for t0 in (100, 250, 500):
            CFG.append(("MPC (receding horizon)", f"MPC lam={lam} t0={t0}", P.race, dict(rule=P.rule_mpc(lam), t0=t0)))
    for algo in ("d3rb", "exp3"):
        for b in (0.2, 0.3, 0.4, 0.5, 0.6):
            CFG.append((f"ORSO ({algo}, post-hoc baseline)", f"ORSO {algo} budget={b}", P.orso, dict(algo=algo, budget=b)))
    for d in (0.05, 0.2, 0.4):
        for b in (0.2, 0.35, 0.5):
            CFG.append(("Thompson BAI (TTTS)", f"TTTS delta={d} budget={b}", P.ttts, dict(delta=d, budget=b)))
    return CFG
