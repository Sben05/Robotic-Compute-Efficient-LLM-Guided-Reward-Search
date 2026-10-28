"""Running ledger of actual GPU work, from the backed-up result files. Usage: python vast/ledger.py  -> results/LEDGER.json

Counts every training phase with a result*.json (s1 = from scratch, cont = resumed), PPO iterations logged, GPU-seconds
(training_seconds / runs sharing the GPU), and distinct (reward, seed) pairs that reached 3,000 iterations.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
STEPS_PER_ITERATION = {
    "ant": 4096 * 16,
    "cartpole": 512 * 16,
    "ballbalance": 4096 * 16,
    "humanoid": 4096 * 32,
}


def main():
    phases, iters, env_steps, gpu_s, wall_s, failed = Counter(), 0, 0, 0.0, 0.0, []
    full = set()
    by_exp = Counter()
    for f in sorted(R.rglob("result*.json")):
        if f.name not in ("result.json", "result_cont.json", "result_full.json"):
            continue
        d = json.loads(f.read_text())
        rel = f.parent.relative_to(R).as_posix()
        if f.name == "result_full.json":
            if d.get("iterations_logged") == 3000:
                full.add(rel)
            continue
        if not d.get("trained"):
            continue
        kind = "cont" if f.name == "result_cont.json" else "s1"
        if d.get("exit_code") not in (0, None):
            failed.append(rel + "/" + f.name)
            continue
        phases[kind] += 1
        by_exp[rel.split("/")[0]] += 1
        n = d.get("iterations_logged") or 0
        iters += n
        task = rel.split("/")[1] if rel.startswith("exp_i_generalize/") else "ant"
        env_steps += n * STEPS_PER_ITERATION[task]
        if d.get("training_seconds"):
            wall_s += d["training_seconds"]
            gpu_s += d["training_seconds"] / max(1, d.get("concurrent_runs_on_gpu") or 1)
        if kind == "s1" and n == 3000:
            full.add(rel)
    on_vast = {p for p in full if re.match(r"exp_(f_seeds|g_|h_)", p)}
    out = {"phases_completed": dict(phases), "failed_phases": failed, "ppo_iterations": iters,
           "env_steps": env_steps, "gpu_hours_shared_estimate": gpu_s / 3600, "run_wall_hours": wall_s / 3600,
           "reward_seed_pairs_at_3000": len(full), "of_which_on_vast": len(on_vast), "phases_by_experiment": dict(by_exp)}
    (R / "LEDGER.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
