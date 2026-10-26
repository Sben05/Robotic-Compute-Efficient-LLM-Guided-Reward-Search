"""Zero-training reward probe (exploratory): score generated rewards on scripted Ant rollouts, no PPO.

One official Ant simulation runs a population of scripted behaviours (standing still, random
actions, and randomized sine-wave gaits). Every candidate reward is evaluated on the same states
the environment hands its own compute_reward. A reward is "aligned" when its per-episode return
ranks behaviours the way true forward progress (the consecutive_successes quantity) does.

Usage: python probe.py OUT.json REWARD_FILE...   (each file is an env_rewardonly.py)
"""
import isaacgym  # noqa: F401  (must be imported before torch)
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parent
ISAAC = ROOT / "Eureka/isaacgymenvs/isaacgymenvs"
import isaacgymenvs  # noqa: E402,F401  (registers the config resolvers)
from isaacgymenvs.tasks.ant import Ant  # noqa: E402
from isaacgymenvs.utils.reformat import omegaconf_to_dict  # noqa: E402
from isaacgymenvs.utils.utils import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "Eureka/eureka"))
from utils.extract_task_code import get_function_signature  # noqa: E402

NUM_ENVS, STEPS, SEED = 4096, 500, 0
HEADER = "from typing import Tuple, Dict\nimport math\nimport torch\nfrom torch import Tensor\n"


def load_reward(path, tmp):
    tmp.write_text(HEADER + Path(path).read_text())
    spec = importlib.util.spec_from_file_location("probe_reward_%d" % abs(hash(str(path))), str(tmp))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Same function the environment calls (upstream takes the first def's arguments as self.<name>).
    name = get_function_signature(Path(path).read_text())[0].split("(")[0]
    return getattr(module, name), get_function_signature(Path(path).read_text())[1]


def ranks(x):
    return torch.argsort(torch.argsort(x)).float()


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / torch.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def main():
    out = Path(sys.argv[1])
    files = [str(f) for a in sys.argv[2:] for f in (sorted(Path().glob(a)) if "*" in a else [a])]
    set_seed(SEED)
    with initialize_config_dir(config_dir=str(ISAAC / "cfg"), version_base="1.1"):
        cfg = compose(config_name="config", overrides=["task=Ant", "num_envs=%d" % NUM_ENVS, "headless=True"])
    env = Ant(cfg=omegaconf_to_dict(cfg.task), rl_device="cuda:0", sim_device="cuda:0", graphics_device_id=-1,
              headless=True, virtual_screen_capture=False, force_render=False)
    dev, n = env.device, NUM_ENVS
    g = torch.Generator(device=dev).manual_seed(SEED)
    rand = lambda *s: torch.rand(*s, generator=g, device=dev)
    # Behaviour population: 1/16 zero action, 3/16 i.i.d. random, 12/16 sine gaits with random params.
    kind = torch.full((n,), 2, device=dev)
    kind[: n // 16] = 0
    kind[n // 16: n // 4] = 1
    amp, freq = rand(n, 1), 0.5 + 2.5 * rand(n, 1)
    phase, bias = 2 * math.pi * rand(n, 8), 0.6 * rand(n, 8) - 0.3

    rewards = []
    tmp = out.parent / "_probe_reward_tmp.py"
    for f in files:
        try:
            fn, args = load_reward(f, tmp)
            rewards.append({"file": f, "fn": fn, "args": args, "error": None})
        except Exception as e:
            rewards.append({"file": f, "fn": None, "args": [], "error": "load: %r" % e})
    tmp.unlink(missing_ok=True)
    returns = torch.zeros(len(rewards), n, device=dev)
    progress = torch.zeros(n, device=dev)
    human = torch.zeros(n, device=dev)
    falls = torch.zeros(n, device=dev)
    env.reset_idx(torch.arange(n, device=dev))
    started = time.time()
    for t in range(STEPS):
        sine = amp * torch.sin(2 * math.pi * freq * t * env.dt + phase) + bias
        actions = torch.where(kind[:, None] == 0, torch.zeros_like(sine),
                              torch.where(kind[:, None] == 1, 2 * rand(n, 8) - 1, sine)).clamp(-1, 1)
        env.step(actions)
        progress += env.potentials - env.prev_potentials
        human += env.rew_buf
        falls += env.reset_buf.float()
        for i, r in enumerate(rewards):
            if r["fn"] is None or r["error"]:
                continue
            try:
                total, _ = r["fn"](*[getattr(env, a) for a in r["args"]])
                returns[i] += total
            except Exception as e:
                r["error"] = "step %d: %r" % (t, e)
    sim_seconds = time.time() - started
    k = n // 10
    top_true = set(torch.topk(progress, k).indices.tolist())
    result = {"num_envs": n, "steps": STEPS, "seed": SEED, "sim_seconds": sim_seconds,
              "env_steps": n * STEPS, "falls_per_env": float(falls.mean()),
              "progress_mean_by_kind": {name: float(progress[kind == i].mean()) for i, name in
                                        enumerate(["zero", "random", "sine"])},
              "human_reward": {"spearman_vs_progress": spearman(human, progress)}, "candidates": {}}
    for i, r in enumerate(rewards):
        entry = {"error": r["error"]}
        if not r["error"]:
            ret = returns[i]
            finite = bool(torch.isfinite(ret).all())
            entry.update(finite=finite)
            if finite:
                top_cand = set(torch.topk(ret, k).indices.tolist())
                entry.update(spearman_vs_progress=spearman(ret, progress),
                             top10_precision=len(top_cand & top_true) / k,
                             mean_return_by_kind={name: float(ret[kind == j].mean()) for j, name in
                                                  enumerate(["zero", "random", "sine"])})
        result["candidates"][r["file"]] = entry
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
