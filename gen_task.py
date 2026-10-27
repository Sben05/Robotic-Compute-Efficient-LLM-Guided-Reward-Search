"""Generalisation study (exp_i): LLM reward generation for other Isaac Gym tasks with upstream Eureka prompts, two arms.

  indep  : one reward per call (upstream initial prompt, as in the Ant search)
  batch4 : four rewards per call (the Ant trial's best arm B: same prompt + "write FOUR ... compute_reward_1..4")
Model, temperature, max_tokens and endpoint are those of the Ant runs (eureka_ab: MiniMax M3.1 Flash via Vercel, 32,768 tokens).

  python gen_task.py generate TASK ARM CALLS     (Mac; needs the API key; idempotent per call file; prints tokens and cost)
  python gen_task.py build TASK                  (GPU box; validates every reply -> candidate dirs, jobs.json)
TASK in cartpole | ballbalance | humanoid. Writes results/exp_i_generalize/<task>/<arm>/call<i>[_f<k>]/.
"""
import concurrent.futures as cf
import json
import re
import sys
import traceback
from pathlib import Path

import eureka_ab as e

TASKS = {"ant": dict(task="Ant", env="ant", cfg="ant", iters=3000, num_envs=4096),
         "cartpole": dict(task="Cartpole", env="cartpole", cfg="cartpole", iters=100),
         "ballbalance": dict(task="BallBalance", env="ball_balance", cfg="ballbalance", iters=250),
         "humanoid": dict(task="Humanoid", env="humanoid", cfg="humanoid", iters=1000)}
import os
TAG = os.environ.get("GEN_TAG", "")  # robustness study: one output tree + task-name tag per LLM
OUT = e.ROOT / ("results/exp_k_llm_robustness/" + TAG if TAG else "results/exp_i_generalize")
BATCH = "Write FOUR different reward functions in this one reply, each in its own ```python block, each named compute_reward_1 ... compute_reward_4."
P_IN, P_OUT = 0.15e-6, 0.60e-6


def prompts(t):
    import yaml
    from utils.extract_task_code import file_to_string
    spec = TASKS[t]
    d = e.EUREKA / "utils/prompts"
    read = lambda n: file_to_string(str(d / (n + ".txt")))
    env_cfg = yaml.safe_load((e.EUREKA / f"cfg/env/{spec['cfg']}.yaml").read_text())
    obs = file_to_string(str(e.EUREKA / f"envs/isaac/{spec['env']}_obs.py"))
    system = read("initial_system").format(task_reward_signature_string=read("reward_signature")) + read("code_output_tip")
    user = read("initial_user").format(task_obs_code_string=obs, task_description=env_cfg["description"])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], obs


def generate(t, arm, calls):
    msgs, _ = prompts(t)
    if arm == "batch4":
        msgs = [msgs[0], {"role": "user", "content": msgs[1]["content"] + "\n\n" + BATCH}]
    d = OUT / t / arm
    d.mkdir(parents=True, exist_ok=True)
    e.save_json(d / "messages.json", msgs)
    todo = [i for i in range(calls) if not (d / f"reply{i}.json").exists()]

    def one(i):
        payload, raw, attempt, errors, seconds = e.llm(msgs)
        rec = {"call": i, "attempts": attempt, "errors": errors, "seconds": seconds, "request": payload}
        if raw:
            rec["response"] = json.loads(raw)
        e.save_json(d / f"reply{i}.json", rec)
        return rec
    with cf.ThreadPoolExecutor(min(8, max(1, len(todo)))) as ex:
        list(ex.map(one, todo))
    usd = 0.0
    for f in sorted(d.glob("reply*.json")):
        u = (json.loads(f.read_text()).get("response") or {}).get("usage") or {}
        usd += u.get("prompt_tokens", 0) * P_IN + u.get("completion_tokens", 0) * P_OUT
    print(f"{t} {arm}: {len(list(d.glob('reply*.json')))} replies, est ${usd:.3f}")


def pieces(arm, text):
    if arm != "batch4":
        return [(None, text)]
    out = []
    for block in re.findall(r"```python(.*?)```", text, re.DOTALL):
        m = re.search(r"def (compute_reward_(\d+))\s*\(", block)
        if m:
            out.append((int(m.group(2)), "```python\n" + block.replace(m.group(1) + "(", "compute_reward(", 1) + "\n```"))
    return out


def env_attributes(spec, obs):
    import ast
    from utils.extract_task_code import file_to_string
    src = "\n".join(file_to_string(str(p)) for p in [e.EUREKA / f"envs/isaac/{spec['env']}.py", e.ISAAC / "tasks/base/vec_task.py"]) + "\n" + obs
    return {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"}


def build_one(spec, cdir, text, obs, suffix):
    """Upstream injection (eureka.py lines ~150-170) for any task; generated code is never edited."""
    import importlib.util
    from utils.extract_task_code import file_to_string, get_function_signature
    rec = {"valid": False}
    code = e.extract_code(text)
    try:
        sig, inputs = get_function_signature(code)
    except Exception as ex:
        rec["error"] = "Upstream cannot parse function signature: %r" % ex
        return rec
    task_code = file_to_string(str(e.EUREKA / f"envs/isaac/{spec['env']}.py")).replace(spec["task"], spec["task"] + suffix)
    lines = [f"self.rew_buf[:], self.rew_dict = {sig}", "self.extras['gpt_reward'] = self.rew_buf.mean()",
             "for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()"]
    inject = "\n".join(" " * 8 + l for l in lines)
    for head in ("def compute_reward(self):", "def compute_reward(self, actions):"):
        if head in task_code:
            task_code_iter = task_code.replace(head, head + "\n" + inject)
            break
    else:
        rec["error"] = "no compute_reward hook in env file"
        return rec
    if "@torch.jit.script" not in code:
        code = "@torch.jit.script\n" + code
    header = "from typing import Tuple, Dict\nimport math\nimport torch\nfrom torch import Tensor\n"
    (cdir / "env_rewardonly.py").write_text(code + "\n")
    (cdir / "env_generated.py").write_text(task_code_iter + "\n" + header + code + "\n")
    rec.update(suffix=suffix, environment_call=sig, inputs=inputs)
    missing = sorted(set(inputs) - env_attributes(spec, obs))
    if missing:
        rec["error"] = "Inputs are not environment attributes: %s" % missing
        return rec
    chk = cdir / "_torchscript_check.py"
    chk.write_text(header + code + "\n")
    try:
        sp = importlib.util.spec_from_file_location("chk_" + suffix, str(chk))
        sp.loader.exec_module(importlib.util.module_from_spec(sp))
    except Exception:
        rec["error"] = "TorchScript compilation failed:\n" + traceback.format_exc()
        return rec
    finally:
        chk.unlink(missing_ok=True)
    rec["valid"] = True
    return rec


def build(t):
    spec = TASKS[t]
    _, obs = prompts(t)
    jobs, summary = [], {}
    for arm_dir in sorted((OUT / t).iterdir()):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        s = summary.setdefault(arm, {"calls": 0, "replies_with_text": 0, "functions": 0, "valid": 0, "completion_tokens": 0, "prompt_tokens": 0})
        for f in sorted(arm_dir.glob("reply*.json")):
            i = int(f.stem[5:])
            rec = json.loads(f.read_text())
            s["calls"] += 1
            raw = rec.get("response") or {}
            u = raw.get("usage") or {}
            s["completion_tokens"] += u.get("completion_tokens", 0); s["prompt_tokens"] += u.get("prompt_tokens", 0)
            ch = (raw.get("choices") or [{}])[0]
            text = (ch.get("message") or {}).get("content") or ""
            s["replies_with_text"] += bool(text.strip())
            for k, piece in (pieces(arm, text) if text.strip() else []):
                name = f"call{i}" + (f"_f{k}" if k is not None else "")
                cdir = arm_dir / name
                cdir.mkdir(exist_ok=True)
                suffix = "GPTMI" + re.sub(r"[^A-Za-z0-9]", "", f"{TAG}{t}{arm}{name}")
                gen = {"task_spec": {"task": spec["task"], "env": spec["env"], "num_envs": spec.get("num_envs")}, "task": t, "arm": arm, "call": i,
                       "model_requested": e.MODEL,
                       "function": k, "model_used": raw.get("model"), "usage": u, "finish_reason": ch.get("finish_reason"),
                       "functions_in_reply": len(pieces(arm, text)), "generation_seconds": rec.get("seconds"), "response_text": piece}
                gen["validation"] = build_one(spec, cdir, piece, obs, suffix)
                e.save_json(cdir / "generation.json", gen)
                s["functions"] += 1
                if gen["validation"]["valid"]:
                    s["valid"] += 1
                    jobs.append(str(cdir))
                print(t, arm, name, "valid" if gen["validation"]["valid"] else gen["validation"]["error"][:70], flush=True)
    e.save_json(OUT / t / "build_summary.json", summary)
    e.save_json(OUT / t / "valid_candidates.json", jobs)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        build(sys.argv[2])
