"""Experiments A-C: 3,000-iteration human baseline, 3+2 Mini-Eureka, retrospective screening.

Upstream Eureka (eureka/eureka.py) is followed line for line for prompt assembly,
code extraction, reward injection, task creation, feedback construction, and
best-sample selection. Deliberate differences: MiniMax M3.1 Flash (NVIDIA NIM) instead of GPT-4, 3 then 2
samples over two rounds, seed 0, one Slurm job (one A100) per training run, and
unique per-candidate task names so runs can train in parallel without sharing files.

Subcommands (run by eureka.sbatch):
  generate ROUND N   query the LLM N times and validate each reward (CPU)
  train NAME         train results/<NAME> for 3,000 iterations (GPU); NAME=human for Exp A
  analyze            Experiment C screening analysis, figures, scores (CPU)
"""
import ast
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EUREKA = ROOT / "Eureka/eureka"
ISAAC = ROOT / "Eureka/isaacgymenvs/isaacgymenvs"
HUMAN = ROOT / "results/exp_a_human_3000"
# LLM arms (EUREKA_LLM, default minimax); each has its own result directories and task-name tag.
#  minimax: MiniMax M3.1 Flash on NVIDIA NIM's free endpoint (OpenAI chat API, 40 RPM);
#           key in ~/.config/msthesis/nvidia_api_key_minimax (mode 600, never logged).
#           EUREKA_ENDPOINT=vercel routes the same model through Vercel AI Gateway (vercel_ai_gateway_key).
ARMS = {
    "minimax": {"model": "minimaxai/minimax-m3.1-flash", "url": "https://integrate.api.nvidia.com/v1/chat/completions",
                "key_file": "nvidia_api_key_minimax", "min_interval": 2, "tag": "M", "timeout": 420},
}
LLM = os.environ.get("EUREKA_LLM", "minimax")
ARM = ARMS[LLM]
if LLM == "minimax" and os.environ.get("EUREKA_ENDPOINT") == "vercel":  # same model, Vercel AI Gateway route (free credits)
    ARM = dict(ARM, model="minimax/minimax-m3.1-flash", url="https://ai-gateway.vercel.sh/v1/chat/completions",
               key_file="vercel_ai_gateway_key", min_interval=0)
MODEL, URL, MIN_INTERVAL = ARM["model"], ARM["url"], ARM["min_interval"]
MODEL = os.environ.get("EUREKA_MODEL", MODEL)  # robustness study: other gateway models (e.g. deepseek/deepseek-v4.1-flash)
KEY_FILE = Path.home() / ".config/msthesis" / ARM["key_file"] if ARM["key_file"] else None
MAX_TOKENS = int(os.environ.get("EUREKA_MAX_TOKENS", "16384"))  # explicit room for reasoning plus code
OUT = ROOT / ("results/exp_b_" + LLM)
SCREEN = ROOT / ("results/exp_c_screening_" + LLM)
TEMPERATURE = 1.0
ITERATIONS = 3000
SEED = int(os.environ.get("EUREKA_SEED", "0"))  # extra seeds retrain the same rewards into *_seedN directories
NUM_ENVS = 4096
STEPS_PER_ITER = NUM_ENVS * 16  # horizon_length 16 in AntPPO.yaml
CHECKPOINTS = [100, 250, 500, 1000, 1500, 3000]
DUMMY_FAILURE = -10000.0  # upstream value

sys.path.insert(0, str(EUREKA))
from utils.extract_task_code import file_to_string, get_function_signature  # noqa: E402
from utils.misc import filter_traceback  # noqa: E402


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def suffix_for(name):
    # Upstream uses suffix "GPT"; a unique suffix per candidate keeps parallel runs apart.
    return "GPT" + ARM["tag"] + name.replace("round", "R").replace("/cand", "C")


# ---------------------------------------------------------------- generation

def prompt_parts():
    """Exactly the prompt strings eureka.py loads and formats."""
    import yaml
    d = EUREKA / "utils/prompts"
    read = lambda n: file_to_string(str(d / n))
    p = {n: read(n + ".txt") for n in ["initial_system", "code_output_tip", "code_feedback", "initial_user",
                                     "reward_signature", "policy_feedback", "execution_error_feedback"]}
    env_cfg = yaml.safe_load((EUREKA / "cfg/env/ant.yaml").read_text())
    obs = file_to_string(str(EUREKA / "envs/isaac/ant_obs.py"))
    p["initial_system"] = p["initial_system"].format(task_reward_signature_string=p["reward_signature"]) + p["code_output_tip"]
    p["initial_user"] = p["initial_user"].format(task_obs_code_string=obs, task_description=env_cfg["description"])
    p["obs"] = obs
    return p


def llm(messages):
    """One sample, OpenAI chat format as upstream eureka.py sends it: (request body, raw response or None,
    attempts, errors, seconds). Transient failures are retried like upstream's retry loop (bounded)."""
    key = KEY_FILE.read_text().strip() if KEY_FILE else ""
    payload = {"model": MODEL, "messages": messages, "temperature": TEMPERATURE, "n": 1, "max_tokens": MAX_TOKENS}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(URL, data=json.dumps(payload).encode(), method="POST", headers=headers)
    redact = (lambda s: s.replace(key, "[REDACTED]")) if key else (lambda s: s)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None

    errors, started = [], time.monotonic()
    for attempt in range(1, 13):
        time.sleep(MIN_INTERVAL)
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=ARM["timeout"]) as response:
                raw = response.read().decode()
            return payload, redact(raw), attempt, errors, time.monotonic() - started
        except urllib.error.HTTPError as e:
            errors.append({"attempt": attempt, "http": e.code, "body": redact(e.read().decode()[:2000])})
            if e.code not in (429, 500, 502, 503, 504):
                break
        except Exception as e:  # timeouts / connection resets
            errors.append({"attempt": attempt, "error": redact(repr(e))})
        time.sleep(min(120, 5 * 2 ** (attempt - 1)))
    return payload, None, attempt, errors, time.monotonic() - started


def parse_response(raw):
    """(text, finish reason, usage, model used) from a chat-completions response body."""
    body = json.loads(raw)
    choice = body["choices"][0] if body.get("choices") else {}
    return (choice.get("message", {}).get("content"), choice.get("finish_reason"),
            body.get("usage"), body.get("model"))


def extract_code(response_cur):
    """Verbatim from eureka.py."""
    patterns = [r'```python(.*?)```', r'```(.*?)```', r'"""(.*?)"""', r'""(.*?)""', r'"(.*?)"']
    for pattern in patterns:
        code_string = re.search(pattern, response_cur, re.DOTALL)
        if code_string is not None:
            code_string = code_string.group(1).strip()
            break
    code_string = response_cur if not code_string else code_string
    lines = code_string.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            code_string = "\n".join(lines[i:])
    return code_string


def env_attributes(obs):
    sources = [EUREKA / "envs/isaac/ant.py", ISAAC / "tasks/base/vec_task.py"]
    tree = ast.parse("\n".join(file_to_string(str(s)) for s in sources) + "\n" + obs)
    return {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"}


def build_candidate(cdir, response_cur, obs):
    """Upstream reward injection; returns a validation record. Generated code is never edited."""
    record = {"valid": False}
    code_string = extract_code(response_cur)
    try:
        gpt_reward_signature, input_lst = get_function_signature(code_string)
    except Exception as e:
        record["error"] = "Upstream cannot parse function signature: %r" % e
        (cdir / "extracted_code.py").write_text(code_string + "\n")
        return record
    suffix = suffix_for(cdir.relative_to(OUT).as_posix())
    task_code_string = file_to_string(str(EUREKA / "envs/isaac/ant.py")).replace("Ant", "Ant" + suffix)
    reward_signature = [
        f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
        f"self.extras['gpt_reward'] = self.rew_buf.mean()",
        f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
    ]
    reward_signature = "\n".join([" " * 8 + line for line in reward_signature])
    task_code_string_iter = task_code_string.replace(
        "def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
    if "@torch.jit.script" not in code_string:
        code_string = "@torch.jit.script\n" + code_string
    header = "from typing import Tuple, Dict\nimport math\nimport torch\nfrom torch import Tensor\n"
    (cdir / "env_rewardonly.py").write_text(code_string + "\n")
    (cdir / "env_generated.py").write_text(task_code_string_iter + "\n" + header + code_string + "\n")
    record.update(suffix=suffix, environment_call=gpt_reward_signature, inputs=input_lst)
    missing = sorted(set(input_lst) - env_attributes(obs))
    if missing:
        record["error"] = "Inputs are not environment attributes: %s" % missing
        return record
    # TorchScript compile check: import the reward module from a file so TorchScript can read its source.
    check = cdir / "_torchscript_check.py"
    check.write_text(header + code_string + "\n")
    try:
        spec = importlib.util.spec_from_file_location("reward_check_" + suffix, str(check))
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    except Exception:
        record["error"] = "TorchScript compilation failed:\n" + traceback.format_exc()
        return record
    finally:
        check.unlink()
    record["valid"] = True
    return record


def round_messages(round_id, p):
    initial = [{"role": "system", "content": p["initial_system"]}, {"role": "user", "content": p["initial_user"]}]
    if round_id == 1:
        return initial, None
    return feedback(OUT / "round1", p, initial)


def feedback(rdir, p, messages):
    """Reflection prompt from the previous round, as eureka.py builds it after training."""
    from utils.file_utils import load_tensorboard_logs
    import numpy as np
    successes, contents, used = [], [], []
    exec_success = False
    for cdir in sorted(rdir.glob("cand*")):
        gen = load_json(cdir / "generation.json")
        if gen.get("validation", {}).get("error", "").startswith("Upstream cannot parse"):
            continue  # upstream drops signature failures before training
        if not gen.get("response_text"):
            continue  # no LLM output at all
        used.append(cdir.name)
        log = cdir / "train.log"
        stdout_str = log.read_text() if log.exists() else ""
        if not gen["validation"]["valid"]:
            traceback_msg = gen["validation"]["error"]
        else:
            traceback_msg = filter_traceback(stdout_str)
            if not log.exists() or "Tensorboard Directory:" not in stdout_str:
                traceback_msg = traceback_msg or "Training did not start."
        content = ""
        if traceback_msg == "":
            exec_success = True
            lines = stdout_str.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Tensorboard Directory:"):
                    break
            tensorboard_logs = load_tensorboard_logs(line.split(":")[-1].strip())
            max_iterations = np.array(tensorboard_logs["gt_reward"]).shape[0]
            epoch_freq = max(int(max_iterations // 10), 1)
            content += p["policy_feedback"].format(epoch_freq=epoch_freq)
            for metric in tensorboard_logs:
                if "/" not in metric:
                    metric_cur = ["{:.2f}".format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                    metric_cur_max = max(tensorboard_logs[metric])
                    metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                    if "consecutive_successes" == metric:
                        successes.append(metric_cur_max)
                    metric_cur_min = min(tensorboard_logs[metric])
                    if metric != "gt_reward" and metric != "gpt_reward":
                        metric_name = metric if metric != "consecutive_successes" else "task_score"
                        content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                    elif "consecutive_successes" not in tensorboard_logs:
                        content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
            content += p["code_feedback"]
        else:
            successes.append(DUMMY_FAILURE)
            content += p["execution_error_feedback"].format(traceback_msg=traceback_msg)
        content += p["code_output_tip"]
        contents.append(content)
    selection = {"candidates": used, "successes": successes}
    if not exec_success:
        # Upstream: "All code generation failed! Repeat this iteration from the current message checkpoint!"
        selection["note"] = "All round-1 candidates failed; upstream repeats the iteration with the initial messages."
        return messages, selection
    best = int(np.argmax(np.array(successes)))
    selection.update(best_index=best, best_candidate=used[best], best_task_score=successes[best])
    best_gen = load_json(rdir / used[best] / "generation.json")
    return messages + [{"role": "assistant", "content": best_gen["response_text"]},
                       {"role": "user", "content": contents[best]}], selection


def generate(round_id, n):
    p = prompt_parts()
    rdir = OUT / ("round%d" % round_id)
    rdir.mkdir(parents=True, exist_ok=True)
    messages, selection = round_messages(round_id, p)
    save_json(rdir / "messages.json", messages)
    if selection is not None:
        save_json(rdir / "feedback_selection.json", selection)
    for i in range(n):
        cdir = rdir / ("cand%d" % i)
        if (cdir / "generation.json").exists():
            print(cdir, "already generated; not regenerating")
            continue
        cdir.mkdir(exist_ok=True)
        payload, raw, attempt, errors, seconds = llm(messages)
        save_json(cdir / "request.json", payload)  # body only; no auth headers
        gen = {"model_requested": MODEL, "provider": URL,
               "temperature": TEMPERATURE, "candidate_count": 1,
               "round": round_id, "index": i, "request_attempts": attempt, "api_errors": errors,
               "generation_seconds": seconds, "response_text": None}
        if raw is not None:
            (cdir / "raw_response.json").write_text(raw)
            text, finish, usage, used = parse_response(raw)
            gen.update(model_used=used, usage=usage, finish_reason=finish, response_text=text)
            if text:
                (cdir / "response.txt").write_text(text)
        if gen["response_text"]:
            gen["validation"] = build_candidate(cdir, gen["response_text"], p["obs"])
        else:
            gen["validation"] = {"valid": False, "error": "No response text from %s" % MODEL}
        save_json(cdir / "generation.json", gen)
        print(cdir.name, "valid" if gen["validation"]["valid"] else gen["validation"]["error"], flush=True)


# ---------------------------------------------------------------- training

def run_info(path):
    cmds = [["date", "-u"], ["hostname"], ["git", "-C", str(ROOT / "Eureka"), "rev-parse", "HEAD"],
            ["git", "-C", str(ROOT / "Eureka"), "status", "--short"], ["nvidia-smi"],
            [sys.executable, "--version"], [sys.executable, "-m", "pip", "freeze"]]
    with open(path, "w") as f:
        for cmd in cmds:
            f.write("$ " + " ".join(cmd) + "\n")
            f.flush()
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)


def seeded(path):
    """Seed 0 results live in the original directory; seed N in results/<top>_seedN/<same subpath>."""
    if SEED == 0:
        return path
    parts = path.relative_to(ROOT / "results").parts
    return ROOT / "results" / (parts[0] + "_seed%d" % SEED) / Path(*parts[1:])


def train(name):
    human = name == "human"
    cdir0 = HUMAN if human else OUT / name
    cdir = seeded(cdir0)
    cdir.mkdir(parents=True, exist_ok=True)
    if (cdir / "train.log").exists():
        sys.exit("%s already has train.log; not repeating the run" % cdir)
    if human:
        task, created = "Ant", []
    else:
        gen = load_json(cdir0 / "generation.json")
        if not gen["validation"]["valid"]:
            save_json(cdir / "result.json", {"trained": False, "reason": gen["validation"].get("error")})
            print("Invalid candidate; not trained:", gen["validation"].get("error"))
            return
        base = gen["validation"]["suffix"]
        suffix = base + ("S%d" % SEED if SEED else "")  # unique task name so seeds can run concurrently
        task = "Ant" + suffix
        code = (cdir0 / "env_generated.py").read_text().replace("Ant" + base, task)
        if SEED:
            for f in ("generation.json", "env_generated.py", "env_rewardonly.py"):
                (cdir / f).write_text((cdir0 / f).read_text())
        from utils.create_task import create_task
        create_task(str(ISAAC), "Ant", "ant", suffix)  # upstream: writes cfg/task + cfg/train YAMLs
        created = [ISAAC / f"cfg/task/{task}.yaml", ISAAC / f"cfg/train/{task}PPO.yaml",
                   ISAAC / f"tasks/ant{suffix.lower()}.py"]
        created[2].write_text(code)
    run_info(cdir / "run_info.txt")
    # Upstream flags, plus the explicitly requested seed and environment count.
    cmd = [sys.executable, "-u", str(ISAAC / "train.py"), "hydra/output=subprocess", f"task={task}",
           "wandb_activate=False", "wandb_entity=", "wandb_project=", "headless=True", "capture_video=False",
           "force_render=False", f"max_iterations={ITERATIONS}", f"seed={SEED}", f"num_envs={NUM_ENVS}"]
    started = time.time()
    try:
        with open(cdir / "train.log", "w") as f:
            f.write("Command: " + " ".join(cmd) + "\n")
            f.flush()
            code = subprocess.run(cmd, cwd=str(cdir), stdout=f, stderr=subprocess.STDOUT).returncode
    finally:
        for path in created:  # remove the temporary task files; copies stay in the result directory
            path.unlink(missing_ok=True)
    seconds = time.time() - started
    summarize(cdir, code, seconds)


def summarize(cdir, code=None, seconds=None):
    import numpy as np
    from utils.file_utils import load_tensorboard_logs
    stdout_str = (cdir / "train.log").read_text()
    result = {"trained": True, "exit_code": code, "training_seconds": seconds, "seed": SEED,
              "concurrent_runs_on_gpu": int(os.environ.get("PACKED_RUNS", "1")),
              "num_envs": NUM_ENVS, "max_iterations": ITERATIONS, "traceback": filter_traceback(stdout_str)}
    tb = re.search(r"^Tensorboard Directory: (.*)$", stdout_str, re.M)
    if tb:
        logs = load_tensorboard_logs(tb.group(1).strip())
        curves = {k: v for k, v in logs.items() if "/" not in k}
        save_json(cdir / "curves.json", curves)
        cs = curves.get("consecutive_successes", [])
        result.update(iterations_logged=len(cs), peak_task_score=max(cs) if cs else None,
                      final_task_score=cs[-1] if cs else None,
                      peak_iteration=int(np.argmax(cs)) + 1 if cs else None)
        ckpts = sorted(Path(tb.group(1).strip()).parent.glob("nn/last_*_ep_*.pth"))
        if ckpts:
            import torch
            state = torch.load(str(ckpts[-1]), map_location="cpu")
            result.update(checkpoint=str(ckpts[-1].relative_to(cdir)), checkpoint_epoch=state["epoch"],
                          environment_steps=state["frame"])
    save_json(cdir / "result.json", result)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------- analysis (Experiment C)

def candidates():
    rows = []
    for rdir in sorted(OUT.glob("round*")):
        for cdir in sorted(rdir.glob("cand*")):
            res = cdir / "result.json"
            curve = []
            if (cdir / "curves.json").exists():
                curve = load_json(cdir / "curves.json").get("consecutive_successes", [])
            rows.append({"name": "%s/%s" % (rdir.name, cdir.name), "round": rdir.name,
                         "result": load_json(res) if res.exists() else {"trained": False, "reason": "missing"},
                         "curve": curve})
    return rows


def screen(pool, k):
    """Select by best observed task score through iteration k; never looks past k for the choice."""
    live = [c for c in pool if c["curve"]]
    if not live:
        return None
    early = {c["name"]: max(c["curve"][:k]) for c in live}
    full = {c["name"]: max(c["curve"]) for c in live}
    chosen = max(early, key=early.get)
    winner = max(full, key=full.get)
    full_iters = sum(len(c["curve"]) for c in pool)
    screen_iters = sum(min(k, len(c["curve"])) for c in pool) + max(0, len(
        next(c for c in live if c["name"] == chosen)["curve"]) - k)
    return {"checkpoint": k, "early_scores": early, "selected": chosen, "full_winner": winner,
            "winner_retained": chosen == winner, "winner_rejected_early": chosen != winner,
            "selected_full_peak": full[chosen], "winner_full_peak": full[winner],
            "retained_fraction": full[chosen] / full[winner] if full[winner] > 0 else None,
            "full_iterations": full_iters, "screening_iterations": screen_iters,
            "full_env_steps": full_iters * STEPS_PER_ITER, "screening_env_steps": screen_iters * STEPS_PER_ITER,
            "saved_fraction": 1 - screen_iters / full_iters if full_iters else None}


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SCREEN.mkdir(parents=True, exist_ok=True)
    rows = candidates()
    human = load_json(HUMAN / "curves.json")["consecutive_successes"]
    # Checkpoint table: score at iteration k and best observed through k (actual values only).
    with open(SCREEN / "checkpoint_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "iterations_logged"] + ["score_at_%d" % k for k in CHECKPOINTS]
                   + ["best_through_%d" % k for k in CHECKPOINTS])
        for name, curve in [("human", human)] + [(r["name"], r["curve"]) for r in rows]:
            w.writerow([name, len(curve)] + [curve[k - 1] if len(curve) >= k else "" for k in CHECKPOINTS]
                       + [max(curve[:k]) if len(curve) >= k else "" for k in CHECKPOINTS])
    rounds = sorted({r["round"] for r in rows})
    analysis = {"human": {"peak": max(human), "final": human[-1], "iterations": len(human)},
                "candidates": {r["name"]: {"peak": max(r["curve"]) if r["curve"] else None,
                                           "final": r["curve"][-1] if r["curve"] else None,
                                           "iterations": len(r["curve"]),
                                           "trained": r["result"].get("trained"),
                                           "reason": r["result"].get("reason") or r["result"].get("traceback") or None}
                               for r in rows},
                "screening": {}}
    for k in CHECKPOINTS:
        per_round = {rd: screen([r for r in rows if r["round"] == rd], k) for rd in rounds}
        total_full = sum(v["full_iterations"] for v in per_round.values() if v)
        total_screen = sum(v["screening_iterations"] for v in per_round.values() if v)
        best_full = max((v["winner_full_peak"] for v in per_round.values() if v), default=None)
        best_screen = max((v["selected_full_peak"] for v in per_round.values() if v), default=None)
        analysis["screening"][k] = {
            "label": "primary (predetermined)" if k == 500 else "exploratory",
            "rounds": per_round, "total_full_iterations": total_full, "total_screening_iterations": total_screen,
            "total_full_env_steps": total_full * STEPS_PER_ITER, "total_screening_env_steps": total_screen * STEPS_PER_ITER,
            "saved_fraction": 1 - total_screen / total_full if total_full else None,
            "best_peak_full_search": best_full, "best_peak_screened_search": best_screen,
            "overall_retained_fraction": best_screen / best_full if best_full else None}
    save_json(SCREEN / "screening.json", analysis)

    # Figure 1: learning curves.
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(range(1, len(human) + 1), human, color="#0b0b0b", lw=2, label="Human reward")
    for color, r in zip(colors, rows):
        if r["curve"]:
            style = "-" if r["round"] == "round1" else "--"
            ax.plot(range(1, len(r["curve"]) + 1), r["curve"], color=color, lw=1.6, ls=style,
                    label=r["name"].replace("round", "R").replace("/cand", " c"))
    ax.axvline(500, color="#8a8984", lw=1, ls=":")
    ax.text(510, ax.get_ylim()[0], " screening\n checkpoint", color="#52514e", fontsize=8, va="bottom")
    ax.set(xlabel="PPO iteration", ylabel="Task score (consecutive_successes)",
           title="Ant, seed 0: human reward vs MiniMax M3.1 Flash candidates")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(SCREEN / "fig1_learning_curves.png", dpi=160)
    plt.close(fig)

    # Figure 2: screening checkpoint vs retained performance and compute saved (two panels, one axis each).
    ks = [k for k in CHECKPOINTS if analysis["screening"][k]["saved_fraction"] is not None]
    retained = [100 * analysis["screening"][k]["overall_retained_fraction"] for k in ks]
    saved = [100 * analysis["screening"][k]["saved_fraction"] for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, ys, title, color in [(axes[0], retained, "Best reward retained (% of full search)", "#2a78d6"),
                                 (axes[1], saved, "PPO compute saved (%)", "#1baf7a")]:
        ax.plot(ks, ys, color=color, lw=2, marker="o", ms=6)
        for x, y in zip(ks, ys):
            ax.annotate("%.0f" % y, (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.axvline(500, color="#8a8984", lw=1, ls=":")
        ax.set(xscale="log", xticks=ks, xticklabels=[str(k) for k in ks], xlabel="Screening checkpoint (PPO iterations)",
               title=title, ylim=(0, 110))
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(SCREEN / "fig2_screening.png", dpi=160)
    plt.close(fig)
    print(json.dumps(analysis["screening"][500], indent=2))


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "generate":
        generate(int(sys.argv[2]), int(sys.argv[3]))
    elif command == "train":
        train(sys.argv[2])
    elif command == "summarize":
        summarize(Path(sys.argv[2]))
    elif command == "pack":
        # Independent training processes sharing one GPU (fewer Slurm allocations; same runs).
        env = dict(os.environ, PACKED_RUNS=str(len(sys.argv) - 2))
        procs = [subprocess.Popen([sys.executable, "-u", __file__, "train", n], env=env) for n in sys.argv[2:]]
        sys.exit(max(p.wait() for p in procs))
    else:
        analyze()
