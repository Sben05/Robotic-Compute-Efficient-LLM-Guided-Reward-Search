"""Turn the real-LLM trial replies (results/llm_efficiency/trial_plan/result_*.json) into candidate directories and 500-iteration
screen jobs. Run on the GPU box (TorchScript validation needs torch):  python llm_efficiency/trial_candidates.py

Batch arms (B, C) ask for four functions compute_reward_1..4 in one reply: each ```python block that defines one becomes its own
candidate (renamed to compute_reward). Independent arms (D, E) give one candidate per reply. Extraction, injection and validation
are upstream Eureka's (eureka_ab.build_candidate); generated code is otherwise never edited.
Writes results/exp_h_llm_trial/<arm>/call<i>[_f<k>]/ and results/exp_h_llm_trial/jobs.json (valid candidates, 500 its, seed 0).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eureka_ab as e  # noqa: E402

IN = e.ROOT / "results/llm_efficiency/trial_plan"
e.OUT = e.ROOT / "results/exp_h_llm_trial"
e.suffix_for = lambda name: "GPTMH" + re.sub(r"[^A-Za-z0-9]", "", name)


def pieces(arm, text):
    if not arm.startswith(("B_", "C_")):
        return [(None, text)]
    out = []
    for block in re.findall(r"```python(.*?)```", text, re.DOTALL):
        m = re.search(r"def (compute_reward_(\d+))\s*\(", block)
        if m:
            out.append((int(m.group(2)), "```python\n" + block.replace(m.group(1) + "(", "compute_reward(", 1) + "\n```"))
    return out


def main():
    p = e.prompt_parts()
    jobs, summary = [], {}
    for f in sorted(IN.glob("result_*.json")):
        arm, i = f.stem[len("result_"):].rsplit("_", 1)
        rec = json.loads(f.read_text())
        raw = rec["response"]
        choice = raw["choices"][0]
        text = choice["message"].get("content") or ""
        parts = pieces(arm, text)
        summary.setdefault(arm, {"calls": 0, "functions": 0, "valid": 0, "completion_tokens": 0})
        summary[arm]["calls"] += 1
        summary[arm]["completion_tokens"] += raw["usage"]["completion_tokens"]
        summary[arm]["functions"] += len(parts)
        for k, piece in parts:
            cdir = e.OUT / arm / (f"call{i}" + (f"_f{k}" if k is not None else ""))
            cdir.mkdir(parents=True, exist_ok=True)
            gen = {"model_used": raw.get("model"), "provider": "vercel ai gateway", "arm": arm, "call": int(i), "function": k,
                   "functions_in_reply": len(parts), "usage": raw["usage"], "finish_reason": choice.get("finish_reason"),
                   "generation_seconds": rec["seconds"], "response_text": piece}
            gen["validation"] = e.build_candidate(cdir, piece, p["obs"]) if piece.strip() else {"valid": False, "error": "empty reply"}
            e.save_json(cdir / "generation.json", gen)
            if gen["validation"]["valid"]:
                summary[arm]["valid"] += 1
                jobs.append([str(cdir), str(cdir), 0, "s1", 500, 3000])
            print(cdir.relative_to(e.OUT), "valid" if gen["validation"]["valid"] else gen["validation"]["error"][:80], flush=True)
    e.save_json(e.OUT / "jobs.json", jobs)
    e.save_json(e.OUT / "build_summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
