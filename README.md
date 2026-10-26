# Compute-efficient LLM reward search — Ant

MS CS thesis code reproducing the Ant task of [Eureka](https://github.com/eureka-research/Eureka)
(`Eureka/`, upstream commit `9eee428`, unmodified) and testing early screening of LLM-generated rewards.
Everything runs on Northwestern Quest (A100, Isaac Gym Preview 4); nothing trains on this Mac.

## Files

| File | Purpose |
| --- | --- |
| `run_ant.sh`, `quest_ant.sbatch`, `plot_ant.py` | Experiment 1: original 500-iteration Ant PPO baseline |
| `eureka_ab.py` | 3,000-iteration human baseline; 3+2 Mini-Eureka (upstream prompts, extraction, feedback, selection); screening analysis |
| `pool.py` | 16-candidate first-round pool and retrospective screening rules |
| `rank.py` | Blind LLM pre-ranking of the pool |
| `probe.py` | Zero-training probe: score rewards on scripted/sine-wave gaits |
| `eureka.sbatch` | Generic launcher: `sbatch [opts] eureka.sbatch SCRIPT.py ARGS` |

## Setup

- LLM: MiniMax M3.1 Flash (`minimaxai/minimax-m3.1-flash`) via NVIDIA NIM's free endpoint, temperature 1.0.
  Key in `~/.config/msthesis/nvidia_api_key_minimax` on Quest (mode 600).
- PPO: upstream `AntPPO`, 4,096 envs, seed 0, unchanged hyperparameters; reward comparison uses
  `consecutive_successes` (peak over training), as upstream Eureka does.
- Quest: account `e32695`, partition `gengpu`, `--gres=gpu:a100:1`, env `~/MSThesis/.env`.

## Results layout (`results/`)

- `ant_seed0_*` — Experiment 1 (500 iterations, seed 0): peak task score 8.009, A100, 32,768,000 env steps, 185 s.
- `exp_a_human_3000` — human reward, 3,000 iterations.
- `exp_b_minimax/round{1,2}/cand*` — generated reward, prompts, raw response, validation, training log and curves.
- `exp_d_pool_minimax` — extra 13 first-round samples, LLM ranking, screening analysis.
- `exp_c_screening_minimax` — screening analysis of the 3+2 search; `probe/` — probe scores.

Earlier Gemini / gpt-oss / DeepSeek attempts were removed; on Quest they are archived in
`~/archive_gemini_gptoss_deepseek_20261026.tar.gz`.
