# Experiment manifest

Every experiment, what kind of evidence it is, how much was run, and where its raw data and code are. Hardware: Northwestern
Quest A100 (exp_a-exp_e, up to 26 Oct 2026 ~19:00 CDT), then rented Vast.ai A100-PCIE-40GB boxes (box 1, box 2; box 3 was
rented and shut off without training). Software stack identical on all machines (Python 3.8, torch 2.0.0+cu117, Isaac Gym
Preview 4, Eureka @9eee428); 20+10-iteration smoke runs gave identical scores to the last digit on Quest, box 1 and box 2.

| ID | What | Evidence type | Runs (reward × seed) | Results | Code |
| --- | --- | --- | --- | --- | --- |
| exp_a | Human reward, 3,000 its, seeds 0-1 | actual | 2 | results/exp_a_human_3000* | eureka_ab.py |
| exp_b | Mini-Eureka 3+2 search (MiniMax), full training | actual | 5 | results/exp_b_minimax | eureka_ab.py |
| exp_c | Retrospective screening of exp_b | retrospective | - | results/exp_c_screening_minimax | eureka_ab.py |
| exp_d | Extra round-1 pool + LLM pre-ranking + probe | actual / retrospective | 6 | results/exp_d_pool_minimax | pool.py, rank.py, probe.py |
| exp_e | Live early stopping: 2 trials × 2 rounds × 4 rewards, 500-it stops, round 2 from 500-it feedback; audit resumes | actual | 16 (+16 audit) | results/exp_e_live | live.py, analyze_live.py |
| exp_f | 8 rewards (7 LLM + human) on seeds 1-5, 7-9 (500 + resume); seed-1 replication of all 10 viable live-round candidates | actual | 74 | results/exp_f_seeds | live.py, analyze_seeds.py |
| exp_g | Desaturation ablation (straight 3,000; r1c1 τ=4/10 seeds 0-3, τ=6 and r2c0 τ=4 seeds 0-1); resume control (straight human seeds 2-3); solo calibration | actual | 11 + 2 + 1 | results/exp_g_ablation, exp_g_calib | live.py |
| exp_h | LLM-trial arms B/C/D/E (22 calls), 500-it screens on seeds 0 and 1; top 6 resumed to 3,000 | actual | 62 + 6 | results/exp_h_llm_trial*, results/llm_efficiency/trial_plan | llm_efficiency/llm_trial_plan.py, trial_candidates.py |
| exp_i | Generalisation: Cartpole, BallBalance, Humanoid (28 calls), full training | actual | 79 (2 crashed) | results/exp_i_generalize | gen_task.py, live_task.py |
| exp_j | Live pruning confirmation, VoI, seed 6, 8 arms | actual | 8 (7 killed) | results/exp_j_live_prune | live_prune.py |
| exp_k | Other LLMs (Gemini 2.5 Flash, Qwen-3-32B) on Ant, 20 calls, 500-it screens, top 4 each resumed | actual | 24 + 8 | results/exp_k_llm_robustness | gen_task.py, live_task.py |
| pruning | Rule comparison: tuning on seeds 0-3 (frozen), held-out scoring on seeds 4,5,7,8,9; live-round replay | retrospective held-out | 175 held-out sets | results/pruning | pruning/heldout.py, pruning/run_pool_cv.py |
| final | Pre-registered H1-H5 + exploratory E1-E5 | analysis | - | results/final | final_analysis.py, exploratory_analysis.py |

Totals (results/LEDGER.json): 384 completed training phases (280 from scratch, 104 resumed), 451,000 PPO iterations
(3.02 × 10^10 environment steps, using each task's environment count and PPO horizon), 2 failed phases (both LLM-reward bugs on Humanoid). The separate live-pruning run is not included in the ledger because it has no `result.json` phases. LLM calls tonight: 70 ($0.69).
Vast spend: ~$17.5 estimated from wall time (credit $21).

Pre-registration and provenance:
- `STATS_PLAN.md` (commit 4f7c0a1, amendment 0a48a11, substitution 4bd08b2), `results/pruning/frozen_rules.json` (e868927),
  `results/pruning/best_rule_criterion.md` (b9c1fc2): all committed before the data they govern existed.
- Every run directory keeps the exact reward code (`env_generated.py`, `env_rewardonly.py`), the LLM request/response,
  `train.log` (first line = full command incl. seed), Hydra `config.yaml`, tensorboard events and `run_info.txt`
  (pip freeze, GPU, driver). `provenance/` holds per-box snapshots (code hashes, upstream-tree hashes, pip freeze, nvidia-smi,
  queue state); `backups/` holds the sha256 manifests used to verify every copy from the boxes (0 mismatches on settled files).
- Final checkpoints (`last_*.pth`) are kept outside git in `~/MSThesis_checkpoints/`.

Known deviations: box 3 rented and shut off without training (stalled package install); the planned extra seeds for borderline candidates were dropped for 2 extra desaturation seeds (same cost, more power for H4); the second box's direct SSH port
failed mid-night (reached through Vast's proxy, no data lost); DeepSeek/Qwen-3.8 replaced by Gemini/Qwen-3-32B (free-tier
access); one Humanoid run per arm crashed because of the LLM's code; the resume control's strict dip criterion failed on 1 of 8
seeds (see PRUNING_COMPARISON.md §7).
