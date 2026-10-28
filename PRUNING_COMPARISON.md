# Pruning PPO runs of LLM-written rewards: which stopping rule is best?

Task: Isaac Gym Ant, PPO (`AntPPO`, 4,096 envs, 3,000 iterations). Score = peak `consecutive_successes` over training.
"Retained" = the selected reward's 3,000-iteration peak divided by the best reward's peak in the same candidate set.
"Compute" = PPO iterations spent (screening + finishing the pick) divided by training every candidate to 3,000.

Every number is tagged **[actual]** (training runs that really happened as described), **[held-out]** (frozen rules replayed on
curves never used for tuning) or **[simulation]**. p-values appear only for the pre-registered H1 (STATS_PLAN.md).

## 1. Main result: H1 (pre-registered; conservative seed-level test not significant)

Rules were tuned on 65 training curves (seeds 0-3) and frozen in commit `e868927` before any held-out curve existed. The rule
for H1 was then picked by the pre-registered criterion (`results/pruning/best_rule_criterion.md`) on the interim seeds 4-5:
**VoI (expected improvement), λ = 1.0, first decision at 250 iterations**. H1 is tested on held-out seeds 4, 5, 7, 8 and 9
(5 seeds × all 35 four-reward subsets of the 7 strong LLM rewards = 175 candidate sets) [held-out].

| | Compute | Retained [95% CI] | Picks the true winner |
| --- | ---: | ---: | ---: |
| fixed@500, keep 1 (the current rule) | 0.375 | 0.972 [0.952, 0.991] | 59% |
| fixed@698, keep 1 (same compute as VoI) | 0.425 | 0.981 | 65% |
| **VoI λ=1.0, t0=250** | **0.425** | **0.997 [0.993, 0.999]** | **86%** |

- VoI − fixed@500: **+0.025 retained [0.006, 0.046], one-sided p = 0.0019, Holm-adjusted p = 0.0076** (cluster bootstrap over
  seeds). This original bootstrap p-value is optimistic for the overlapping reward subsets; the conservative seed-level result below governs the write-up.
- VoI − equal-compute fixed checkpoint: +0.015 [0.002, 0.031], p = 0.0099. The gain is not just "spend a bit more".
- Per held-out seed (VoI / fixed@500): seed 4 0.997/0.940, seed 5 0.996/0.994, seed 7 0.991/0.962, seed 8 0.999/0.999,
  seed 9 1.000/0.963. VoI is never worse; fixed@500 loses most on seeds where two rewards are close at iteration 500.

- Clean check (the rule was chosen on seeds 4-5, so seeds 7-9 alone are the untouched test): VoI 0.997 vs fixed@500 0.975,
  +0.022 [0.000, 0.039], p = 0.036 (105 sets). PID ties VoI on these seeds (0.997, winner 83%).
- Closest published method, ORSO (Zhang et al., ICLR 2025: bandit allocation of fixed training chunks, judged on the current
  task score), rebuilt from the paper and added post hoc (tuned on training seeds only, same criterion): D3RB variant 0.990
  retained at 0.475 compute (winner 78%), Exp3 0.980 at 0.617. Paired on the same sets: VoI − ORSO = +0.0067 [0.0014,
  0.0127], p = 0.012, with 5% less compute; PID − ORSO = +0.001 [−0.010, +0.011], with 6% less compute.
  `results/pruning/orso_baseline.json`.

**Conservative re-analysis (added after a reviewer-style critique; `results/pruning/conservative_reanalysis.json`).** The 175
held-out sets are built from the same 7 rewards (each reward appears in 20 of the 35 sets per seed), so they are not 175
independent tests, and a cluster bootstrap over 5 seeds understates uncertainty. Treating each seed as one observation
(one-sample t-test over per-seed mean differences): VoI − fixed@500 per seed = +0.057, +0.002, +0.029, 0.000, +0.037;
**p = 0.041 uncorrected (n = 5 seeds), 0.21 after Holm across H1-H5: not significant under this stricter test.** Dropping any one
reward keeps the difference positive (+0.017 to +0.033); PID − fixed@500 behaves the same (p = 0.030 uncorrected). Reading:
the improvement is consistent (never negative on any seed or reward subset) but statistically modest, and it is established
only for this reward pool on new seeds, not for new rewards.

## 2. Every frozen rule on the same held-out sets [held-out, n = 175 sets, 5 seeds]

| Rule (frozen config) | Compute | Retained [95% CI] | Winner picked | Mean peak vs human (same seed) |
| --- | ---: | ---: | ---: | ---: |
| random pick (baseline) | 0.250 | 0.908 [0.871, 0.941] | 26% | 0.36 below |
| **fixed@500 keep-1 (reference)** | 0.375 | 0.972 [0.952, 0.991] | 59% | 0.33 above |
| PID (P+I+D) th=0.4 | 0.413 | 0.991 [0.979, 0.999] | 78% | 0.54 above |
| MPC / receding horizon λ=1.0 t0=500 | 0.418 | 0.995 [0.991, 0.999] | 83% | 0.59 above |
| **VoI λ=1.0 t0=250** | 0.425 | 0.997 [0.993, 0.999] | 86% | 0.61 above |
| PD th=0.4 | 0.440 | 0.991 [0.979, 0.999] | 78% | 0.54 above |
| Bayesian pruning P(best)<0.1 t0=500 | 0.453 | 0.996 [0.992, 0.999] | 85% | 0.60 above |
| Thompson best-arm (bandit) δ=0.2 | 0.454 | 0.996 [0.992, 0.999] | 87% | 0.60 above |
| successive halving t0=500 η=3 | 0.458 | 0.988 [0.969, 0.999] | 75% | 0.51 above |
| PI th=0.25 | 0.476 | 0.997 [0.993, 1.000] | 87% | 0.61 above |
| fixed@1000 keep-1 | 0.500 | 0.992 [0.985, 0.998] | 81% | 0.56 above |
| ratio-bound racing q=0.75 | 0.535 | 0.997 [0.993, 1.000] | 87% | 0.61 above |
| Hyperband, resumable brackets | 0.547 | 0.992 [0.979, 0.999] | 87% | 0.56 above |
| P th=0.25 | 0.560 | 0.997 [0.993, 1.000] | 87% | 0.61 above |
| human-reference threshold ρ=0.8 | 0.615 | 0.997 [0.993, 1.000] | 87% | 0.61 above |
| Hyperband as in Li et al. (2018) | 3.055 | 1.000 | 100% | 0.64 above |

Reading: every forecasting rule (VoI, MPC, Bayesian, Thompson) reaches ~0.995 retained at 0.42-0.45 compute. Rules
that did **not** clearly beat fixed@500: successive halving (CI overlaps) and random. Paper Hyperband is useless at this
scale: with ≤ 12 candidates every bracket retrains the same few rewards from scratch, costing 3× training everything.
"False-prune rate" (a stopped run that would have ended in the top 2) is ~0.34 for every keep-1 rule by construction (the
2nd-best is always stopped), so it does not discriminate rules here; see §4 for the live false-prune rate.

## 3. Live confirmation of the selected rule [actual]

One fresh round (seed 6, 8 strong rewards) where VoI really controlled the processes: all 8 trained concurrently, each was
paused (SIGSTOP) at the rule's checkpoints, the rule decided from the logged prefixes, dropped runs were killed (SIGKILL).

- Compute: **7,769 of 24,000 iterations (32%)**; 6 runs killed at 250-850 iterations, 1 at 1,551 (not selected).
- The survivor, `r2c1` (round-2 candidate 1 of the original search), trained to 3,000 and peaked at **11.09** (human reward on
  seed 0: 10.48).
- **Replay check: the same rule replayed on the recorded curves made identical decisions** (same pick, same stop iteration
  for all 8 arms; overshoot past a checkpoint ≤ 6 iterations, counted). `results/exp_j_live_prune/seed6/live_prune_result.json`.
- Not measured: whether the killed rewards would have beaten r2c1 on seed 6 (they were not trained further). The held-out
  replay (§1) answers that statistically.

## 4. What fixed-500 actually did in the live search [actual, n = 2 search trials]

- Live early stopping (2 trials, 4 rounds of 4 candidates): **73% of PPO iterations saved** (6,500 vs 24,000 per trial).
- The kept candidate retained 0.83, 1.00, 0.94, 0.92 of each round's best (resumed audit). Top-1 correct in 1 of 4 rounds.
- False prunes (a stopped run that beat the kept one at 3,000): **4 of 12 = 33% [Wilson 95% CI 14%-61%]**; in every case the
  kept run led by < 0.8 points at iteration 500.
- Retrospectively (old 3+2 search) fixed-500 had looked perfect (100% retained); the live trials show that was luck of a small
  sample. The held-out replay (§1, fixed@500 retained 0.972) is the better estimate.

## 5. Reproducibility across seeds [actual]

8 rewards (7 LLM + human) on 9 seeds (0-5, 7-9; seed 6 used live).
- The 500-iteration pick equalled the 3,000-iteration winner on 6 of 9 seeds; mean retained 0.977.
- Seed-0 winners regress: the original headline reward r2c0 (10.93 on seed 0) averages **9.15** on new seeds (winner's curse).
- Ranking strong rewards is mostly seed noise (between-seed Spearman 0.1-0.8), while the gap between a clearly good and a
  clearly bad reward is reproducible (pooled 27-curve Spearman between 500-it and 3,000-it rank: 0.82).
- No "slow starters": no reward ranked in the bottom half at 500 iterations ended in the top quarter (A10, n = 27).

## 6. Transfer to other tasks [held-out task; read with GENERALIZATION_RESULTS.md]

The frozen Ant rules, applied unchanged to Cartpole, BallBalance and Humanoid curves (time rescaled, scores scaled to the
human peak): on Cartpole/BallBalance every rule trivially retains 1.0 because nearly every reward reaches the task's score
ceiling. **On Humanoid the transfer is weaker for VoI (0.93 retained vs 0.97 for fixed@500; corrected from an earlier 0.81, a library-leak
artifact in `exploratory_analysis.py` -- see MASTER_LOG.md)** while PID-family rules (0.97 at 0.32 compute) and successive
halving (0.99) hold up. The VoI forecaster uses an Ant library; it transfers less well to a task with different learning-curve
shapes (n = 11 Humanoid curves, 1 seed: exploratory).

## 7. Caveats

- Seeds ≥ 1 of the strong rewards and all held-out curves come from the 500-iteration-then-resume protocol. Resume control
  (same reward and seed, split vs uninterrupted): peaks differ by −0.18, −0.07, −0.30 on seeds 1-3 (within the ±1 criterion),
  but the pre-registered "no reset" check fails on 1 of 8 seeds (seed 7 dipped to 19.7% of its 500-it score for a few
  iterations; threshold 20%). Every resumed run shows this brief dip and recovers within ~20 iterations; peaks are unaffected.
  Resumed numbers are labelled as such.
- One task, one PPO configuration, candidate sets of 4 from 7 strong rewards; the held-out sets share rewards across seeds.
- Prior work exists for every rule family (successive halving, Hyperband, Bayesian early stopping / learning-curve
  extrapolation, best-arm identification, PID and MPC control); no novelty is claimed for the rules themselves.
