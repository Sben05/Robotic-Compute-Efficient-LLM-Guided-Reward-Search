# Do the Ant findings generalise to other Isaac Gym tasks?

Exploratory (STATS_PLAN.md: no significance claims). Tasks: **Cartpole** (100 PPO its), **BallBalance** (250 its),
**Humanoid** (1,000 its), task-default settings. Rewards written by MiniMax M3.1 Flash with upstream Eureka prompts, one per
call (8 calls; Humanoid 6) and four per call (2 calls each); 28 calls, $0.21. Every valid reward and the human reward trained to
full length, Cartpole/BallBalance on seeds 0 and 1, Humanoid on seed 0 (human on 0 and 1). Tags: **[actual]**,
**[retrospective transfer]**.

## 1. Reward quality [actual]

Score = environment success metric (episode length for Cartpole/BallBalance, forward progress for Humanoid), as a fraction of
the same-seed human reward's peak.

| Task | Human peak (seed 0 / 1) | One per call: runs, mean, best | Four per call: runs, mean, best |
| --- | --- | --- | --- |
| Cartpole | 499 / 499 | 12 runs, 1.00, 1.00 | 16 runs, 1.00, 1.00 |
| BallBalance | 499 / 499 | 16 runs, 0.90, 1.00 | 16 runs, 1.00, 1.00 |
| Humanoid | 7.1 / 5.0 | 4 runs, 0.33, 0.76 | 7 runs, 0.12, 0.37 |

- **Cartpole and BallBalance are at the ceiling**: almost every reward, human or LLM, reaches the maximum episode length
  (~500), so these tasks cannot separate reward quality or test pruning. They only show that LLM rewards are valid and solve
  easy tasks (validity 6/8 and 8/8 one per call, 8/8 and 8/8 four per call).
- **Humanoid is a negative result**: no LLM reward came close to the human reward (best 76% of human). Two of 13 valid
  rewards crashed at run time (a wrong `Tensor` annotation on a float, an out-of-range index), counted as failed candidates.
- Batching stayed cheaper per valid reward on all three tasks (2.6×-5.5×, see LLM_EFFICIENCY_RESULTS.md §1); on Humanoid the
  batched rewards were weaker (n = 7 vs 4, one seed).

## 2. Do the frozen Ant pruning rules transfer? [retrospective transfer]

Rules frozen on Ant, applied unchanged; curves stretched to 3,000 points (fraction of training), scores scaled so the
same-seed human peak = 10.48 (Ant's), library = Ant training curves. Candidate sets of 4 from same-seed runs.

| Rule | Cartpole (800 sets) compute / retained | BallBalance (800) | Humanoid (330) |
| --- | --- | --- | --- |
| fixed@500 keep-1 | 0.375 / 1.000 | 0.375 / 1.000 | 0.375 / 0.972 |
| VoI λ=1.0 t0=250 (Ant winner) | 0.326 / 1.000 | 0.376 / 1.000 | 0.320 / **0.925** |
| PID / PD th=0.4 | 0.347 / 1.000 | 0.349 / 1.000 | 0.322 / 0.972 |
| Thompson best-arm | 0.323 / 1.000 | 0.322 / 1.000 | 0.347 / 0.981 |
| MPC λ=0.5 t0=100 | 0.379 / 1.000 | 0.442 / 1.000 | 0.323 / 0.888 |
| Bayesian P(best)<0.2 | 0.390 / 1.000 | 0.452 / 1.000 | 0.396 / 0.968 |
| successive halving t0=500 η=2 | 0.458 / 1.000 | 0.458 / 1.000 | 0.417 / 0.972 |

Corrected 2026-10-28: the Ant library these rules forecast from was leaking the held-out Ant seeds' human curves (the same bug
as MASTER_LOG.md's H1 fix); fixing it changed VoI from 0.814 to 0.925 and MPC from 0.805 to 0.888 on Humanoid (forecasting
rules read that library directly), while the non-forecasting rows (PID, successive halving) were unaffected; their configs and
Thompson's/Bayesian's shifted slightly because `frozen_rules.json` itself was retuned with the same fix. Cartpole/BallBalance
are untouched (every rule already retains 1.0 there).

- Cartpole/BallBalance: retained is 1.0 for every rule because nearly all rewards tie at the ceiling. Uninformative.
- Humanoid: **the Ant-tuned VoI and MPC forecasters transfer less well (0.93 / 0.89 retained)**, because their library-based
  forecasts assume Ant-shaped learning curves. Simpler lag-based rules (PID family) and successive halving keep 0.97-0.99 at
  0.32-0.46 compute. Exploratory: 11 curves on one seed.

## 3. What this means

- The cost saving from batching generalises (4 of 4 tasks).
- Whether adaptive pruning beats fixed checkpoints off Ant is not established: two tasks are too easy to tell, and on the one
  hard task the Ant-tuned forecaster transfers less well. A forecasting rule does better with a library from the same task.
- Next test (not run): Humanoid with several seeds and a Humanoid library, plus a dexterous task (e.g. ShadowHand) where LLM
  rewards are known to vary widely.
