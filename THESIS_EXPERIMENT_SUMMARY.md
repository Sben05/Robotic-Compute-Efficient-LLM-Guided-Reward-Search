# Thesis experiments: summary

## In plain words

An LLM writes reward functions for a simulated robot (Isaac Gym Ant); each reward has to be tested by training a robot with
PPO, which is the expensive part. The thesis asks two things:

1. **Can we stop training bad rewards early without losing the good ones?** Yes. A stopping rule that forecasts each run's
   final score from its learning curve (value of information, "VoI") found the best reward 86% of the time using 42% of the
   compute of training everything, against 59% for the simple rule "train everything 500 iterations, keep the leader". This was
   tested on data the rule had never seen, and once more live, where the rule really killed training processes.
2. **Can we spend less on the LLM?** Yes for cost: asking for four rewards in one call costs 2.6-5.5× fewer tokens per usable
   reward on all four tasks tested, and produced the best reward of the whole project. Whether the batched rewards are equally
   good is not settled. One LLM "refinement" turned out to be mostly a single constant (a speed scale) that can be edited
   without any LLM call.

## Pre-registered results (STATS_PLAN.md; one-sided, Holm-corrected across five)

| | Hypothesis | Effect | n | p | Holm p | Verdict |
| --- | --- | --- | --- | ---: | ---: | --- |
| H1 | VoI pruning beats fixed@500 on held-out curves | retained 0.997 vs 0.972 (+0.025 [0.006, 0.046]); winner 86% vs 59%; compute 0.42 vs 0.375 | 175 sets, 5 held-out seeds | 0.0019 | 0.0076 | **significant** |
| H2 | 4-per-call is cheaper per valid reward | 2.7×, 3.7×, 2.6×, 5.5× fewer tokens | 4 tasks, 70 calls | <0.0001 | <0.0001 | **significant** |
| H3 | 4-per-call rewards are no worse (margin 10%) | −0.023 [−0.120, +0.073] | 4 tasks, 112 runs | 0.064 | 0.128 | inconclusive |
| H4 | zero-token edit recovers the refinement gain | 7.57 → 9.04 (+1.47) | 4 vs 4 seeds | 0.025 | 0.075 | not significant after correction |
| H5 | best reward found (p3) beats the human reward | +0.86 points | 6 seeds, paired | 0.073 | 0.128 | not significant |

H1 also holds against a fixed checkpoint of equal compute (+0.015 [0.002, 0.031], p = 0.0099), and it was confirmed in one
live run (seed 6: 32% of the compute, decisions identical to replay).

## Other findings (exploratory, with uncertainty)

- **Live early stopping** saved 73% of PPO iterations in 2 real searches, but 4 of 12 stopped runs would have won
  (33%, CI 14-61%), always when the leader at 500 iterations was ahead by < 0.8 points.
- **Winner's curse is large**: the original search's headline reward (10.93) averages 9.15 on 8 other seeds. The human reward
  ranges 7.9-10.6 across seeds. Single-seed comparisons of strong rewards are mostly noise.
- **Generation strategies** (Ant): strategy hints made rewards worse and 5× costlier; "think briefly" made them much worse.
- **Other LLMs**: Gemini 2.5 Flash and Qwen-3-32B wrote clearly weaker Ant rewards than MiniMax; batching was again cheaper.
- **Generalisation**: Cartpole and BallBalance are too easy (every reward hits the ceiling); on Humanoid all LLM rewards were
  far below the human reward and the Ant-tuned VoI forecaster did not transfer (0.81 retained), while PID-style rules did (0.97).
- **Generate vs train** [simulation]: at measured prices (one LLM call ≈ 60-230 PPO iterations of A100 time) adaptive
  generate-or-train rules do not beat "generate 4, screen at 500" at equal cost.

## Failed or unsupported hypotheses (report these)

H3, H4 (after correction), H5; Hyperband as published (3× the cost of training everything at this scale); successive halving
did not beat fixed@500; transfer of the Ant-tuned VoI rule to Humanoid; strategy hints and short reasoning; no-feedback control
for the round-2 improvement was never run; slow starters (none found).

## Limitations

One LLM family for the main study (MiniMax M3.1 Flash), one PPO configuration, one task with meaningful reward variation
(Ant); 2 live search trials; held-out pruning sets reuse the same 7 rewards across seeds; most seed replicates use the
500-iteration-then-resume protocol, whose strict "no reset" control failed on 1 of 8 seeds even though peaks match the
uninterrupted runs within 0.3.

## Next steps (in order of value per dollar)

1. Re-run H5 and H4 with 4-6 more seeds each (~$3): both are one seed-set away from surviving correction.
2. A hard second task with real reward variation (Humanoid with 5 seeds and an in-task library, or ShadowHand) to test
   whether VoI pruning generalises when the forecaster is built on that task (~$8-10).
3. H3 with ~3× more rewards per arm on Ant to settle non-inferiority (~$4 of GPU, <$0.5 of LLM).
4. More independent live searches (≥ 5) to put a CI on search-level outcomes.

Details: PRUNING_COMPARISON.md, LLM_EFFICIENCY_RESULTS.md, GENERALIZATION_RESULTS.md, EXPERIMENT_MANIFEST.md, STATS_PLAN.md.
