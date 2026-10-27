# Pre-registration: LLM-as-judge from early training logs (exploratory, separate from the H1-H5 Holm family)

Written 2026-10-27 ~12:35 UTC, before any judge call. Approved by the user (~$1 of MiniMax calls).

- Question: can an LLM that reads each candidate's reward code AND its first 500 PPO iterations of logs pick the reward that
  ends best after 3,000 iterations, better than simply keeping the leader at iteration 500 (fixed@500)?
- Data: Ant, held-out seeds 7, 8, 9 (never used for any tuning or rule selection); all 35 four-reward subsets of the 7 strong
  LLM rewards per seed = 105 sets. Candidate order shuffled per set (fixed RNG), neutral labels C1-C4.
- Model: MiniMax M3.1 Flash (the reward writer), Vercel gateway, temperature 1.0, max_tokens 32,768, one call per set, no retries
  beyond transport errors. Prompt: `llm_judge.py` (SYSTEM + task + per-candidate code and logs every 50 iterations).
- Compute: identical to fixed@500 (all four trained to 500, the pick finished). Unparsable reply -> counted as the fixed@500 pick.
- Primary outcome: mean retained fraction (true 3,000-it peak of the pick / best in set), paired against fixed@500 on the same
  sets; one-sided cluster bootstrap over seeds; success = p < 0.05. Secondary: top-1 hit rate; comparison with VoI, PID and
  ORSO from results/pruning/heldout_scores.json and orso_baseline.json (those spend ~0.42-0.48 compute vs 0.375).
- Reported regardless of outcome.
