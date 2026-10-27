# Pre-registered: which frozen rule gets the live confirmation (A-5)

Written 2026-10-27 ~04:52 UTC, after freezing (frozen_rules.json, commit e868927) and BEFORE any seed-4/5 curve exists.

1. Score every frozen family config on the held-out worlds (seeds 4-5) with `pruning/heldout.py score`.
2. Eligible: held-out mean compute <= 0.50 of train-all (at least half the PPO compute saved). Paper Hyperband is not eligible.
3. Best = highest held-out mean retained fraction among eligible rules; ties (within 0.002) -> lower compute.
4. If no eligible rule beats fixed@500 keep-1 on retained, the live confirmation runs fixed@500 keep-1 instead and the result
   is reported as "no adaptive rule beat fixed-500 on held-out data".
5. Live run: seed 6, pool of 8 = r2c0, r2c1, p3, t2r2c2, t2r1c3, t2r2c3, t1r2c0, t1r1c2; library = results/pruning/live_lib_pool8.npz
   (TRAIN curves of all other rewards + human seeds 0-3). Labelled "actual run".
