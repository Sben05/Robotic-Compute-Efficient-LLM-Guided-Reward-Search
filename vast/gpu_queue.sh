#!/bin/bash
# Sequential GPU queue after the seed stage: G1 ablation -> H LLM-trial screens -> G2 seeds 4,5 if spend allows -> H again (late candidates) -> G3 solo calibration.
# Spend is estimated from wall time: container boot (23:47:21 UTC) minus 10 min for creation, at $0.70/h incl. storage.
cd ~/MSThesis && . .env/bin/activate
START=$((1793058441 - 600)); RATE=0.70
spend() { python3 -c "import time; print(round((time.time()-$START)/3600*$RATE, 2))"; }
below() { python3 -c "import sys; sys.exit(0 if $(spend) + $2 < $1 else 1)"; }  # below CAP NEED
log() { echo "$(date -u +%FT%TZ) spend~\$$(spend) $*" >> ~/queue.log; }
log "queue armed; waiting for seeds_done.flag"
while [ ! -f ~/seeds_done.flag ]; do sleep 15; done
log "seeds done -> G1 start"
LIVE_WORKERS=3 python -u live.py pack results/exp_g_ablation/jobs.json > ~/g1.log 2>&1
log "G1 done"
trial() {  # LLM-trial 500-iteration screens (user-approved); idempotent, runs whatever candidates are ready
  if [ -f results/exp_h_llm_trial/jobs.json ] && below 8 0.2; then
    log "H trial screens start"; LIVE_WORKERS=3 python -u live.py pack results/exp_h_llm_trial/jobs.json >> ~/h.log 2>&1; log "H trial screens done"; fi
}
trial
if below 6 0; then
  for x in 4 5; do
    if below 8 1.1; then log "G2 seeds $x start"; LIVE_WORKERS=3 python -u live.py seeds $x > ~/seeds$x.log 2>&1; log "G2 seeds $x done"
    else log "G2 seeds $x skipped (cap)"; fi
  done
else log "G2 skipped (spend >= 6 at G1 end)"; fi
trial
if below 8 0.05; then log "G3 start"; python -u live.py run results/exp_g_calib/human_solo human 0 s1 500 3000 > ~/g3.log 2>&1; log "G3 done"; fi
touch ~/queue_done.flag; log "QUEUE DONE - GPU idle, instance can be stopped"
