#!/bin/bash
# GPU batch runner: after the seed stage, repeatedly runs the lowest-numbered pending batch in ~/MSThesis/queue/*.json
# ({"name", "est_hours", "jobs": [[cdir, src, seed, phase, s1, total], ...]}) via live.py pack, then moves it to queue/done/.
# A batch starts only if estimated spend + est_hours*RATE < CAP. Spend = wall time since rental start at $0.70/h (incl. storage).
cd ~/MSThesis && . .env/bin/activate
START=$((1793058441 - 600)); RATE=0.70; CAP=${CAP:-8.0}
spend() { python3 -c "import time; print(round((time.time()-$START)/3600*$RATE, 2))"; }
log() { echo "$(date -u +%FT%TZ) spend~\$$(spend) $*" >> ~/queue.log; }
mkdir -p queue/done queue/skipped
log "runner armed; waiting for seeds_done.flag"
while [ ! -f ~/seeds_done.flag ]; do sleep 15; done
idle=0
while [ ! -f queue/STOP ]; do
  f=$(ls queue/*.json 2>/dev/null | sort | head -1)
  if [ -z "$f" ]; then idle=$((idle+1)); [ $idle -eq 1 ] && log "IDLE: queue empty"; sleep 30; continue; fi
  idle=0
  est=$(python3 -c "import json; print(json.load(open('$f')).get('est_hours', 1))")
  if ! python3 -c "import sys; sys.exit(0 if $(spend) + $est * $RATE < $CAP else 1)"; then log "SKIP $f (est ${est}h would pass cap \$$CAP)"; mv "$f" queue/skipped/; continue; fi
  python3 -c "import json; json.dump(json.load(open('$f'))['jobs'], open('/tmp/batch.json','w'))"
  n=$(python3 -c "import json; print(len(json.load(open('/tmp/batch.json'))))")
  log "START $f ($n jobs, est ${est}h)"
  t0=$(date +%s)
  LIVE_WORKERS=${LIVE_WORKERS:-3} python -u live.py pack /tmp/batch.json >> ~/batches.log 2>&1
  log "DONE $f in $(( ($(date +%s)-t0)/60 )) min"
  mv "$f" queue/done/
done
log "runner stopped (STOP file)"
