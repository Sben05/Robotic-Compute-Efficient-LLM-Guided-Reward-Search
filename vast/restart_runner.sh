#!/bin/bash
# Restart a box's batch runner (safe mid-batch: runner v2 waits for any running 'live.py pack').
# Usage: bash vast/restart_runner.sh SSH_WRAPPER user@host WAIT_SEEDS CAP "log note"
W=$1; H=$2; WAIT=$3; CAP=$4; NOTE=$5
$W $H "cd ~/MSThesis && for p in \$(ps -eo pid,args | grep '[/]bin/bash ./runner.sh' | awk '{print \$1}'); do kill \$p; done; echo \"\$(date -u +%FT%TZ) runner restarted: CAP=$CAP WAIT_SEEDS=$WAIT ($NOTE)\" >> ~/queue.log" 2>&1 | grep -v "Welcome\|Have fun"
$W $H "cd ~/MSThesis && setsid -f bash -c 'WAIT_SEEDS=$WAIT CAP=$CAP LIVE_WORKERS=3 ./runner.sh > ~/runner.out 2>&1' </dev/null >/dev/null 2>&1; sleep 3; tail -1 ~/queue.log; ls queue | tr '\n' ' '; echo; echo runners=\$(ps -eo args | grep -cE '^/bin/bash ./runner.sh') packs=\$(ps -eo args | grep -c '[l]ive.py pack')" 2>&1 | grep -v "Welcome\|Have fun"
