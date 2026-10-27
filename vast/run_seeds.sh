#!/bin/bash
cd ~/MSThesis && . .env/bin/activate && export LIVE_WORKERS=3
for s in 1 2 3; do python -u live.py seeds $s > ~/seeds$s.log 2>&1; done
echo ALL_SEEDS_DONE > ~/seeds_done.flag
