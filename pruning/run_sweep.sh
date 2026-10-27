#!/bin/bash
# Offline pruning comparison sweep (retrospective simulation on actual curves). Usage: PY=python bash run_sweep.sh
PY=${PY:-python3}
for a in "4 30 1.0 0.98" "4 30 0.0 0.98" "4 30 2.0 0.98" "4 30 1.0 0.95" "8 20 1.0 0.98"; do
  set -- $a
  $PY run_pool_cv.py $1 $2 $3 $4 > ../results/pruning/cv_N$1_sig$3_thr$4.txt 2>&1
  echo "done N=$1 sig=$3 thr=$4: $(head -1 ../results/pruning/cv_N$1_sig$3_thr$4.txt)"
done
