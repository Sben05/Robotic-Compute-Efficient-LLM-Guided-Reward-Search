#!/bin/bash
# Build a fresh Vast A100 box into the exact environment used for all runs, then smoke-test it.
# Usage (from the repo root on the Mac): bash vast/setup_box.sh SSH_WRAPPER user@host
# Steps: apt (python3.8 venv/dev, build-essential, git, rsync, python3-tk, libx11-6) -> copy Isaac Gym Preview 4 tarball,
# upstream Eureka @9eee428, thesis scripts and every reward source dir -> venv with the exact Quest pip freeze -> editable
# installs -> 'live.py smoke' (3 packed 20+10-iteration runs) whose scores must equal the reference values below.
set -euo pipefail
W=$1; H=$2; M=$(pwd); F() { grep -v "Welcome\|Have fun\|Warning: Permanently" || true; }
$W $H 'touch ~/.no_auto_tmux; git config --global --add safe.directory "*"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; mkdir -p ~/MSThesis/results ~/MSThesis/queue
export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq python3.8-venv python3.8-dev python3-pip build-essential git rsync python3-tk libx11-6 > ~/apt.log 2>&1; echo apt ok' 2>&1 | F
rsync -a -e "$W" ~/Downloads/IsaacGym_Preview_4_Package.tar vast/quest_freeze.txt $H:MSThesis/ 2>&1 | F
rsync -a --exclude=__pycache__ -e "$W" Eureka/ $H:MSThesis/Eureka/ 2>&1 | F
rsync -a -e "$W" live.py eureka_ab.py pool.py vast/runner.sh vast/provenance.sh $H:MSThesis/ 2>&1 | F
rsync -a -e "$W" llm_efficiency $H:MSThesis/ 2>&1 | F
(cd results && rsync -a --relative --exclude='*.pth' --exclude='policy-*' --exclude='train*.log' --exclude='curves*.json' --exclude='raw_response.json' \
  -e "$W" exp_g_ablation exp_f_seeds/rewards.json exp_b_minimax exp_d_pool_minimax exp_e_live $H:MSThesis/results/ 2>&1 | F)
$W $H 'cd ~/MSThesis && chmod +x runner.sh provenance.sh && tar xf IsaacGym_Preview_4_Package.tar && python3.8 -m venv .env && . .env/bin/activate \
  && pip install -q pip==24.0 setuptools==75.3.0 wheel==0.45.1 && pip install -q -r quest_freeze.txt \
  && pip install -q --no-deps -e Eureka/rl_games -e Eureka/isaacgymenvs -e Eureka -e isaacgym/python > ~/env.log 2>&1 && echo env ok \
  && LIVE_WORKERS=3 python -u live.py smoke > ~/smoke.log 2>&1; grep -E "^(human|cand0|seed1) " ~/smoke.log | cut -c1-200; rm -rf results/_smoke' 2>&1 | F
echo "reference: human peak 0.4689645767211914, cand0 0.13611412048339844, seed1 0.06449031829833984"
