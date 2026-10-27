#!/bin/bash
# Provenance snapshot of the Vast GPU box (run on the box): exact code hashes, environment, hardware, queue state.
# Writes ~/MSThesis/provenance/<UTC>/. Environment variables are filtered so no credential is ever recorded.
cd ~/MSThesis
D=provenance/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $D
sha256sum live.py eureka_ab.py pool.py runner.sh run_seeds.sh gpu_queue.sh IsaacGym_Preview_4_Package.tar quest_freeze.txt 2>/dev/null > $D/code_sha256.txt
(cd Eureka && git rev-parse HEAD && git status --short && git diff --stat) > $D/eureka_git.txt 2>&1
find Eureka isaacgym/python -name "*.py" -o -name "*.yaml" | sort | xargs sha256sum > $D/upstream_tree_sha256.txt
.env/bin/python -m pip freeze > $D/pip_freeze.txt; .env/bin/python --version > $D/python.txt 2>&1
nvidia-smi -q > $D/nvidia_smi_q.txt; nvidia-smi > $D/nvidia_smi.txt
{ uname -a; cat /etc/os-release; nproc; free -g; df -h /; lscpu | head -20; } > $D/system.txt 2>&1
env | grep -viE "key|token|secret|pass|auth|cred" | sort > $D/env_filtered.txt
cp ~/queue.log $D/ 2>/dev/null; ls -la queue queue/done queue/skipped > $D/queue_state.txt 2>&1
cp -r queue $D/queue_snapshot
find results -name "train*.log" -newer IsaacGym_Preview_4_Package.tar | sort | xargs sha256sum > $D/raw_logs_sha256.txt
find results -name "*.tfevents.*" | sort | xargs sha256sum > $D/tfevents_sha256.txt
find results -name "config.yaml" -path "*policy-*" | sort | xargs sha256sum > $D/hydra_config_sha256.txt
echo $D
