#!/bin/bash
# Incremental, append-only backup Vast -> Mac, verified by sha256.  Usage: bash vast/backup.sh   (run from the repo root)
# - results/: everything except *.pth (raw train logs, tfevents, Hydra config.yaml, curves, results, reward code); never --delete
# - provenance/: the box's provenance snapshots
# - ~/MSThesis_checkpoints/: final + stop checkpoints (last_*.pth) only, outside git
# Verification: the box hashes every settled file (unchanged for 5+ min) in run dirs that have a result*.json; the Mac re-hashes
# its copies and compares. Files still being written by a running phase are picked up by a later backup.
set -euo pipefail
SD=/private/tmp/claude-501/-Users-shreeniketbendre-Desktop-MSThesis/7c0491b1-a48c-4290-8b6e-d1aeecc9838a/scratchpad
mkdir -p backups
for BOX in "1 $SD/vssh.sh root@66.160.146.100" "2 $SD/vssh2.sh root@36.212.175.216"; do
set -- $BOX; B=$1; SSH=$2; H=$3
TS=$(date -u +%Y%m%dT%H%M%SZ)_box$B
$SSH $H 'cd ~/MSThesis && for d in $(find results -name "result*.json" -printf "%h\n" | sort -u); do find "$d" -type f -mmin +5 ! -name "*.pth" ! -path "*/__pycache__/*"; done | sort | xargs -r sha256sum' 2>/dev/null | grep -v "Welcome\|Have fun" > backups/manifest_$TS.sha256
rsync -a --exclude='*.pth' --exclude='__pycache__' -e "$SSH" $H:MSThesis/results/ results/ 2>&1 | grep -v "Welcome\|Have fun" || true
rsync -a -e "$SSH" $H:MSThesis/provenance/ provenance/box$B/ 2>&1 | grep -v "Welcome\|Have fun" || true
mkdir -p ~/MSThesis_checkpoints
rsync -a --prune-empty-dirs --include='*/' --include='last_*.pth' --exclude='*' -e "$SSH" $H:MSThesis/results/ ~/MSThesis_checkpoints/results/ 2>&1 | grep -v "Welcome\|Have fun" || true
n=$(wc -l < backups/manifest_$TS.sha256 | tr -d ' ')
bad=$(shasum -a 256 -c backups/manifest_$TS.sha256 2>/dev/null | grep -vc ': OK$' || true)
echo "$TS backup: $n files of finished runs verified, $bad mismatches; checkpoints: $(find ~/MSThesis_checkpoints -name '*.pth' | wc -l | tr -d ' ')" | tee -a backups/backup.log
done
