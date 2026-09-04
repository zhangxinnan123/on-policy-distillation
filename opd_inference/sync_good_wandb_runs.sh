#!/bin/bash
# Sync only the offline wandb runs that actually trained, per the inventory produced by
# opd_inference/list_offline_wandb_runs.py.
#
# Keep rule: has_val  OR  last_step >= MIN_STEP.
#   - has_val          -> completed at least one validation pass (covers eval-only runs,
#                         which finish at _step 0)
#   - last_step >= 20  -> real training progress even if it died before the next val
#                         (e.g. the 4B aopd run that reached step 98 of 100)
# Everything else is a failed launch (OOM / EngineDead / config error), which still leaves a
# wandb dir behind but carries no usable data.
#
# Both fields come from files/wandb-summary.json, never from scanning the binary .wandb
# datastore -- see the note in list_offline_wandb_runs.py for why scanning under-reports.

set -u

INVENTORY=${1:-/fsx/xinnanzh/logs/wandb_inventory.json}
WANDB_DIR=${2:-/fsx/xinnanzh/on-policy-distillation/wandb}
MIN_STEP=${MIN_STEP:-20}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opd

unset WANDB_MODE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export WANDB_API_KEY=c4b67c713ad88ef65b62908bcaa8b5c5cb72d1a9
export WANDB_ENTITY=rl_agent

mapfile -t KEEP < <(python - "$INVENTORY" "$MIN_STEP" <<'PYEOF'
import json, sys
rows = json.load(open(sys.argv[1]))
min_step = int(sys.argv[2])
for r in rows:
    step = r.get("last_step")
    try:
        step = int(float(step)) if step is not None else -1
    except (TypeError, ValueError):
        step = -1
    if r.get("has_val") or step >= min_step:
        print(r["dir"])
PYEOF
)

cd "$WANDB_DIR" || exit 1
N=${#KEEP[@]}
echo "syncing $N run(s) (keep rule: val_pts>0 or last_step>=$MIN_STEP)"

ok=0
fail=0
for i in "${!KEEP[@]}"; do
    r=${KEEP[$i]}
    echo "[$((i + 1))/$N] $r"
    if wandb sync "$r" 2>&1 | tail -2 | grep -q "done"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        echo "  ^^ FAILED"
    fi
done

echo "ALL_DONE ok=$ok fail=$fail total=$N"
