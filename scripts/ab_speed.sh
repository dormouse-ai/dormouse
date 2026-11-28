#!/usr/bin/env bash
# Speed table for the A/B: three clean runs per arm at b8 x s1024.
# A profile that trips the router-collapse guard is not a measurement, so retry the
# fresh process (up to five times) until three clean transcripts exist. The r192
# value is attempted once to capture the kernel's verbatim rank refusal.
set -euo pipefail
cd /home/sehaxe/dormouse2/app
RUNS=runs/ab_top1_r192
CFG=/tmp/opencode/ab_cfg
mkdir -p "$RUNS" "$CFG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

load() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
echo "LOAD before: $(load)"
.venv/bin/python scripts/ab_make_configs.py "$CFG" --steps 4000 --batch 4 --seq 1024 \
    --lr-muon 0.002 --lr-adamw 0.0003 >/dev/null

for fam in ctrl topk1; do
  ok=0
  for attempt in 1 2 3 4 5; do
    [ "$ok" -ge 3 ] && break
    rep=$((ok + 1))
    dest="$RUNS/speed_${fam}_${rep}"
    mkdir -p "$dest"
    echo "START speed ${fam}_${rep} attempt ${attempt}"
    .venv-cuda/bin/python tools/profile_step.py --config "$CFG/${fam}_s0.toml" \
        --batch 8 --seq 1024 --steps 3 --dtype bf16 > "$dest/profile.txt" 2>&1 || true
    if grep -q "^  total " "$dest/profile.txt"; then
      grep -E "^  total|kernel launches/step" "$dest/profile.txt"
      ok=$((ok + 1))
    else
      echo "  retry: $(grep -oE 'routing needs [0-9]+ dispatch passes|OutOfMemoryError' "$dest/profile.txt" | tail -1)"
    fi
  done
  [ "$ok" -ge 3 ] || { echo "only $ok clean runs for $fam" >&2; exit 1; }
done

dest="$RUNS/speed_arm_refused"
mkdir -p "$dest"
.venv-cuda/bin/python tools/profile_step.py --config configs/dormouse_1b_top1_r192.toml \
    --batch 8 --seq 1024 --steps 3 --dtype bf16 > "$dest/profile.txt" 2>&1 || true
grep -E "power of two|Error" "$dest/profile.txt" | tail -3 || true
echo "LOAD after: $(load)"
