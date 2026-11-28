#!/usr/bin/env bash
# The whole FLOPs-per-byte A/B on one card, in one process chain.
#
# Everything that touches the GPU is here so the lock is held across the full chain:
# no other job can interleave. The combined arm's rank-192 value is REFUSED by the
# fused bank kernel (bank.py:675-680 requires a power-of-two rank >= 16), so it is
# attempted once for the verbatim refusal and otherwise not run; the runnable lever
# topk=1 is paired against the control on three seeds.
set -euo pipefail
cd /home/sehaxe/dormouse2/app

RUNS=runs/ab_top1_r192
CFG=/tmp/opencode/ab_cfg
DATA=/tmp/opencode/ab_data
mkdir -p "$RUNS" "$CFG" "$DATA"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { printf '[%s] %s\n' "$(date +%FT%T)" "$*" | tee -a "$RUNS/chain.log"; }
load() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
trap 'log "LOAD after: $(load)"' EXIT

log "LOAD before: $(load)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

.venv/bin/python scripts/ab_make_data.py "$DATA" | tee -a "$RUNS/chain.log"
# b4 x s1024 with a 10x lower peak rate: the shipped 0.02 Muon rate collapses the
# router past the 4-pass dispatch ceiling within ~20 steps on this corpus, and fp32
# cannot run the fused bank kernel at all (shared memory 151 KiB > 101 KiB). Both
# arms share the lowered rate, so the A/B stays paired; this is recorded as a recipe
# deviation in the verdict. Speed below stays at the specified b8 x s1024 shape.
.venv/bin/python scripts/ab_make_configs.py "$CFG" --batch 4 --seq 1024 \
    --lr-muon 0.002 --lr-adamw 0.0003 | tee -a "$RUNS/chain.log"

for fam in ctrl topk1; do
  for seed in 0 1 2; do
    log "START quality ${fam}_s${seed}"
    .venv-cuda/bin/python scripts/ab_train.py "$CFG/${fam}_s${seed}.toml" \
        --runs "$RUNS" --steps 4000 >> "$RUNS/chain.log" 2>&1
    log "DONE quality ${fam}_s${seed}"
  done
done

for fam in ctrl topk1; do
  for rep in 1 2 3; do
    dest="$RUNS/speed_${fam}_${rep}"
    mkdir -p "$dest"
    log "START speed ${fam}_${rep}"
    cfg="$CFG/${fam}_s0.toml"
    .venv-cuda/bin/python tools/profile_step.py --config "$cfg" \
        --batch 8 --seq 1024 --steps 3 --dtype bf16 > "$dest/profile.txt" 2>&1
    log "DONE speed ${fam}_${rep}"
  done
done

dest="$RUNS/speed_arm_refused"
mkdir -p "$dest"
log "START speed arm (expected refusal: rank 192)"
.venv-cuda/bin/python tools/profile_step.py --config configs/dormouse_1b_top1_r192.toml \
    --batch 8 --seq 1024 --steps 3 --dtype bf16 > "$dest/profile.txt" 2>&1 || true
log "DONE speed arm (refusal captured in $dest/profile.txt)"

log "CHAIN COMPLETE"
