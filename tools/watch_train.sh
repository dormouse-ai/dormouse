#!/usr/bin/env bash
# Run one training config until it reaches `kind: end`; resume after a crash.
#
# WHY THIS FILE EXISTS. The last CSA run died at step 2306 with six skipped
# (NaN) steps and nobody was watching: a launch without a monitor is a wasted
# GPU day. This loop is the monitor: it resumes from the last checkpoint, caps
# the number of restarts, and leaves a named line in the log when it gives up.
set -euo pipefail
cd /home/sehaxe/dormouse2/app

config="${1:?config}"
run_dir="${2:?run_dir}"
log="${3:?log}"
max_restarts="${4:-3}"

export PYTHONPATH=src:../dormouse-kernels/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DORMOUSE_FFN_CHECKPOINT=0

attempt=0
while true; do
  if grep -q '"kind": "end"' "$run_dir/run.jsonl" 2>/dev/null; then
    echo "WATCH: end reached" >> "$log"
    exit 0
  fi
  attempt=$((attempt + 1))
  if [ "$attempt" -gt "$max_restarts" ]; then
    echo "WATCH: giving up after $max_restarts restarts" >> "$log"
    exit 2
  fi
  echo "WATCH: attempt $attempt" >> "$log"
  if [ -f "$run_dir/ckpt.pt" ]; then
    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python -m dormouse.cli train "$config" --resume --yes >> "$log" 2>&1 || true
  else
    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python -m dormouse.cli train "$config" --yes >> "$log" 2>&1 || true
  fi
  sleep 20
done
