#!/usr/bin/env bash
# One holder at a time on the card.
#
#     tools/gpu_lock.sh -- python tools/profile_step.py
#     tools/gpu_lock.sh --wait -- python -m pytest -m gpu
#
# The card is shared by several lanes. Two processes timing the same GPU is not two
# measurements, it is one measurement of the contention: the numbers are wrong and
# nothing says so. This is a single flock on a fixed path, which is the smallest thing
# that makes "one holder" true — no daemon, no pid file to go stale, no state.
#
# Without --wait a taken lock refuses in one line and exits 75 (EX_TEMPFAIL), so a
# script that cannot get the card says that instead of blocking a CI job for an hour.

set -euo pipefail

LOCK="${DORMOUSE_GPU_LOCK:-/tmp/opencode/dormouse-gpu.lock}"
WAIT=0
if [ "${1:-}" = "--wait" ]; then
  WAIT=1
  shift
fi
if [ "${1:-}" = "--" ]; then
  shift
fi
if [ "$#" -eq 0 ]; then
  echo "usage: $0 [--wait] -- <command...>" >&2
  exit 64
fi

mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if [ "$WAIT" -eq 1 ]; then
  flock 9
else
  if ! flock -n 9; then
    holder=$(cat "$LOCK.holder" 2>/dev/null || echo "unknown")
    echo "card is held by: $holder (lock $LOCK); rerun with --wait to queue" >&2
    exit 75
  fi
fi
echo "$$ $(date -Is) $*" >"$LOCK.holder"
export DORMOUSE_GPU_LOCK_HELD=$$
trap 'rm -f "$LOCK.holder"' EXIT
"$@"
