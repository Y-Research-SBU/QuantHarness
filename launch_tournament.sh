#!/usr/bin/env bash
# Launch the 3 NEW tournament instances alongside the existing baseline.
# The original baseline whitelist runner (run_continuous.py) is expected
# to already be running — this script does NOT start a baseline.
#
# Logs:   /tmp/quantagent-<profile>.log
# PIDs:   /tmp/quantagent-<profile>.pid
# DBs:    paper_trades_<profile>.db   (per-profile)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
PROFILES=("baseline" "crypto_aggro" "forex_focus" "top25_only" "crypto_kronos_only" "crypto_breakout_vol")

start_profile() {
  local name="$1"
  local log="/tmp/quantagent-${name}.log"
  local pid="/tmp/quantagent-${name}.pid"

  if [[ -f "$pid" ]]; then
    local existing
    existing="$(cat "$pid")"
    if kill -0 "$existing" 2>/dev/null; then
      echo "[skip] $name already running (pid=$existing)"
      return 0
    fi
    rm -f "$pid"
  fi

  echo "[start] $name -> log=$log"
  nohup "$PYTHON" run_instance.py --profile "$name" \
    > "$log" 2>&1 &
  local newpid=$!
  echo "$newpid" > "$pid"
  echo "[ok] $name pid=$newpid"
}

for p in "${PROFILES[@]}"; do
  start_profile "$p"
done

echo
echo "Launched. Tail logs with:"
for p in "${PROFILES[@]}"; do
  echo "  tail -f /tmp/quantagent-${p}.log"
done
