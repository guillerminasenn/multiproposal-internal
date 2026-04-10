#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PY="$SCRIPT_DIR/run.py"
ROOT_DIR="$SCRIPT_DIR/../../.."
ENV_BIN="$ROOT_DIR/.multip-env/bin"

GRID_COUNT=${1:-""}
PCN_COUNT=${2:-100}
MPCN_COUNT=${3:-100}
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

if [[ -z "$GRID_COUNT" ]]; then
  GRID_COUNT=$(
    GRID_HINT=$((PCN_COUNT > MPCN_COUNT ? PCN_COUNT : MPCN_COUNT)) \
    PYTHON_BIN="$ENV_BIN/python" \
    "$ENV_BIN/python" - <<'PY'
import os
hint = int(os.environ.get("GRID_HINT", "1"))
workers = max(1, min(hint, os.cpu_count() or 1))
print(workers)
PY
  )
fi

if [[ "$GRID_COUNT" -lt 1 ]]; then
  echo "GRID_COUNT must be >= 1" >&2
  exit 1
fi

if [[ "$PCN_COUNT" -lt 1 || "$MPCN_COUNT" -lt 1 ]]; then
  echo "PCN_COUNT and MPCN_COUNT must be >= 1" >&2
  exit 1
fi

echo "Launching $GRID_COUNT workers (pCN=$PCN_COUNT, mPCN=$MPCN_COUNT)..."
for i in $(seq 0 $((GRID_COUNT - 1))); do
  LOG_FILE="$LOG_DIR/independent_worker_${i}.log"
  echo "  worker $i -> $LOG_FILE"
  "$ENV_BIN/python" -u "$RUN_PY" \
    --pcn-count "$PCN_COUNT" \
    --mpcn-count "$MPCN_COUNT" \
    --grid-count "$GRID_COUNT" \
    --grid-index "$i" > "$LOG_FILE" 2>&1 &
done

echo "Done. Use 'jobs' to see running workers, or 'tail -f logs/independent_worker_0.log' to watch one."
