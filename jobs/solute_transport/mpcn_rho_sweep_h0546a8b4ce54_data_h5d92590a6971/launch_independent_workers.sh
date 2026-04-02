#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PY="$SCRIPT_DIR/run.py"

GRID_COUNT=${1:-""}
INDEPENDENT_COUNT=${2:-500}
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

if [[ -z "$GRID_COUNT" ]]; then
  GRID_COUNT=$(RUN_PY="$RUN_PY" python - <<'PY'
import ast
import os
from pathlib import Path

run_py = Path(os.environ["RUN_PY"]).resolve()
tree = ast.parse(run_py.read_text(encoding="utf-8"))

rho_list = None
for node in ast.walk(tree):
  if isinstance(node, ast.Assign) and len(node.targets) == 1:
    target = node.targets[0]
    if isinstance(target, ast.Name) and target.id == "rho_list":
      try:
        rho_list = ast.literal_eval(node.value)
      except Exception:
        pass

rho_list = rho_list or []
workers = max(1, min(len(rho_list) or 1, os.cpu_count() or 1))
print(workers)
PY
)
fi

if [[ "$GRID_COUNT" -lt 1 ]]; then
  echo "GRID_COUNT must be >= 1" >&2
  exit 1
fi

if [[ "$INDEPENDENT_COUNT" -lt 1 ]]; then
  echo "INDEPENDENT_COUNT must be >= 1" >&2
  exit 1
fi

echo "Launching $GRID_COUNT independent-chain workers (P=$INDEPENDENT_COUNT)..."
for i in $(seq 0 $((GRID_COUNT - 1))); do
  LOG_FILE="$LOG_DIR/independent_worker_${i}.log"
  echo "  worker $i -> $LOG_FILE"
  python "$RUN_PY" \
    --skip-mpcn \
    --skip-pcn \
    --skip-mess \
    --independent-pcn-count "$INDEPENDENT_COUNT" \
    --grid-count "$GRID_COUNT" \
    --grid-index "$i" > "$LOG_FILE" 2>&1 &
done

echo "Done. Use 'jobs' to see running workers, or 'tail -f logs/independent_worker_0.log' to watch one."
