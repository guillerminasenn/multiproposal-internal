#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PY="$SCRIPT_DIR/run.py"

GRID_COUNT=${1:-""}
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

if [[ -z "$GRID_COUNT" ]]; then
  GRID_COUNT=$(RUN_PY="$RUN_PY" python - <<'PY'
import ast
import os
from pathlib import Path

run_py = Path(os.environ["RUN_PY"]).resolve()
tree = ast.parse(run_py.read_text(encoding="utf-8"))

values = {"P_list": None, "rho_list": None}
for node in ast.walk(tree):
  if isinstance(node, ast.Assign) and len(node.targets) == 1:
    target = node.targets[0]
    if isinstance(target, ast.Name) and target.id in values:
      try:
        values[target.id] = ast.literal_eval(node.value)
      except Exception:
        pass

P_list = values.get("P_list") or []
rho_list = values.get("rho_list") or []
grid_size = len(P_list) * len(rho_list)
cpu_count = os.cpu_count() or 1
workers = max(1, min(grid_size or 1, cpu_count))
print(workers)
PY
)
fi

if [[ "$GRID_COUNT" -lt 1 ]]; then
  echo "GRID_COUNT must be >= 1" >&2
  exit 1
fi

echo "Launching $GRID_COUNT workers..."
for i in $(seq 0 $((GRID_COUNT - 1))); do
  LOG_FILE="$LOG_DIR/worker_${i}.log"
  echo "  worker $i -> $LOG_FILE"
  python "$RUN_PY" --grid-count "$GRID_COUNT" --grid-index "$i" > "$LOG_FILE" 2>&1 &
done

echo "Done. Use 'jobs' to see running workers, or 'tail -f logs/worker_0.log' to watch one."
