from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _stable_hash(payload: Dict[str, Any], length: int = 12) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def format_float_tag(value: float, precision: int = 5) -> str:
    return f"{float(value):.{precision}f}".replace(".", "p")


def build_run_dirs(
    repo_root: Path,
    dataset: str,
    algorithm: str,
    data_config: Dict[str, Any],
    algorithm_config: Dict[str, Any],
    algorithm_dir: Optional[str] = None,
    sweep_config: Optional[Dict[str, Any]] = None,
    tag_parts: Optional[Iterable[str]] = None,
) -> Tuple[Path, Path, str, Dict[str, Any]]:
    run_config: Dict[str, Any] = {
        "dataset": dataset,
        "algorithm": algorithm,
        "data": data_config,
        "algorithm_config": algorithm_config,
        "sweep": sweep_config,
    }
    run_hash = _stable_hash(run_config)
    name_parts = [dataset, algorithm]
    if tag_parts:
        name_parts.extend(tag_parts)
    run_name = "_".join(name_parts) + f"_h{run_hash}"

    algo_dir = algorithm if algorithm_dir is None else algorithm_dir
    if algo_dir:
        estimations_dir = repo_root / "estimations" / dataset / algo_dir / run_name
        reports_dir = repo_root / "reports" / dataset / algo_dir / run_name
    else:
        estimations_dir = repo_root / "estimations" / dataset / run_name
        reports_dir = repo_root / "reports" / dataset / run_name
    estimations_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config_path = estimations_dir / "config.json"
    if not config_path.exists():
        payload = dict(run_config)
        payload["run_name"] = run_name
        payload["run_hash"] = run_hash
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return estimations_dir, reports_dir, run_name, run_config
