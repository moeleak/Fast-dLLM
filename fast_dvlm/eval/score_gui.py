#!/usr/bin/env python3
"""Merge Fast-dVLM GUI prediction shards and write validation/test metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fast_dvlm.gui_finetune.metrics import score_grounding_records, score_planner_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("planner", "grounder"), required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=100)
    parser.add_argument("--step", type=int)
    parser.add_argument("--algorithm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.expected_samples <= 100:
        raise ValueError("--expected-samples must be in [1, 100]")
    rows = []
    seen = set()
    paths = sorted(args.predictions_dir.glob("part-*.jsonl"))
    if not paths:
        raise FileNotFoundError("no prediction shards found")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                if sample_id in seen:
                    raise RuntimeError(f"duplicate sample {sample_id} in {path}:{line_number}")
                seen.add(sample_id)
                rows.append(row)
    if len(rows) != args.expected_samples:
        raise RuntimeError(f"expected {args.expected_samples} predictions, got {len(rows)}")
    runtime_errors = [row for row in rows if row.get("error")]
    if runtime_errors:
        raise RuntimeError(f"{len(runtime_errors)} runtime errors; first={runtime_errors[0]['error']}")
    metrics = (
        score_planner_records(rows)
        if args.task == "planner"
        else score_grounding_records(rows)
    )
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(
        args.predictions_dir.glob("run-config-*.json")
    )]
    if not configs:
        raise RuntimeError("run configs are missing")
    sample_hashes = {config["sample_ids_sha256"] for config in configs}
    if len(sample_hashes) != 1:
        raise RuntimeError("shards used different ordered sample sets")
    result = {
        "schema_version": 1,
        "task": args.task,
        "metrics": metrics,
        "step": args.step,
        "algorithm": args.algorithm,
        "runtime_errors": 0,
        "sample_ids_sha256": next(iter(sample_hashes)),
        "prediction_ids_sha256": hashlib.sha256(
            "".join(f"{row['sample_id']}\n" for row in sorted(rows, key=lambda row: row["sample_id"])).encode(
                "utf-8"
            )
        ).hexdigest(),
        "run_configs": configs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
