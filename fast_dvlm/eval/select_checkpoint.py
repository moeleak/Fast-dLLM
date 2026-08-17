#!/usr/bin/env python3
"""Select Planner or Grounder checkpoints from validation-only score files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fast_dvlm.gui_finetune.metrics import (
    select_grounder_checkpoint,
    select_planner_checkpoint,
    validate_planner_acceptance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("planner", "grounder"), required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.candidate:
        value = json.loads(path.read_text(encoding="utf-8"))
        metrics = value.get("metrics", value)
        if not isinstance(metrics, dict):
            raise ValueError(f"candidate metrics must be an object: {path}")
        candidate = {**value, **metrics, "score_path": str(path.resolve())}
        rows.append(candidate)
    selected = (
        select_planner_checkpoint(rows)
        if args.stage == "planner"
        else select_grounder_checkpoint(rows)
    )
    acceptance_error = None
    if args.stage == "planner":
        try:
            validate_planner_acceptance(selected)
        except RuntimeError as exc:
            acceptance_error = str(exc)
    result = {
        "schema_version": 1,
        "stage": args.stage,
        "selected": selected,
        "candidates": rows,
        "acceptance": {
            "passed": acceptance_error is None,
            "error": acceptance_error,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if acceptance_error is not None:
        raise RuntimeError(acceptance_error)


if __name__ == "__main__":
    main()
