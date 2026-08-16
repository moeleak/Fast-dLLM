#!/usr/bin/env python3
"""Combine the two independent Grounder validation score files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--mind2web", type=Path, required=True)
    parser.add_argument("--mobile", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("mdm", "spec"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = {}
    for name, path in (("mind2web", args.mind2web), ("mobile", args.mobile)):
        value = json.loads(path.read_text(encoding="utf-8"))
        metrics = value["metrics"]
        if int(metrics["num_samples"]) != 100:
            raise RuntimeError(f"{name} validation must contain exactly 100 samples")
        values[name] = value
    result = {
        "schema_version": 1,
        "step": args.step,
        "algorithm": args.algorithm,
        "mind2web_ssr": values["mind2web"]["metrics"]["ssr_point_only"],
        "mobile_ssr": values["mobile"]["metrics"]["ssr_point_only"],
        "mind2web_joint_ssr": values["mind2web"]["metrics"]["joint_step_success"],
        "mobile_joint_ssr": values["mobile"]["metrics"]["joint_step_success"],
        "mind2web_parse": values["mind2web"]["metrics"]["parse_rate"],
        "mobile_parse": values["mobile"]["metrics"]["parse_rate"],
        "mean_latency_seconds": sum(
            float(values[name]["metrics"]["latency_seconds"]["mean"])
            for name in ("mind2web", "mobile")
        ) / 2.0,
        "score_paths": {
            "mind2web": str(args.mind2web.resolve()),
            "mobile": str(args.mobile.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
