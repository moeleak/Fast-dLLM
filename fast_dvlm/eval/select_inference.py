#!/usr/bin/env python3
"""Select MDM versus speculative inference on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.candidate]
    algorithms = {str(row["algorithm"]) for row in rows}
    if algorithms != {"mdm", "spec"}:
        raise RuntimeError(f"expected one MDM and one speculative candidate, got {algorithms}")
    selected = max(
        rows,
        key=lambda row: (
            min(float(row["mind2web_ssr"]), float(row["mobile_ssr"])),
            (float(row["mind2web_ssr"]) + float(row["mobile_ssr"])) / 2.0,
            min(float(row["mind2web_joint_ssr"]), float(row["mobile_joint_ssr"])),
            -float(row["mean_latency_seconds"]),
        ),
    )
    result = {"schema_version": 1, "selected": selected, "candidates": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
