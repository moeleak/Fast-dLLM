#!/usr/bin/env python3
"""Prepare deterministic Planner or two-domain Grounder training data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fast_dvlm.gui_finetune.data import (
    GROUNDER_EXPECTED_COUNTS,
    PLANNER_EXPECTED_COUNTS,
    PLANNER_EXPECTED_MANIFEST_SHA256,
    convert_grounder_dataset,
    convert_planner_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    planner = subparsers.add_parser("planner")
    planner.add_argument("--source-dir", type=Path, required=True)
    planner.add_argument("--image-root", type=Path, required=True)
    planner.add_argument("--output-dir", type=Path, required=True)
    planner.add_argument(
        "--expected-manifest-sha256",
        default=PLANNER_EXPECTED_MANIFEST_SHA256,
    )
    planner.add_argument("--allow-count-drift", action="store_true")

    grounder = subparsers.add_parser("grounder")
    grounder.add_argument("--mind2web-dir", type=Path, required=True)
    grounder.add_argument("--mobile-dir", type=Path, required=True)
    grounder.add_argument("--output-dir", type=Path, required=True)
    grounder.add_argument("--allow-count-drift", action="store_true")
    for name in ("mind2web-validation", "mobile-validation", "mind2web-test"):
        grounder.add_argument(f"--{name}-root", type=Path)
        grounder.add_argument(f"--{name}-key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "planner":
        audit = convert_planner_dataset(
            args.source_dir,
            args.image_root,
            args.output_dir,
            expected_counts=None if args.allow_count_drift else PLANNER_EXPECTED_COUNTS,
            expected_manifest_sha256=(
                None if args.expected_manifest_sha256.lower() == "none"
                else args.expected_manifest_sha256
            ),
        )
    else:
        heldout = {}
        for name in ("mind2web_validation", "mobile_validation", "mind2web_test"):
            root, key = getattr(args, f"{name}_root"), getattr(args, f"{name}_key")
            if (root is None) != (key is None):
                raise ValueError(f"--{name.replace('_', '-')}-root/key must be supplied together")
            if root is not None:
                heldout[name] = (root, key)
        audit = convert_grounder_dataset(
            args.mind2web_dir,
            args.mobile_dir,
            args.output_dir,
            expected_counts=None if args.allow_count_drift else GROUNDER_EXPECTED_COUNTS,
            heldout_benchmarks=heldout,
        )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
