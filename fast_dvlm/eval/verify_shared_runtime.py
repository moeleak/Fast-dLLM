#!/usr/bin/env python3
"""Verify adapter switching on one persistent Planner backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for value in (_REPO_ROOT, _REPO_ROOT / "third_party" / "sglang" / "python"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from fast_dvlm.gui_finetune.runtime import SharedBackboneEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--planner-prompt", default="Open Settings")
    parser.add_argument("--grounder-prompt", default="Click on Settings.")
    parser.add_argument("--algorithm", choices=("mdm", "spec"), default="mdm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with SharedBackboneEngine(
        args.model_path,
        adapter_path=args.adapter_path,
        processor_path=args.processor_path,
        algorithm=args.algorithm,
    ) as engine:
        identity = engine.backbone_identity
        planner_before = engine.generate(args.image, args.planner_prompt, mode="planner")
        grounder = engine.generate(args.image, args.grounder_prompt, mode="grounder")
        planner_after = engine.generate(args.image, args.planner_prompt, mode="planner")
        if {planner_before["backbone_identity"], grounder["backbone_identity"], planner_after["backbone_identity"]} != {identity}:
            raise RuntimeError("Planner and Grounder did not share one engine object")
        if planner_before["text"] != planner_after["text"]:
            raise RuntimeError("disabling LoRA did not restore the Planner output")
        if planner_before["adapter_enabled"] or not grounder["adapter_enabled"] or planner_after["adapter_enabled"]:
            raise RuntimeError("adapter enable/disable audit is inconsistent")
        result = {
            "schema_version": 1,
            "one_backbone": True,
            "backbone_identity": identity,
            "planner_output_restored": True,
            "planner_before": planner_before,
            "grounder": grounder,
            "planner_after": planner_after,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
