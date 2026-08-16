#!/usr/bin/env python3
"""Create the final fixed-100 quality/latency comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINES = [
    {
        "model": "LLaDA-o 8B full Planner + residual LoRA",
        "planner_validation": "published selected backbone",
        "ocr_ssr_percent": 80.0,
        "joint_ssr_percent": 80.0,
        "action_f1_percent": 100.0,
        "parse_percent": 100.0,
        "mean_seconds": 1.282,
        "reference": "A800 BF16 OCR-aligned test-100",
    },
    {
        "model": "LLaDA-o 8B Q3 edge reference",
        "planner_validation": "published selected backbone",
        "ocr_ssr_percent": 79.0,
        "joint_ssr_percent": 79.0,
        "action_f1_percent": 100.0,
        "parse_percent": 100.0,
        "mean_seconds": 2.734,
        "reference": "Q3 component-exact reference",
    },
]


def directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in {".safetensors", ".bin"}
        and "optimizer" not in item.name.lower()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-selection", type=Path, required=True)
    parser.add_argument("--grounder-selection", type=Path, required=True)
    parser.add_argument("--inference-selection", type=Path, required=True)
    parser.add_argument("--final-score", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--gpu-memory-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    planner = json.loads(args.planner_selection.read_text(encoding="utf-8"))["selected"]
    grounder = json.loads(args.grounder_selection.read_text(encoding="utf-8"))["selected"]
    inference = json.loads(args.inference_selection.read_text(encoding="utf-8"))["selected"]
    score = json.loads(args.final_score.read_text(encoding="utf-8"))
    metrics = score["metrics"]
    memory = json.loads(args.gpu_memory_audit.read_text(encoding="utf-8"))
    mean = float(metrics["latency_seconds"]["mean"])
    candidate = {
        "model": "Fast-dVLM 3B full Planner + residual LoRA",
        "planner_validation": {
            "step": planner["step"],
            "content_action_exact_percent": 100 * float(planner["content_action_exact"]),
            "action_macro_recall_percent": 100 * float(planner["action_macro_recall"]),
            "schema_valid_percent": 100 * float(planner["schema_valid_rate"]),
        },
        "grounder_validation": grounder,
        "algorithm": inference["algorithm"],
        "ocr_ssr_percent": 100 * float(metrics["ssr_point_only"]),
        "joint_ssr_percent": 100 * float(metrics["joint_step_success"]),
        "action_f1_percent": 100 * float(metrics["action_f1_macro_present"]),
        "parse_percent": 100 * float(metrics["parse_rate"]),
        "mean_seconds": mean,
        "p50_seconds": metrics["latency_seconds"]["p50"],
        "p95_seconds": metrics["latency_seconds"]["p95"],
        "backbone_size_bytes": directory_size(args.model_dir),
        "adapter_size_bytes": directory_size(args.adapter_dir),
        "peak_gpu_memory_used_mib": memory["peak_memory_used_mib"],
        "speedup_vs_8b": 1.282 / mean,
        "target_pass": bool(
            metrics["ssr_point_only"] >= 0.78
            and metrics["action_f1_macro_present"] >= 0.99
            and metrics["parse_rate"] >= 0.99
            and mean < 1.282
        ),
        "sample_ids_sha256": score["sample_ids_sha256"],
    }
    result = {"schema_version": 1, "baselines": BASELINES, "candidate": candidate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
