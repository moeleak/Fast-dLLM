#!/usr/bin/env python3
"""Run one persistent Fast-dVLM Planner or Grounder evaluation worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIRD_PARTY = _REPO_ROOT / "third_party"
for value in (str(_REPO_ROOT), str(_THIRD_PARTY)):
    if value not in sys.path:
        sys.path.insert(0, value)

from fast_dvlm.gui_finetune.metrics import parse_grounding_action
from fast_dvlm.gui_finetune.runtime import SharedBackboneEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("planner", "grounder"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--processor-path", type=Path)
    parser.add_argument("--dataset", type=Path, help="converted Planner JSON")
    parser.add_argument("--benchmark-root", type=Path, help="Grounder benchmark root")
    parser.add_argument("--benchmark", help="benchmark key from manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--algorithm", choices=("mdm", "spec"), default="mdm")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-sample-ids-sha256")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be in [1, 100]")
    if not 0 <= args.rank < args.world_size:
        parser.error("rank must satisfy 0 <= rank < world-size")
    if args.task == "planner" and not args.dataset:
        parser.error("Planner evaluation requires --dataset")
    if args.task == "grounder":
        if not args.adapter_path:
            parser.error("Grounder evaluation requires --adapter-path")
        if not args.benchmark_root or not args.benchmark:
            parser.error("Grounder evaluation requires --benchmark-root and --benchmark")
    return args


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"expected object at {path}:{line_number}")
            yield value


def _planner_samples(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("Planner dataset root must be a list")
    result = []
    for row in rows:
        conversations = row["conversations"]
        prompt = str(conversations[0]["value"]).replace("<image>", "", 1).lstrip()
        result.append(
            {
                "sample_id": str(row["id"]),
                "image": str(row["image"]),
                "prompt": prompt,
                "target": str(conversations[-1]["value"]),
                "benchmark": "planner_validation",
            }
        )
    return result


def _grounder_samples(root: Path, benchmark: str) -> list[dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if benchmark not in manifest.get("benchmarks", {}):
        raise RuntimeError(f"benchmark is not present: {benchmark}")
    entry = manifest["benchmarks"][benchmark]
    rows = list(_read_jsonl(root / entry["path"]))
    expected = int(entry["rows"])
    if len(rows) != expected:
        raise RuntimeError(f"benchmark row count mismatch: expected {expected}, got {len(rows)}")
    sample_hash = hashlib.sha256(
        "".join(f"{row['sample_id']}\n" for row in rows).encode("utf-8")
    ).hexdigest()
    if sample_hash != entry["sample_ids_sha256"]:
        raise RuntimeError("benchmark sample ID hash mismatch")
    for row in rows:
        row["image"] = str((root / row["image"]).resolve())
    return rows


def _seed_for(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}\0{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def main() -> None:
    args = parse_args()
    if args.task == "planner":
        all_rows = _planner_samples(args.dataset.resolve())
    else:
        all_rows = _grounder_samples(args.benchmark_root.resolve(), args.benchmark)
    selected = all_rows[: args.limit]
    if len(selected) != args.limit:
        raise RuntimeError(f"requested {args.limit} samples, found {len(selected)}")
    selected_hash = hashlib.sha256(
        "".join(f"{row['sample_id']}\n" for row in selected).encode("utf-8")
    ).hexdigest()
    if (
        args.expected_sample_ids_sha256
        and selected_hash != args.expected_sample_ids_sha256
    ):
        raise RuntimeError(
            "ordered sample ID hash mismatch: "
            f"expected {args.expected_sample_ids_sha256}, got {selected_hash}"
        )
    rows = [row for index, row in enumerate(selected) if index % args.world_size == args.rank]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"part-{args.rank:05d}.jsonl"
    config = {
        "schema_version": 1,
        "task": args.task,
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "processor_path": str(args.processor_path.resolve()) if args.processor_path else None,
        "dataset": str(args.dataset.resolve()) if args.dataset else None,
        "benchmark_root": str(args.benchmark_root.resolve()) if args.benchmark_root else None,
        "benchmark": args.benchmark,
        "limit": args.limit,
        "rank": args.rank,
        "world_size": args.world_size,
        "algorithm": args.algorithm,
        "dtype": "bfloat16",
        "latency_scope": "synchronized image preprocessing, prefill, and generation; model load excluded",
        "sample_ids_sha256": selected_hash,
    }
    (output_dir / f"run-config-{args.rank:05d}.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    import torch

    with SharedBackboneEngine(
        args.model_path,
        adapter_path=args.adapter_path,
        processor_path=args.processor_path,
        algorithm=args.algorithm,
        mem_fraction_static=args.mem_fraction_static,
    ) as engine, output_path.open("w", encoding="utf-8") as handle:
        identity = engine.backbone_identity
        for sample in rows:
            sample_id = str(sample["sample_id"])
            try:
                seed = _seed_for(sample_id, args.seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                result = engine.generate(
                    sample["image"],
                    str(sample["prompt"]),
                    mode=args.task,
                    max_new_tokens=args.max_new_tokens,
                )
                if result["backbone_identity"] != identity:
                    raise RuntimeError("shared backbone engine identity changed")
                row = {
                    **sample,
                    "prediction": result["text"],
                    "latency_seconds": result["latency_seconds"],
                    "meta_info": result["meta_info"],
                    "adapter_enabled": result["adapter_enabled"],
                    "backbone_identity": result["backbone_identity"],
                    "inference_seed": seed,
                    "error": None,
                }
                if args.task == "grounder":
                    parsed = parse_grounding_action(result["text"])
                    row.update(
                        {
                            "predicted_action": parsed.action,
                            "predicted_bbox_1000": list(parsed.bbox_1000) if parsed.bbox_1000 else None,
                            "predicted_value": parsed.value,
                            "parse_error": parsed.error,
                        }
                    )
            except BaseException as exc:
                row = {
                    **sample,
                    "prediction": "",
                    "latency_seconds": None,
                    "adapter_enabled": args.task == "grounder",
                    "backbone_identity": identity,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=20),
                }
                if args.fail_fast:
                    raise
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
