"""Deterministic data conversion and auditing for GUI fine-tuning.

The training stack consumes LMFlow's ``custom_multi_modal`` JSON format.  The
Planner source is already JSONL plus image paths, while the residual Grounder
sources are Parquet shards with embedded image bytes.  This module converts
both without changing prompts or targets and writes a provenance audit beside
the converted data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PLANNER_EXPECTED_COUNTS = {"train": 13_004, "validation": 1_639, "test": 1_501}
PLANNER_EXPECTED_MANIFEST_SHA256 = (
    "c64bede96c021260438b12a5b252ba693ed87ffa7a97a254db40c8c7cea4ad11"
)
GROUNDER_EXPECTED_COUNTS = {"mind2web": 7_341, "mobile": 8_264}
SUPPORTED_GROUNDING_ACTIONS = ("lclick", "hover", "type_in")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_model_manifest(path: Path) -> dict[str, Any] | None:
    """Hash local model/tokenizer artifacts while excluding optimizer state."""

    if not path.is_dir():
        return None
    suffixes = {".json", ".model", ".safetensors", ".bin", ".txt"}
    files = []
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = candidate.relative_to(path).as_posix()
        if candidate.suffix.lower() not in suffixes or "optimizer" in relative.lower():
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    digest = sha256_values(
        f"{item['path']}\0{item['size_bytes']}\0{item['sha256']}" for item in files
    )
    return {"root": str(path.resolve()), "files": files, "tree_sha256": digest}


def sha256_values(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_sample_key(sample_id: str) -> tuple[str, str]:
    return (hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), sample_id)


def select_planner_validation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 100,
    minimum_per_action: int = 4,
) -> list[dict[str, Any]]:
    """Choose a fixed, action-stratified Planner validation subset.

    Selection depends only on sample IDs and actions, not file order or model
    output.  A small per-action floor protects macro recall; the remainder is
    filled globally by a stable SHA-256 ordering.
    """

    if not 1 <= limit <= len(rows):
        raise ValueError(f"validation limit must be in [1, {len(rows)}]")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    ids: set[str] = set()
    for row in rows:
        sample_id = str(row.get("id", "")).strip()
        action = str(row.get("_gui_balance_key", "")).strip().lower()
        if not sample_id or sample_id in ids or not action:
            raise ValueError("validation rows require unique IDs and actions")
        ids.add(sample_id)
        groups.setdefault(action, []).append(row)
    floor = min(minimum_per_action, limit // len(groups))
    chosen: dict[str, Mapping[str, Any]] = {}
    for action in sorted(groups):
        ordered = sorted(groups[action], key=lambda row: _stable_sample_key(str(row["id"])))
        for row in ordered[: min(floor, len(ordered))]:
            chosen[str(row["id"])] = row
    remainder = sorted(
        (row for row in rows if str(row["id"]) not in chosen),
        key=lambda row: _stable_sample_key(str(row["id"])),
    )
    for row in remainder[: limit - len(chosen)]:
        chosen[str(row["id"])] = row
    selected = sorted(chosen.values(), key=lambda row: _stable_sample_key(str(row["id"])))
    if len(selected) != limit:
        raise RuntimeError(f"selected {len(selected)} validation rows, expected {limit}")
    return [dict(row) for row in selected]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def load_benchmark_rows(
    root: Path,
    benchmark: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and authenticate a benchmark manifest entry.

    Older OCR-aligned benchmark manifests authenticate the JSONL file itself,
    while newer manifests also record the ordered sample-ID digest. Accept
    either evidence format, verify every digest that is present, and always
    return the computed ordered-ID digest for downstream pinning.
    """

    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if benchmark not in manifest.get("benchmarks", {}):
        raise ValueError(f"benchmark {benchmark!r} is absent from {manifest_path}")
    entry = manifest["benchmarks"][benchmark]
    data_path = (root / str(entry["path"])).resolve()
    if data_path != root and root not in data_path.parents:
        raise ValueError(f"benchmark path escapes its root: {data_path}")

    rows = list(iter_jsonl(data_path))
    ids = [str(row.get("sample_id", "")).strip() for row in rows]
    expected_rows = int(entry["rows"])
    if len(rows) != expected_rows:
        raise ValueError(
            f"benchmark row count mismatch: expected {expected_rows}, got {len(rows)}"
        )
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("benchmark sample IDs must be non-empty and unique")

    sample_ids_sha256 = sha256_values(ids)
    expected_ids_sha256 = entry.get("sample_ids_sha256")
    expected_file_sha256 = entry.get("sha256")
    if expected_ids_sha256 is None and expected_file_sha256 is None:
        raise ValueError("benchmark manifest requires a file or sample-ID SHA-256")
    if expected_ids_sha256 is not None and sample_ids_sha256 != expected_ids_sha256:
        raise ValueError("benchmark sample ID hash mismatch")
    data_sha256 = sha256_file(data_path)
    if expected_file_sha256 is not None and data_sha256 != expected_file_sha256:
        raise ValueError("benchmark data file hash mismatch")

    return rows, {
        "root": str(root),
        "benchmark": benchmark,
        "count": len(rows),
        "sample_ids_sha256": sample_ids_sha256,
        "data_path": str(data_path),
        "data_sha256": data_sha256,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"unsupported message content: {type(content).__name__}")
    texts: list[str] = []
    for item in content:
        if isinstance(item, Mapping) and item.get("type") == "text":
            texts.append(str(item.get("text", "")))
    return "\n".join(texts).strip()


def _planner_conversations(row: Mapping[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"planner sample {row.get('id')} has no user/assistant messages")
    user = next((item for item in messages if item.get("role") == "user"), None)
    assistant = next(
        (item for item in reversed(messages) if item.get("role") == "assistant"),
        None,
    )
    if user is None or assistant is None:
        raise ValueError(f"planner sample {row.get('id')} is missing a role")
    prompt = _message_text(user.get("content"))
    target = _message_text(assistant.get("content"))
    if not prompt or not target:
        raise ValueError(f"planner sample {row.get('id')} has an empty turn")
    try:
        parsed = json.loads(target)
    except json.JSONDecodeError as exc:
        raise ValueError(f"planner target is not JSON for {row.get('id')}") from exc
    action = str(parsed.get("action", "")).strip().lower()
    if not action:
        raise ValueError(f"planner target has no action for {row.get('id')}")
    return (
        [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": target},
        ],
        action,
    )


def convert_planner_dataset(
    source_dir: Path,
    image_root: Path,
    output_dir: Path,
    *,
    expected_counts: Mapping[str, int] | None = PLANNER_EXPECTED_COUNTS,
    expected_manifest_sha256: str | None = PLANNER_EXPECTED_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Convert all Planner splits and fail closed on count/hash/leakage drift."""

    source_dir = source_dir.resolve()
    image_root = image_root.resolve()
    manifest_path = source_dir / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        raise ValueError(
            "planner manifest SHA256 mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha}"
        )

    split_ids: dict[str, set[str]] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_audits: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        rows: list[dict[str, Any]] = []
        ids: list[str] = []
        seen_ids: set[str] = set()
        action_counts: Counter[str] = Counter()
        source_path = source_dir / f"{split}.jsonl"
        for source in iter_jsonl(source_path):
            sample_id = str(source.get("id", "")).strip()
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"missing or duplicate planner id in {source_path}: {sample_id}")
            image_value = str(source.get("image", "")).strip()
            image_path = (image_root / image_value).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"planner image does not exist: {image_path}")
            conversations, action = _planner_conversations(source)
            rows.append(
                {
                    "id": sample_id,
                    "image": str(image_path),
                    "conversations": conversations,
                    "_gui_balance_key": action,
                    "_gui_domain": "planner",
                    "_gui_metadata": {
                        "sample_id": sample_id,
                        "split": split,
                        "source_image": image_value,
                    },
                }
            )
            ids.append(sample_id)
            seen_ids.add(sample_id)
            action_counts[action] += 1
        expected = expected_counts.get(split) if expected_counts else None
        if expected is not None and len(rows) != expected:
            raise ValueError(
                f"planner {split} count mismatch: expected {expected}, got {len(rows)}"
            )
        _atomic_json(output_dir / f"{split}.json", rows)
        split_rows[split] = rows
        split_ids[split] = set(ids)
        split_audits[split] = {
            "count": len(rows),
            "sample_ids_sha256": sha256_values(ids),
            "source_sha256": sha256_file(source_path),
            "action_counts": dict(sorted(action_counts.items())),
            "output": str((output_dir / f"{split}.json").resolve()),
        }

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            raise ValueError(f"planner split leakage {left}/{right}: {sorted(overlap)[:5]}")

    validation_rows = select_planner_validation_rows(split_rows["validation"])
    validation_ids = [str(row["id"]) for row in validation_rows]
    validation_actions = Counter(str(row["_gui_balance_key"]) for row in validation_rows)
    validation_path = output_dir / "validation-100.json"
    _atomic_json(validation_path, validation_rows)

    audit = {
        "schema_version": 1,
        "kind": "fast_dvlm_gui_planner",
        "source_dir": str(source_dir),
        "image_root": str(image_root),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha,
        "splits": split_audits,
        "validation_selection": {
            "method": "action_floor_then_sha256_sample_id",
            "minimum_per_action": 4,
            "count": len(validation_rows),
            "sample_ids_sha256": sha256_values(validation_ids),
            "action_counts": dict(sorted(validation_actions.items())),
            "output": str(validation_path.resolve()),
            "source_split": "validation",
        },
        "assistant_only_labels": True,
    }
    _atomic_json(output_dir / "audit.json", audit)
    return audit


def _image_suffix(path_hint: str, payload: bytes) -> str:
    suffix = Path(path_hint).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8"):
        return ".jpg"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("cannot infer embedded image format")


def _safe_image_name(sample_id: str, suffix: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")[:96]
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    return f"{readable}-{digest}{suffix}"


def _iter_parquet_rows(directory: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised on training hosts
        raise RuntimeError("pyarrow is required to convert Grounder Parquet data") from exc
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Parquet shards below {directory}")
    columns = ["sample_id", "source", "image", "conversations", "metadata"]
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                yield path, row


def _grounding_target(conversations: Sequence[Mapping[str, Any]], sample_id: str) -> str:
    if len(conversations) != 2:
        raise ValueError(f"grounder sample {sample_id} must contain exactly two turns")
    if conversations[0].get("from") != "human" or conversations[1].get("from") != "gpt":
        raise ValueError(f"grounder sample {sample_id} has invalid roles")
    target = str(conversations[1].get("value", "")).strip()
    action = target.split(maxsplit=1)[0].lower() if target else ""
    if action not in SUPPORTED_GROUNDING_ACTIONS:
        raise ValueError(f"grounder sample {sample_id} has unsupported target: {target}")
    return action


def convert_grounder_dataset(
    mind2web_dir: Path,
    mobile_dir: Path,
    output_dir: Path,
    *,
    expected_counts: Mapping[str, int] | None = GROUNDER_EXPECTED_COUNTS,
    heldout_benchmarks: Mapping[
        str,
        tuple[Path, str] | tuple[Path, str, int],
    ]
    | None = None,
) -> dict[str, Any]:
    """Extract embedded images and create one two-domain Grounder dataset."""

    output_dir = output_dir.resolve()
    image_root = output_dir / "images"
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    domain_ids: dict[str, list[str]] = {"mind2web": [], "mobile": []}
    action_counts: dict[str, Counter[str]] = {
        "mind2web": Counter(),
        "mobile": Counter(),
    }
    shard_hashes: dict[str, dict[str, str]] = {"mind2web": {}, "mobile": {}}

    for domain, directory in (("mind2web", mind2web_dir), ("mobile", mobile_dir)):
        directory = directory.resolve()
        for shard, source in _iter_parquet_rows(directory):
            shard_key = str(shard)
            if shard_key not in shard_hashes[domain]:
                shard_hashes[domain][shard_key] = sha256_file(shard)
            sample_id = str(source.get("sample_id", "")).strip()
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"missing or duplicate grounder sample id: {sample_id}")
            conversations = source.get("conversations")
            if not isinstance(conversations, list):
                raise ValueError(f"grounder sample {sample_id} has no conversations")
            action = _grounding_target(conversations, sample_id)
            image = source.get("image")
            if not isinstance(image, Mapping) or not isinstance(image.get("bytes"), bytes):
                raise ValueError(f"grounder sample {sample_id} has no embedded image")
            payload = image["bytes"]
            suffix = _image_suffix(str(image.get("path") or ""), payload)
            relative = Path("images") / domain / _safe_image_name(sample_id, suffix)
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.read_bytes() != payload:
                    raise ValueError(f"existing extracted image differs: {destination}")
            else:
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_bytes(payload)
                os.replace(temporary, destination)
            metadata_raw = source.get("metadata")
            metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
            rows.append(
                {
                    "id": sample_id,
                    "image": str(destination),
                    "conversations": conversations,
                    "_gui_balance_key": domain,
                    "_gui_domain": domain,
                    "_gui_metadata": metadata,
                }
            )
            seen_ids.add(sample_id)
            domain_ids[domain].append(sample_id)
            action_counts[domain][action] += 1

    for domain, ids in domain_ids.items():
        expected = expected_counts.get(domain) if expected_counts else None
        if expected is not None and len(ids) != expected:
            raise ValueError(
                f"grounder {domain} count mismatch: expected {expected}, got {len(ids)}"
            )

    heldout_audit: dict[str, Any] = {}
    heldout_ids: dict[str, set[str]] = {}
    for name, specification in (heldout_benchmarks or {}).items():
        root, benchmark = specification[:2]
        limit = specification[2] if len(specification) == 3 else None
        rows, benchmark_audit = load_benchmark_rows(root, benchmark)
        if limit is not None:
            if not 1 <= int(limit) <= len(rows):
                raise ValueError(f"held-out benchmark limit is invalid: {name}={limit}")
            rows = rows[: int(limit)]
            benchmark_audit = {
                **benchmark_audit,
                "manifest_count": benchmark_audit["count"],
                "count": len(rows),
                "selection": "ordered_prefix",
                "selection_limit": int(limit),
                "sample_ids_sha256": sha256_values(str(row["sample_id"]) for row in rows),
            }
        ids = [str(row["sample_id"]) for row in rows]
        overlap = seen_ids & set(ids)
        if overlap:
            raise ValueError(f"Grounder train/{name} leakage: {sorted(overlap)[:5]}")
        heldout_ids[name] = set(ids)
        heldout_audit[name] = benchmark_audit
    names = sorted(heldout_ids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = heldout_ids[left] & heldout_ids[right]
            if overlap:
                raise ValueError(f"Grounder held-out leakage {left}/{right}: {sorted(overlap)[:5]}")

    _atomic_json(output_dir / "train.json", rows)
    audit = {
        "schema_version": 1,
        "kind": "fast_dvlm_gui_grounder",
        "assistant_only_labels": True,
        "sampling": "domain_balanced_with_replacement",
        "image_root": str(image_root),
        "output": str((output_dir / "train.json").resolve()),
        "total_count": len(rows),
        "domains": {
            domain: {
                "count": len(domain_ids[domain]),
                "sample_ids_sha256": sha256_values(domain_ids[domain]),
                "action_counts": dict(sorted(action_counts[domain].items())),
                "source_shards": shard_hashes[domain],
            }
            for domain in ("mind2web", "mobile")
        },
        "heldout_benchmarks": heldout_audit,
    }
    _atomic_json(output_dir / "audit.json", audit)
    return audit


def audit_converted_training_file(path: Path) -> dict[str, Any]:
    """Return a lightweight audit used by launch preflight and unit tests."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"training file must contain a non-empty list: {path}")
    ids: list[str] = []
    seen_ids: set[str] = set()
    balances: Counter[str] = Counter()
    for row in value:
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"missing or duplicate id in {path}: {sample_id}")
        if not Path(str(row.get("image", ""))).is_file():
            raise FileNotFoundError(f"missing image for {sample_id}: {row.get('image')}")
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise ValueError(f"invalid conversations for {sample_id}")
        ids.append(sample_id)
        seen_ids.add(sample_id)
        balances[str(row.get("_gui_balance_key", ""))] += 1
    return {
        "count": len(value),
        "sample_ids_sha256": sha256_values(ids),
        "balance_counts": dict(sorted(balances.items())),
        "file_sha256": sha256_file(path),
    }
