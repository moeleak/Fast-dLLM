"""Planner and GUI-grounding metrics shared by validation and test runners."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


GROUNDING_ACTIONS = ("lclick", "hover", "type_in")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_GROUNDING_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(lclick|hover|type_in)\s*"
    rf"\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]"
    rf"(?:\s+([^\r\n<]*))?",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class GroundingAction:
    action: str | None
    bbox_1000: tuple[float, float, float, float] | None
    value: str
    valid: bool
    error: str | None


def parse_grounding_action(text: Any) -> GroundingAction:
    if not isinstance(text, str) or not text.strip():
        return GroundingAction(None, None, "", False, "empty_prediction")
    match = _GROUNDING_RE.search(text)
    if match is None:
        return GroundingAction(None, None, "", False, "action_or_bbox_not_found")
    action = match.group(1).lower()
    coordinates = tuple(float(value) for value in match.groups()[1:5])
    value = (match.group(6) or "").strip()
    if not all(math.isfinite(item) for item in coordinates):
        return GroundingAction(action, None, value, False, "non_finite_bbox")
    if not all(0.0 <= item <= 1000.0 for item in coordinates):
        return GroundingAction(action, coordinates, value, False, "bbox_out_of_range")
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        return GroundingAction(action, coordinates, value, False, "degenerate_bbox")
    return GroundingAction(action, coordinates, value, True, None)


def _bbox_center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_in_box(point: Sequence[float], box: Sequence[float]) -> bool:
    x, y = (float(value) for value in point)
    x1, y1, x2, y2 = (float(value) for value in box)
    return x1 <= x <= x2 and y1 <= y <= y2


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _f1(targets: Sequence[str], predictions: Sequence[str | None], label: str) -> float:
    tp = sum(t == label and p == label for t, p in zip(targets, predictions))
    fp = sum(t != label and p == label for t, p in zip(targets, predictions))
    fn = sum(t == label and p != label for t, p in zip(targets, predictions))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def score_grounding_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    targets: list[str] = []
    predictions: list[str | None] = []
    point_hits: list[bool] = []
    joint_hits: list[bool] = []
    errors: Counter[str] = Counter()
    latencies: list[float] = []
    for row in rows:
        target_action = str(row["target_action"]).lower()
        if target_action not in GROUNDING_ACTIONS:
            raise ValueError(f"unsupported target action: {target_action}")
        target_box = tuple(float(value) for value in row["target_bbox_1000"])
        if len(target_box) != 4:
            raise ValueError("target_bbox_1000 must contain four coordinates")
        parsed = parse_grounding_action(row.get("prediction"))
        if not parsed.valid:
            errors[str(parsed.error)] += 1
        action_hit = parsed.action == target_action
        point_hit = bool(
            parsed.valid
            and parsed.bbox_1000
            and _point_in_box(_bbox_center(parsed.bbox_1000), target_box)
        )
        targets.append(target_action)
        predictions.append(parsed.action)
        point_hits.append(point_hit)
        joint_hits.append(action_hit and point_hit)
        latency = row.get("latency_seconds")
        if isinstance(latency, (int, float)) and math.isfinite(float(latency)):
            latencies.append(float(latency))

    count = len(rows)
    per_action = {label: _f1(targets, predictions, label) for label in GROUNDING_ACTIONS}
    present = [label for label in GROUNDING_ACTIONS if label in targets]
    correct_actions = sum(t == p for t, p in zip(targets, predictions))
    parsed_count = count - sum(errors.values())
    return {
        "num_samples": count,
        "num_parsed": parsed_count,
        "parse_rate": parsed_count / count if count else 0.0,
        "parse_errors": dict(sorted(errors.items())),
        "ssr_point_only": sum(point_hits) / count if count else 0.0,
        "joint_step_success": sum(joint_hits) / count if count else 0.0,
        "action_accuracy": correct_actions / count if count else 0.0,
        "action_f1_macro_present": (
            statistics.fmean(per_action[label] for label in present) if present else 0.0
        ),
        "action_f1_macro_all": statistics.fmean(per_action.values()),
        "action_f1_per_type": per_action,
        "action_support": dict(Counter(targets)),
        "latency_seconds": {
            "count": len(latencies),
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_planner(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "empty_prediction"
    text = value.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        first, last = text.find("{"), text.rfind("}")
        if first < 0 or last <= first:
            return None, "invalid_json"
        try:
            parsed = json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            return None, "invalid_json"
    if not isinstance(parsed, dict) or not isinstance(parsed.get("action"), str):
        return None, "invalid_schema"
    return parsed, None


def score_planner_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    target_actions: list[str] = []
    predicted_actions: list[str | None] = []
    schema_valid = 0
    content_exact = 0
    parse_errors: Counter[str] = Counter()
    latencies: list[float] = []
    for row in rows:
        target = row.get("target")
        if isinstance(target, str):
            target = json.loads(target)
        if not isinstance(target, Mapping) or not isinstance(target.get("action"), str):
            raise ValueError("planner target must be a JSON object with an action")
        prediction, error = _parse_planner(row.get("prediction"))
        action = str(target["action"]).lower()
        target_actions.append(action)
        if prediction is None:
            predicted_actions.append(None)
            parse_errors[str(error)] += 1
        else:
            schema_valid += 1
            predicted_actions.append(str(prediction["action"]).lower())
            content_exact += _canonical_json(prediction) == _canonical_json(target)
        latency = row.get("latency_seconds")
        if isinstance(latency, (int, float)) and math.isfinite(float(latency)):
            latencies.append(float(latency))

    labels = sorted(set(target_actions))
    recalls = {}
    for label in labels:
        support = sum(value == label for value in target_actions)
        hit = sum(t == label and p == label for t, p in zip(target_actions, predicted_actions))
        recalls[label] = hit / support if support else 0.0
    count = len(rows)
    return {
        "num_samples": count,
        "schema_valid_rate": schema_valid / count if count else 0.0,
        "content_action_exact": content_exact / count if count else 0.0,
        "action_accuracy": (
            sum(t == p for t, p in zip(target_actions, predicted_actions)) / count
            if count
            else 0.0
        ),
        "action_macro_recall": statistics.fmean(recalls.values()) if recalls else 0.0,
        "action_recall_per_type": recalls,
        "action_support": dict(Counter(target_actions)),
        "parse_errors": dict(sorted(parse_errors.items())),
        "latency_seconds": {
            "count": len(latencies),
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
    }


def select_planner_checkpoint(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("no Planner validation rows")
    return max(
        rows,
        key=lambda row: (
            float(row["content_action_exact"]),
            float(row["action_macro_recall"]),
            float(row["schema_valid_rate"]),
            -int(row["step"]),
        ),
    )


def validate_planner_acceptance(row: Mapping[str, Any]) -> None:
    thresholds = {
        "schema_valid_rate": 0.98,
        "content_action_exact": 0.50,
        "action_macro_recall": 0.50,
    }
    failed = [
        f"{name}={float(row[name]):.4f} < {minimum:.4f}"
        for name, minimum in thresholds.items()
        if float(row[name]) < minimum
    ]
    if failed:
        raise RuntimeError("Planner validation acceptance failed: " + "; ".join(failed))


def select_grounder_checkpoint(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("no Grounder validation rows")
    return max(
        rows,
        key=lambda row: (
            min(float(row["mind2web_ssr"]), float(row["mobile_ssr"])),
            (float(row["mind2web_ssr"]) + float(row["mobile_ssr"])) / 2.0,
            min(float(row["mind2web_joint_ssr"]), float(row["mobile_joint_ssr"])),
            -int(row["step"]),
        ),
    )
