"""Training contracts and pure helpers for the two-stage GUI workflow."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


FAST_DVLM_MODEL = "Efficient-Large-Model/Fast_dVLM_3B"
FAST_DVLM_REVISION = "d7977da26e374f3ef7c96c1700b2ebab50ff62fc"
FAST_DVLM_MASK_TOKEN_ID = 151_665
EXPECTED_TEXT_LAYERS = 36
EXPECTED_LORA_MODULES = EXPECTED_TEXT_LAYERS * 4
EXPECTED_LORA_PARAMETERS = 14_745_600
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

_TEXT_LORA_RE = re.compile(
    r"(?:^|\.)(?:language_model\.(?:model\.)?|model\.)layers\."
    r"(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)(?:\.|$)"
)


@dataclass(frozen=True)
class LearningRates:
    language: float
    connector: float
    vision: float


def processor_artifact_source(model_path: str, tokenizer_name: str | None) -> str:
    """Keep checkpoint weights separate from the shared processor artifacts."""

    return tokenizer_name or model_path


def parameter_role(name: str) -> str:
    lowered = name.lower()
    if "lora_" in lowered:
        return "adapter"
    connector_markers = (
        "visual.merger",
        "vision.merger",
        "multi_modal_projector",
        "mm_projector",
        "connector",
        "language_projection",
    )
    if any(marker in lowered for marker in connector_markers):
        return "connector"
    if any(marker in lowered for marker in (".visual.", ".vision_tower.", ".vision_model.")):
        return "vision"
    return "language"


def learning_rate_for(name: str, rates: LearningRates) -> float:
    role = parameter_role(name)
    if role == "connector":
        return rates.connector
    if role == "vision":
        return rates.vision
    return rates.language


def balance_weights(keys: Sequence[str], power: float) -> list[float]:
    if not 0.0 <= power <= 1.0:
        raise ValueError("balance power must be in [0, 1]")
    counts = Counter(keys)
    if not keys or "" in counts:
        raise ValueError("every training row must have a non-empty balance key")
    return [math.pow(counts[key], -power) for key in keys]


def exact_balanced_epoch_indices(
    keys: Sequence[str],
    *,
    epoch_samples: int,
    seed: int,
) -> list[int]:
    """Draw an exactly balanced, deterministic epoch across all domains."""

    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        if not key:
            raise ValueError("every training row must have a non-empty balance key")
        groups.setdefault(key, []).append(index)
    if not groups or epoch_samples <= 0 or epoch_samples % len(groups):
        raise ValueError("epoch size must be positive and divisible by the domain count")
    per_group = epoch_samples // len(groups)
    generator = random.Random(seed)
    selected: list[int] = []
    for key in sorted(groups):
        source = groups[key]
        domain_indices: list[int] = []
        while len(domain_indices) < per_group:
            cycle = source.copy()
            generator.shuffle(cycle)
            domain_indices.extend(cycle[: per_group - len(domain_indices)])
        selected.extend(domain_indices)
    generator.shuffle(selected)
    return selected


def is_text_lora_parameter(name: str) -> bool:
    return bool(_TEXT_LORA_RE.search(name)) and "lora_" in name.lower()


def lora_module_keys(parameter_names: Iterable[str]) -> set[tuple[int, str]]:
    modules: set[tuple[int, str]] = set()
    for name in parameter_names:
        if "lora_" not in name.lower():
            continue
        match = _TEXT_LORA_RE.search(name)
        if match:
            modules.add((int(match.group(1)), match.group(2)))
    return modules


def expected_epoch_samples(stage: str, source_audit: Mapping[str, Any] | None) -> int | None:
    """Return the exact logical epoch size required by the training recipe."""

    if stage == "planner":
        if source_audit:
            return int(source_audit["splits"]["train"]["count"])
        return 13_004
    if stage == "grounder":
        if source_audit:
            counts = [int(value["count"]) for value in source_audit["domains"].values()]
        else:
            counts = [7_341, 8_264]
        return 2 * max(counts)
    raise ValueError(f"unknown stage: {stage}")


def audit_zero_delta(named_parameters: Iterable[tuple[str, Any]]) -> dict[str, int]:
    """Fail unless every trainable LoRA B matrix starts at exactly zero."""

    matrices = 0
    parameters = 0
    nonzero = 0
    for name, parameter in named_parameters:
        if not parameter.requires_grad or "lora_b" not in name.lower():
            continue
        matrices += 1
        parameters += int(parameter.numel())
        nonzero += int(parameter.detach().count_nonzero().item())
    if matrices == 0:
        raise RuntimeError("Grounder has no trainable LoRA B matrices")
    if nonzero:
        raise RuntimeError(f"Grounder LoRA is not zero-delta: {nonzero} nonzero B values")
    return {"lora_b_matrices": matrices, "lora_b_parameters": parameters, "nonzero": nonzero}


def audit_parameters(named_parameters: Iterable[tuple[str, Any]], stage: str) -> dict[str, Any]:
    if stage not in {"planner", "grounder"}:
        raise ValueError(f"unknown stage: {stage}")
    total = 0
    trainable = 0
    roles: dict[str, dict[str, int]] = {}
    trainable_names: list[str] = []
    all_names: list[str] = []
    for name, parameter in named_parameters:
        count = int(parameter.numel())
        requires_grad = bool(parameter.requires_grad)
        role = parameter_role(name)
        roles.setdefault(role, {"parameters": 0, "trainable_parameters": 0, "tensors": 0})
        roles[role]["parameters"] += count
        roles[role]["tensors"] += 1
        total += count
        all_names.append(name)
        if requires_grad:
            roles[role]["trainable_parameters"] += count
            trainable += count
            trainable_names.append(name)

    if stage == "planner":
        frozen = total - trainable
        if frozen:
            raise RuntimeError(f"Planner must be full-parameter, but {frozen:,} parameters are frozen")
        for required in ("language", "vision", "connector"):
            if roles.get(required, {}).get("trainable_parameters", 0) <= 0:
                raise RuntimeError(f"Planner has no trainable {required} parameters")
        lora_modules: set[tuple[int, str]] = set()
    else:
        invalid = [name for name in trainable_names if not is_text_lora_parameter(name)]
        if invalid:
            raise RuntimeError(
                "Grounder has trainable non-text-LoRA parameters: " + ", ".join(invalid[:8])
            )
        lora_modules = lora_module_keys(all_names)
        expected = {(layer, target) for layer in range(EXPECTED_TEXT_LAYERS) for target in LORA_TARGETS}
        if lora_modules != expected:
            missing = sorted(expected - lora_modules)
            extra = sorted(lora_modules - expected)
            raise RuntimeError(f"Grounder LoRA target mismatch: missing={missing[:8]} extra={extra[:8]}")
        if trainable <= 0:
            raise RuntimeError("Grounder has no trainable adapter parameters")
        if trainable != EXPECTED_LORA_PARAMETERS:
            raise RuntimeError(
                "Grounder trainable parameter mismatch: "
                f"expected {EXPECTED_LORA_PARAMETERS:,}, got {trainable:,}"
            )

    return {
        "schema_version": 1,
        "stage": stage,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "roles": roles,
        "lora_module_count": len(lora_modules),
        "expected_lora_module_count": EXPECTED_LORA_MODULES if stage == "grounder" else 0,
    }


def validate_stage_hyperparameters(stage: str, values: Mapping[str, Any]) -> None:
    expected = {
        "planner": {
            "num_train_epochs": 2.0,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-6,
            "save_steps": 813,
        },
        "grounder": {
            "num_train_epochs": 3.0,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "save_steps": 1033,
            "max_steps": 3099,
        },
    }
    if stage not in expected:
        raise ValueError(f"unknown stage: {stage}")
    mismatches = []
    for key, wanted in expected[stage].items():
        actual = values.get(key)
        if isinstance(wanted, float):
            equal = actual is not None and math.isclose(float(actual), wanted, rel_tol=0, abs_tol=1e-12)
        else:
            equal = actual == wanted
        if not equal:
            mismatches.append(f"{key}={actual!r} (expected {wanted!r})")
    if mismatches:
        raise ValueError(f"{stage} hyperparameter contract mismatch: " + "; ".join(mismatches))
