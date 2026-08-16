"""Training contracts and pure helpers for the two-stage GUI workflow."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FAST_DVLM_MODEL = "Efficient-Large-Model/Fast_dVLM_3B"
FAST_DVLM_REVISION = "d7977da26e374f3ef7c96c1700b2ebab50ff62fc"
FAST_DVLM_MASK_TOKEN_ID = 151_665
EXPECTED_TEXT_LAYERS = 36
EXPECTED_LORA_MODULES = EXPECTED_TEXT_LAYERS * 4
EXPECTED_LORA_PARAMETERS = 14_745_600
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
EXPECTED_OPTIMIZER_STEPS = {"planner": 1_626, "grounder": 3_099}
RESUME_MAX_ABS_TOLERANCE = 5e-6
RESUME_RELATIVE_L2_TOLERANCE = 5e-7
RESUME_MISMATCHED_FRACTION_TOLERANCE = 1e-4

_TEXT_LORA_RE = re.compile(
    r"(?:^|\.)(?:language_model\.(?:model\.)?|model\.)layers\."
    r"(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)(?:\.|$)"
)


@dataclass(frozen=True)
class LearningRates:
    language: float
    connector: float
    vision: float


class GradientRoleTracker:
    """Record first-step gradients at backward-hook time under ZeRO."""

    def __init__(self, stage: str):
        self.stage = stage
        self.roles = {"language": 0, "connector": 0, "vision": 0, "adapter": 0}
        self.invalid: list[str] = []
        self.seen: set[str] = set()
        self.handles: list[Any] = []

    def attach(self, named_parameters: Iterable[tuple[str, Any]]) -> None:
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue

            def record(gradient, *, parameter_name=name, value=parameter):
                if gradient is None or parameter_name in self.seen:
                    return None
                self.seen.add(parameter_name)
                role = parameter_role(parameter_name)
                self.roles[role] = self.roles.get(role, 0) + int(value.numel())
                if self.stage == "grounder" and role != "adapter":
                    self.invalid.append(parameter_name)
                return None

            self.handles.append(parameter.register_hook(record))

    def audit(self, step: int) -> dict[str, Any]:
        if self.invalid:
            raise RuntimeError(
                "Grounder backbone received gradients: " + ", ".join(self.invalid[:8])
            )
        required = (
            ("language", "connector", "vision")
            if self.stage == "planner"
            else ("adapter",)
        )
        missing = [role for role in required if self.roles.get(role, 0) == 0]
        if missing:
            raise RuntimeError(f"no first-step gradients for parameter roles: {missing}")
        return {
            "step": int(step),
            "gradient_parameters_by_role": dict(self.roles),
            "gradient_parameter_tensors": len(self.seen),
            "capture": "backward_hooks_before_zero_partition",
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def compare_saved_model_weights(left_root: Path, right_root: Path) -> dict[str, Any]:
    """Compare resumed weights without requiring nondeterministic GPU bit identity."""

    import torch
    from safetensors import safe_open

    def model_files(root: Path) -> list[Path]:
        candidates = sorted(root.glob("*.safetensors"))
        if not candidates:
            candidates = sorted(
                path for path in root.glob("*.bin") if path.name != "training_args.bin"
            )
        if not candidates:
            raise RuntimeError(f"no saved model weights in {root}")
        if any(path.suffix != ".safetensors" for path in candidates):
            raise RuntimeError("numeric resume comparison requires safetensors weights")
        return candidates

    def digest(path: Path) -> str:
        result = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                result.update(chunk)
        return result.hexdigest()

    def tensor_locations(files: Sequence[Path]) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in result:
                        raise RuntimeError(f"duplicate saved tensor: {key}")
                    result[key] = path
        return result

    left_files = model_files(left_root)
    right_files = model_files(right_root)
    left_hashes = {path.name: digest(path) for path in left_files}
    right_hashes = {path.name: digest(path) for path in right_files}
    left_locations = tensor_locations(left_files)
    right_locations = tensor_locations(right_files)
    if left_locations.keys() != right_locations.keys():
        raise RuntimeError("saved tensor key sets differ across resume smoke")

    total = 0
    mismatched = 0
    mismatched_tensors = 0
    nonfinite_differences = 0
    nonfloating_mismatches = 0
    squared_error = 0.0
    squared_reference = 0.0
    max_abs = 0.0
    handles: dict[tuple[str, Path], Any] = {}
    try:
        for name in sorted(left_locations):
            left_key = ("left", left_locations[name])
            right_key = ("right", right_locations[name])
            if left_key not in handles:
                handles[left_key] = safe_open(
                    left_locations[name], framework="pt", device="cpu"
                )
            if right_key not in handles:
                handles[right_key] = safe_open(
                    right_locations[name], framework="pt", device="cpu"
                )
            left = handles[left_key].get_tensor(name)
            right = handles[right_key].get_tensor(name)
            if left.shape != right.shape or left.dtype != right.dtype:
                raise RuntimeError(f"saved tensor contract differs for {name}")
            total += left.numel()
            different = int(torch.count_nonzero(left != right).item())
            mismatched += different
            mismatched_tensors += int(different > 0)
            if left.is_floating_point():
                left_double = left.double()
                squared_reference += float(torch.sum(left_double.square()).item())
                if different:
                    delta = left_double - right.double()
                    finite = torch.isfinite(delta)
                    nonfinite_differences += int(torch.count_nonzero(~finite).item())
                    if bool(finite.any()):
                        finite_delta = delta[finite]
                        max_abs = max(max_abs, float(finite_delta.abs().max().item()))
                        squared_error += float(torch.sum(finite_delta.square()).item())
            else:
                nonfloating_mismatches += different
            del left, right
    finally:
        handles.clear()

    mismatched_fraction = mismatched / total if total else 0.0
    relative_l2 = (
        math.sqrt(squared_error / squared_reference) if squared_reference else 0.0
    )
    numeric_consistent = (
        nonfinite_differences == 0
        and nonfloating_mismatches == 0
        and max_abs <= RESUME_MAX_ABS_TOLERANCE
        and relative_l2 <= RESUME_RELATIVE_L2_TOLERANCE
        and mismatched_fraction <= RESUME_MISMATCHED_FRACTION_TOLERANCE
    )
    return {
        "schema_version": 2,
        "left": str(left_root.resolve()),
        "right": str(right_root.resolve()),
        "left_sha256": left_hashes,
        "right_sha256": right_hashes,
        "byte_identical": left_hashes == right_hashes,
        "numeric_consistent": numeric_consistent,
        "accepted": left_hashes == right_hashes or numeric_consistent,
        "tensor_count": len(left_locations),
        "elements": total,
        "mismatched_elements": mismatched,
        "mismatched_tensors": mismatched_tensors,
        "mismatched_fraction": mismatched_fraction,
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "nonfinite_differences": nonfinite_differences,
        "nonfloating_mismatches": nonfloating_mismatches,
        "tolerances": {
            "max_abs": RESUME_MAX_ABS_TOLERANCE,
            "relative_l2": RESUME_RELATIVE_L2_TOLERANCE,
            "mismatched_fraction": RESUME_MISMATCHED_FRACTION_TOLERANCE,
        },
    }


def processor_artifact_source(model_path: str, tokenizer_name: str | None) -> str:
    """Keep checkpoint weights separate from the shared processor artifacts."""

    return tokenizer_name or model_path


def validate_multimodal_processor(processor: Any) -> dict[str, str]:
    """Reject tokenizer-only AutoProcessor fallbacks before collator execution."""

    processor_type = f"{type(processor).__module__}.{type(processor).__qualname__}"
    image_processor = getattr(processor, "image_processor", None)
    required_callables = ("apply_chat_template", "batch_decode")
    missing = [
        name
        for name in required_callables
        if not callable(getattr(processor, name, None))
    ]
    if image_processor is None or missing:
        details = ", ".join(missing) if missing else "image_processor"
        raise TypeError(
            f"Fast-dVLM requires a multimodal processor; got {processor_type} "
            f"missing {details}"
        )
    return {
        "processor_type": processor_type,
        "image_processor_type": (
            f"{type(image_processor).__module__}.{type(image_processor).__qualname__}"
        ),
    }


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


def validate_preflight_stop_step(
    stop_after_steps: int | None,
    *,
    max_steps: int,
    allow_recipe_override: bool,
) -> None:
    """Keep resume smoke scheduling identical to its uninterrupted control."""

    if stop_after_steps is None:
        return
    if not allow_recipe_override:
        raise ValueError("--preflight_stop_after_steps is only valid for recipe smoke runs")
    if max_steps <= 1 or not 0 < stop_after_steps < max_steps:
        raise ValueError(
            "preflight stop step must be positive and strictly below max_steps; "
            f"got stop={stop_after_steps}, max_steps={max_steps}"
        )


def audit_training_schedule(
    stage: str,
    *,
    state_max_steps: int,
    world_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    allow_recipe_override: bool,
    requested_max_steps: int,
) -> dict[str, int]:
    """Fail before training when data sharding changes the advertised recipe."""

    expected_steps = (
        requested_max_steps if allow_recipe_override else EXPECTED_OPTIMIZER_STEPS[stage]
    )
    global_batch_size = (
        world_size * per_device_batch_size * gradient_accumulation_steps
    )
    if state_max_steps != expected_steps:
        raise RuntimeError(
            f"{stage} schedule resolved to {state_max_steps} optimizer steps; "
            f"expected {expected_steps}"
        )
    if world_size != 2 or global_batch_size != 16:
        raise RuntimeError(
            f"{stage} schedule resolved to world_size={world_size}, "
            f"global_batch_size={global_batch_size}; expected 2 and 16"
        )
    return {
        "optimizer_steps": state_max_steps,
        "world_size": world_size,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "global_batch_size": global_batch_size,
    }
