#!/usr/bin/env python3
"""Train Fast-dVLM as a full Planner or a frozen-backbone Grounder LoRA."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIRD_PARTY = _REPO_ROOT / "third_party"
for value in (str(_REPO_ROOT), str(_THIRD_PARTY)):
    if value not in sys.path:
        sys.path.insert(0, value)

from fast_dvlm.gui_finetune.data import audit_converted_training_file, local_model_manifest
from fast_dvlm.gui_finetune.training import (
    EXPECTED_LORA_MODULES,
    FAST_DVLM_MASK_TOKEN_ID,
    FAST_DVLM_MODEL,
    FAST_DVLM_REVISION,
    LORA_TARGETS,
    LearningRates,
    audit_parameters,
    audit_zero_delta,
    balance_weights,
    expected_epoch_samples,
    learning_rate_for,
    parameter_role,
    validate_stage_hyperparameters,
)


@dataclass
class GuiWorkflowArguments:
    stage: str = field(metadata={"choices": ["planner", "grounder"]})
    source_audit: Optional[str] = None
    training_audit: Optional[str] = None
    max_pixels: int = 705_600
    min_lr_ratio: float = 0.1
    language_learning_rate: Optional[float] = None
    connector_learning_rate: float = 2e-6
    vision_learning_rate: float = 1e-7
    balance_power: Optional[float] = None
    allow_recipe_override: bool = False
    expected_lora_modules: int = EXPECTED_LORA_MODULES


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = []
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint-*"):
            try:
                step = int(path.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, path))
    return str(max(checkpoints)[1]) if checkpoints else None


def _code_revision() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=_REPO_ROOT, text=True
            ).strip()
        )
        return {"git_revision": revision, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "git_dirty": None}


def _find_subsequence(values, needle) -> int:
    width = len(needle)
    for index in range(0, len(values) - width + 1):
        if values[index : index + width] == needle:
            return index
    return -1


def main() -> None:
    import torch
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import Sampler
    from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainerCallback

    from lmflow.args import ModelArguments, MultiModalDatasetArguments, AutoArguments
    from lmflow.datasets.dataset import Dataset
    from lmflow.datasets.multi_modal_dataset import DataCollatorForQwenVL
    from lmflow.models.auto_model import AutoModel

    PipelineArguments = AutoArguments.get_pipeline_args_class("finetuner")
    parser = HfArgumentParser(
        (GuiWorkflowArguments, ModelArguments, MultiModalDatasetArguments, PipelineArguments)
    )
    workflow, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if workflow.stage not in {"planner", "grounder"}:
        parser.error("--stage must be planner or grounder")

    model_args.model_name_or_path = model_args.model_name_or_path or FAST_DVLM_MODEL
    model_args.tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    if model_args.model_name_or_path == FAST_DVLM_MODEL:
        model_args.model_revision = FAST_DVLM_REVISION
    model_args.trust_remote_code = True
    model_args.torch_dtype = "bfloat16"
    model_args.mdm = True
    model_args.bd_size = 32
    model_args.use_qlora = False
    model_args.use_dora = False
    data_args.return_as_qwen_messages = True

    if workflow.stage == "planner":
        model_args.use_lora = False
        default_balance_power = 0.25
        expected_epochs = 2.0
    else:
        model_args.use_lora = True
        model_args.lora_r = 32
        model_args.lora_alpha = 32
        model_args.lora_dropout = 0.1
        model_args.lora_target_modules = list(LORA_TARGETS)
        default_balance_power = 1.0
        expected_epochs = 3.0
    if workflow.balance_power is None:
        workflow.balance_power = default_balance_power
    if workflow.language_learning_rate is None:
        workflow.language_learning_rate = float(training_args.learning_rate)

    recipe_values = {
        "num_train_epochs": float(training_args.num_train_epochs),
        "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(training_args.gradient_accumulation_steps),
        "learning_rate": float(training_args.learning_rate),
        "save_steps": int(training_args.save_steps),
        "max_steps": int(training_args.max_steps),
    }
    if not workflow.allow_recipe_override:
        validate_stage_hyperparameters(workflow.stage, recipe_values)
    if float(training_args.num_train_epochs) != expected_epochs and not workflow.allow_recipe_override:
        raise ValueError("unexpected epoch count")
    if not 0.0 <= workflow.min_lr_ratio <= 1.0:
        raise ValueError("--min_lr_ratio must be in [0,1]")

    dataset_audit = audit_converted_training_file(Path(data_args.dataset_path))
    source_audit = None
    if workflow.source_audit:
        source_audit_path = Path(workflow.source_audit)
        source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))

    dataset = Dataset(data_args, backend="custom_multi_modal")
    model = AutoModel.get_model(model_args)
    backend_model = model.get_backend_model()
    if workflow.stage == "planner":
        for parameter in backend_model.parameters():
            parameter.requires_grad_(True)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision if model_args.model_name_or_path == FAST_DVLM_MODEL else None,
        trust_remote_code=True,
        max_pixels=workflow.max_pixels,
    )
    processor.tokenizer = model.tokenizer
    mask_id = model.tokenizer.convert_tokens_to_ids("|<MASK>|")
    if mask_id != FAST_DVLM_MASK_TOKEN_ID:
        raise RuntimeError(
            f"Fast-dVLM mask token mismatch: expected {FAST_DVLM_MASK_TOKEN_ID}, got {mask_id}"
        )
    dataset.backend_dataset.register_tokenizer(
        model.tokenizer,
        getattr(model, "image_processor", None),
    )
    parameter_audit = audit_parameters(backend_model.named_parameters(), workflow.stage)
    zero_delta_audit = (
        audit_zero_delta(backend_model.named_parameters())
        if workflow.stage == "grounder"
        else None
    )
    if workflow.stage == "grounder" and (
        parameter_audit["lora_module_count"] != workflow.expected_lora_modules
    ):
        raise RuntimeError(
            f"expected {workflow.expected_lora_modules} LoRA modules, "
            f"got {parameter_audit['lora_module_count']}"
        )
    if zero_delta_audit and zero_delta_audit["lora_b_matrices"] != workflow.expected_lora_modules:
        raise RuntimeError(
            f"expected {workflow.expected_lora_modules} zero LoRA B matrices, "
            f"got {zero_delta_audit['lora_b_matrices']}"
        )

    class DistributedWeightedSampler(Sampler[int]):
        def __init__(
            self,
            weights,
            *,
            replicas: int,
            rank: int,
            seed: int,
            epoch_samples: int,
        ):
            self.weights = torch.as_tensor(weights, dtype=torch.double)
            self.replicas = replicas
            self.rank = rank
            self.seed = seed
            self.epoch = 0
            self.samples_per_rank = math.ceil(epoch_samples / replicas)

        def __iter__(self):
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            total = self.samples_per_rank * self.replicas
            indices = torch.multinomial(
                self.weights,
                total,
                replacement=True,
                generator=generator,
            ).tolist()
            return iter(indices[self.rank:total:self.replicas])

        def __len__(self):
            return self.samples_per_rank

        def set_epoch(self, epoch: int):
            self.epoch = epoch

    class GradientAuditCallback(TrainerCallback):
        def __init__(self, stage: str):
            self.stage = stage
            self.result = None

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if self.result is not None or model is None:
                return
            roles = {"language": 0, "connector": 0, "vision": 0, "adapter": 0}
            invalid = []
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                role = parameter_role(name)
                roles[role] = roles.get(role, 0) + int(parameter.numel())
                if self.stage == "grounder" and role != "adapter":
                    invalid.append(name)
            if invalid:
                raise RuntimeError("Grounder backbone received gradients: " + ", ".join(invalid[:8]))
            required = ("language", "connector", "vision") if self.stage == "planner" else ("adapter",)
            missing = [role for role in required if roles.get(role, 0) == 0]
            if missing:
                raise RuntimeError(f"no first-step gradients for parameter roles: {missing}")
            self.result = {"step": int(state.global_step), "gradient_parameters_by_role": roles}

    gradient_audit = GradientAuditCallback(workflow.stage)
    rates = LearningRates(
        language=float(workflow.language_learning_rate),
        connector=float(workflow.connector_learning_rate),
        vision=float(workflow.vision_learning_rate),
    )

    class GuiTrainer(Trainer):
        optimizer_group_audit: list[dict[str, Any]]

        def _get_train_sampler(self, train_dataset=None):
            selected = train_dataset if train_dataset is not None else self.train_dataset
            rows = getattr(selected, "data_dict", None)
            if not rows or workflow.balance_power == 0:
                try:
                    return super()._get_train_sampler(train_dataset)
                except TypeError:
                    return super()._get_train_sampler()
            keys = [str(row.get("_gui_balance_key", "")) for row in rows]
            weights = balance_weights(keys, float(workflow.balance_power))
            return DistributedWeightedSampler(
                weights,
                replicas=max(1, int(self.args.world_size)),
                rank=int(self.args.process_index),
                seed=int(self.args.seed),
                epoch_samples=int(expected_epoch_samples(workflow.stage, source_audit)),
            )

        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            decay_names = set(self.get_decay_parameter_names(self.model))
            grouped: dict[tuple[str, bool, float], list[Any]] = {}
            counts: Counter[tuple[str, bool, float]] = Counter()
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                role = parameter_role(name)
                learning_rate = (
                    rates.language
                    if workflow.stage == "grounder"
                    else learning_rate_for(name, rates)
                )
                key = (role, name in decay_names, learning_rate)
                grouped.setdefault(key, []).append(parameter)
                counts[key] += int(parameter.numel())
            optimizer_groups = []
            self.optimizer_group_audit = []
            for (role, decay, learning_rate), parameters in sorted(
                grouped.items(), key=lambda item: item[0]
            ):
                weight_decay = self.args.weight_decay if decay else 0.0
                optimizer_groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rate,
                        "weight_decay": weight_decay,
                    }
                )
                self.optimizer_group_audit.append(
                    {
                        "role": role,
                        "decay": decay,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "parameters": counts[(role, decay, learning_rate)],
                    }
                )
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, self.model
            )
            self.optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)
            return self.optimizer

        def create_scheduler(self, num_training_steps: int, optimizer=None):
            if self.lr_scheduler is not None:
                return self.lr_scheduler
            optimizer = optimizer or self.optimizer
            warmup_steps = self.args.get_warmup_steps(num_training_steps)
            minimum = float(workflow.min_lr_ratio)

            def multiplier(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return max(1e-12, step / warmup_steps)
                denominator = max(1, num_training_steps - warmup_steps)
                progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return minimum + (1.0 - minimum) * cosine

            self.lr_scheduler = LambdaLR(optimizer, multiplier)
            return self.lr_scheduler

    data_collator = DataCollatorForQwenVL(processor=processor, tokenizer=model.tokenizer)
    first_source = dataset.backend_dataset.data_dict[0]
    first_messages = first_source["conversations"]
    first_prompt = str(first_messages[0]["value"]).replace("<image>", "").strip()
    first_target = str(first_messages[-1]["value"]).strip()
    masking_batch = data_collator([dataset.get_backend_dataset()[0]])
    input_values = masking_batch["input_ids"][0].tolist()
    label_values = masking_batch["labels"][0].tolist()
    prompt_values = model.tokenizer.encode(first_prompt, add_special_tokens=False)
    target_values = model.tokenizer.encode(first_target, add_special_tokens=False)
    prompt_start = _find_subsequence(input_values, prompt_values)
    target_start = _find_subsequence(input_values, target_values)
    if prompt_start < 0 or target_start < 0:
        raise RuntimeError("assistant-only audit could not locate prompt/target tokens")
    if any(value != -100 for value in label_values[prompt_start : prompt_start + len(prompt_values)]):
        raise RuntimeError("assistant-only audit found supervised prompt tokens")
    if label_values[target_start : target_start + len(target_values)] != target_values:
        raise RuntimeError("assistant-only audit found masked assistant target tokens")
    masking_audit = {
        "sample_id": str(first_source["id"]),
        "prompt_tokens": len(prompt_values),
        "target_tokens": len(target_values),
        "supervised_tokens": sum(value != -100 for value in label_values),
        "prompt_fully_masked": True,
        "target_fully_supervised": True,
    }
    trainer = GuiTrainer(
        model=backend_model,
        args=training_args,
        train_dataset=dataset.get_backend_dataset(),
        tokenizer=model.tokenizer,
        data_collator=data_collator,
        callbacks=[gradient_audit],
    )
    resume = training_args.resume_from_checkpoint
    if resume is None:
        resume = _latest_checkpoint(Path(training_args.output_dir))
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    audit_path = Path(
        workflow.training_audit or (Path(training_args.output_dir) / "training-audit.json")
    )
    audit = {
        "schema_version": 1,
        "stage": workflow.stage,
        "model": model_args.model_name_or_path,
        "model_revision": model_args.model_revision,
        "model_artifacts": local_model_manifest(Path(model_args.model_name_or_path)),
        "tokenizer": model_args.tokenizer_name,
        "code": _code_revision(),
        "mask_token_id": mask_id,
        "native_objective": {"mdm": True, "block_size": 32, "causal_auxiliary": True},
        "dataset": dataset_audit,
        "assistant_only_masking": masking_audit,
        "source_audit": source_audit,
        "parameters": parameter_audit,
        "zero_delta": zero_delta_audit,
        "optimizer_groups": getattr(trainer, "optimizer_group_audit", []),
        "first_step_gradients": gradient_audit.result,
        "recipe": {
            **recipe_values,
            "warmup_ratio": float(training_args.warmup_ratio),
            "min_lr_ratio": float(workflow.min_lr_ratio),
            "weight_decay": float(training_args.weight_decay),
            "adam_beta1": float(training_args.adam_beta1),
            "adam_beta2": float(training_args.adam_beta2),
            "adam_epsilon": float(training_args.adam_epsilon),
            "max_grad_norm": float(training_args.max_grad_norm),
            "balance_power": float(workflow.balance_power),
            "max_pixels": int(workflow.max_pixels),
            "deepspeed": str(training_args.deepspeed),
        },
        "resume_from_checkpoint": resume,
        "train_metrics": result.metrics,
        "peak_gpu_allocated_gib": (
            torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
        ),
        "peak_gpu_reserved_gib": (
            torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else None
        ),
    }
    if trainer.is_world_process_zero():
        _atomic_json(audit_path, audit)


if __name__ == "__main__":
    main()
