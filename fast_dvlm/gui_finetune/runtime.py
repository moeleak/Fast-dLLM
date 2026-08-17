"""One-backbone SGLang runtime with request-scoped Grounder LoRA switching."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


ALGORITHMS = {"mdm": "HierarchyBlock", "spec": "SpeculativeBlock"}


def engine_arguments(
    model_path: str,
    *,
    processor_path: str,
    adapter_path: str | None,
    adapter_name: str,
    algorithm: str,
    mem_fraction_static: float,
    max_lora_rank: int,
) -> dict[str, Any]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown Fast-dVLM algorithm: {algorithm}")
    result: dict[str, Any] = {
        "model_path": model_path,
        "tokenizer_path": processor_path,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "mem_fraction_static": mem_fraction_static,
        "max_running_requests": 1,
        "chunked_prefill_size": 16_384,
        "dllm_algorithm": ALGORITHMS[algorithm],
        "disable_cuda_graph": False,
        "log_level": "warning",
        "enable_metrics": True,
        "mm_attention_backend": "triton_attn",
    }
    if adapter_path:
        result.update(
            {
                "enable_lora": True,
                "lora_paths": [
                    {
                        "lora_name": adapter_name,
                        "lora_path": adapter_path,
                        "pinned": True,
                    }
                ],
                "max_lora_rank": max_lora_rank,
                "max_loras_per_batch": 1,
                "max_loaded_loras": 1,
            }
        )
    return result


def register_fast_dvlm_processor(
    processor_class: type[Any],
    registry: dict[str, type[Any]] | None = None,
) -> None:
    """Make SGLang load Fast-dVLM's Qwen2.5-VL processor, not a tokenizer.

    Loading the trust-remote-code checkpoint registers ``fast_dvlm`` as a
    custom Transformers config.  ``AutoProcessor`` has no mapping for that
    config and silently falls back to ``Qwen2TokenizerFast``.  SGLang then
    fails when its multimodal wrapper accesses ``processor.tokenizer``.
    """

    if registry is None:
        from sglang.srt.multimodal.customized_mm_processor_utils import (
            _CUSTOMIZED_MM_PROCESSOR,
        )

        registry = _CUSTOMIZED_MM_PROCESSOR
    existing = registry.get("fast_dvlm")
    if existing is not None and existing is not processor_class:
        raise RuntimeError(
            "SGLang already registered a different processor for fast_dvlm: "
            f"{existing}"
        )
    registry["fast_dvlm"] = processor_class


def build_inputs(processor: Any, image: str | Path, prompt: str) -> list[int]:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.input_ids[0].tolist()


class SharedBackboneEngine:
    """Own exactly one SGLang engine and optionally one pinned Grounder adapter."""

    adapter_name = "gui_grounder"

    def __init__(
        self,
        model_path: str | Path,
        *,
        adapter_path: str | Path | None = None,
        processor_path: str | Path | None = None,
        algorithm: str = "mdm",
        mem_fraction_static: float = 0.75,
        max_lora_rank: int = 32,
    ) -> None:
        os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
        import sglang as sgl
        from transformers import AutoTokenizer, Qwen2_5_VLProcessor

        self.model_path = str(model_path)
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        processor_source = str(processor_path or model_path)
        register_fast_dvlm_processor(Qwen2_5_VLProcessor)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            processor_source,
            trust_remote_code=True,
            use_fast=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(processor_source, trust_remote_code=True)
        self.processor.tokenizer = tokenizer
        engine_kwargs = engine_arguments(
            self.model_path,
            processor_path=processor_source,
            adapter_path=self.adapter_path,
            adapter_name=self.adapter_name,
            algorithm=algorithm,
            mem_fraction_static=mem_fraction_static,
            max_lora_rank=max_lora_rank,
        )
        self.engine = sgl.Engine(**engine_kwargs)
        self.backbone_identity = id(self.engine)
        self.algorithm = algorithm

    def generate(
        self,
        image: str | Path,
        prompt: str,
        *,
        mode: str,
        max_new_tokens: int = 64,
    ) -> dict[str, Any]:
        if mode not in {"planner", "grounder"}:
            raise ValueError("mode must be planner or grounder")
        if mode == "grounder" and not self.adapter_path:
            raise RuntimeError("Grounder mode requires an adapter")
        input_ids = build_inputs(self.processor, image, prompt)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            output = self.engine.generate(
                input_ids=input_ids,
                image_data=[str(image)],
                sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
                lora_path=self.adapter_name if mode == "grounder" else None,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        except BaseException:
            raise
        if isinstance(output, list):
            output = output[0]
        return {
            "text": output["text"],
            "latency_seconds": elapsed,
            "meta_info": output.get("meta_info", {}),
            "mode": mode,
            "adapter_enabled": mode == "grounder",
            "backbone_identity": self.backbone_identity,
        }

    def shutdown(self) -> None:
        self.engine.shutdown()

    def __enter__(self) -> "SharedBackboneEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.shutdown()
        return False
