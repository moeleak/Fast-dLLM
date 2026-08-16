from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # Local source-only validation does not install the CUDA stack.
    torch = None


@unittest.skipIf(torch is None, "torch is required for the fused-QKV numerical test")
class SGLangLoRAContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "third_party" / "sglang" / "python"))
        os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")

    @staticmethod
    def load_lora_adapter():
        root = Path(__file__).resolve().parents[2]

        def module(name, **values):
            value = types.ModuleType(name)
            for key, item in values.items():
                setattr(value, key, item)
            sys.modules[name] = value

        module("sglang")
        module("sglang.srt")
        module("sglang.srt.configs")
        module("sglang.srt.configs.load_config", LoadConfig=object)
        module("sglang.srt.layers")
        module("sglang.srt.layers.utils", get_layer_id=lambda _: None)
        module("sglang.srt.lora")
        module("sglang.srt.lora.backend")
        module("sglang.srt.lora.backend.base_backend", BaseLoRABackend=object)
        module("sglang.srt.lora.backend.lora_registry", LORA_SUPPORTED_BACKENDS=set())
        module("sglang.srt.lora.lora_config", LoRAConfig=object)
        module("sglang.srt.model_loader")
        module("sglang.srt.model_loader.loader", DefaultModelLoader=object)
        module("sglang.srt.utils")
        module("sglang.srt.utils.hf_transformers_utils", AutoConfig=object)
        path = root / "third_party" / "sglang" / "python" / "sglang" / "srt" / "lora" / "lora.py"
        spec = importlib.util.spec_from_file_location("_fast_dvlm_sglang_lora", path)
        assert spec and spec.loader
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded.LoRAAdapter

    def test_separate_peft_qkv_matches_sglang_fused_layout(self):
        LoRAAdapter = self.load_lora_adapter()

        torch.manual_seed(42)
        rank, input_size = 2, 5
        outputs = {"q_proj": 4, "k_proj": 2, "v_proj": 2}
        weights = {}
        for projection, output_size in outputs.items():
            weights[f"layer.{projection}.lora_A.weight"] = torch.randn(rank, input_size)
            weights[f"layer.{projection}.lora_B.weight"] = torch.randn(output_size, rank)
        original = {name: value.clone() for name, value in weights.items()}
        LoRAAdapter.normalize_qkv_proj(object(), list(weights), weights)
        self.assertEqual(
            set(weights),
            {"layer.qkv_proj.lora_A.weight", "layer.qkv_proj.lora_B.weight"},
        )
        fused_a = weights["layer.qkv_proj.lora_A.weight"]
        fused_b = weights["layer.qkv_proj.lora_B.weight"]
        x = torch.randn(3, input_size)
        peft = []
        fused = []
        a_offset = b_offset = 0
        for projection, output_size in outputs.items():
            peft.append(
                (x @ original[f"layer.{projection}.lora_A.weight"].T)
                @ original[f"layer.{projection}.lora_B.weight"].T
            )
            a = fused_a[a_offset : a_offset + rank]
            b = fused_b[b_offset : b_offset + output_size]
            fused.append((x @ a.T) @ b.T)
            a_offset += rank
            b_offset += output_size
        torch.testing.assert_close(torch.cat(peft, dim=-1), torch.cat(fused, dim=-1))

    def test_zero_delta_is_identity(self):
        torch.manual_seed(7)
        x = torch.randn(4, 8)
        base = torch.randn(8, 8)
        lora_a = torch.randn(2, 8)
        lora_b = torch.zeros(8, 2)
        torch.testing.assert_close(x @ base.T + (x @ lora_a.T) @ lora_b.T, x @ base.T)


if __name__ == "__main__":
    unittest.main()
