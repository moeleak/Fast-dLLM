from __future__ import annotations

import os
import sys
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

    def test_separate_peft_qkv_matches_sglang_fused_layout(self):
        from sglang.srt.lora.lora import LoRAAdapter

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
