from __future__ import annotations

import unittest

from fast_dvlm.gui_finetune.metrics import (
    parse_grounding_action,
    score_grounding_records,
    score_planner_records,
    select_grounder_checkpoint,
    select_planner_checkpoint,
)
from fast_dvlm.gui_finetune.runtime import engine_arguments
from fast_dvlm.gui_finetune.training import (
    EXPECTED_LORA_PARAMETERS,
    LearningRates,
    audit_parameters,
    balance_weights,
    expected_epoch_samples,
    learning_rate_for,
    validate_stage_hyperparameters,
)


class GuiContractsTest(unittest.TestCase):
    class Parameter:
        def __init__(self, count: int, requires_grad: bool):
            self.count = count
            self.requires_grad = requires_grad

        def numel(self):
            return self.count

    def test_learning_rates_and_epoch_lengths(self):
        rates = LearningRates(1e-6, 2e-6, 1e-7)
        self.assertEqual(learning_rate_for("model.layers.0.self_attn.q_proj.weight", rates), 1e-6)
        self.assertEqual(learning_rate_for("model.visual.merger.weight", rates), 2e-6)
        self.assertEqual(learning_rate_for("model.visual.blocks.0.weight", rates), 1e-7)
        self.assertEqual(expected_epoch_samples("planner", None), 13004)
        self.assertEqual(expected_epoch_samples("grounder", None), 16528)
        weights = balance_weights(["mind2web"] * 2 + ["mobile"] * 4, 1.0)
        self.assertAlmostEqual(sum(weights[:2]), sum(weights[2:]))

    def test_stage_recipes(self):
        validate_stage_hyperparameters(
            "planner",
            {
                "num_train_epochs": 2,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "learning_rate": 1e-6,
                "save_steps": 813,
                "max_steps": -1,
            },
        )
        with self.assertRaises(ValueError):
            validate_stage_hyperparameters("planner", {"num_train_epochs": 1})

    def test_metrics_and_selection(self):
        parsed = parse_grounding_action("lclick [100, 100, 200, 200]")
        self.assertTrue(parsed.valid)
        grounding = score_grounding_records(
            [{"target_action": "lclick", "target_bbox_1000": [90, 90, 210, 210], "prediction": "lclick [100,100,200,200]", "latency_seconds": 1.0}]
        )
        self.assertEqual(grounding["ssr_point_only"], 1.0)
        planner = score_planner_records(
            [{"target": '{"action":"click","target":"X"}', "prediction": '{"action":"click","target":"X"}'}]
        )
        self.assertEqual(planner["content_action_exact"], 1.0)
        selected = select_planner_checkpoint(
            [
                {"step": 813, "content_action_exact": 0.5, "action_macro_recall": 0.6, "schema_valid_rate": 1.0},
                {"step": 1626, "content_action_exact": 0.5, "action_macro_recall": 0.6, "schema_valid_rate": 1.0},
            ]
        )
        self.assertEqual(selected["step"], 813)
        selected_grounder = select_grounder_checkpoint(
            [
                {"step": 1033, "mind2web_ssr": 0.8, "mobile_ssr": 0.7, "mind2web_joint_ssr": 0.8, "mobile_joint_ssr": 0.7},
                {"step": 2066, "mind2web_ssr": 0.75, "mobile_ssr": 0.75, "mind2web_joint_ssr": 0.75, "mobile_joint_ssr": 0.75},
            ]
        )
        self.assertEqual(selected_grounder["step"], 2066)

    def test_shared_engine_uses_one_pinned_adapter(self):
        args = engine_arguments(
            "/model",
            adapter_path="/adapter",
            adapter_name="gui_grounder",
            algorithm="mdm",
            mem_fraction_static=0.75,
            max_lora_rank=32,
        )
        self.assertEqual(args["model_path"], "/model")
        self.assertNotIn("lora_path", args)
        self.assertEqual(
            args["lora_paths"],
            [{"lora_name": "gui_grounder", "lora_path": "/adapter", "pinned": True}],
        )
        self.assertEqual(args["max_loaded_loras"], 1)

    def test_grounder_audit_accepts_only_36_layer_qkvo_lora(self):
        per_module = EXPECTED_LORA_PARAMETERS // (36 * 4)
        parameters = [("model.visual.blocks.0.weight", self.Parameter(100, False))]
        for layer in range(36):
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                parameters.append(
                    (
                        f"base_model.model.model.language_model.layers.{layer}.self_attn.{projection}.lora_A.default.weight",
                        self.Parameter(per_module // 2, True),
                    )
                )
                parameters.append(
                    (
                        f"base_model.model.model.language_model.layers.{layer}.self_attn.{projection}.lora_B.default.weight",
                        self.Parameter(per_module - per_module // 2, True),
                    )
                )
        audit = audit_parameters(parameters, "grounder")
        self.assertEqual(audit["lora_module_count"], 144)
        self.assertEqual(audit["trainable_parameters"], EXPECTED_LORA_PARAMETERS)


if __name__ == "__main__":
    unittest.main()
