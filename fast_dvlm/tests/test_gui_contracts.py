from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

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
    GradientRoleTracker,
    LearningRates,
    audit_training_schedule,
    audit_parameters,
    balance_weights,
    exact_balanced_epoch_indices,
    expected_epoch_samples,
    learning_rate_for,
    processor_artifact_source,
    validate_multimodal_processor,
    validate_preflight_stop_step,
    validate_stage_hyperparameters,
)


class GuiContractsTest(unittest.TestCase):
    class Parameter:
        def __init__(self, count: int, requires_grad: bool):
            self.count = count
            self.requires_grad = requires_grad

        def numel(self):
            return self.count

    class MultimodalProcessor:
        image_processor = object()

        def apply_chat_template(self):
            return None

        def batch_decode(self):
            return None

    class TextTokenizer:
        def apply_chat_template(self):
            return None

    class HookParameter(Parameter):
        def __init__(self, count: int, requires_grad: bool):
            super().__init__(count, requires_grad)
            self.hook = None

        def register_hook(self, hook):
            self.hook = hook

            class Handle:
                def remove(inner_self):
                    self.hook = None

            return Handle()

        def emit_gradient(self):
            if self.hook is not None:
                self.hook(object())

    def test_resource_waiter_is_fail_closed(self):
        root = Path(__file__).resolve().parents[2]
        script = root / "fast_dvlm" / "train_scripts" / "wait_and_run_gui_pipeline.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn('git -C "${repo_root}" status --porcelain', text)
        self.assertIn("flock -n 9", text)
        self.assertIn("if (( status == 75 ))", text)
        self.assertIn('minimum_gpu_mib="${MINIMUM_GPU_FREE_MIB:-71680}"', text)
        self.assertIn('minimum_disk_gib="${MINIMUM_DISK_GIB:-300}"', text)

    def test_pipeline_gates_final_test_on_shared_runtime(self):
        root = Path(__file__).resolve().parents[2]
        pipeline = root / "fast_dvlm" / "train_scripts" / "run_gui_pipeline.sh"
        text = pipeline.read_text(encoding="utf-8")
        runtime_gate = text.index("verify_shared_runtime.py")
        heldout_comment = text.index("The held-out test set is touched exactly once")
        final_eval = text.index('final_predictions="${work_root}/final/mind2web-ocr-test100"')
        self.assertLess(runtime_gate, heldout_comment)
        self.assertLess(heldout_comment, final_eval)
        self.assertIn('--shared-runtime-audit "${shared_runtime_audit}"', text)
        self.assertIn('planner_validation_ids_sha256=', text)
        self.assertIn('EXPECTED_SAMPLE_IDS_SHA256="${expected_hash}"', text)

    def test_completed_evaluation_is_reused_without_worker_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            output.mkdir()
            for rank in range(2):
                rows = [
                    {"sample_id": f"sample-{index}", "error": None}
                    for index in range(rank, 100, 2)
                ]
                (output / f"part-{rank:05d}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                (output / f"run-config-{rank:05d}.json").write_text(
                    "{}\n", encoding="utf-8"
                )
            (output / "gpu-memory.csv").write_text(
                "2026-01-01T00:00:00,0,123\n2026-01-01T00:00:00,1,456\n",
                encoding="utf-8",
            )
            script = Path(__file__).resolve().parents[1] / "eval" / "run_gui_eval_2gpu.sh"
            env = {
                **os.environ,
                "TASK": "planner",
                "MODEL_PATH": "/unused/model",
                "OUTPUT_DIR": str(output),
            }
            completed = subprocess.run(
                ["bash", str(script)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Reusing completed 100-sample evaluation", completed.stdout)
            audit = json.loads((output / "gpu-memory-audit.json").read_text())
            self.assertEqual(audit["peak_memory_used_mib"], {"0": 123, "1": 456})

    def test_learning_rates_and_epoch_lengths(self):
        rates = LearningRates(1e-6, 2e-6, 1e-7)
        self.assertEqual(learning_rate_for("model.layers.0.self_attn.q_proj.weight", rates), 1e-6)
        self.assertEqual(learning_rate_for("model.visual.merger.weight", rates), 2e-6)
        self.assertEqual(learning_rate_for("model.visual.blocks.0.weight", rates), 1e-7)
        self.assertEqual(expected_epoch_samples("planner", None), 13004)
        self.assertEqual(expected_epoch_samples("grounder", None), 16528)
        weights = balance_weights(["mind2web"] * 2 + ["mobile"] * 4, 1.0)
        self.assertAlmostEqual(sum(weights[:2]), sum(weights[2:]))
        keys = ["mind2web"] * 3 + ["mobile"] * 4
        indices = exact_balanced_epoch_indices(keys, epoch_samples=8, seed=42)
        selected = [keys[index] for index in indices]
        self.assertEqual(selected.count("mind2web"), 4)
        self.assertEqual(selected.count("mobile"), 4)
        self.assertEqual(
            indices,
            exact_balanced_epoch_indices(keys, epoch_samples=8, seed=42),
        )
        self.assertNotEqual(
            indices,
            exact_balanced_epoch_indices(keys, epoch_samples=8, seed=43),
        )
        self.assertEqual(
            processor_artifact_source("/planner/checkpoint-813", "/planner"),
            "/planner",
        )
        self.assertEqual(processor_artifact_source("base", None), "base")

    def test_multimodal_processor_contract_rejects_tokenizer_fallback(self):
        audit = validate_multimodal_processor(self.MultimodalProcessor())
        self.assertIn("MultimodalProcessor", audit["processor_type"])
        with self.assertRaisesRegex(TypeError, "multimodal processor"):
            validate_multimodal_processor(self.TextTokenizer())

        root = Path(__file__).resolve().parents[2]
        entrypoint = root / "fast_dvlm" / "train_scripts" / "train_gui.py"
        text = entrypoint.read_text(encoding="utf-8")
        self.assertIn("Qwen2_5_VLProcessor.from_pretrained", text)
        self.assertIn("use_fast=False", text)

    def test_gradient_tracker_captures_before_zero_partitions_gradients(self):
        parameters = [
            ("model.layers.0.weight", self.HookParameter(11, True)),
            ("model.visual.blocks.0.weight", self.HookParameter(13, True)),
            ("model.visual.merger.weight", self.HookParameter(17, True)),
        ]
        tracker = GradientRoleTracker("planner")
        tracker.attach(parameters)
        for _, parameter in parameters:
            parameter.emit_gradient()
        audit = tracker.audit(step=0)
        self.assertEqual(
            audit["gradient_parameters_by_role"],
            {"language": 11, "connector": 17, "vision": 13, "adapter": 0},
        )
        self.assertEqual(audit["capture"], "backward_hooks_before_zero_partition")
        tracker.close()

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

    def test_resume_smoke_keeps_the_uninterrupted_scheduler(self):
        validate_preflight_stop_step(
            1,
            max_steps=2,
            allow_recipe_override=True,
        )
        with self.assertRaises(ValueError):
            validate_preflight_stop_step(
                1,
                max_steps=1,
                allow_recipe_override=True,
            )
        with self.assertRaises(ValueError):
            validate_preflight_stop_step(
                1,
                max_steps=2,
                allow_recipe_override=False,
            )
        root = Path(__file__).resolve().parents[2]
        script = (
            root
            / "fast_dvlm"
            / "train_scripts"
            / "run_gui_stage_with_preflight.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("RECIPE_SMOKE_STEPS=2 RECIPE_SMOKE_SAVE_STEPS=1", script)
        self.assertIn("RECIPE_SMOKE_STOP_AFTER_STEPS=1", script)
        self.assertIn("compare_saved_model_weights", script)
        self.assertIn('result["accepted"]', script)

    def test_schedule_requires_one_accelerate_shard_and_exact_steps(self):
        self.assertEqual(
            audit_training_schedule(
                "planner",
                state_max_steps=1626,
                world_size=2,
                per_device_batch_size=1,
                gradient_accumulation_steps=8,
                allow_recipe_override=False,
                requested_max_steps=-1,
            )["global_batch_size"],
            16,
        )
        with self.assertRaisesRegex(RuntimeError, "resolved to 814"):
            audit_training_schedule(
                "planner",
                state_max_steps=814,
                world_size=2,
                per_device_batch_size=1,
                gradient_accumulation_steps=8,
                allow_recipe_override=False,
                requested_max_steps=-1,
            )
        root = Path(__file__).resolve().parents[2]
        entrypoint = (
            root / "fast_dvlm" / "train_scripts" / "train_gui.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Accelerator shards the prepared DataLoader", entrypoint)
        self.assertIn("replicas=1", entrypoint)
        self.assertIn("rank=0", entrypoint)

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
