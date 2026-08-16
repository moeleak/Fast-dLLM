from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from fast_dvlm.gui_finetune.data import (
    audit_converted_training_file,
    convert_grounder_dataset,
    convert_planner_dataset,
    load_benchmark_rows,
    select_planner_validation_rows,
    sha256_file,
)


class GuiDataTest(unittest.TestCase):
    def test_benchmark_manifest_accepts_authenticated_legacy_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            samples.mkdir()
            data_path = samples / "mind2web.jsonl"
            data_path.write_text(
                "".join(
                    json.dumps({"sample_id": f"sample-{index}", "image": "x.png"}) + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "benchmarks": {
                            "mind2web": {
                                "path": "samples/mind2web.jsonl",
                                "rows": 3,
                                "sha256": sha256_file(data_path),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            rows, audit = load_benchmark_rows(root, "mind2web")
            self.assertEqual(
                [row["sample_id"] for row in rows],
                ["sample-0", "sample-1", "sample-2"],
            )
            self.assertEqual(audit["data_sha256"], sha256_file(data_path))

            data_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row count mismatch"):
                load_benchmark_rows(root, "mind2web")

    def test_planner_conversion_and_validation_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, images, output = root / "source", root / "images", root / "out"
            source.mkdir()
            images.mkdir()
            (source / "manifest.json").write_text('{"fixture":true}\n', encoding="utf-8")
            counts = {"train": 12, "validation": 120, "test": 10}
            next_id = 0
            for split, count in counts.items():
                lines = []
                for index in range(count):
                    sample_id = f"{split}-{next_id}"
                    next_id += 1
                    image = f"{sample_id}.png"
                    (images / image).write_bytes(b"fixture")
                    action = ("click", "type", "swipe")[index % 3]
                    lines.append(
                        json.dumps(
                            {
                                "id": sample_id,
                                "image": image,
                                "messages": [
                                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Do it"}]},
                                    {"role": "assistant", "content": json.dumps({"action": action})},
                                ],
                            }
                        )
                    )
                (source / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            audit = convert_planner_dataset(
                source,
                images,
                output,
                expected_counts=counts,
                expected_manifest_sha256=sha256_file(source / "manifest.json"),
            )
            selected = json.loads((output / "validation-100.json").read_text())
            self.assertEqual(len(selected), 100)
            self.assertEqual(set(audit["validation_selection"]["action_counts"]), {"click", "swipe", "type"})
            self.assertTrue(all(row["conversations"][-1]["from"] == "gpt" for row in selected))
            self.assertEqual(audit_converted_training_file(output / "train.json")["count"], 12)

    def test_stable_validation_selection_is_order_independent(self):
        rows = [
            {"id": f"sample-{index}", "_gui_balance_key": f"a{index % 5}"}
            for index in range(140)
        ]
        forward = [row["id"] for row in select_planner_validation_rows(rows)]
        backward = [row["id"] for row in select_planner_validation_rows(list(reversed(rows)))]
        self.assertEqual(forward, backward)

    def test_grounder_conversion_is_two_domain_and_direct_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for domain in ("mind2web", "mobile"):
                directory = root / domain
                directory.mkdir()
                sources[domain] = directory
                table = pa.Table.from_pylist(
                    [
                        {
                            "sample_id": f"{domain}-{index}",
                            "source": domain,
                            "image": {"bytes": b"\x89PNG\r\n\x1a\nfixture", "path": "x.png"},
                            "conversations": [
                                {"from": "human", "value": "<image>\nClick on X."},
                                {"from": "gpt", "value": "lclick [1,2,3,4]"},
                            ],
                            "metadata": "{}",
                        }
                        for index in range(2)
                    ]
                )
                pq.write_table(table, directory / "part.parquet")
            with mock.patch(
                "fast_dvlm.gui_finetune.data.sha256_file",
                wraps=sha256_file,
            ) as hash_file:
                audit = convert_grounder_dataset(
                    sources["mind2web"],
                    sources["mobile"],
                    root / "out",
                    expected_counts={"mind2web": 2, "mobile": 2},
                )
            parquet_hashes = [
                call.args[0]
                for call in hash_file.call_args_list
                if call.args[0].suffix == ".parquet"
            ]
            self.assertEqual(
                parquet_hashes,
                [
                    (sources["mind2web"] / "part.parquet").resolve(),
                    (sources["mobile"] / "part.parquet").resolve(),
                ],
            )
            rows = json.loads((root / "out" / "train.json").read_text())
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["_gui_domain"] for row in rows}, {"mind2web", "mobile"})
            self.assertEqual(audit["sampling"], "domain_balanced_with_replacement")

    def test_grounder_leakage_uses_the_declared_test_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for domain in ("mind2web", "mobile"):
                directory = root / domain
                directory.mkdir()
                sources[domain] = directory
                table = pa.Table.from_pylist(
                    [
                        {
                            "sample_id": f"{domain}-train",
                            "source": domain,
                            "image": {"bytes": b"\x89PNG\r\n\x1a\nfixture", "path": "x.png"},
                            "conversations": [
                                {"from": "human", "value": "<image>\nClick on X."},
                                {"from": "gpt", "value": "lclick [1,2,3,4]"},
                            ],
                            "metadata": "{}",
                        }
                    ]
                )
                pq.write_table(table, directory / "part.parquet")

            heldout = root / "heldout"
            heldout.mkdir()
            data_path = heldout / "samples.jsonl"
            data_path.write_text(
                "".join(
                    json.dumps({"sample_id": sample_id}) + "\n"
                    for sample_id in ("test-only", "mind2web-train")
                ),
                encoding="utf-8",
            )
            (heldout / "manifest.json").write_text(
                json.dumps(
                    {
                        "benchmarks": {
                            "test": {
                                "path": data_path.name,
                                "rows": 2,
                                "sha256": sha256_file(data_path),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            audit = convert_grounder_dataset(
                sources["mind2web"],
                sources["mobile"],
                root / "out",
                expected_counts={"mind2web": 1, "mobile": 1},
                heldout_benchmarks={"test": (heldout, "test", 1)},
            )
            test_audit = audit["heldout_benchmarks"]["test"]
            training_rows = json.loads((root / "out" / "train.json").read_text())
            self.assertEqual(audit["total_count"], 2)
            self.assertEqual(len(training_rows), 2)
            self.assertEqual(test_audit["manifest_count"], 2)
            self.assertEqual(test_audit["count"], 1)
            self.assertEqual(test_audit["selection"], "ordered_prefix")


if __name__ == "__main__":
    unittest.main()
