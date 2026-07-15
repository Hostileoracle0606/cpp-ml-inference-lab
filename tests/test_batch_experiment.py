"""Unit tests for the frozen v1.2 experiment aggregator."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import run_batch_experiment as experiment  # noqa: E402


def record_for(run: int, ratio: float) -> dict:
    serial_samples = [8.0] * 200
    batch_samples = [8.0 / ratio] * 200
    serial_throughput = 1_600_000.0 / sum(serial_samples)
    batch_throughput = 1_600_000.0 / sum(batch_samples)
    return {
        "schema": experiment.EXPECTED_SCHEMA,
        "run": run,
        "order": "serial_then_batch" if run % 2 == 1 else "batch_then_serial",
        "batch_size": 8,
        "warmup_workloads_per_mode": 20,
        "measured_workloads_per_mode": 200,
        "input_recipe": experiment.EXPECTED_RECIPE,
        "model": "/tmp/frozen.onnx",
        "build": {
            "type": "Release",
            "system": experiment.EXPECTED_BUILD_SYSTEM,
            "compiler": experiment.EXPECTED_COMPILER,
            "cxx": 201703,
        },
        "serial_eight": {
            "runtime_calls_per_group": 8,
            "items_per_second": serial_throughput,
            "group_latency_ms": serial_samples,
        },
        "batch_eight": {
            "runtime_calls_per_group": 1,
            "items_per_second": batch_throughput,
            "group_latency_ms": batch_samples,
        },
        "batch_to_serial_items_per_second_ratio": batch_throughput
        / serial_throughput,
    }


class BatchExperimentTests(unittest.TestCase):
    def test_metadata_preflight_rejects_non_release_build(self) -> None:
        valid = (
            "build_type: Release\n"
            f"system: {experiment.EXPECTED_BUILD_SYSTEM}\n"
            f"compiler: {experiment.EXPECTED_COMPILER}\n"
            "cxx: 201703\n"
            "inference: skipped (pass --model <path.onnx>)\n"
        )
        experiment.validate_metadata_probe(valid)
        for old, new in (
            ("Release", "ReleaseWithDebInfo"),
            (experiment.EXPECTED_BUILD_SYSTEM, "Darwin/arm64-extra"),
            (experiment.EXPECTED_COMPILER, experiment.EXPECTED_COMPILER + "-suffix"),
            ("cxx: 201703", "cxx: 2017030"),
            ("inference: skipped (pass --model <path.onnx>)",
             "inference: skipped-but-not-really"),
        ):
            with self.subTest(replacement=new):
                with self.assertRaisesRegex(ValueError, "missing"):
                    experiment.validate_metadata_probe(valid.replace(old, new))

    def test_validates_and_aggregates_the_exact_d33_math(self) -> None:
        ratios = [1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85]
        model = Path("/tmp/frozen.onnx")
        validated = [
            experiment.validate_run_record(record_for(run, ratio), run=run,
                                           model_path=model)
            for run, ratio in enumerate(ratios, start=1)
        ]
        summary = experiment.summarize_runs(validated)

        self.assertAlmostEqual(summary["median_items_per_second_ratio"], 1.625)
        self.assertEqual(summary["favorable_run_count"], 10)
        self.assertEqual(summary["pooled_sample_count_per_mode"], 2000)
        self.assertLess(
            summary["batch_eight_group_p95_ms"],
            summary["serial_eight_group_p95_ms"],
        )
        self.assertTrue(summary["accepted"])

    def test_rejects_order_sample_formula_and_run_set_drift(self) -> None:
        model = Path("/tmp/frozen.onnx")
        wrong_order = record_for(1, 2.0)
        wrong_order["order"] = "batch_then_serial"
        with self.assertRaisesRegex(ValueError, "order"):
            experiment.validate_run_record(wrong_order, run=1, model_path=model)

        wrong_build = record_for(1, 2.0)
        wrong_build["build"]["type"] = "Debug"
        with self.assertRaisesRegex(ValueError, "Release"):
            experiment.validate_run_record(wrong_build, run=1, model_path=model)

        missing_sample = record_for(1, 2.0)
        missing_sample["serial_eight"]["group_latency_ms"].pop()
        with self.assertRaisesRegex(ValueError, "200"):
            experiment.validate_run_record(missing_sample, run=1, model_path=model)

        wrong_formula = record_for(1, 2.0)
        wrong_formula["batch_eight"]["items_per_second"] += 1.0
        with self.assertRaisesRegex(ValueError, "formula"):
            experiment.validate_run_record(wrong_formula, run=1, model_path=model)

        nine = [
            experiment.validate_run_record(record_for(run, 2.0), run=run,
                                           model_path=model)
            for run in range(1, 10)
        ]
        with self.assertRaisesRegex(ValueError, "ten"):
            experiment.summarize_runs(nine)

    def test_rejects_non_finite_and_non_positive_samples(self) -> None:
        model = Path("/tmp/frozen.onnx")
        for invalid in (0.0, -1.0, float("inf"), float("nan"), True):
            with self.subTest(invalid=invalid):
                record = copy.deepcopy(record_for(1, 2.0))
                record["serial_eight"]["group_latency_ms"][0] = invalid
                with self.assertRaisesRegex(ValueError, "positive and finite|numeric"):
                    experiment.validate_run_record(record, run=1, model_path=model)

    def test_execute_launches_exactly_ten_runs_and_writes_sealed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark"
            model = root / "model.onnx"
            runtime = root / "libonnxruntime.dylib"
            output = root / "evidence"

            def fake_run(command, **_kwargs):
                run = int(command[3])
                record = record_for(run, 2.0)
                record["model"] = command[5]
                Path(command[7]).write_text(json.dumps(record), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "synthetic stdout\n", "")

            with mock.patch.object(
                experiment, "preflight", return_value={"source_commit": "a" * 40}
            ), mock.patch.object(
                experiment.subprocess, "run", side_effect=fake_run
            ) as run_mock:
                summary = experiment.execute(benchmark, model, runtime, output)

            self.assertEqual(run_mock.call_count, 10)
            self.assertTrue(summary["accepted"])
            self.assertTrue((output / "commands.json").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())
            self.assertFalse((output / "ABORTED.json").exists())
            self.assertEqual(len(list(output.glob("run-*.json"))), 10)

    def test_execute_aborts_without_launching_a_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "evidence"
            failed = subprocess.CompletedProcess(["benchmark"], 9, "", "failed\n")
            with mock.patch.object(
                experiment, "preflight", return_value={"source_commit": "b" * 40}
            ), mock.patch.object(
                experiment.subprocess, "run", return_value=failed
            ) as run_mock:
                with self.assertRaisesRegex(RuntimeError, "process 1 failed"):
                    experiment.execute(
                        root / "benchmark",
                        root / "model.onnx",
                        root / "libonnxruntime.dylib",
                        output,
                    )

            self.assertEqual(run_mock.call_count, 1)
            self.assertTrue((output / "ABORTED.json").is_file())
            self.assertFalse((output / "summary.json").exists())
            with mock.patch.object(experiment, "preflight") as preflight_mock:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    experiment.execute(
                        root / "benchmark",
                        root / "model.onnx",
                        root / "libonnxruntime.dylib",
                        output,
                    )
                preflight_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
