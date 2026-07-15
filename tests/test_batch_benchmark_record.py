"""Validate the frozen D-33 paired-benchmark record without judging performance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess


EXPECTED_RECIPE = "byte_i_row_r=(i*37+r*17)%256"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_mode(mode: dict, expected_calls: int) -> float:
    require(mode["runtime_calls_per_group"] == expected_calls, "wrong call accounting")
    samples = mode["group_latency_ms"]
    require(len(samples) == 200, "paired mode must retain exactly 200 group samples")
    require(
        all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in samples),
        "paired samples must be positive and finite",
    )
    expected_throughput = 1_600_000.0 / sum(samples)
    actual_throughput = mode["items_per_second"]
    require(
        math.isclose(actual_throughput, expected_throughput, rel_tol=1.0e-12),
        "items/s did not use 1600 items divided by total milliseconds",
    )
    return actual_throughput


def run_and_validate(
    executable: Path, model: Path, output: Path, run: int, expected_order: str
) -> None:
    output.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            str(executable),
            "--paired-batch",
            "--paired-run",
            str(run),
            "--model",
            str(model),
            "--json-out",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    require(
        completed.returncode == 0,
        f"paired benchmark run {run} failed:\n{completed.stdout}\n{completed.stderr}",
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    require(record["schema"] == "cpp_ml.paired_batch.v1", "wrong record schema")
    require(record["run"] == run, "wrong paired run number")
    require(record["order"] == expected_order, "odd/even order was not frozen")
    require(record["batch_size"] == 8, "candidate batch size must remain eight")
    require(record["warmup_workloads_per_mode"] == 20, "wrong warm-up count")
    require(record["measured_workloads_per_mode"] == 200, "wrong sample count")
    require(record["input_recipe"] == EXPECTED_RECIPE, "wrong deterministic input recipe")
    require(Path(record["model"]) == model, "record did not identify its model")

    serial_throughput = validate_mode(record["serial_eight"], expected_calls=8)
    batch_throughput = validate_mode(record["batch_eight"], expected_calls=1)
    require(
        math.isclose(
            record["batch_to_serial_items_per_second_ratio"],
            batch_throughput / serial_throughput,
            rel_tol=1.0e-12,
        ),
        "paired ratio did not use the recorded mode throughputs",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    run_and_validate(
        args.benchmark,
        args.model,
        args.work_dir / "paired-run-1.json",
        run=1,
        expected_order="serial_then_batch",
    )
    run_and_validate(
        args.benchmark,
        args.model,
        args.work_dir / "paired-run-2.json",
        run=2,
        expected_order="batch_then_serial",
    )
    print("D-33 paired benchmark records passed structural and accounting checks")


if __name__ == "__main__":
    main()
