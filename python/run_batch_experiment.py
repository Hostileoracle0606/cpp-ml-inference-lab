"""Execute and aggregate the frozen v1.2 paired batch experiment exactly once."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_SHA256 = (
    "fec6ac786c75d2e328618a021763b408b993b9d6246772bf305eb3cd996b255c"
)
EXPECTED_ORT_SHA256 = (
    "183dca3132c8f0e2f0a1a30da3bcf7dee634cd6a7d5bf99b47f3d7ffe2791799"
)
EXPECTED_ORT_INSTALL_NAME = "@rpath/libonnxruntime.1.19.2.dylib"
EXPECTED_BUILD_SYSTEM = "Darwin/arm64"
EXPECTED_COMPILER = "AppleClang 21.0.0.21000099"
EXPECTED_OS_VERSION = "26.4"
EXPECTED_CPU = "Apple M4"
EXPECTED_SCHEMA = "cpp_ml.paired_batch.v1"
EXPECTED_RECIPE = "byte_i_row_r=(i*37+r*17)%256"
RUN_COUNT = 10
WARMUP_WORKLOADS = 20
MEASURED_WORKLOADS = 200
BATCH_SIZE = 8
ITEMS_PER_MODE = BATCH_SIZE * MEASURED_WORKLOADS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _run_checked(command: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        list(command), cwd=cwd, capture_output=True, check=False, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(samples: Sequence[float], quantile: float) -> float:
    _require(bool(samples), "cannot calculate a percentile without samples")
    _require(0.0 < quantile <= 1.0, "quantile must be in (0,1]")
    ordered = sorted(samples)
    rank = math.ceil(quantile * len(ordered))
    return ordered[rank - 1]


def even_median(values: Sequence[float]) -> float:
    _require(len(values) == RUN_COUNT, "the frozen median requires exactly ten ratios")
    ordered = sorted(values)
    return (ordered[4] + ordered[5]) / 2.0


def validate_metadata_probe(output: str) -> None:
    lines = {line.strip() for line in output.splitlines()}
    for expected_line in (
        "build_type: Release",
        f"system: {EXPECTED_BUILD_SYSTEM}",
        f"compiler: {EXPECTED_COMPILER}",
        "cxx: 201703",
        "inference: skipped (pass --model <path.onnx>)",
    ):
        _require(expected_line in lines,
                 f"benchmark metadata preflight is missing: {expected_line}")


def _positive_finite_number(value: Any, description: str) -> float:
    _require(type(value) in (int, float), f"{description} must be numeric")
    converted = float(value)
    _require(math.isfinite(converted) and converted > 0.0,
             f"{description} must be positive and finite")
    return converted


def _validate_mode(mode: Mapping[str, Any], expected_calls: int) -> Dict[str, Any]:
    _require(mode.get("runtime_calls_per_group") == expected_calls,
             "runtime call accounting differs from D-33")
    raw_samples = mode.get("group_latency_ms")
    _require(isinstance(raw_samples, list), "group latency samples must be a list")
    _require(len(raw_samples) == MEASURED_WORKLOADS,
             "each mode must retain exactly 200 measured workloads")
    samples = [
        _positive_finite_number(value, "group latency sample")
        for value in raw_samples
    ]
    expected_throughput = ITEMS_PER_MODE * 1000.0 / sum(samples)
    recorded_throughput = _positive_finite_number(
        mode.get("items_per_second"), "recorded items/s"
    )
    _require(
        math.isclose(recorded_throughput, expected_throughput, rel_tol=1.0e-12),
        "recorded items/s differs from the frozen aggregate formula",
    )
    return {"samples": samples, "items_per_second": recorded_throughput}


def validate_run_record(
    record: Mapping[str, Any], *, run: int, model_path: Path
) -> Dict[str, Any]:
    expected_order = "serial_then_batch" if run % 2 == 1 else "batch_then_serial"
    _require(record.get("schema") == EXPECTED_SCHEMA, "unexpected run schema")
    _require(record.get("run") == run, "run number differs from its invocation")
    _require(record.get("order") == expected_order, "run order differs from D-33")
    _require(record.get("batch_size") == BATCH_SIZE, "batch candidate is not eight")
    _require(record.get("warmup_workloads_per_mode") == WARMUP_WORKLOADS,
             "warm-up count differs from D-33")
    _require(record.get("measured_workloads_per_mode") == MEASURED_WORKLOADS,
             "measured workload count differs from D-33")
    _require(record.get("input_recipe") == EXPECTED_RECIPE,
             "input recipe differs from D-33")
    _require(Path(str(record.get("model"))).resolve() == model_path.resolve(),
             "run record identifies a different model")
    build = record.get("build")
    _require(isinstance(build, dict), "run record has no build metadata")
    _require(build.get("type") == "Release", "official run must use a Release build")
    _require(build.get("system") == EXPECTED_BUILD_SYSTEM,
             "build system differs from the frozen reference")
    _require(build.get("compiler") == EXPECTED_COMPILER,
             "compiler differs from the frozen reference")
    _require(build.get("cxx") == 201703,
             "official run must use the frozen C++17 language mode")

    serial = _validate_mode(record["serial_eight"], expected_calls=8)
    batch = _validate_mode(record["batch_eight"], expected_calls=1)
    ratio = batch["items_per_second"] / serial["items_per_second"]
    recorded_ratio = _positive_finite_number(
        record.get("batch_to_serial_items_per_second_ratio"), "paired ratio"
    )
    _require(math.isclose(recorded_ratio, ratio, rel_tol=1.0e-12),
             "recorded paired ratio differs from its mode throughputs")
    return {"run": run, "order": expected_order, "serial": serial,
            "batch": batch, "ratio": ratio}


def summarize_runs(validated_runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _require(len(validated_runs) == RUN_COUNT,
             "the frozen experiment requires exactly ten valid process runs")
    _require([entry["run"] for entry in validated_runs] == list(range(1, 11)),
             "run records must be complete and ordered 1 through 10")

    ratios = [float(entry["ratio"]) for entry in validated_runs]
    serial_samples = [
        sample
        for entry in validated_runs
        for sample in entry["serial"]["samples"]
    ]
    batch_samples = [
        sample
        for entry in validated_runs
        for sample in entry["batch"]["samples"]
    ]
    _require(len(serial_samples) == 2000 and len(batch_samples) == 2000,
             "pooled tail distributions must contain 2,000 samples per mode")

    median_ratio = even_median(ratios)
    favorable_runs = sum(ratio > 1.0 for ratio in ratios)
    serial_p95 = nearest_rank(serial_samples, 0.95)
    batch_p95 = nearest_rank(batch_samples, 0.95)
    gates = {
        "median_items_per_second_ratio_at_least_1_5": median_ratio >= 1.5,
        "at_least_8_of_10_runs_favor_batch": favorable_runs >= 8,
        "batch_group_p95_not_above_serial_group_p95": batch_p95 <= serial_p95,
    }
    return {
        "schema": "cpp_ml.v1_2_batch_experiment_summary.v1",
        "run_count": RUN_COUNT,
        "paired_items_per_second_ratios": ratios,
        "median_items_per_second_ratio": median_ratio,
        "median_items_per_second_improvement_percent": (median_ratio - 1.0) * 100.0,
        "favorable_run_count": favorable_runs,
        "pooled_sample_count_per_mode": len(serial_samples),
        "serial_eight_group_p95_ms": serial_p95,
        "batch_eight_group_p95_ms": batch_p95,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mark_aborted(output_dir: Path, value: Mapping[str, Any]) -> None:
    """Preserve the abort reason when storage is still writable."""

    try:
        _write_json(output_dir / "ABORTED.json", value)
    except OSError:
        pass


def preflight(benchmark: Path, model: Path, onnxruntime_library: Path) -> Dict[str, Any]:
    _require(benchmark.is_file(), f"benchmark executable does not exist: {benchmark}")
    _require(model.is_file(), f"ONNX model does not exist: {model}")
    _require(onnxruntime_library.is_file(),
             f"ONNX Runtime library does not exist: {onnxruntime_library}")
    model_sha256 = sha256_file(model)
    _require(model_sha256 == EXPECTED_MODEL_SHA256,
             "ONNX SHA-256 differs from the frozen v1.1 r2 artifact")
    onnxruntime_sha256 = sha256_file(onnxruntime_library)
    _require(onnxruntime_sha256 == EXPECTED_ORT_SHA256,
             "ONNX Runtime library SHA-256 differs from the frozen 1.19.2 SDK")
    linkage = _run_checked(["otool", "-L", str(benchmark.resolve())], cwd=REPO_ROOT)
    _require(EXPECTED_ORT_INSTALL_NAME in linkage,
             "benchmark is not linked to the frozen ONNX Runtime 1.19.2 install name")
    load_commands = _run_checked(
        ["otool", "-l", str(benchmark.resolve())], cwd=REPO_ROOT
    )
    _require(str(onnxruntime_library.resolve().parent) in load_commands,
             "benchmark rpath does not select the frozen ONNX Runtime directory")
    os_version = platform.mac_ver()[0]
    cpu_brand = _run_checked(
        ["sysctl", "-n", "machdep.cpu.brand_string"], cwd=REPO_ROOT
    )
    _require(os_version == EXPECTED_OS_VERSION,
             "macOS version differs from the frozen reference environment")
    _require(platform.machine() == "arm64" and cpu_brand == EXPECTED_CPU,
             "machine differs from the frozen Apple M4 arm64 reference")
    source_root = Path(
        _run_checked(["git", "rev-parse", "--show-toplevel"], cwd=REPO_ROOT)
    ).resolve()
    _require(source_root == REPO_ROOT.resolve(), "script is not running in its repository")
    status = _run_checked(["git", "status", "--porcelain=v1"], cwd=REPO_ROOT)
    _require(status == "", "official measurement requires a clean source worktree")
    commit = _run_checked(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    metadata_probe = _run_checked(
        [str(benchmark.resolve()), "--warmup", "1", "--iterations", "1"],
        cwd=REPO_ROOT,
    )
    validate_metadata_probe(metadata_probe)
    return {
        "schema": "cpp_ml.v1_2_batch_experiment_preflight.v1",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": commit,
        "benchmark_path": str(benchmark.resolve()),
        "benchmark_sha256": sha256_file(benchmark),
        "benchmark_metadata_probe": metadata_probe.splitlines(),
        "model_path": str(model.resolve()),
        "model_sha256": model_sha256,
        "onnxruntime_library_path": str(onnxruntime_library.resolve()),
        "onnxruntime_library_sha256": onnxruntime_sha256,
        "onnxruntime_install_name": EXPECTED_ORT_INSTALL_NAME,
        "benchmark_linkage": linkage.splitlines(),
        "onnxruntime_rpath": str(onnxruntime_library.resolve().parent),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_brand": cpu_brand,
        "os_version": os_version,
        "processor": platform.processor(),
        "policy": {
            "runs": RUN_COUNT,
            "warmup_workloads_per_mode": WARMUP_WORKLOADS,
            "measured_workloads_per_mode": MEASURED_WORKLOADS,
            "input_recipe": EXPECTED_RECIPE,
            "failure_policy": "abort_without_selective_replacement",
        },
    }


def execute(
    benchmark: Path, model: Path, onnxruntime_library: Path, output_dir: Path
) -> Dict[str, Any]:
    _require(not output_dir.exists(),
             "output directory already exists; official runs are never overwritten")
    preflight_record = preflight(benchmark, model, onnxruntime_library)
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "preflight.json", preflight_record)

    validated_runs: List[Mapping[str, Any]] = []
    commands = [
        [
            str(benchmark.resolve()),
            "--paired-batch",
            "--paired-run",
            str(run),
            "--model",
            str(model.resolve()),
            "--json-out",
            str((output_dir / f"run-{run:02d}.json").resolve()),
        ]
        for run in range(1, RUN_COUNT + 1)
    ]
    _write_json(output_dir / "commands.json", {"commands": commands})
    for run, command in enumerate(commands, start=1):
        record_path = output_dir / f"run-{run:02d}.json"
        try:
            completed = subprocess.run(
                command, cwd=REPO_ROOT, capture_output=True, check=False, text=True
            )
            (output_dir / f"run-{run:02d}.stdout.log").write_text(
                completed.stdout, encoding="utf-8"
            )
            (output_dir / f"run-{run:02d}.stderr.log").write_text(
                completed.stderr, encoding="utf-8"
            )
        except OSError as error:
            _mark_aborted(output_dir, {
                "run": run,
                "reason": "process launch or log capture failed; do not replace this run",
                "error": str(error),
            })
            raise RuntimeError(
                f"paired benchmark process {run} could not be launched or recorded"
            ) from error
        if completed.returncode != 0:
            _mark_aborted(output_dir, {
                "run": run,
                "returncode": completed.returncode,
                "reason": "benchmark process failed; do not selectively replace this run",
            })
            raise RuntimeError(f"paired benchmark process {run} failed; experiment aborted")
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            validated_runs.append(
                validate_run_record(record, run=run, model_path=model)
            )
        except Exception as error:
            _mark_aborted(output_dir, {
                "run": run,
                "reason": "malformed benchmark record; do not selectively replace this run",
                "error": str(error),
            })
            raise RuntimeError(
                f"paired benchmark record {run} was invalid; experiment aborted"
            ) from error

    summary = summarize_runs(validated_runs)
    summary["preflight"] = preflight_record
    summary["commands"] = commands
    summary["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json(output_dir / "summary.json", summary)

    checksum_lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen ten-process v1.2 batch experiment once."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--onnxruntime-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = execute(
            args.benchmark, args.model, args.onnxruntime_library, args.output_dir
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"[batch-experiment] FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(
        "[batch-experiment] "
        f"median_ratio={summary['median_items_per_second_ratio']:.6f} "
        f"favorable={summary['favorable_run_count']}/10 "
        f"serial_p95={summary['serial_eight_group_p95_ms']:.6f}ms "
        f"batch_p95={summary['batch_eight_group_p95_ms']:.6f}ms "
        f"result={'PASS' if summary['accepted'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
