"""Generate deterministic valid and contract-negative ONNX models for C++ tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OPSET = 17
IR_VERSION = 8


def save_linear_model(
    path: Path,
    *,
    input_name: str = "input",
    output_name: str = "logits",
    input_type: int = TensorProto.FLOAT,
    input_shape=("batch", 3, 32, 32),
    output_shape=("batch", 10),
    actual_classes: int = 10,
    extra_output: bool = False,
) -> None:
    fixed_input_elements = 1
    for dimension, fallback in zip(input_shape[1:], (3, 32, 32)):
        fixed_input_elements *= dimension if isinstance(dimension, int) else fallback

    numpy_dtype = np.float64 if input_type == TensorProto.DOUBLE else np.float32
    weights = numpy_helper.from_array(
        np.zeros((fixed_input_elements, actual_classes), dtype=numpy_dtype),
        name="weights",
    )
    nodes = [
        helper.make_node("Flatten", [input_name], ["flattened"], axis=1),
        helper.make_node("MatMul", ["flattened", "weights"], [output_name]),
    ]
    outputs = [helper.make_tensor_value_info(output_name, input_type, list(output_shape))]
    if extra_output:
        nodes.append(helper.make_node("Identity", [output_name], ["auxiliary"]))
        outputs.append(
            helper.make_tensor_value_info(
                "auxiliary", input_type, ["batch", actual_classes]
            )
        )

    graph = helper.make_graph(
        nodes,
        path.stem,
        [helper.make_tensor_value_info(input_name, input_type, list(input_shape))],
        outputs,
        initializer=[weights],
    )
    model = helper.make_model(
        graph,
        producer_name="cpp-ml-inference-lab-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, path)


def save_dynamic_class_model(path: Path, declared_shape=(1, "classes")) -> None:
    initializers = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="ends"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps"),
    ]
    graph = helper.make_graph(
        [
            helper.make_node("NonZero", ["input"], ["indices"]),
            helper.make_node(
                "Slice",
                ["indices", "starts", "ends", "axes", "steps"],
                ["sliced"],
            ),
            helper.make_node(
                "Cast", ["sliced"], ["logits"], to=TensorProto.FLOAT
            ),
        ],
        path.stem,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, list(declared_shape))],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="cpp-ml-inference-lab-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, path)


def save_runtime_batch_mismatch_model(path: Path) -> None:
    pads = numpy_helper.from_array(
        np.array([0, 0, 0, 6], dtype=np.int64), name="pads"
    )
    pad_value = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="pad_value")
    graph = helper.make_graph(
        [
            helper.make_node("NonZero", ["input"], ["indices"]),
            helper.make_node("Transpose", ["indices"], ["transposed"], perm=[1, 0]),
            helper.make_node(
                "Cast", ["transposed"], ["float_indices"], to=TensorProto.FLOAT
            ),
            helper.make_node(
                "Pad", ["float_indices", "pads", "pad_value"], ["logits"]
            ),
        ],
        path.stem,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 10])],
        initializer=[pads, pad_value],
    )
    model = helper.make_model(
        graph,
        producer_name="cpp-ml-inference-lab-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, path)


def save_wrong_output_dtype_model(path: Path) -> None:
    weights = numpy_helper.from_array(
        np.zeros((3 * 32 * 32, 10), dtype=np.float32), name="weights"
    )
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["input"], ["flattened"], axis=1),
            helper.make_node("MatMul", ["flattened", "weights"], ["raw_logits"]),
            helper.make_node(
                "Cast", ["raw_logits"], ["logits"], to=TensorProto.DOUBLE
            ),
        ],
        path.stem,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("logits", TensorProto.DOUBLE, ["batch", 10])],
        initializer=[weights],
    )
    model = helper.make_model(
        graph,
        producer_name="cpp-ml-inference-lab-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, path)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_fixture in output_dir.glob("*.onnx"):
        stale_fixture.unlink()

    save_linear_model(output_dir / "valid.onnx")
    save_linear_model(output_dir / "wrong_input_name.onnx", input_name="features")
    save_linear_model(output_dir / "wrong_output_name.onnx", output_name="scores")
    save_linear_model(
        output_dir / "wrong_input_dtype.onnx", input_type=TensorProto.DOUBLE
    )
    save_linear_model(
        output_dir / "wrong_input_rank.onnx",
        input_shape=("batch", 3, 32),
    )
    save_linear_model(
        output_dir / "wrong_input_batch.onnx",
        input_shape=(2, 3, 32, 32),
        output_shape=(2, 10),
    )
    save_linear_model(
        output_dir / "wrong_input_channels.onnx",
        input_shape=("batch", 1, 32, 32),
    )
    save_linear_model(
        output_dir / "wrong_input_height.onnx",
        input_shape=("batch", 3, 31, 32),
    )
    save_linear_model(
        output_dir / "wrong_input_width.onnx",
        input_shape=("batch", 3, 32, 31),
    )
    save_linear_model(
        output_dir / "dynamic_channel.onnx",
        input_shape=("batch", "channels", 32, 32),
    )
    save_linear_model(
        output_dir / "dynamic_height.onnx",
        input_shape=("batch", 3, "height", 32),
    )
    save_linear_model(
        output_dir / "dynamic_width.onnx",
        input_shape=("batch", 3, 32, "width"),
    )
    save_linear_model(output_dir / "extra_output.onnx", extra_output=True)
    save_wrong_output_dtype_model(output_dir / "wrong_output_dtype.onnx")
    save_linear_model(
        output_dir / "wrong_output_shape.onnx",
        output_shape=("batch", 9),
        actual_classes=9,
    )
    save_linear_model(
        output_dir / "wrong_output_rank.onnx",
        output_shape=("batch", 1, 10),
    )
    save_dynamic_class_model(output_dir / "dynamic_classes.onnx")
    save_runtime_batch_mismatch_model(output_dir / "runtime_batch_mismatch.onnx")
    save_dynamic_class_model(
        output_dir / "runtime_class_mismatch.onnx", declared_shape=(1, 10)
    )
    (output_dir / "corrupt.onnx").write_bytes(b"not-an-onnx-model\x00\xff")

    generated = sorted(path.name for path in output_dir.glob("*.onnx"))
    if len(generated) != 20:
        raise RuntimeError(f"expected 20 ONNX fixtures, generated {len(generated)}")
    print(f"generated {len(generated)} deterministic ONNX fixtures in {output_dir}")
    for name in generated:
        print(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
