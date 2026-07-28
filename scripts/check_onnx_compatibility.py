#!/usr/bin/env python3
"""Validate an ONNX model against OpenCV DNN and TensorRT's ONNX parser."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Shape must contain comma-separated integers."
        ) from error

    if not shape or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError(
            "Every input-shape dimension must be greater than zero."
        )
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether an ONNX model loads and runs with OpenCV DNN, "
            "and whether TensorRT can parse it before engine creation."
        )
    )
    parser.add_argument("model", type=Path, help="Path to the ONNX model.")
    parser.add_argument(
        "--input-shape",
        type=parse_shape,
        default=None,
        metavar="N,C,H,W",
        help=(
            "Run an OpenCV dummy inference with this NCHW input shape. "
            "Example for PFLD: 1,3,112,112."
        ),
    )
    parser.add_argument(
        "--expect-output-elements",
        type=int,
        default=None,
        help=(
            "Require the OpenCV output to contain this many values. "
            "PFLD with 68 (x, y) landmarks uses 136."
        ),
    )
    parser.add_argument(
        "--skip-opencv-run",
        action="store_true",
        help="Only load the model in OpenCV; do not run dummy inference.",
    )
    return parser.parse_args()


def element_count(shape: Sequence[int]) -> int | None:
    if any(dimension <= 0 for dimension in shape):
        return None
    return math.prod(shape)


def check_onnx_structure(model_path: Path) -> bool:
    print("\n=== ONNX structure check ===")
    try:
        import onnx
    except ImportError:
        print(
            "[SKIP] The optional onnx package is not installed. "
            "OpenCV and TensorRT checks will still run."
        )
        return True

    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
    except Exception as error:
        print(f"[FAIL] ONNX structural validation failed:\n{error}")
        return False

    opsets = ", ".join(
        f"{item.domain or 'ai.onnx'}:{item.version}"
        for item in model.opset_import
    )
    print(f"ONNX: {onnx.__version__}")
    print(f"Opset: {opsets}")
    print("[PASS] onnx.checker accepted the model structure.")
    return True


def check_opencv(
    model_path: Path,
    input_shape: tuple[int, ...] | None,
    expected_output_elements: int | None,
    skip_run: bool,
) -> bool:
    print("\n=== OpenCV DNN check ===")
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        print(f"[FAIL] OpenCV/NumPy import failed: {error}")
        return False

    print(f"OpenCV: {cv2.__version__}")
    try:
        network = cv2.dnn.readNetFromONNX(str(model_path))
    except cv2.error as error:
        print(f"[FAIL] OpenCV could not load the ONNX model:\n{error}")
        return False

    print("[PASS] OpenCV loaded the ONNX model.")
    if skip_run:
        print("[SKIP] OpenCV dummy inference was disabled.")
        return True
    if input_shape is None:
        print(
            "[SKIP] OpenCV dummy inference needs --input-shape "
            "(for example, 1,3,112,112)."
        )
        return True

    dummy_input = np.zeros(input_shape, dtype=np.float32)
    try:
        network.setInput(dummy_input)
        output = np.asarray(network.forward())
    except cv2.error as error:
        print(f"[FAIL] OpenCV dummy inference failed:\n{error}")
        return False

    print(f"OpenCV output shape: {tuple(output.shape)}")
    print(f"OpenCV output dtype: {output.dtype}")
    if not np.isfinite(output).all():
        print("[FAIL] OpenCV output contains NaN or infinity.")
        return False

    if (
        expected_output_elements is not None
        and output.size != expected_output_elements
    ):
        print(
            "[FAIL] Unexpected OpenCV output size: "
            f"expected {expected_output_elements}, got {output.size}."
        )
        return False

    print("[PASS] OpenCV dummy inference returned finite output.")
    return True


def check_tensorrt(
    model_path: Path,
    expected_output_elements: int | None,
) -> bool:
    print("\n=== TensorRT ONNX parser check ===")
    try:
        import tensorrt as trt
    except ImportError as error:
        print(f"[FAIL] TensorRT Python import failed: {error}")
        print(
            "Use the JetPack-provided Python environment that contains "
            "the tensorrt package."
        )
        return False

    print(f"TensorRT: {trt.__version__}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    )
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    model_bytes = model_path.read_bytes()
    if not parser.parse(model_bytes):
        print(f"[FAIL] TensorRT reported {parser.num_errors} parser error(s).")
        for index in range(parser.num_errors):
            print(f"  [{index + 1}] {parser.get_error(index)}")
        return False

    print("[PASS] TensorRT parsed the ONNX graph.")
    print(f"Inputs: {network.num_inputs}")
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        print(
            f"  - {tensor.name}: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}"
        )

    print(f"Outputs: {network.num_outputs}")
    output_sizes: list[int | None] = []
    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        shape = tuple(tensor.shape)
        output_sizes.append(element_count(shape))
        print(
            f"  - {tensor.name}: shape={shape}, "
            f"dtype={tensor.dtype}"
        )

    if expected_output_elements is not None:
        static_sizes = [size for size in output_sizes if size is not None]
        if expected_output_elements not in static_sizes:
            if not static_sizes:
                print(
                    "[WARN] TensorRT output is dynamic, so the expected "
                    "output size cannot be confirmed until engine creation."
                )
            else:
                print(
                    "[FAIL] No TensorRT output has the expected element "
                    f"count {expected_output_elements}; got {static_sizes}."
                )
                return False

    return True


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()

    print(f"Model: {model_path}")
    if not model_path.is_file():
        print(f"[FAIL] ONNX model not found: {model_path}")
        return 1
    if model_path.suffix.lower() != ".onnx":
        print("[FAIL] Model path must end with .onnx.")
        return 1
    if (
        args.expect_output_elements is not None
        and args.expect_output_elements <= 0
    ):
        print("[FAIL] --expect-output-elements must be greater than zero.")
        return 1

    structure_ok = check_onnx_structure(model_path)
    opencv_ok = check_opencv(
        model_path=model_path,
        input_shape=args.input_shape,
        expected_output_elements=args.expect_output_elements,
        skip_run=args.skip_opencv_run,
    )
    tensorrt_ok = check_tensorrt(
        model_path=model_path,
        expected_output_elements=args.expect_output_elements,
    )

    print("\n=== Result ===")
    if structure_ok and opencv_ok and tensorrt_ok:
        print(
            "[PASS] ONNX model passed structural, OpenCV, "
            "and TensorRT checks."
        )
        return 0

    print("[FAIL] ONNX compatibility check failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

