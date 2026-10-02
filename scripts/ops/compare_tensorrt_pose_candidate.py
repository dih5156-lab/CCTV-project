#!/usr/bin/env python3
"""Compare raw Pose tensors on paired local video inputs; not an accuracy gate.

Run in a network-none Docker container with read-only engine/video mounts.
No operational application, event publisher or device client is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def pose_tensor_difference(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if (
        reference.shape != candidate.shape
        or reference.ndim != 3
        or reference.shape[1] != 56
    ):
        raise ValueError("expected matching [batch, 56, proposals] Pose tensor shapes")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Pose outputs must be finite")
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    foreground = np.maximum(reference[:, 4, :], candidate[:, 4, :]) >= 0.25
    foreground_difference = difference.transpose(0, 2, 1)[foreground]
    return {
        "shape": list(reference.shape),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "p99_abs": float(np.quantile(difference, 0.99)),
        "person_score_max_abs": float(difference[:, 4, :].max()),
        "foreground_proposals": int(foreground.sum()),
        "foreground_max_abs": float(foreground_difference.max())
        if foreground.any()
        else None,
        "numerically_close": bool(
            np.allclose(reference, candidate, atol=1e-3, rtol=1e-3)
        ),
        "tolerance": {"atol": 1e-3, "rtol": 1e-3},
    }


def letterbox_pose(frame, size=640):
    import cv2

    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = cv2.resize(frame, (resized_width, resized_height))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = canvas[:, :, ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(rgb, dtype=np.float32) / 255.0


def load_runtime(path):
    import tensorrt as trt

    from src.core.ai._attribute_runtimes import TensorRTNamedOutputsRuntime

    serialized = path.read_bytes()
    owner = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = owner.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError(f"cannot load engine: {path.name}")
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    inputs = [
        name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    ]
    outputs = [
        name
        for name in names
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
    ]
    if len(inputs) != 1 or "output" not in outputs:
        raise ValueError("expected one input and a Pose output named 'output'")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("cannot create TensorRT execution context")
    adapter = TensorRTNamedOutputsRuntime(engine, context, inputs[0], outputs)
    details = {
        "path": str(path),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "tensorrt_version": trt.__version__,
        "input_name": inputs[0],
        "input_shape": list(engine.get_tensor_shape(inputs[0])),
        "profile": [
            list(shape) for shape in engine.get_tensor_profile_shape(inputs[0], 0)
        ],
        "outputs": outputs,
    }
    return {"owner": owner, "adapter": adapter, "details": details}


def infer_pose(model, tensor):
    adapter = model["adapter"]
    if not adapter.context.set_input_shape(adapter.input_name, tuple(tensor.shape)):
        raise ValueError(f"unsupported input shape: {tensor.shape}")
    output = adapter.run_named(tensor)["output"]
    if output.shape != (tensor.shape[0], 56, 8400) or not np.isfinite(output).all():
        raise ValueError(f"invalid Pose tensor: {output.shape}")
    return output


def run(args):
    import cv2

    if not Path("/.dockerenv").exists() or {
        path.name for path in Path("/sys/class/net").iterdir()
    } != {"lo"}:
        raise ValueError("network-none Docker isolation is required")
    if not args.video.is_file():
        raise ValueError("an existing local video is required")
    if args.output.exists():
        raise ValueError("output report already exists")
    reference = load_runtime(args.reference)
    candidate = load_runtime(args.candidate)
    capture = cv2.VideoCapture(str(args.video))
    inputs, samples = [], []
    try:
        for seconds in (5, 15, 25, 35):
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"cannot read frame at {seconds}s")
            tensor = letterbox_pose(frame)
            inputs.append(tensor)
            samples.append(
                {
                    "requested_seconds": seconds,
                    "input_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
                }
            )
    finally:
        capture.release()
    reference_outputs = np.concatenate(
        [infer_pose(reference, tensor) for tensor in inputs]
    )
    candidate_outputs = np.concatenate(
        [infer_pose(candidate, tensor) for tensor in inputs]
    )
    comparisons = {}
    for batch_size in (2, 4):
        outputs = np.concatenate(
            [
                infer_pose(
                    candidate, np.concatenate(inputs[index : index + batch_size])
                )
                for index in range(0, len(inputs), batch_size)
            ]
        )
        comparisons[str(batch_size)] = pose_tensor_difference(
            candidate_outputs, outputs
        )
    report = {
        "scope": "paired_raw_tensor_diagnostic",
        "quality_acceptance": "NOT_EVALUATED",
        "preprocessing": "640 RGB float32 /255; aspect-preserving resize; symmetric 114 padding",
        "reference": reference["details"],
        "candidate": candidate["details"],
        "samples": samples,
        "reference_vs_candidate_batch1": pose_tensor_difference(
            reference_outputs, candidate_outputs
        ),
        "candidate_batch1_vs_batched": comparisons,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
