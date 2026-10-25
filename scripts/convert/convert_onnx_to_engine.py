"""고정한 ONNX 한 개를 검증하고 대상 Jetson에서 FP16 엔진을 생성한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_shape(value: str) -> tuple[int, ...]:
    dimensions = tuple(int(part) for part in value.split("x"))
    if len(dimensions) != 4 or any(part <= 0 for part in dimensions):
        raise ValueError(f"양수 NCHW 4차원 shape가 필요합니다: {value}")
    return dimensions


def has_external_data(message) -> bool:
    # 단일 ONNX 해시로 별도 가중치 파일까지 보증할 수 없으므로 중첩 그래프도 거부한다.
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        return bool(message.external_data) or message.data_location == 1
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            children = [value] if hasattr(value, "ListFields") else value
            if any(has_external_data(child) for child in children):
                return True
    return False


def validate_onnx_artifact(
    path: Path, expected_sha256: str, input_name: str, shapes: list[tuple[int, ...]],
) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"비어 있지 않은 ONNX 파일이 필요합니다: {path}")
    if file_sha256(path) != expected_sha256.lower():
        raise ValueError(f"ONNX SHA256 불일치: {path}")
    if any(not low <= optimum <= high for low, optimum, high in zip(*shapes)):
        raise ValueError("shape는 차원별 min <= opt <= max 조건이 필요합니다.")
    try:
        import onnx

        model = onnx.load(path, load_external_data=False)
    except Exception as exc:
        raise ValueError(f"ONNX 읽기 실패(변환 환경의 onnx 패키지 필요): {exc}") from exc
    if has_external_data(model):
        raise ValueError("외부 가중치 ONNX는 지원하지 않습니다. 단일 파일 원본을 준비하세요.")
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    if [item.name for item in inputs] != [input_name]:
        raise ValueError(f"ONNX 입력 이름/개수 불일치: {[item.name for item in inputs]}")
    dimensions = inputs[0].type.tensor_type.shape.dim
    if len(dimensions) != 4:
        raise ValueError("NCHW 4차원 입력 ONNX만 지원합니다.")
    for shape in shapes:
        if any(dim.HasField("dim_value") and dim.dim_value != size for dim, size in zip(dimensions, shape)):
            raise ValueError(f"ONNX 고정 차원과 shape 불일치: {shape}")
    return any(not dim.HasField("dim_value") for dim in dimensions)


def build_trtexec_command(args, engine_path: Path, help_text: str | None = None) -> list[str]:
    command = [
        args.trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={engine_path}",
        "--fp16",
    ]
    # 고정 입력 ONNX에 동적 profile을 지정하면 TRT 8.5가 생성을 거부한다.
    if getattr(args, "dynamic_input", True):
        command.extend([
            f"--minShapes={args.input_name}:{args.min_shape}",
            f"--optShapes={args.input_name}:{args.opt_shape}",
            f"--maxShapes={args.input_name}:{args.max_shape}",
        ])
    command.extend(["--memPoolSize=workspace:1024", "--avgTiming=1"])
    # JetPack 5의 TRT 8.5는 buildOnly를 사용하며 builderOptimizationLevel이 없다.
    if help_text is None or "--builderOptimizationLevel" in help_text:
        command.append("--builderOptimizationLevel=0")
    if help_text is None or "--skipInference" in help_text:
        command.append("--skipInference")
    elif "--buildOnly" in help_text:
        command.append("--buildOnly")
    else:
        raise ValueError("trtexec에서 추론 없이 엔진을 생성하는 옵션을 확인하지 못했습니다.")
    return command


def convert_to_engine(args) -> dict:
    if not os.path.isfile(args.trtexec) or not os.access(args.trtexec, os.X_OK):
        raise FileNotFoundError(f"실행 가능한 trtexec가 필요합니다: {args.trtexec}")
    help_text = subprocess.run(
        [args.trtexec, "--help"], check=True, capture_output=True, text=True, timeout=15,
    ).stdout
    build_trtexec_command(args, args.engine, help_text)
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trt-build-", dir=args.engine.parent) as directory:
        temporary_engine = Path(directory) / args.engine.name
        command = build_trtexec_command(args, temporary_engine, help_text)
        subprocess.run(command, check=True, stdout=sys.stderr)
        if not temporary_engine.is_file() or temporary_engine.stat().st_size == 0:
            raise RuntimeError("trtexec가 비어 있지 않은 엔진을 생성하지 않았습니다.")
        if file_sha256(args.onnx) != args.sha256.lower():
            raise ValueError("생성 중 ONNX가 변경되어 엔진을 보존하지 않습니다.")
        engine_sha256 = file_sha256(temporary_engine)
        # 실패 중간 파일은 격리하고, 검사 이후 다른 프로세스가 만든 엔진도 덮어쓰지 않는다.
        os.link(temporary_engine, args.engine)
    return {"state": "BUILT_NOT_VALIDATED", "command": command, "engine_sha256": engine_sha256}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--sha256", required=True, help="확인된 원본의 SHA256")
    parser.add_argument("--input-name", required=True)
    parser.add_argument("--min-shape", required=True)
    parser.add_argument("--opt-shape", required=True)
    parser.add_argument("--max-shape", required=True)
    parser.add_argument("--trtexec", default=os.environ.get("TRTEXEC"))
    parser.add_argument("--build", action="store_true", help="대상 Jetson에서 실제 생성; 기본은 검증/명령 출력만")
    args = parser.parse_args(argv)
    try:
        args.onnx = args.onnx.resolve(strict=True)
        args.engine = args.engine.absolute()
        if os.path.lexists(args.engine):
            raise FileExistsError(f"기존 엔진/링크/디렉터리는 덮어쓰지 않습니다: {args.engine}")
        shapes = [parse_shape(value) for value in (args.min_shape, args.opt_shape, args.max_shape)]
        args.dynamic_input = validate_onnx_artifact(args.onnx, args.sha256, args.input_name, shapes)
        # 로컬 dry-run은 SDK 탐색 성공 여부와 관계없이 GPU 없이 실행한다.
        args.trtexec = args.trtexec or shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
        result = convert_to_engine(args) if args.build else {
            "state": "DRY_RUN", "command": build_trtexec_command(args, args.engine),
        }
        result.update(onnx_sha256=args.sha256.lower(), engine=str(args.engine))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"엔진 준비 실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
