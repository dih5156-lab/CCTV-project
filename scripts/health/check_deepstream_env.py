"""Jetson DeepStream 런타임 환경과 nvinfer 설정 파일을 점검한다."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_CONFIGS = (
    "config/deepstream/config_infer_primary.txt",
    "config/deepstream/config_infer_helmet.txt",
    "config/deepstream/config_infer_pphuman.txt",
)
DEFAULT_GST_PLUGINS = ("nvstreammux", "nvinfer", "nvtracker", "nvv4l2decoder", "nvvideoconvert")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _print_result(result: CheckResult) -> None:
    prefix = "PASS" if result.ok else "FAIL"
    print(f"[{prefix}] {result.name}: {result.detail}")


def _check_python_module(module_name: str) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return CheckResult(
            f"Python import {module_name}",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    location = getattr(module, "__file__", None)
    detail = str(location) if location else "import 성공"
    return CheckResult(f"Python import {module_name}", True, detail)


def _check_gstreamer_python() -> CheckResult:
    try:
        gi = importlib.import_module("gi")
        gi.require_version("Gst", "1.0")
        importlib.import_module("gi.repository.Gst")
    except Exception as exc:
        return CheckResult("GStreamer Python Gst", False, f"{type(exc).__name__}: {exc}")
    return CheckResult("GStreamer Python Gst", True, "Gst 1.0 import 성공")


def _check_gst_plugin(plugin_name: str, timeout: float) -> CheckResult:
    try:
        completed = subprocess.run(
            ["gst-inspect-1.0", plugin_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return CheckResult(f"GStreamer plugin {plugin_name}", False, "gst-inspect-1.0 없음")
    except subprocess.TimeoutExpired:
        return CheckResult(f"GStreamer plugin {plugin_name}", False, f"{timeout:g}초 timeout")

    if completed.returncode == 0:
        return CheckResult(f"GStreamer plugin {plugin_name}", True, "gst-inspect 성공")

    output = (completed.stderr or completed.stdout).strip().splitlines()
    detail = output[-1] if output else f"exit={completed.returncode}"
    return CheckResult(f"GStreamer plugin {plugin_name}", False, detail)


def _check_deepstream_app(timeout: float) -> CheckResult:
    # Python import 성공만으로는 호스트 SDK의 공유 라이브러리 누락을 찾을 수 없다.
    # 실제 실행 파일을 로드하고 지원 중인 SDK 7.1 계열인지 함께 확인한다.
    name = "DeepStream native runtime"
    # 현재 이미지는 SDK를 호스트에서 연결하므로 /usr/bin 링크 없이도 실제 SDK를 검사한다.
    executable = shutil.which("deepstream-app") or "/opt/nvidia/deepstream/deepstream/bin/deepstream-app"
    try:
        completed = subprocess.run(
            [executable, "--version-all"],
            check=False, capture_output=True, text=True, timeout=timeout,
        )
    except OSError as exc:
        return CheckResult(name, False, str(exc))
    except subprocess.TimeoutExpired:
        return CheckResult(name, False, f"{timeout:g}초 timeout")
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0:
        return CheckResult(name, False, f"exit={completed.returncode}: {output[:1600]}")
    version = re.search(r"deepstream-app version\s+(\d+\.\d+(?:\.\d+)*)", output)
    if version is None or version.group(1).split(".")[:2] != ["7", "1"]:
        return CheckResult(name, False, f"DeepStream 7.1 필요: {output[:1600]}")
    return CheckResult(name, True, f"SDK {version.group(1)} 로드 성공; 영상 처리 검증은 별도")


def _parse_property_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    in_properties = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        # 클래스별 속성이 모델 경로를 덮거나 필수 [property] 누락을 숨기지 않게 한다.
        if line.startswith("["):
            in_properties = line == "[property]"
            continue
        if not in_properties or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _resolve_config_value(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _check_path(name: str, path: Path) -> CheckResult:
    # 빈 파일·자동 생성된 디렉터리를 준비된 모델/설정으로 오인하지 않는다.
    try:
        if not path.is_file():
            return CheckResult(name, False, f"파일 없음: {path}")
        size = path.stat().st_size
    except OSError as exc:
        return CheckResult(name, False, f"{path}: {exc}")
    return CheckResult(name, size > 0, f"{path} ({size} bytes)")


def _check_engine_load(path: Path, *, batch_size: Optional[int] = None) -> CheckResult:
    """엔진 로드와 지정한 배치의 profile 범위를 검사한다. 실제 추론 검증은 별도다."""
    name = f"TensorRT engine load {path}"
    batch_detail = "배치 미검증"
    try:
        trt = importlib.import_module("tensorrt")
        logger = trt.Logger(trt.Logger.ERROR)
        if not trt.init_libnvinfer_plugins(logger, ""):
            return CheckResult(name, False, "TensorRT 플러그인 초기화 실패")
        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(path.read_bytes())
            if engine is None:
                return CheckResult(name, False, "역직렬화 실패: 대상 GPU/TensorRT에서 엔진 재생성 필요")
            try:
                if batch_size is not None:
                    if batch_size <= 0:
                        raise ValueError("batch-size는 양수여야 합니다")
                    names = (engine.get_tensor_name(index) for index in range(engine.num_io_tensors))
                    inputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
                    if len(inputs) != 1 or len(engine.get_tensor_shape(inputs[0])) != 4:
                        raise ValueError("배치 검사는 단일 4차원 영상 입력 엔진만 지원합니다")
                    # nvinfer가 사용하는 profile 0을 검사해야 다른 profile의 범위로 오판하지 않는다.
                    shapes = engine.get_tensor_profile_shape(inputs[0], 0)
                    minimum, optimum, maximum = (int(shape[0]) for shape in shapes)
                    batch_detail = (
                        f"요청 batch={batch_size}; {inputs[0]} profile=0 "
                        f"min/opt/max={minimum}/{optimum}/{maximum}"
                    )
                    if not 0 < minimum <= optimum <= maximum:
                        raise ValueError(f"잘못된 배치 profile: {batch_detail}")
                    if not minimum <= batch_size <= maximum:
                        raise ValueError(f"배치 불일치: {batch_detail}; 요청 범위를 지원하는 엔진 필요")
            finally:
                # 실패한 배치 검사도 다음 모델 검사 전에 공유 GPU 메모리를 해제한다.
                del engine
    except Exception as exc:
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}")
    return CheckResult(name, True, f"대상 GPU 로드 성공; {batch_detail}; 실제 추론·메모리/성능 검증은 별도")


def _check_infer_config(
    config_path: Path, *, load_engines: bool = False, batch_size: Optional[int] = None,
) -> list[CheckResult]:
    results = [_check_path(f"nvinfer config {config_path.name}", config_path)]
    if not results[0].ok:
        return results

    try:
        values = _parse_property_file(config_path)
    except (OSError, UnicodeError) as exc:
        return [
            *results,
            CheckResult(f"nvinfer config parse {config_path.name}", False, str(exc)),
        ]

    if load_engines:
        try:
            # 앱의 카메라 수 override를 명시하지 않으면 설정 파일의 배치를 검사한다.
            batch_size = int(values.get("batch-size", "1")) if batch_size is None else batch_size
            if batch_size <= 0:
                raise ValueError("batch-size는 양수여야 합니다")
        except ValueError as exc:
            results.append(CheckResult(f"{config_path.name} batch-size", False, str(exc)))
            return results

    for key in ("model-engine-file", "labelfile-path"):
        value = values.get(key)
        if not value:
            if key == "labelfile-path":
                continue
            results.append(CheckResult(f"{config_path.name} {key}", False, "설정값 없음"))
            continue
        resolved = _resolve_config_value(config_path, value)
        path_result = _check_path(f"{config_path.name} {key}", resolved)
        results.append(path_result)
        if key == "model-engine-file" and load_engines and path_result.ok:
            results.append(_check_engine_load(resolved, batch_size=batch_size))
    return results


def build_checks(
    *,
    root: Path,
    config_paths: Iterable[str],
    gst_plugins: Iterable[str],
    timeout: float,
    skip_gst_plugins: bool,
    native_only: bool = False,
    load_engines: bool = False,
    files_only: bool = False,
    batch_size: Optional[int] = None,
) -> list[CheckResult]:
    results = []
    # NX가 없어도 복사할 파일을 검사하며 SDK/GPU 접근이나 엔진 생성을 하지 않는다.
    if not files_only:
        results.append(_check_deepstream_app(timeout))
        if not native_only:
            results.extend([_check_gstreamer_python(), _check_python_module("pyds")])

        if not skip_gst_plugins:
            results.extend(_check_gst_plugin(plugin, timeout) for plugin in gst_plugins)

    # 호스트 준비 단계에는 모델·pyds가 없어도 SDK 의존성을 먼저 검사할 수 있어야 한다.
    if native_only:
        return results

    for config_path_text in config_paths:
        config_path = Path(config_path_text)
        if not config_path.is_absolute():
            config_path = root / config_path
        results.extend(_check_infer_config(
            config_path.resolve(), load_engines=load_engines, batch_size=batch_size,
        ))

    # 실제 DeepStreamProcessor의 tracker 설정과 같은 파일을 검사한다.
    tracker_config = root / "config/deepstream/config_tracker_iou.yml"
    results.append(_check_path("DeepStream tracker config", tracker_config))
    return results


def _summarize(results: list[CheckResult], *, files_only: bool = False) -> int:
    failures = [result for result in results if not result.ok]
    if not failures:
        print("\nDeepStream 파일 점검 통과 (SDK/GPU 미검증)" if files_only else "\nDeepStream 환경 점검 통과")
        return 0

    print("\n확인이 필요한 항목")
    for result in failures:
        print(f"- {result.name}: {result.detail}")
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Jetson DeepStream 환경 점검")
    parser.add_argument("--root", default=str(_repo_root()), help="프로젝트 루트 경로")
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="추가 또는 대체할 nvinfer config 경로. 여러 번 지정 가능",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="gst-inspect timeout(초)")
    runtime_mode = parser.add_mutually_exclusive_group()
    runtime_mode.add_argument(
        "--native-only", action="store_true",
        help="SDK 실행·GStreamer 플러그인만 검사 (모델/Python 바인딩 검사 제외)",
    )
    runtime_mode.add_argument(
        "--load-engines", action="store_true",
        help="엔진 로드와 설정 배치를 검사 (GPU 메모리 사용, 실제 추론/성능 검사는 별도)",
    )
    runtime_mode.add_argument(
        "--files-only", action="store_true",
        help="SDK/GPU 없이 설정·엔진·라벨·tracker 파일만 검사 (엔진 생성/로드 제외)",
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="앱이 실제 요청하는 배치로 검사 (--load-engines와 단일 --config 필요)",
    )
    parser.add_argument(
        "--skip-gst-plugins",
        action="store_true",
        help="gst-inspect 플러그인 점검을 건너뜀",
    )
    parser.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력")
    args = parser.parse_args(argv)
    if args.batch_size is not None and (
        args.batch_size <= 0 or not args.load_engines or not args.configs or len(args.configs) != 1
    ):
        parser.error("--batch-size는 양수이며 --load-engines와 단일 --config가 필요합니다")

    root = Path(args.root).resolve()
    config_paths = args.configs if args.configs else DEFAULT_CONFIGS
    results = build_checks(
        root=root,
        config_paths=config_paths,
        gst_plugins=DEFAULT_GST_PLUGINS,
        timeout=args.timeout,
        skip_gst_plugins=args.skip_gst_plugins,
        native_only=args.native_only,
        load_engines=args.load_engines,
        files_only=args.files_only,
        batch_size=args.batch_size,
    )

    if args.json:
        payload = {
            "passed": all(result.ok for result in results),
            "files_only": args.files_only,
            "results": [
                {"name": result.name, "ok": result.ok, "detail": result.detail}
                for result in results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["passed"] else 1

    for result in results:
        _print_result(result)
    return _summarize(results, files_only=args.files_only)


if __name__ == "__main__":
    sys.exit(main())
