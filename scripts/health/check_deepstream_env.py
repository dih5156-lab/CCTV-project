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
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("[") or "=" not in line:
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
    if not path.exists():
        return CheckResult(name, False, f"없음: {path}")
    if path.is_file():
        return CheckResult(name, True, f"{path} ({path.stat().st_size} bytes)")
    return CheckResult(name, True, str(path))


def _check_infer_config(config_path: Path) -> list[CheckResult]:
    results = [_check_path(f"nvinfer config {config_path.name}", config_path)]
    if not config_path.exists():
        return results

    try:
        values = _parse_property_file(config_path)
    except OSError as exc:
        return [
            *results,
            CheckResult(f"nvinfer config parse {config_path.name}", False, str(exc)),
        ]

    for key in ("model-engine-file", "labelfile-path"):
        value = values.get(key)
        if not value:
            if key == "labelfile-path":
                continue
            results.append(CheckResult(f"{config_path.name} {key}", False, "설정값 없음"))
            continue
        resolved = _resolve_config_value(config_path, value)
        results.append(_check_path(f"{config_path.name} {key}", resolved))
    return results


def build_checks(
    *,
    root: Path,
    config_paths: Iterable[str],
    gst_plugins: Iterable[str],
    timeout: float,
    skip_gst_plugins: bool,
    native_only: bool = False,
) -> list[CheckResult]:
    results = [_check_deepstream_app(timeout)]
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
        results.extend(_check_infer_config(config_path.resolve()))

    tracker_config = root / "config/deepstream/config_tracker.txt"
    results.append(_check_path("DeepStream tracker config", tracker_config))
    return results


def _summarize(results: list[CheckResult]) -> int:
    failures = [result for result in results if not result.ok]
    if not failures:
        print("\nDeepStream 환경 점검 통과")
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
    parser.add_argument(
        "--native-only", action="store_true",
        help="SDK 실행·GStreamer 플러그인만 검사 (모델/Python 바인딩 검사 제외)",
    )
    parser.add_argument(
        "--skip-gst-plugins",
        action="store_true",
        help="gst-inspect 플러그인 점검을 건너뜀",
    )
    parser.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    config_paths = args.configs if args.configs else DEFAULT_CONFIGS
    results = build_checks(
        root=root,
        config_paths=config_paths,
        gst_plugins=DEFAULT_GST_PLUGINS,
        timeout=args.timeout,
        skip_gst_plugins=args.skip_gst_plugins,
        native_only=args.native_only,
    )

    if args.json:
        payload = {
            "passed": all(result.ok for result in results),
            "results": [
                {"name": result.name, "ok": result.ok, "detail": result.detail}
                for result in results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["passed"] else 1

    for result in results:
        _print_result(result)
    return _summarize(results)


if __name__ == "__main__":
    sys.exit(main())
