#!/usr/bin/env python3
"""호스트 환경에 맞는 Compose 파일만 실행하는 안전 진입점."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_CONFIG = {
    "jetson": ("docker-compose.jetson.yml", ".env.jetson", "edgex-jetson"),
    "server": ("docker-compose.yml", ".env", "edgex"),
    "windows": ("docker-compose.yml", ".env", "edgex"),
}


def is_jetson_host(
    *,
    system: str | None = None,
    machine: str | None = None,
    marker_paths: Sequence[Path] | None = None,
) -> bool:
    system_name = (system or platform.system()).lower()
    machine_name = (machine or platform.machine()).lower()
    if system_name != "linux" or machine_name not in {"aarch64", "arm64"}:
        return False

    markers = marker_paths or (
        Path("/etc/nv_tegra_release"),
        Path("/proc/device-tree/model"),
    )
    for marker in markers:
        try:
            if marker.name == "model":
                return "jetson" in marker.read_text(errors="ignore").lower()
            if marker.exists():
                return True
        except OSError:
            continue
    return False


def detect_target() -> str:
    if platform.system().lower() == "windows":
        return "windows"
    return "jetson" if is_jetson_host() else "server"


def build_compose_command(target: str, compose_args: Sequence[str]) -> list[str]:
    compose_file, env_file, project_name = TARGET_CONFIG[target]
    command = [
        "docker", "compose", "--project-name", project_name,
        "-f", str(PROJECT_ROOT / compose_file),
    ]
    env_path = PROJECT_ROOT / env_file
    if env_path.exists():
        command.extend(("--env-file", str(env_path)))
    command.extend(compose_args or ("ps",))
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jetson/서버/Windows에 맞는 Compose 파일만 실행합니다."
    )
    parser.add_argument(
        "--target", choices=("auto", "jetson", "server", "windows"), default="auto"
    )
    parser.add_argument(
        "--force-target", action="store_true", help="호스트/target 불일치 차단 해제"
    )
    parser.add_argument("compose_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    detected = detect_target()
    target = detected if args.target == "auto" else args.target

    if not args.force_target:
        if detected == "jetson" and target != "jetson":
            print("[차단] Jetson에서는 jetson Compose만 사용할 수 있습니다.", file=sys.stderr)
            return 2
        if detected != "jetson" and target == "jetson":
            print(f"[차단] {detected}에서 Jetson Compose 실행을 거부했습니다.", file=sys.stderr)
            return 2

    command = build_compose_command(target, args.compose_args)
    compose_file, env_file, project_name = TARGET_CONFIG[target]
    print(
        f"[compose] target={target} project={project_name} "
        f"file={compose_file} env={env_file}",
        flush=True,
    )
    try:
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    except FileNotFoundError:
        print("[오류] docker 명령을 찾을 수 없습니다.", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
