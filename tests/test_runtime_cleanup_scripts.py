"""런타임 정리 shell 스크립트의 가벼운 회귀 테스트."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_install_timer_dry_run_renders_project_root(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["SYSTEMD_DIR"] = str(tmp_path / "systemd")

    result = subprocess.run(
        ["bash", "scripts/ops/install_runtime_cleanup_timer.sh", "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert str(PROJECT_ROOT) in result.stdout
    assert "@PROJECT_ROOT@" not in result.stdout
    assert f"=== {tmp_path / 'systemd' / 'cctv-runtime-cleanup.service'} ===" in result.stdout
    assert "Unit=cctv-runtime-cleanup.service" in result.stdout


def test_cleanup_runtime_preview_uses_python_bin_override(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    runtime_dir = data_dir / "runtime"
    crop_dir = runtime_dir / "appearance_crops"
    data_dir.mkdir()
    crop_dir.mkdir(parents=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": os.environ.get("PYTHON", "python3"),
            "RUNTIME_DATA_DIR": str(data_dir),
            "RUNTIME_DIR": str(runtime_dir),
            "APPEARANCES_DB": str(runtime_dir / "appearances.db"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/cleanup/cleanup_runtime_data.sh"],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "모드: 미리보기" in result.stdout
    assert f"runtime 산출물 경로: {runtime_dir}" in result.stdout
    assert f"crop 경로: {crop_dir}" in result.stdout
    assert "실제 반영하려면 --apply를 추가하세요." in result.stdout
