"""런타임 정리 shell 스크립트의 가벼운 회귀 테스트."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_script(
    command: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _read_env_values(env_file: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in env_file.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    )


def test_install_timer_dry_run_renders_project_root(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["SYSTEMD_DIR"] = str(tmp_path / "systemd")

    result = _run_script(
        ["bash", "scripts/ops/install_runtime_cleanup_timer.sh", "--dry-run"],
        env=env,
    )

    assert result.returncode == 0
    assert str(PROJECT_ROOT) in result.stdout
    assert "@PROJECT_ROOT@" not in result.stdout
    assert (
        f"=== {tmp_path / 'systemd' / 'cctv-runtime-cleanup.service'} ==="
        in result.stdout
    )
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

    result = _run_script(
        ["bash", "scripts/cleanup/cleanup_runtime_data.sh"],
        env=env,
    )

    assert result.returncode == 0
    assert "모드: 미리보기" in result.stdout
    assert f"runtime 산출물 경로: {runtime_dir}" in result.stdout
    assert f"crop 경로: {crop_dir}" in result.stdout
    assert "=== SQLite outbox cleanup ===" in result.stdout
    assert "실제 반영하려면 --apply를 추가하세요." in result.stdout


def test_outbox_cleanup_deletes_only_expired_sent_rows(tmp_path: Path) -> None:
    http_db = tmp_path / "http.db"
    mqtt_db = tmp_path / "mqtt.db"
    now_ms = int(time.time() * 1000)
    old_ms = now_ms - (8 * 24 * 60 * 60 * 1000)
    recent_ms = now_ms - (2 * 24 * 60 * 60 * 1000)

    for db_path, table_name in (
        (http_db, "http_event_outbox"),
        (mqtt_db, "mqtt_event_outbox"),
    ):
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    sent_at_ms INTEGER
                )
                """
            )
            connection.executemany(
                f"INSERT INTO {table_name} (id, status, sent_at_ms) VALUES (?, ?, ?)",
                [
                    (1, "sent", old_ms),
                    (2, "sent", recent_ms),
                    (3, "pending", old_ms),
                ],
            )

    result = _run_script(
        [
            os.environ.get("PYTHON", "python3"),
            "scripts/cleanup/cleanup_outbox_databases.py",
            "--apply",
            "--http-db",
            str(http_db),
            "--mqtt-db",
            str(mqtt_db),
            "--retention-days",
            "7",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "deleted=1" in result.stdout
    for db_path, table_name in (
        (http_db, "http_event_outbox"),
        (mqtt_db, "mqtt_event_outbox"),
    ):
        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(
                f"SELECT id, status FROM {table_name} ORDER BY id"
            ).fetchall()
        assert rows == [(2, "sent"), (3, "pending")]


def test_outbox_cleanup_preview_does_not_delete_rows(tmp_path: Path) -> None:
    http_db = tmp_path / "http.db"
    old_ms = int(time.time() * 1000) - (8 * 24 * 60 * 60 * 1000)
    with sqlite3.connect(http_db) as connection:
        connection.execute(
            """
            CREATE TABLE http_event_outbox (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                sent_at_ms INTEGER
            )
            """
        )
        connection.execute(
            "INSERT INTO http_event_outbox VALUES (1, 'sent', ?)",
            (old_ms,),
        )

    result = _run_script(
        [
            os.environ.get("PYTHON", "python3"),
            "scripts/cleanup/cleanup_outbox_databases.py",
            "--http-db",
            str(http_db),
            "--mqtt-db",
            str(tmp_path / "missing.db"),
            "--retention-days",
            "7",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "eligible=1, deleted=0" in result.stdout
    with sqlite3.connect(http_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM http_event_outbox"
        ).fetchone()[0] == 1


def test_ensure_public_api_key_preserves_existing_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PUBLIC_API_KEY=existing-public\nINTERNAL_SERVICE_TOKEN=existing-internal\n",
        encoding="utf-8",
    )

    result = _run_script(
        ["sh", "scripts/ops/ensure_public_api_key.sh", str(env_file)],
    )

    assert result.returncode == 0
    assert env_file.read_text(encoding="utf-8") == (
        "PUBLIC_API_KEY=existing-public\nINTERNAL_SERVICE_TOKEN=existing-internal\n"
    )
    assert "PUBLIC_API_KEY already set" in result.stdout
    assert "INTERNAL_SERVICE_TOKEN already set" in result.stdout
    assert "existing-public" not in result.stdout
    assert "existing-internal" not in result.stdout


def test_ensure_public_api_key_generates_empty_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PUBLIC_API_KEY=\nINTERNAL_SERVICE_TOKEN=\n",
        encoding="utf-8",
    )

    result = _run_script(
        ["sh", "scripts/ops/ensure_public_api_key.sh", str(env_file)],
    )

    assert result.returncode == 0
    env_values = _read_env_values(env_file)
    assert env_values["PUBLIC_API_KEY"]
    assert env_values["INTERNAL_SERVICE_TOKEN"]
    assert env_values["PUBLIC_API_KEY"] not in result.stdout
    assert env_values["INTERNAL_SERVICE_TOKEN"] not in result.stdout
    assert "PUBLIC_API_KEY generated" in result.stdout
    assert "INTERNAL_SERVICE_TOKEN generated" in result.stdout


def test_ensure_public_api_key_rotates_existing_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PUBLIC_API_KEY=existing-public\nINTERNAL_SERVICE_TOKEN=existing-internal\n",
        encoding="utf-8",
    )

    result = _run_script(
        ["sh", "scripts/ops/ensure_public_api_key.sh", "--rotate", str(env_file)],
    )

    assert result.returncode == 0
    env_values = _read_env_values(env_file)
    assert env_values["PUBLIC_API_KEY"] != "existing-public"
    assert env_values["INTERNAL_SERVICE_TOKEN"] != "existing-internal"
    assert env_values["PUBLIC_API_KEY"] != env_values["INTERNAL_SERVICE_TOKEN"]
    assert len(env_values["PUBLIC_API_KEY"]) == 43
    assert len(env_values["INTERNAL_SERVICE_TOKEN"]) == 43
    assert env_values["PUBLIC_API_KEY"] not in result.stdout
    assert env_values["INTERNAL_SERVICE_TOKEN"] not in result.stdout
    assert "PUBLIC_API_KEY rotated" in result.stdout
    assert "INTERNAL_SERVICE_TOKEN rotated" in result.stdout


def test_ensure_public_api_key_generates_placeholder_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PUBLIC_API_KEY=${PUBLIC_API_KEY:-}\n"
        "INTERNAL_SERVICE_TOKEN=${INTERNAL_SERVICE_TOKEN:-}\n",
        encoding="utf-8",
    )

    result = _run_script(
        ["sh", "scripts/ops/ensure_public_api_key.sh", str(env_file)],
    )

    assert result.returncode == 0
    env_values = _read_env_values(env_file)
    assert env_values["PUBLIC_API_KEY"]
    assert env_values["PUBLIC_API_KEY"] != "${PUBLIC_API_KEY:-}"
    assert env_values["INTERNAL_SERVICE_TOKEN"]
    assert env_values["INTERNAL_SERVICE_TOKEN"] != "${INTERNAL_SERVICE_TOKEN:-}"


def test_ensure_public_api_key_copies_example_from_script_location(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / "generated.env"
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()

    result = _run_script(
        [
            "sh",
            str(PROJECT_ROOT / "scripts/ops/ensure_public_api_key.sh"),
            str(env_file),
        ],
        cwd=other_cwd,
    )

    assert result.returncode == 0
    env_values = _read_env_values(env_file)
    assert env_values["PUBLIC_API_KEY"]
    assert env_values["INTERNAL_SERVICE_TOKEN"]
