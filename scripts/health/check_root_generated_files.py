#!/usr/bin/env python3
"""루트 디렉터리에 생성물이 남지 않았는지 검사한다."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GENERATED_SUFFIXES = {".db", ".jsonl", ".log", ".pyc", ".pyo"}
ALLOWED_ROOT_FILES = {
    ".gitignore",
    ".env",
    ".env.example",
    ".env.jetson",
    ".env.jetson.example",
    "LICENSE",
    "README.md",
    "COMMANDS.md",
    "pyproject.toml",
    "Dockerfile",
    "Dockerfile.jetson",
    "docker-compose.yml",
    "docker-compose.jetson.yml",
    "main.py",
    "run_external_ingest.py",
    "conftest.py",
    "pytest.ini",
    "cameras.example.json",
    "known_faces.example.json",
    "zones_config.json",
}


def main() -> int:
    violations: list[str] = []

    for path in PROJECT_ROOT.iterdir():
        if path.name in {".git", ".github", ".codex", ".venv", "__pycache__"}:
            continue
        if path.is_dir():
            continue
        if path.name in ALLOWED_ROOT_FILES:
            continue
        if path.suffix.lower() in GENERATED_SUFFIXES:
            violations.append(path.name)

    pycache_dir = PROJECT_ROOT / "__pycache__"
    if pycache_dir.exists():
        violations.append("__pycache__/")

    if violations:
        print("루트 생성물 금지 규칙 위반:")
        for item in sorted(violations):
            print(f"- {item}")
        return 1

    print("루트 생성물 금지 규칙 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
