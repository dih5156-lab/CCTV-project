"""Check that local Dockerfile COPY sources exist before attempting a build."""

from __future__ import annotations

import shlex
from pathlib import Path


DOCKERFILES = (
    Path("Dockerfile"),
    Path("Dockerfile.action"),
    Path("Dockerfile.jetson"),
    Path("Dockerfile.parser"),
)


def _logical_lines(path: Path) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        current += line
        lines.append(current)
        current = ""
    if current:
        lines.append(current)
    return lines


def _copy_sources(line: str) -> list[str]:
    try:
        tokens = shlex.split(line)
    except ValueError:
        return []
    if not tokens or tokens[0].upper() != "COPY":
        return []
    if any(token.startswith("--from=") for token in tokens[1:]):
        return []

    paths = [token for token in tokens[1:] if not token.startswith("--")]
    if len(paths) < 2:
        return []
    return paths[:-1]


def find_missing_sources() -> list[str]:
    missing: list[str] = []
    root = Path.cwd()
    for dockerfile in DOCKERFILES:
        if not dockerfile.exists():
            missing.append(f"{dockerfile}: Dockerfile not found")
            continue
        for line in _logical_lines(dockerfile):
            for source in _copy_sources(line):
                if source.startswith(("http://", "https://")):
                    continue
                if not (root / source).exists():
                    missing.append(f"{dockerfile}: COPY source not found: {source}")
    return missing


def main() -> int:
    missing = find_missing_sources()
    if missing:
        print("Missing Dockerfile COPY sources:")
        for item in missing:
            print(f"- {item}")
        return 1
    print("All Dockerfile COPY sources exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
