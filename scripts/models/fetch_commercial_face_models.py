"""Download and verify pinned OpenCV face model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("config/models/commercial_face_models.json")
DEFAULT_OUTPUT_DIR = Path("models/commercial_face")
_TOP_LEVEL_FIELDS = {"schema_version", "models"}
_REQUIRED_ARTIFACTS = {"model", "license"}


def verify_file(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    unknown_fields = set(payload) - _TOP_LEVEL_FIELDS
    if unknown_fields:
        raise ValueError(f"unknown manifest fields: {sorted(unknown_fields)}")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema_version")
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("manifest models must be a non-empty object")

    for model_name, model in models.items():
        revision = model.get("revision", "")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"invalid revision for {model_name}")
        artifacts = model.get("artifacts", {})
        missing = _REQUIRED_ARTIFACTS - set(artifacts)
        if missing:
            raise ValueError(f"missing artifacts for {model_name}: {sorted(missing)}")
        for artifact_name in _REQUIRED_ARTIFACTS:
            artifact = artifacts[artifact_name]
            url = artifact.get("url", "")
            if f"/resolve/{revision}/" not in url:
                raise ValueError(
                    f"{model_name}.{artifact_name} URL must use pinned revision"
                )
            sha256 = artifact.get("sha256", "")
            if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
                raise ValueError(f"invalid SHA-256 for {model_name}.{artifact_name}")
    return models


def _download_verified(url: str, destination: Path, sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        urllib.request.urlretrieve(url, temporary_path)
        verify_file(temporary_path, sha256)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def fetch_models(manifest_path: Path, output_dir: Path) -> list[Path]:
    installed: list[Path] = []
    for model in load_manifest(manifest_path).values():
        for artifact in model["artifacts"].values():
            destination = output_dir / artifact["filename"]
            if destination.is_file():
                verify_file(destination, artifact["sha256"])
            else:
                _download_verified(artifact["url"], destination, artifact["sha256"])
            installed.append(destination)
    return installed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    installed = fetch_models(args.manifest, args.output_dir)
    for path in installed:
        print(f"verified: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
