#!/usr/bin/env python3
"""Inspect the public fall-detection package without extracting it.

This check is intentionally lightweight by default. The bundled test.zip files
are about 3.5 GiB each, so deep zip inspection is opt-in.
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

PACKAGE_TARS = {
    "fall_binary": Path("낙상방향-003.tar"),
    "fall_direction": Path("낙상유무탐지-002.tar"),
}

EXPECTED_TEST_ZIPS = {
    "fall_binary": "SCH_FN/test.zip",
    "fall_direction": "SCH_FNF/test.zip",
}


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _count_extensions(names: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for name in names:
        suffix = Path(name).suffix.lower() or "<dir/no-ext>"
        counts[suffix] += 1
    return counts


def _inspect_tar(tar_path: Path, package_key: str, deep_test_zip: bool) -> list[str]:
    lines = [f"[{package_key}] {tar_path}"]
    if not tar_path.exists():
        return lines + ["  status: missing"]

    lines.append(f"  size: {_human_size(tar_path.stat().st_size)}")
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        lines.append(f"  entries: {len(members)}")

        ext_counts = _count_extensions(names)
        interesting = [".pkl", ".sav", ".h5", ".py", ".zip", ".xlsx", ".npy"]
        for suffix in interesting:
            if ext_counts[suffix]:
                lines.append(f"  {suffix}: {ext_counts[suffix]}")

        test_zip_name = EXPECTED_TEST_ZIPS[package_key]
        if test_zip_name in names:
            member = tar.getmember(test_zip_name)
            lines.append(f"  test_zip: {test_zip_name} ({_human_size(member.size)})")
            if deep_test_zip:
                lines.extend(_inspect_test_zip(tar, member))
        else:
            lines.append(f"  test_zip: missing ({test_zip_name})")

        root_dirs = sorted({name.split("/", 1)[0] for name in names if "/" in name})
        if root_dirs:
            lines.append(f"  root_dirs: {', '.join(root_dirs)}")

    return lines


def _inspect_test_zip(tar: tarfile.TarFile, member: tarfile.TarInfo) -> list[str]:
    """Read bundled test.zip and report npy metadata.

    The zip member is large and ZipFile needs a seekable object, so this loads
    the zip into memory. Keep this behind --deep-test-zip.
    """
    lines = ["  deep_test_zip: enabled"]
    extracted = tar.extractfile(member)
    if extracted is None:
        return lines + ["    status: unreadable"]

    data = extracted.read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        npy_names = [name for name in names if name.endswith(".npy")]
        clip_dirs = {
            parts[1]
            for name in names
            if name.startswith("test/") and len((parts := name.split("/"))) > 2
        }
        lines.append(f"    zip_entries: {len(names)}")
        lines.append(f"    npy_count: {len(npy_names)}")
        lines.append(f"    clip_dir_count: {len(clip_dirs)}")

        if npy_names:
            lines.extend(_inspect_first_npy(archive, npy_names[0]))

    return lines


def _inspect_first_npy(archive: zipfile.ZipFile, name: str) -> list[str]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"    first_npy: {name}", f"    first_npy_shape: skipped ({exc})"]

    with archive.open(name) as file:
        array = np.load(io.BytesIO(file.read()))
    return [
        f"    first_npy: {name}",
        f"    first_npy_shape: {tuple(array.shape)}",
        f"    first_npy_dtype: {array.dtype}",
    ]


def _inspect_model_dir(root: Path) -> list[str]:
    model_root = root / "2. AI학습모델파일"
    lines = [f"[model_dir] {model_root}"]
    if not model_root.exists():
        return lines + ["  status: missing"]

    files = [path for path in model_root.rglob("*") if path.is_file()]
    lines.append(f"  files: {len(files)}")
    counts = _count_extensions(str(path.relative_to(model_root)) for path in files)
    for suffix in (".pkl", ".sav", ".h5"):
        lines.append(f"  {suffix}: {counts[suffix]}")
    lines.append(f"  size: {_human_size(sum(path.stat().st_size for path in files))}")
    return lines


def _inspect_duplicate_candidates(root: Path) -> list[str]:
    pairs = [
        (
            root / "낙상방향-003.tar",
            root / "3. 도커이미지" / "낙상방향.tar",
        ),
        (
            root / "낙상유무탐지-002.tar",
            root / "3. 도커이미지" / "낙상유무탐지.tar",
        ),
    ]
    lines = ["[duplicates]"]
    for left, right in pairs:
        if not left.exists() or not right.exists():
            lines.append(f"  skipped: {left.name} / {right.name} (one side missing)")
            continue
        same_size = left.stat().st_size == right.stat().st_size
        lines.append(
            f"  candidate: {left.name} <-> {right.relative_to(root)} "
            f"(same_size={same_size})"
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("falldata"),
        help="Path to the falldata package root.",
    )
    parser.add_argument(
        "--deep-test-zip",
        action="store_true",
        help="Read bundled 3.5 GiB test.zip files to count npy samples.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    if not root.exists():
        print(f"falldata root not found: {root}", file=sys.stderr)
        return 2

    print(f"falldata_root: {root.resolve()}")
    print()
    for package_key, relative_path in PACKAGE_TARS.items():
        print("\n".join(_inspect_tar(root / relative_path, package_key, args.deep_test_zip)))
        print()
    print("\n".join(_inspect_model_dir(root)))
    print()
    print("\n".join(_inspect_duplicate_candidates(root)))
    print()
    print("[integration_hint]")
    print("  current_project: YOLO pose keypoints -> heuristic FallDetector")
    print("  falldata_package: 600-frame feature sequences -> sklearn/keras models")
    print("  recommendation: validate as isolated POC before wiring into runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
