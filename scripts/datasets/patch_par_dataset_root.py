#!/usr/bin/env python3
"""Patch dataset.root inside a PAR dataset_all.pkl file."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def patch_dataset_root(pkl_path: Path, image_root: Path) -> tuple[str, str]:
    with pkl_path.open("rb") as handle:
        dataset = pickle.load(handle)

    old_root = str(getattr(dataset, "root", ""))
    dataset.root = str(image_root.resolve())

    backup_path = pkl_path.with_suffix(pkl_path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_bytes(pkl_path.read_bytes())

    with pkl_path.open("wb") as handle:
        pickle.dump(dataset, handle)
    return old_root, dataset.root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch PAR dataset pkl image root.")
    parser.add_argument("--pkl", type=Path, required=True, help="dataset_all.pkl path.")
    parser.add_argument("--image-root", type=Path, required=True, help="Directory containing dataset images.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not args.pkl.exists():
        raise SystemExit(f"pkl not found: {args.pkl}")
    if not args.image_root.exists():
        raise SystemExit(f"image root not found: {args.image_root}")
    old_root, new_root = patch_dataset_root(args.pkl, args.image_root)
    print(f"patched {args.pkl}")
    print(f"old_root: {old_root}")
    print(f"new_root: {new_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
