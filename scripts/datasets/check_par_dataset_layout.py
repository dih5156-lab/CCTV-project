#!/usr/bin/env python3
"""Check whether PAR dataset folders contain images and annotation metadata."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def _pkl_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    return {
        "exists": True,
        "root": str(getattr(dataset, "root", "")),
        "images": len(getattr(dataset, "image_name", [])),
        "attributes": len(getattr(dataset, "attr_name", [])),
        "first_image": (getattr(dataset, "image_name", [""]) or [""])[0],
    }


def check_dataset(dataset_root: Path, dataset_name: str) -> dict[str, Any]:
    name = dataset_name.upper()
    if name in {"PA100K", "PA100K"}:
        image_dir = dataset_root / "PA100k" / "data"
        pkl_path = dataset_root / "PA100k" / "dataset_all.pkl"
        annotation_path = dataset_root / "PA100k" / "annotation.mat"
    elif name in {"RAP2", "RAPV2"}:
        image_dir = dataset_root / "RAP2" / "RAP_dataset"
        pkl_path = dataset_root / "RAP2" / "dataset_all.pkl"
        annotation_path = dataset_root / "RAP2" / "RAP_annotation" / "RAP_annotation.mat"
    elif name in {"RAP", "RAP1", "RAPV1"}:
        image_dir = dataset_root / "RAP" / "RAP_dataset"
        pkl_path = dataset_root / "RAP" / "dataset_all.pkl"
        annotation_path = dataset_root / "RAP" / "RAP_annotation" / "RAP_annotation.mat"
    elif name == "PETA":
        image_dir = dataset_root / "PETA" / "images"
        pkl_path = dataset_root / "PETA" / "dataset_all.pkl"
        annotation_path = dataset_root / "PETA" / "PETA.mat"
    else:
        raise SystemExit(f"unsupported dataset: {dataset_name}")

    image_count = _count_images(image_dir)
    pkl = _pkl_summary(pkl_path)
    expected = int(pkl.get("images") or 0)
    pkl_root = Path(str(pkl.get("root") or ""))
    first_image = str(pkl.get("first_image") or "")
    pkl_first_image_exists = bool(
        pkl.get("exists")
        and first_image
        and (pkl_root / first_image).exists()
    )
    return {
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "image_dir": str(image_dir),
        "image_count": image_count,
        "annotation_path": str(annotation_path),
        "annotation_exists": annotation_path.exists(),
        "pkl_path": str(pkl_path),
        "pkl": pkl,
        "pkl_first_image_exists": pkl_first_image_exists,
        "ready": (
            image_count > 0
            and (not expected or image_count >= min(expected, 10))
            and (not pkl.get("exists") or pkl_first_image_exists)
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check PA100K/PETA/RAP dataset layout.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root containing PA100k/PETA/RAP folders.")
    parser.add_argument("--dataset", default="PA100K", help="PA100K, PETA, RAP, or RAP2.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = check_dataset(args.dataset_root, args.dataset)
    print(f"dataset: {result['dataset']}")
    print(f"image_dir: {result['image_dir']}")
    print(f"image_count: {result['image_count']}")
    print(f"annotation_exists: {result['annotation_exists']} ({result['annotation_path']})")
    pkl = result["pkl"]
    print(f"pkl_exists: {pkl['exists']} ({result['pkl_path']})")
    if pkl["exists"]:
        print(f"pkl_root: {pkl['root']}")
        print(f"pkl_images: {pkl['images']}")
        print(f"pkl_attributes: {pkl['attributes']}")
        print(f"pkl_first_image: {pkl['first_image']}")
        print(f"pkl_first_image_exists: {result['pkl_first_image_exists']}")
    print(f"ready: {result['ready']}")
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
