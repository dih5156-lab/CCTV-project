#!/usr/bin/env python3
"""Build a storage-bounded appearance manifest from AI-Hub Korean Re-ID ZIPs."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

SUPPORTED_COLORS = {
    "black",
    "white",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "purple",
    "navy",
    "orange",
}
COLOR_ALIASES = {"흰색": "white"}


def map_color(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in COLOR_ALIASES:
        return COLOR_ALIASES[normalized]
    if normalized in SUPPORTED_COLORS:
        return normalized
    return "other"


def _text(element: ET.Element, tag: str) -> str:
    return (element.findtext(tag) or "").strip().lower()


def _select_evenly(items: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if len(items) <= limit:
        return items

    by_color_pair: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for item in items:
        by_color_pair[(item["upper_color"], item["lower_color"])].append(item)
    representatives = [
        rows[len(rows) // 2]
        for _, rows in sorted(by_color_pair.items())
    ]
    if len(representatives) >= limit:
        if limit == 1:
            return [representatives[len(representatives) // 2]]
        return [
            representatives[round(index * (len(representatives) - 1) / (limit - 1))]
            for index in range(limit)
        ]

    selected_names = {item["image_name"] for item in representatives}
    remaining = [item for item in items if item["image_name"] not in selected_names]
    slots = limit - len(representatives)
    if slots == 1:
        fillers = [remaining[len(remaining) // 2]]
    else:
        fillers = [
            remaining[round(index * (len(remaining) - 1) / (slots - 1))]
            for index in range(slots)
        ]
    return sorted(representatives + fillers, key=lambda item: item["image_name"])


def _read_labels(label_zip: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    people: dict[str, list[dict[str, str]]] = defaultdict(list)
    stats = {"xml_files": 0, "malformed_xml": 0, "undefined_color": 0}
    with zipfile.ZipFile(label_zip) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".xml"):
                continue
            stats["xml_files"] += 1
            try:
                root = ET.fromstring(archive.read(member))
            except ET.ParseError:
                stats["malformed_xml"] += 1
                continue
            person = root.find("OBJECT")
            if person is None:
                stats["malformed_xml"] += 1
                continue
            if _text(person, "defined_upperclothes_color") != "true" or _text(
                person, "defined_lowerclothes_color"
            ) != "true":
                stats["undefined_color"] += 1
                continue
            person_id = (person.get("ID") or "").strip()
            image_name = (root.findtext("FILE/name") or "").strip()
            if not person_id or not image_name:
                stats["malformed_xml"] += 1
                continue
            item_kinds = {
                (item.findtext("kind_of") or "").strip().lower()
                for item in root.findall("./ITEM_LIST/ITEM")
            }
            people[person_id].append(
                {
                    "person_id": person_id,
                    "image_name": image_name,
                    "gender": _text(person, "gender"),
                    "upper_clothes": _text(person, "upperclothes"),
                    "upper_color": map_color(_text(person, "upperclothes_color")),
                    "lower_clothes": _text(person, "lowerclothes"),
                    "lower_color": map_color(_text(person, "lowerclothes_color")),
                    "bag": "yes" if item_kinds & {"bag", "backpack"} else "no",
                    "hat": "yes" if "hat" in item_kinds else "no",
                }
            )
    for person_rows in people.values():
        person_rows.sort(key=lambda row: row["image_name"])
    return people, stats


def prepare_split(
    *,
    label_zip: Path,
    source_zip: Path,
    output_dir: Path,
    split: str,
    max_images_per_person: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    people, stats = _read_labels(label_zip)
    selected = [
        row
        for person_id in sorted(people)
        for row in _select_evenly(people[person_id], max_images_per_person)
    ]
    image_dir = output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    missing_images = 0

    with zipfile.ZipFile(source_zip) as archive:
        members = {PurePosixPath(name).name: name for name in archive.namelist()}
        for row in selected:
            source_member = members.get(PurePosixPath(row["image_name"]).name)
            if source_member is None:
                missing_images += 1
                continue
            destination = image_dir / PurePosixPath(row["image_name"]).name
            if not destination.exists():
                with archive.open(source_member) as source, destination.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
            rows.append(
                {
                    "image_path": str(destination.relative_to(output_dir)),
                    "person_id": row["person_id"],
                    "split": split,
                    "gender": row["gender"],
                    "upper_clothes": row["upper_clothes"],
                    "upper_color": row["upper_color"],
                    "lower_clothes": row["lower_clothes"],
                    "lower_color": row["lower_color"],
                    "bag": row["bag"],
                    "hat": row["hat"],
                }
            )

    summary = {
        **stats,
        "split": split,
        "person_ids": len(people),
        "selected_images": len(rows),
        "missing_images": missing_images,
    }
    return rows, summary


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "image_path", "person_id", "split", "gender", "upper_clothes",
        "upper_color", "lower_clothes", "lower_color", "bag", "hat",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-label-zip", type=Path, required=True)
    parser.add_argument("--train-source-zip", type=Path, required=True)
    parser.add_argument("--validation-label-zip", type=Path, required=True)
    parser.add_argument("--validation-source-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images-per-person", type=int, default=30)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.max_images_per_person < 1:
        raise SystemExit("--max-images-per-person must be at least 1")
    train_rows, train_summary = prepare_split(
        label_zip=args.train_label_zip,
        source_zip=args.train_source_zip,
        output_dir=args.output_dir,
        split="train",
        max_images_per_person=args.max_images_per_person,
    )
    validation_rows, validation_summary = prepare_split(
        label_zip=args.validation_label_zip,
        source_zip=args.validation_source_zip,
        output_dir=args.output_dir,
        split="validation",
        max_images_per_person=args.max_images_per_person,
    )
    rows = train_rows + validation_rows
    _write_manifest(args.output_dir / "manifest.csv", rows)
    summary = {"train": train_summary, "validation": validation_summary, "total_rows": len(rows)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
