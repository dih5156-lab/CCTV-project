#!/usr/bin/env python3
"""Inspect AI-Hub Korean Re-ID labels without extracting their ZIP files."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


def _text(element: ET.Element, tag: str) -> str:
    return (element.findtext(tag) or "").strip().lower()


def _is_defined(value: str) -> bool:
    return value.strip().lower() == "true"


def inspect_label_zip(label_zip: Path) -> dict[str, Any]:
    upper_colors: Counter[str] = Counter()
    lower_colors: Counter[str] = Counter()
    upper_clothes: Counter[str] = Counter()
    lower_clothes: Counter[str] = Counter()
    person_ids: set[str] = set()
    xml_files = 0
    parsed_labels = 0
    malformed_xml = 0
    undefined_upper_colors = 0
    undefined_lower_colors = 0

    with zipfile.ZipFile(label_zip) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".xml"):
                continue
            xml_files += 1
            try:
                root = ET.fromstring(archive.read(member))
            except ET.ParseError:
                malformed_xml += 1
                continue

            object_element = root.find("OBJECT")
            if object_element is None:
                malformed_xml += 1
                continue

            parsed_labels += 1
            person_id = (object_element.get("ID") or "").strip()
            if person_id:
                person_ids.add(person_id)

            upper_clothes_value = _text(object_element, "upperclothes")
            lower_clothes_value = _text(object_element, "lowerclothes")
            if upper_clothes_value:
                upper_clothes[upper_clothes_value] += 1
            if lower_clothes_value:
                lower_clothes[lower_clothes_value] += 1

            if _is_defined(_text(object_element, "defined_upperclothes_color")):
                upper_color = _text(object_element, "upperclothes_color")
                if upper_color:
                    upper_colors[upper_color] += 1
            else:
                undefined_upper_colors += 1

            if _is_defined(_text(object_element, "defined_lowerclothes_color")):
                lower_color = _text(object_element, "lowerclothes_color")
                if lower_color:
                    lower_colors[lower_color] += 1
            else:
                undefined_lower_colors += 1

    return {
        "path": str(label_zip),
        "xml_files": xml_files,
        "parsed_labels": parsed_labels,
        "malformed_xml": malformed_xml,
        "person_ids": len(person_ids),
        "upper_colors": dict(sorted(upper_colors.items())),
        "lower_colors": dict(sorted(lower_colors.items())),
        "undefined_upper_colors": undefined_upper_colors,
        "undefined_lower_colors": undefined_lower_colors,
        "upper_clothes": dict(sorted(upper_clothes.items())),
        "lower_clothes": dict(sorted(lower_clothes.items())),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_zips", type=Path, nargs="+", help="AI-Hub label ZIP files")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = {"label_zips": [inspect_label_zip(path) for path in args.label_zips]}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
