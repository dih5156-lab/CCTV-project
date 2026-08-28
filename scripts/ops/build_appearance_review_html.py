#!/usr/bin/env python3
"""Create a local, dependency-free HTML page for reviewing color crops."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

COLOR_OPTIONS = (
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "navy",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
)
REVIEW_FIELDS = ("upper_color", "lower_color")

DOWNLOAD_SCRIPT = """
function downloadLabels() {
  const byId = new Map();
  for (const select of document.querySelectorAll("select[data-id][data-field]")) {
    const id = Number(select.dataset.id);
    const item = byId.get(id) || {id, upper_color:null, lower_color:null};
    item[select.dataset.field] = select.value || null;
    byId.set(id, item);
  }
  const payload = {schema_version:1, items:[...byId.values()]};
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], {type:"application/json"})
  );
  anchor.download = "appearance_color_review_labels.json";
  anchor.click();
}
"""


def _review_select(item_id: int, field: str) -> str:
    options = ["<option value=''>변경 안 함</option>"]
    options.extend(f"<option>{color}</option>" for color in COLOR_OPTIONS)
    options.append("<option>exclude</option>")
    return (
        f"<select data-id='{item_id}' data-field='{field}'>"
        f"{''.join(options)}</select>"
    )


def _candidate_cells(item: dict, field: str) -> list[str]:
    stored = item.get("stored", {})
    candidate = item.get("candidates", {}).get(field, {})
    model = (
        f"{candidate.get('model_color')} "
        f"({candidate.get('model_confidence')})"
    )
    return [
        f"<td>{html.escape(str(stored.get(field)))}</td>",
        f"<td>{html.escape(str(candidate.get('hsv_color')))}</td>",
        f"<td>{html.escape(str(candidate.get('lab_color')))}</td>",
        f"<td>{html.escape(model)}</td>",
        f"<td>{_review_select(int(item['id']), field)}</td>",
    ]


def build_document(payload: dict, *, base_dir: Path | None = None) -> str:
    rows = []
    for item in payload.get("items", []):
        crop_path = Path(item["crop_path"]).resolve()
        crop_uri = (
            Path(os.path.relpath(crop_path, base_dir.resolve())).as_posix()
            if base_dir is not None
            else crop_path.as_uri()
        )
        cells = [
            f"<td>{item['id']}</td>",
            (
                f"<td><img src='{html.escape(crop_uri)}' "
                "loading='lazy'></td>"
            ),
        ]
        for field in REVIEW_FIELDS:
            cells.extend(_candidate_cells(item, field))
        rows.append("<tr>" + "".join(cells) + "</tr>")

    headers = (
        "<th>ID</th><th>crop</th>"
        "<th>상의 DB</th><th>상의 HSV</th><th>상의 LAB</th>"
        "<th>상의 model</th><th>상의 정답</th>"
        "<th>하의 DB</th><th>하의 HSV</th><th>하의 LAB</th>"
        "<th>하의 model</th><th>하의 정답</th>"
    )
    return f"""<!doctype html><meta charset='utf-8'><title>Appearance color review</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee}} table{{border-collapse:collapse}}td{{border:1px solid #555;padding:6px}}img{{max-width:360px;max-height:260px}}button{{margin:12px;padding:8px}}</style>
<h1>상의·하의 색상 검수 ({len(rows)}건)</h1>
<p>이미지를 보고 정답을 선택하세요. 브라우저를 닫기 전에 반드시 검수 JSON을 다운로드해야 합니다.</p>
<button onclick='downloadLabels()'>검수 라벨 JSON 다운로드</button>
<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<script>{DOWNLOAD_SCRIPT}</script>"""


def build(manifest_path: Path, output_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_document(payload, base_dir=output_path.parent), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.manifest, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
