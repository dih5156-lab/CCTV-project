#!/usr/bin/env python3
"""Build a browser review page for false-positive/false-negative fall videos."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = (
    PROJECT_ROOT
    / "models/experiments/yolo_pose_fall_extratrees_full_metrics.json"
)
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data/fall_eval/auto/train_manifest.jsonl"
DEFAULT_VALIDATION_MANIFEST = (
    PROJECT_ROOT / "data/fall_eval/auto/validation_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/fall_error_review/full_14702"
DEFAULT_CONTAINER_DATA_ROOT = Path("/app/낙상학습데이터")
DEFAULT_HOST_DATA_ROOT = PROJECT_ROOT.parent / "낙상학습데이터"


def read_jsonl_by_scene(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file_pointer:
        for line_number, line in enumerate(file_pointer, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            scene_id = str(row.get("scene_id") or "")
            if not scene_id:
                raise ValueError(f"scene_id missing at {path}:{line_number}")
            if scene_id in rows:
                raise ValueError(f"duplicate scene_id in {path}: {scene_id}")
            rows[scene_id] = row
    return rows


def resolve_video_path(
    raw_path: str,
    *,
    container_data_root: Path = DEFAULT_CONTAINER_DATA_ROOT,
    host_data_root: Path = DEFAULT_HOST_DATA_ROOT,
) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path.resolve()
    try:
        relative_path = path.relative_to(container_data_root)
    except ValueError:
        relative_path = None
    if relative_path is not None:
        return (host_data_root / relative_path).resolve()
    if not path.is_absolute():
        return (PROJECT_ROOT / path).resolve()
    return path


def _error_rows(metrics: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    sections = [
        ("holdout", metrics.get("holdout_errors") or {}),
        ("validation", ((metrics.get("validation") or {}).get("errors") or {})),
    ]
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for split, errors in sections:
        for error_type in ("false_positives", "false_negatives"):
            for error in errors.get(error_type) or []:
                rows.append((split, error_type[:-1], error))
    return rows


def build_error_candidates(
    metrics: dict[str, Any],
    *,
    train_rows: dict[str, dict[str, Any]],
    validation_rows: dict[str, dict[str, Any]],
    container_data_root: Path = DEFAULT_CONTAINER_DATA_ROOT,
    host_data_root: Path = DEFAULT_HOST_DATA_ROOT,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    missing_scene_ids: list[str] = []
    for split, error_type, error in _error_rows(metrics):
        scene_id = str(error.get("scene_id") or "")
        manifest_rows = train_rows if split == "holdout" else validation_rows
        manifest_row = manifest_rows.get(scene_id)
        if manifest_row is None:
            missing_scene_ids.append(f"{split}:{scene_id}")
            continue
        probability = error.get("probability") or []
        fall_probability = float(probability[1]) if len(probability) > 1 else None
        video_path = resolve_video_path(
            str(manifest_row.get("video_path") or ""),
            container_data_root=container_data_root,
            host_data_root=host_data_root,
        )
        true_label = "fall" if int(error.get("true", 0)) == 1 else "non_fall"
        predicted_label = (
            "fall" if int(error.get("predicted", 0)) == 1 else "non_fall"
        )
        candidates.append(
            {
                **manifest_row,
                "review_id": f"{split}:{error_type}:{scene_id}",
                "evaluation_split": split,
                "error_type": error_type,
                "original_label": true_label,
                "predicted_label": predicted_label,
                "fall_probability": fall_probability,
                "video_path": str(video_path),
                "video_exists": video_path.is_file(),
            }
        )
    if missing_scene_ids:
        preview = ", ".join(missing_scene_ids[:3])
        raise ValueError(f"error scenes missing from manifests: {preview}")
    return candidates


def write_candidates_jsonl(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_pointer:
        for candidate in candidates:
            file_pointer.write(json.dumps(candidate, ensure_ascii=False) + "\n")


def filter_error_candidates(
    candidates: list[dict[str, Any]],
    *,
    error_type: str | None = None,
    scene_id_pattern: str | None = None,
) -> list[dict[str, Any]]:
    scene_regex = re.compile(scene_id_pattern) if scene_id_pattern else None
    return [
        candidate
        for candidate in candidates
        if (error_type is None or candidate["error_type"] == error_type)
        and (
            scene_regex is None
            or scene_regex.search(str(candidate.get("scene_id") or ""))
        )
    ]


def select_evaluation_metrics(
    payload: dict[str, Any], evaluation_name: str | None
) -> dict[str, Any]:
    if evaluation_name is None:
        return payload
    evaluations = payload.get("evaluations") or {}
    if evaluation_name not in evaluations:
        available = ", ".join(sorted(evaluations)) or "none"
        raise ValueError(
            f"evaluation not found: {evaluation_name}; available: {available}"
        )
    return {
        "dataset_version": evaluation_name,
        "model_params": {"decision_threshold": payload.get("threshold")},
        "validation": evaluations[evaluation_name],
    }


def build_review_document(
    candidates: list[dict[str, Any]],
    *,
    base_dir: Path,
    dataset_version: str,
    page_size: int = 24,
) -> str:
    cards: list[str] = []
    for index, candidate in enumerate(candidates):
        review_id = str(candidate["review_id"])
        video_path = Path(str(candidate["video_path"])).resolve()
        video_url = Path(os.path.relpath(video_path, base_dir.resolve())).as_posix()
        fall_probability = candidate.get("fall_probability")
        probability_text = (
            "-" if fall_probability is None else f"{float(fall_probability):.3f}"
        )
        error_label = "오탐(FP)" if candidate["error_type"] == "false_positive" else "미탐(FN)"
        cards.append(
            f"<article class=\"card\" data-index=\"{index}\" "
            f"data-review-id=\"{html.escape(review_id, quote=True)}\">"
            f"<video controls muted preload=\"none\" data-src=\"{html.escape(video_url, quote=True)}\"></video>"
            f"<div class=\"meta\"><strong>{html.escape(str(candidate['scene_id']))}</strong> "
            f"<span class=\"error\">{error_label}</span><br>"
            f"구간: {html.escape(str(candidate['evaluation_split']))} · "
            f"기존 정답: {html.escape(str(candidate['original_label']))} · "
            f"모델 예측: {html.escape(str(candidate['predicted_label']))} · "
            f"낙상 확률: {probability_text}</div>"
            "<div class=\"choices\">"
            "<button data-label=\"fall\">낙상</button>"
            "<button data-label=\"non_fall\">비낙상</button>"
            "<button data-label=\"ambiguous\">판단보류</button>"
            "<button data-label=\"exclude\">학습제외</button>"
            "<button data-label=\"\">선택 취소</button>"
            "</div></article>"
        )

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>낙상 오류 재검수</title>
<style>
body{{font-family:sans-serif;background:#111;color:#eee;margin:20px}}
.toolbar{{position:sticky;top:0;background:#111;padding:10px 0;z-index:2}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px}}
.card{{display:none;background:#222;border:2px solid #444;border-radius:8px;padding:12px}}
.card.visible{{display:block}} .card[data-selected="fall"]{{border-color:#ef4444}}
.card[data-selected="non_fall"]{{border-color:#22c55e}}
.card[data-selected="ambiguous"]{{border-color:#eab308}}
.card[data-selected="exclude"]{{border-color:#a855f7}}
video{{width:100%;max-height:480px;background:#000}}
.meta{{font-size:14px;line-height:1.6;overflow-wrap:anywhere;margin:8px 0}}
.error{{color:#fca5a5}} button{{padding:9px 14px;margin:3px;cursor:pointer}}
button.selected{{outline:3px solid #38bdf8}}
</style></head><body>
<h1>오탐·미탐 원본 영상 재검수</h1>
<p>데이터셋: {html.escape(dataset_version)} · 총 {len(cards)}건 · 페이지당 {page_size}개</p>
<p>원본 영상을 확인하고 실제 정답을 선택하세요. 판단이 어렵거나 영상에 문제가 있으면 판단보류 또는 학습제외를 선택하세요.</p>
<div class="toolbar"><button id="prev">이전</button><button id="next">다음</button>
<button onclick="downloadLabels()">검수 JSON 다운로드</button>
<span id="page"></span> · <span id="progress"></span></div>
<main class="grid">{''.join(cards)}</main>
<script>
const pageSize = {page_size};
const storageKey = "fall-error-review-{html.escape(dataset_version, quote=True)}";
const labels = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
const cards = [...document.querySelectorAll(".card")];
const reviewIds = new Set(cards.map(card => card.dataset.reviewId));
let currentPage = 0;
const pageCount = Math.max(1, Math.ceil(cards.length / pageSize));

function refreshCard(card) {{
  const selected = labels[card.dataset.reviewId] || "";
  card.dataset.selected = selected;
  for (const button of card.querySelectorAll("button[data-label]")) {{
    button.classList.toggle("selected", Boolean(selected) && button.dataset.label === selected);
  }}
}}
function showPage({{ scrollToTop = false }} = {{}}) {{
  cards.forEach((card, index) => {{
    const visible = Math.floor(index / pageSize) === currentPage;
    card.classList.toggle("visible", visible);
    const video = card.querySelector("video");
    if (visible && !video.src) video.src = video.dataset.src;
    if (!visible && video.src) {{ video.pause(); video.removeAttribute("src"); video.load(); }}
  }});
  document.getElementById("page").textContent = `${{currentPage + 1}} / ${{pageCount}} 페이지`;
  document.getElementById("progress").textContent =
    `${{[...reviewIds].filter(id => labels[id]).length}} / ${{reviewIds.size}} 선택`;
  if (scrollToTop) window.scrollTo(0, 0);
}}
for (const card of cards) {{
  for (const button of card.querySelectorAll("button[data-label]")) {{
    button.addEventListener("click", () => {{
      if (button.dataset.label) labels[card.dataset.reviewId] = button.dataset.label;
      else delete labels[card.dataset.reviewId];
      localStorage.setItem(storageKey, JSON.stringify(labels));
      refreshCard(card); showPage();
    }});
  }}
  refreshCard(card);
}}
document.getElementById("prev").onclick = () => {{ currentPage = Math.max(0, currentPage - 1); showPage({{scrollToTop: true}}); }};
document.getElementById("next").onclick = () => {{ currentPage = Math.min(pageCount - 1, currentPage + 1); showPage({{scrollToTop: true}}); }};
function downloadLabels() {{
  const items = Object.entries(labels).filter(([id, label]) => reviewIds.has(id) && label)
    .map(([review_id, label]) => ({{review_id, label}}));
  const payload = {{schema_version: 1, dataset_version: {json.dumps(dataset_version)}, items}};
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {{type:"application/json"}}));
  anchor.download = "fall_error_review_labels.json"; anchor.click(); URL.revokeObjectURL(anchor.href);
}}
showPage();
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument(
        "--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host-data-root", type=Path, default=DEFAULT_HOST_DATA_ROOT)
    parser.add_argument("--container-data-root", type=Path, default=DEFAULT_CONTAINER_DATA_ROOT)
    parser.add_argument("--page-size", type=int, default=24)
    parser.add_argument(
        "--error-type",
        choices=("false_positive", "false_negative"),
        help="Include only one error type.",
    )
    parser.add_argument(
        "--scene-id-pattern",
        help="Include only scene IDs matching this regular expression.",
    )
    parser.add_argument(
        "--evaluation-name",
        help="Select one entry from a comparison JSON evaluations object.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.page_size < 1:
        raise ValueError("page-size must be at least 1")
    metrics_payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    metrics = select_evaluation_metrics(metrics_payload, args.evaluation_name)
    candidates = build_error_candidates(
        metrics,
        train_rows=read_jsonl_by_scene(args.train_manifest),
        validation_rows=read_jsonl_by_scene(args.validation_manifest),
        container_data_root=args.container_data_root,
        host_data_root=args.host_data_root,
    )
    candidates = filter_error_candidates(
        candidates,
        error_type=args.error_type,
        scene_id_pattern=args.scene_id_pattern,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "fall_error_review_manifest.jsonl"
    html_path = args.output_dir / "index.html"
    write_candidates_jsonl(manifest_path, candidates)
    html_path.write_text(
        build_review_document(
            candidates,
            base_dir=args.output_dir,
            dataset_version=str(metrics.get("dataset_version") or args.metrics.stem),
            page_size=args.page_size,
        ),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = f"{candidate['evaluation_split']}:{candidate['error_type']}"
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "candidates": len(candidates),
        "missing_videos": sum(not row["video_exists"] for row in candidates),
        "counts": counts,
        "manifest": str(manifest_path),
        "html": str(html_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["missing_videos"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
