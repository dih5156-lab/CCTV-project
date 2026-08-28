#!/usr/bin/env python3
"""Interactively label fall shadow review clips with OpenCV.

Keys: F=fall, N=non-fall, S=needs review, SPACE=pause, R=replay, Q=quit.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/fall_dataset"
DEFAULT_REVIEW_LOG = DEFAULT_DATASET_ROOT / "annotations/review.jsonl"
DEFAULT_CLIP_DIR = DEFAULT_DATASET_ROOT / "clips/pending"
DEFAULT_LABELED_DIR = DEFAULT_DATASET_ROOT / "clips/labeled"
ALLOWED_WEB_LABELS = {"fall", "non_fall", "needs_review"}
SHADOW_DISAGREEMENT_LABELS = {
    "primary_fall_shadow_non_fall": "기존 낙상 / Shadow 비낙상",
    "primary_non_fall_shadow_fall": "기존 비낙상 / Shadow 낙상",
}


def classify_shadow_disagreement(row: dict[str, Any]) -> str | None:
    """Return the primary/Shadow disagreement direction for a valid result."""
    aux = row.get("falldata_aux")
    if not isinstance(aux, dict) or aux.get("status") != "ok":
        return None
    confirmed = aux.get("confirmed")
    if not isinstance(confirmed, bool):
        return None
    event_type = row.get("event_type")
    if event_type == "fall_detected" and not confirmed:
        return "primary_fall_shadow_non_fall"
    if event_type == "fall_shadow_window" and confirmed:
        return "primary_non_fall_shadow_fall"
    return None


def build_review_document(
    candidates: list[dict[str, Any]], *, base_dir: Path | None = None
) -> str:
    """Build a dependency-free browser review page for selected clips."""
    rows: list[str] = []
    for candidate in candidates:
        event_id = str(candidate.get("event_id") or "")
        clip_path = Path(str(candidate["local_clip_path"])).resolve()
        clip_uri = (
            Path(os.path.relpath(clip_path, base_dir.resolve())).as_posix()
            if base_dir is not None
            else clip_path.as_uri()
        )
        aux = candidate.get("falldata_aux") or {}
        probability = aux.get("fall_probability")
        if probability is None:
            probability = (aux.get("temporal_compare_model") or {}).get(
                "fall_probability"
            )
        probability_text = "-" if probability is None else f"{float(probability):.3f}"
        disagreement_type = str(candidate.get("disagreement_type") or "")
        disagreement_text = SHADOW_DISAGREEMENT_LABELS.get(disagreement_type)
        if disagreement_text is None and candidate.get("review_reason") == "threshold_boundary":
            disagreement_text = "임계값 인접 장면"
        disagreement_text = disagreement_text or "-"
        rows.append(
            "<article class='card' data-event-id='{}'>"
            "<video controls muted loop preload='metadata' src='{}'></video>"
            "<div class='meta'><strong>{}</strong><br>카메라: {} · 유형: {} · "
            "낙상 확률: {} · 불일치: {}</div>"
            "<div class='choices'>"
            "<button data-label='fall'>낙상</button>"
            "<button data-label='non_fall'>비낙상</button>"
            "<button data-label='needs_review'>보류</button>"
            "<button data-label=''>선택 취소</button>"
            "</div></article>".format(
                html.escape(event_id, quote=True),
                html.escape(clip_uri, quote=True),
                html.escape(event_id),
                html.escape(str(candidate.get("camera_id") or "-")),
                html.escape(str(candidate.get("event_type") or "-")),
                html.escape(probability_text),
                html.escape(disagreement_text),
            )
        )

    script = """
const storageKey = "fall-review-labels-v1";
const labels = JSON.parse(localStorage.getItem(storageKey) || "{}");
const eventIds = new Set(
  [...document.querySelectorAll(".card")].map(card => card.dataset.eventId)
);

function refreshCard(card) {
  const selected = labels[card.dataset.eventId] || "";
  card.dataset.selected = selected;
  for (const button of card.querySelectorAll("button[data-label]")) {
    button.classList.toggle("selected", button.dataset.label === selected && selected);
  }
  document.getElementById("progress").textContent =
    `${[...eventIds].filter(eventId => labels[eventId]).length} / ${eventIds.size} 선택`;
}

for (const card of document.querySelectorAll(".card")) {
  for (const button of card.querySelectorAll("button[data-label]")) {
    button.addEventListener("click", () => {
      if (button.dataset.label) labels[card.dataset.eventId] = button.dataset.label;
      else delete labels[card.dataset.eventId];
      localStorage.setItem(storageKey, JSON.stringify(labels));
      refreshCard(card);
    });
  }
  refreshCard(card);
}

function downloadLabels() {
  const items = Object.entries(labels)
    .filter(([eventId, label]) => eventIds.has(eventId) && label)
    .map(([event_id, label]) => ({event_id, label}));
  const payload = {schema_version: 1, items};
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"})
  );
  anchor.download = "fall_review_labels.json";
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}
"""
    return f"""<!doctype html>
<html lang='ko'><head><meta charset='utf-8'><title>낙상 후보 검수</title>
<style>
body{{font-family:sans-serif;background:#111;color:#eee;margin:20px}}
.toolbar{{position:sticky;top:0;background:#111;padding:10px 0;z-index:1}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}
.card{{background:#222;border:2px solid #444;border-radius:8px;padding:12px}}
.card[data-selected='fall']{{border-color:#ef4444}}
.card[data-selected='non_fall']{{border-color:#22c55e}}
.card[data-selected='needs_review']{{border-color:#eab308}}
video{{width:100%;max-height:420px;background:#000}}
.meta{{font-size:14px;line-height:1.5;overflow-wrap:anywhere;margin:8px 0}}
button{{padding:9px 14px;margin:3px;border:1px solid #777;border-radius:5px;cursor:pointer}}
button.selected{{outline:3px solid #38bdf8}}
</style></head><body>
<h1>낙상 후보 영상 검수 ({len(rows)}건)</h1>
<p>영상을 끝까지 확인한 뒤 낙상·비낙상·보류 중 하나를 선택하세요. 선택값은 이 브라우저에 자동 임시저장됩니다.</p>
<div class='toolbar'><button onclick='downloadLabels()'>검수 JSON 다운로드</button> <span id='progress'></span></div>
<main class='grid'>{''.join(rows)}</main><script>{script}</script></body></html>"""


def write_review_html(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_review_document(candidates, base_dir=path.parent), encoding="utf-8"
    )


def convert_clip_to_h264(source: Path, destination: Path) -> None:
    """Create a browser-compatible H.264 copy without modifying the source clip."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ],
        check=True,
    )
    temporary_path.replace(destination)


def prepare_browser_clips(
    candidates: list[dict[str, Any]],
    output_dir: Path,
    *,
    convert: Callable[[Path, Path], None] = convert_clip_to_h264,
) -> list[dict[str, Any]]:
    """Return candidates pointing at browser-compatible H.264 clip copies."""
    prepared: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        source = Path(str(candidate["local_clip_path"]))
        event_id = str(candidate.get("event_id") or f"candidate-{index}")
        safe_event_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in event_id
        )
        destination = output_dir / f"{safe_event_id}.mp4"
        if not destination.is_file():
            convert(source, destination)
        browser_candidate = dict(candidate)
        browser_candidate["source_clip_path"] = str(source)
        browser_candidate["local_clip_path"] = str(destination)
        prepared.append(browser_candidate)
    return prepared


def read_review_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at line {line_number}")
            rows.append(row)
    return rows


def resolve_clip_path(
    raw_path: str, *, clip_dir: Path = DEFAULT_CLIP_DIR
) -> Path:
    """Map container clip paths to the local project clip directory."""
    path = Path(raw_path)
    if path.exists():
        return path
    if not path.is_absolute():
        project_path = PROJECT_ROOT / path
        if project_path.exists():
            return project_path
    return clip_dir / path.name


def select_candidates(
    rows: list[dict[str, Any]],
    *,
    camera: str | None,
    include_sample_eval: bool,
    min_fall_probability: float = 0.0,
    only_disagreements: bool = False,
    include_threshold_boundary: float = 0.0,
    clip_dir: Path = DEFAULT_CLIP_DIR,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if (
            row.get("label") is not None
            or row.get("review_status") == "needs_review"
            or not row.get("clip_path")
        ):
            continue
        camera_id = str(row.get("camera_id") or "")
        if camera and camera_id != camera:
            continue
        if not include_sample_eval and camera_id == "sample_eval":
            continue
        disagreement_type = classify_shadow_disagreement(row)
        aux_value = row.get("falldata_aux")
        aux = aux_value if isinstance(aux_value, dict) else {}
        probability_value = aux.get("fall_probability")
        try:
            fall_probability = (
                float(probability_value) if probability_value is not None else 0.0
            )
        except (TypeError, ValueError):
            fall_probability = 0.0
        threshold_value = aux.get("threshold")
        try:
            threshold = float(threshold_value) if threshold_value is not None else None
        except (TypeError, ValueError):
            threshold = None
        is_threshold_boundary = bool(
            include_threshold_boundary > 0
            and threshold is not None
            and abs(fall_probability - threshold) <= include_threshold_boundary
        )
        if only_disagreements and disagreement_type is None and not is_threshold_boundary:
            continue
        if fall_probability < min_fall_probability:
            continue
        local_clip = resolve_clip_path(str(row["clip_path"]), clip_dir=clip_dir)
        if not local_clip.is_file():
            continue
        candidate = dict(row)
        candidate["local_clip_path"] = str(local_clip)
        if disagreement_type is not None:
            candidate["disagreement_type"] = disagreement_type
            candidate["review_reason"] = "shadow_disagreement"
        elif is_threshold_boundary:
            candidate["review_reason"] = "threshold_boundary"
        candidates.append(candidate)
    return candidates


def write_review_log_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as fp:
        temporary_path = Path(fp.name)
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        fp.flush()
    temporary_path.replace(path)


def apply_label(
    path: Path,
    *,
    event_id: str,
    label: str | None,
    review_status: str,
    clip_dir: Path = DEFAULT_CLIP_DIR,
    labeled_dir: Path | None = None,
) -> None:
    """Reload and update one event so each key press is saved immediately."""
    rows = read_review_log(path)
    matches = [row for row in rows if row.get("event_id") == event_id]
    if len(matches) != 1:
        raise ValueError(f"expected one event_id={event_id!r}, found {len(matches)}")
    row = matches[0]
    original_clip: Path | None = None
    labeled_clip: Path | None = None
    if label in {"fall", "non_fall"} and labeled_dir is not None and row.get("clip_path"):
        original_clip = resolve_clip_path(str(row["clip_path"]), clip_dir=clip_dir)
        if not original_clip.is_file():
            raise FileNotFoundError(f"clip not found: {original_clip}")
        label_directory = labeled_dir / label
        label_directory.mkdir(parents=True, exist_ok=True)
        labeled_clip = label_directory / original_clip.name
        if labeled_clip.exists() and labeled_clip != original_clip:
            raise FileExistsError(f"labeled clip already exists: {labeled_clip}")
        if labeled_clip != original_clip:
            shutil.move(str(original_clip), str(labeled_clip))
        row["clip_path"] = str(labeled_clip)

    row["label"] = label
    row["review_status"] = review_status
    try:
        write_review_log_atomic(path, rows)
    except Exception:
        if original_clip and labeled_clip and labeled_clip != original_clip and labeled_clip.exists():
            shutil.move(str(labeled_clip), str(original_clip))
        raise


def create_backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = path.with_name(f"{path.name}.{stamp}.bak")
    shutil.copy2(path, backup)
    return backup


def import_review_labels(
    review_log: Path,
    labels_path: Path,
    *,
    clip_dir: Path = DEFAULT_CLIP_DIR,
    labeled_dir: Path = DEFAULT_LABELED_DIR,
) -> dict[str, Any]:
    """Validate and atomically apply labels exported by the browser page."""
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("items"), list):
        raise ValueError("expected fall review label schema_version=1 with items")

    labels_by_event: dict[str, str] = {}
    for item in payload["items"]:
        if not isinstance(item, dict):
            raise ValueError("each label item must be an object")
        event_id = str(item.get("event_id") or "")
        label = str(item.get("label") or "")
        if not event_id or label not in ALLOWED_WEB_LABELS:
            raise ValueError(f"invalid fall review label: event_id={event_id!r}, label={label!r}")
        if event_id in labels_by_event:
            raise ValueError(f"duplicate event_id in labels: {event_id}")
        labels_by_event[event_id] = label

    rows = read_review_log(review_log)
    rows_by_event: dict[str, dict[str, Any]] = {}
    for row in rows:
        event_id = str(row.get("event_id") or "")
        if event_id in rows_by_event:
            raise ValueError(f"duplicate event_id in review log: {event_id}")
        rows_by_event[event_id] = row
    unknown = sorted(set(labels_by_event) - set(rows_by_event))
    if unknown:
        raise ValueError(f"unknown event_id in labels: {unknown[0]}")

    updates: list[tuple[dict[str, Any], str, Path | None, Path | None]] = []
    for event_id, label in labels_by_event.items():
        row = rows_by_event[event_id]
        existing_label = row.get("label")
        existing_status = row.get("review_status")
        if existing_label is not None or existing_status in {"reviewed", "needs_review"}:
            expected_label = None if label == "needs_review" else label
            expected_status = "needs_review" if label == "needs_review" else "reviewed"
            if existing_label == expected_label and existing_status == expected_status:
                continue
            raise ValueError(f"event already reviewed with a different label: {event_id}")

        source: Path | None = None
        destination: Path | None = None
        if label in {"fall", "non_fall"}:
            if not row.get("clip_path"):
                raise ValueError(f"clip_path is missing: {event_id}")
            source = resolve_clip_path(str(row["clip_path"]), clip_dir=clip_dir)
            if not source.is_file():
                raise FileNotFoundError(f"clip not found: {source}")
            destination = labeled_dir / label / source.name
            if destination.exists() and destination != source:
                raise FileExistsError(f"labeled clip already exists: {destination}")
        updates.append((row, label, source, destination))

    backup = create_backup(review_log)
    moved: list[tuple[Path, Path]] = []
    try:
        for row, label, source, destination in updates:
            if source is not None and destination is not None and source != destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
                row["clip_path"] = str(destination)
            row["label"] = None if label == "needs_review" else label
            row["review_status"] = (
                "needs_review" if label == "needs_review" else "reviewed"
            )
        write_review_log_atomic(review_log, rows)
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
        raise

    return {
        "requested": len(labels_by_event),
        "updated": len(updates),
        "unchanged": len(labels_by_event) - len(updates),
        "backup": str(backup),
    }


def _draw_overlay(frame: Any, candidate: dict[str, Any], index: int, total: int) -> None:
    import cv2

    lines = [
        f"[{index + 1}/{total}] {candidate.get('event_id')}",
        "F: fall   N: non-fall   S: skip/review   SPACE: pause   R: replay   Q: quit",
    ]
    for line_index, text in enumerate(lines):
        y = 32 + line_index * 30
        cv2.putText(
            frame,
            text,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def review_candidate(
    candidate: dict[str, Any], index: int, total: int
) -> tuple[str | None, str] | None:
    import cv2

    clip_path = str(candidate["local_clip_path"])
    capture = cv2.VideoCapture(clip_path)
    if not capture.isOpened():
        raise RuntimeError(f"could not open clip: {clip_path}")

    paused = False
    last_frame = None
    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                last_frame = frame
            if last_frame is None:
                continue
            display_frame = last_frame.copy()
            _draw_overlay(display_frame, candidate, index, total)
            cv2.imshow("Fall shadow clip labeler", display_frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("f"), ord("F")):
                return "fall", "reviewed"
            if key in (ord("n"), ord("N")):
                return "non_fall", "reviewed"
            if key in (ord("s"), ord("S")):
                return None, "needs_review"
            if key in (ord("q"), ord("Q"), 27):
                return None
            if key == ord(" "):
                paused = not paused
            if key in (ord("r"), ord("R")):
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-log", type=Path, default=DEFAULT_REVIEW_LOG)
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR)
    parser.add_argument("--labeled-dir", type=Path, default=DEFAULT_LABELED_DIR)
    parser.add_argument("--camera", default="camera_1")
    parser.add_argument("--include-sample-eval", action="store_true")
    parser.add_argument(
        "--only-disagreements",
        action="store_true",
        help="Only include valid primary/Shadow classification disagreements.",
    )
    parser.add_argument(
        "--include-threshold-boundary",
        type=float,
        default=0.0,
        metavar="MARGIN",
        help="With disagreement filtering, also include scores within MARGIN of the runtime threshold.",
    )
    parser.add_argument(
        "--since",
        help="Only review clips created at or after this ISO-8601 timestamp.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Review at most this many candidates (0 means all).",
    )
    parser.add_argument(
        "--min-fall-probability",
        type=float,
        default=0.0,
        help="Only include clips whose falldata runtime probability reaches this value.",
    )
    parser.add_argument(
        "--export-jsonl",
        type=Path,
        help="Export selected candidates as JSONL and exit without opening the GUI.",
    )
    parser.add_argument(
        "--export-html",
        type=Path,
        help="Export selected candidates as a browser review page and exit.",
    )
    parser.add_argument(
        "--import-labels",
        type=Path,
        help="Apply a fall_review_labels.json file and exit without opening the GUI.",
    )
    parser.add_argument(
        "--dedupe-seconds",
        type=float,
        default=0.0,
        help="Keep one representative clip per camera within this time window.",
    )
    parser.add_argument("--list", action="store_true", help="List candidates without GUI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.import_labels:
        summary = import_review_labels(
            args.review_log,
            args.import_labels,
            clip_dir=args.clip_dir,
            labeled_dir=args.labeled_dir,
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    rows = read_review_log(args.review_log)
    candidates = select_candidates(
        rows,
        camera=None if args.include_sample_eval else args.camera,
        include_sample_eval=args.include_sample_eval,
        min_fall_probability=max(args.min_fall_probability, 0.0),
        only_disagreements=args.only_disagreements,
        include_threshold_boundary=max(args.include_threshold_boundary, 0.0),
        clip_dir=args.clip_dir,
    )
    if args.since:
        candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("created_at") or "") >= args.since
        ]
    if args.dedupe_seconds > 0:
        deduped: list[dict[str, Any]] = []
        last_by_camera: dict[str, float] = {}
        for candidate in candidates:
            try:
                created = datetime.fromisoformat(
                    str(candidate.get("created_at", "")).replace("Z", "+00:00")
                ).astimezone(timezone.utc).timestamp()
            except (TypeError, ValueError):
                created = float("inf")
            camera_id = str(candidate.get("camera_id") or "")
            if created - last_by_camera.get(camera_id, float("-inf")) < args.dedupe_seconds:
                continue
            deduped.append(candidate)
            last_by_camera[camera_id] = created
        candidates = deduped
    if args.limit > 0:
        candidates = candidates[: args.limit]
    print(f"Unlabeled clips: {len(candidates)}")
    if args.export_html:
        browser_candidates = prepare_browser_clips(
            candidates, args.export_html.parent / "browser_clips"
        )
        write_review_html(args.export_html, browser_candidates)
        print(f"Exported: {args.export_html}")
        return 0
    if args.export_jsonl:
        args.export_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.export_jsonl.open("w", encoding="utf-8") as fp:
            for candidate in candidates:
                fp.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"Exported: {args.export_jsonl}")
        return 0
    if args.list:
        for candidate in candidates[:20]:
            print(candidate["event_id"], candidate["local_clip_path"])
        return 0
    if not candidates:
        return 0

    backup = create_backup(args.review_log)
    print(f"Backup: {backup}")
    for index, candidate in enumerate(candidates):
        result = review_candidate(candidate, index, len(candidates))
        if result is None:
            print("Saved. Resume with the same command.")
            break
        label, review_status = result
        apply_label(
            args.review_log,
            event_id=str(candidate["event_id"]),
            label=label,
            review_status=review_status,
            clip_dir=args.clip_dir,
            labeled_dir=args.labeled_dir,
        )
        print(f"{candidate['event_id']}: {label or review_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
