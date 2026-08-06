# Fall accuracy baseline — 2026-07-23

## Repository checks

- Pytest: 1322 passed, 75 skipped, 1 stale Compose assertion failed before baseline correction.
- Jetson Compose config: passed with `.env.jetson` explicitly supplied.
- Ruff: 6 pre-existing import-order issues outside the fall path.

## Model quality

| Candidate | Validation size | Fall precision | Fall recall | Fall F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| `open_fall_rf_500_val80` | 120 | 0.655 | 0.760 | 0.704 | 20 | 12 |
| `yolo_pose_fall_rf_20_uniform` | 20 | 1.000 | 0.500 | 0.667 | 0 | 5 |

The Falldata candidate uses `scene_base` group separation with no overlap. The
YOLO-Pose candidate report lacks group-split metadata and is not eligible for
deployment comparison until retrained with grouped validation.

## Field data readiness

- Collected clips: 577
- Review records: 5,612
- Labeled review records: 0

Field clips must not be used as supervised targets until reviewed as `fall`,
`not_fall`, or `ambiguous`.
