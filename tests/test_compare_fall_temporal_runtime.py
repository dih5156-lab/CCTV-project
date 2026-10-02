import csv
import json
import subprocess
from pathlib import Path

import pytest

from scripts.ops import compare_fall_temporal_runtime as runtime_compare


def test_parse_smoke_output_extracts_required_prediction_fields() -> None:
    result = runtime_compare.parse_smoke_output(
        "prediction: [1]\n"
        "fall_probability: 0.92340000\n"
        "threshold: 0.71000000\n"
        "frames_with_pose: 42\n"
    )

    assert result == {
        "prediction": 1,
        "fall_probability": 0.9234,
        "threshold": 0.71,
        "frames_with_pose": 42,
    }


def test_resolve_container_video_path_requires_video_inside_project_root(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    video_path = project_root / "data" / "fall.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()

    assert runtime_compare.resolve_container_video_path(
        video_path,
        project_root=project_root,
        container_project_root=Path("/app"),
    ) == Path("/app/data/fall.mp4")


def test_summarize_model_results_keeps_no_result_outside_confusion_matrix() -> None:
    rows = [
        {"is_fall": True, "baseline": {"prediction": 1}},
        {"is_fall": True, "baseline": {"prediction": 0}},
        {"is_fall": False, "baseline": {"prediction": 1}},
        {"is_fall": False, "baseline": {"prediction": 0}},
        {"is_fall": True, "baseline": {"prediction": None, "error": "failed"}},
    ]

    summary = runtime_compare.summarize_model_results(rows, "baseline")

    assert summary["counts"] == {
        "TP": 1,
        "FN": 1,
        "FP": 1,
        "TN": 1,
        "NO_RESULT": 1,
    }
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["f1"] == 0.5


def test_run_comparison_writes_intermediate_json_and_csv_after_each_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    videos = project_root / "data"
    videos.mkdir(parents=True)
    for name in ("fall.mp4", "walk.mp4"):
        (videos / name).touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"scene_id": "fall", "video_path": "data/fall.mp4", "is_fall": True}),
                json.dumps({"scene_id": "walk", "video_path": "data/walk.mp4", "label": "notfall"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "results.json"
    output_csv = tmp_path / "results.csv"
    persisted_counts: list[int] = []
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, "prediction: [1]\nfall_probability: 0.9\nthreshold: 0.7\nframes_with_pose: 31\n", ""),
            subprocess.CompletedProcess([], 0, "prediction: [1]\nfall_probability: 0.8\nthreshold: 0.7\nframes_with_pose: 30\n", ""),
            subprocess.CompletedProcess([], 0, "prediction: [0]\nfall_probability: 0.1\nthreshold: 0.7\nframes_with_pose: 28\n", ""),
            subprocess.CompletedProcess([], 1, "", "pose model unavailable"),
        ]
    )

    def fake_run(*_args, **_kwargs):
        return next(responses)

    original_write_outputs = runtime_compare.write_outputs

    def recording_write_outputs(*args, **kwargs):
        original_write_outputs(*args, **kwargs)
        persisted_counts.append(len(json.loads(output_json.read_text(encoding="utf-8"))["videos"]))

    monkeypatch.setattr(runtime_compare.subprocess, "run", fake_run)
    monkeypatch.setattr(runtime_compare, "write_outputs", recording_write_outputs)

    result = runtime_compare.run_comparison(
        manifest=manifest,
        baseline_model=Path("models/baseline.pt"),
        candidate_model=Path("models/candidate.pt"),
        pose_model=Path("models/pose.pt"),
        output_json=output_json,
        output_csv=output_csv,
        python_executable="python",
        project_root=project_root,
        container_project_root=Path("/app"),
        device="cpu",
        max_videos=0,
    )

    assert persisted_counts == [1, 2]
    assert result["summary"]["baseline"]["counts"] == {
        "TP": 1,
        "FN": 0,
        "FP": 0,
        "TN": 0,
        "NO_RESULT": 1,
    }
    assert result["summary"]["candidate"]["counts"] == {
        "TP": 1,
        "FN": 0,
        "FP": 0,
        "TN": 1,
        "NO_RESULT": 0,
    }
    assert result["videos"][1]["baseline"]["error"] == "pose model unavailable"
    assert result["videos"][0]["delta"] == {"prediction": 0, "fall_probability": -0.1}
    with output_csv.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert [row["scene_id"] for row in rows] == ["fall", "walk"]
    assert rows[1]["baseline_result"] == "NO_RESULT"
    assert [row["execution_order"] for row in rows] == [
        "baseline,candidate",
        "candidate,baseline",
    ]


def test_run_comparison_alternates_model_execution_order_per_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    videos = project_root / "data"
    videos.mkdir(parents=True)
    for name in ("one.mp4", "two.mp4"):
        (videos / name).touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"scene_id": "one", "video_path": "data/one.mp4", "is_fall": True}),
                json.dumps({"scene_id": "two", "video_path": "data/two.mp4", "is_fall": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        prediction = 1 if any("baseline.pt" in argument for argument in command) else 0
        return subprocess.CompletedProcess(
            command,
            0,
            f"prediction: [{prediction}]\n"
            f"fall_probability: {float(prediction):.1f}\n"
            "threshold: 0.7\n"
            "frames_with_pose: 30\n",
            "",
        )

    monkeypatch.setattr(runtime_compare.subprocess, "run", fake_run)

    result = runtime_compare.run_comparison(
        manifest=manifest,
        baseline_model=Path("models/baseline.pt"),
        candidate_model=Path("models/candidate.pt"),
        pose_model=Path("models/pose.pt"),
        output_json=tmp_path / "results.json",
        output_csv=tmp_path / "results.csv",
        python_executable="python",
        project_root=project_root,
        container_project_root=Path("/app"),
        device="cpu",
    )

    assert [video["execution_order"] for video in result["videos"]] == [
        ["baseline", "candidate"],
        ["candidate", "baseline"],
    ]
    assert [command[3] for command in commands] == [
        "models/baseline.pt",
        "models/candidate.pt",
        "models/candidate.pt",
        "models/baseline.pt",
    ]
    assert result["videos"][1]["baseline"]["prediction"] == 1
    assert result["videos"][1]["candidate"]["prediction"] == 0


def test_run_comparison_forwards_and_records_pose_extraction_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    video_path = project_root / "data" / "fall.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"scene_id": "fall", "video_path": "data/fall.mp4", "is_fall": True}
        )
        + "\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            "prediction: [1]\n"
            "fall_probability: 0.9\n"
            "threshold: 0.5\n"
            "frames_with_pose: 30\n",
            "",
        )

    monkeypatch.setattr(runtime_compare.subprocess, "run", fake_run)

    result = runtime_compare.run_comparison(
        manifest=manifest,
        baseline_model=Path("models/baseline.pt"),
        candidate_model=Path("models/candidate.pt"),
        pose_model=Path("models/pose.pt"),
        output_json=tmp_path / "results.json",
        output_csv=tmp_path / "results.csv",
        python_executable="python",
        project_root=project_root,
        container_project_root=Path("/app"),
        device="cpu",
        imgsz=640,
        confidence_threshold=0.35,
    )

    for command in commands:
        assert command[-4:] == [
            "--imgsz",
            "640",
            "--confidence-threshold",
            "0.35",
        ]
    assert result["pose_extraction"] == {
        "imgsz": 640,
        "confidence_threshold": 0.35,
    }


def test_run_comparison_rejects_empty_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest contains no videos"):
        runtime_compare.run_comparison(
            manifest=manifest,
            baseline_model=Path("models/baseline.pt"),
            candidate_model=Path("models/candidate.pt"),
            pose_model=Path("models/pose.pt"),
            output_json=tmp_path / "results.json",
            output_csv=tmp_path / "results.csv",
            project_root=tmp_path,
        )


def test_summarize_model_results_separates_successful_and_attempted_average_time() -> None:
    rows = [
        {
            "is_fall": True,
            "baseline": {"prediction": 1, "elapsed_seconds": 0.2},
        },
        {
            "is_fall": False,
            "baseline": {
                "prediction": None,
                "error": "timeout",
                "elapsed_seconds": 0.8,
            },
        },
    ]

    summary = runtime_compare.summarize_model_results(rows, "baseline")

    assert summary["average_execution_seconds"] == 0.2
    assert summary["attempted_average_execution_seconds"] == 0.5


def test_read_manifest_rejects_negative_max_videos(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"scene_id": "one", "video_path": "one.mp4", "is_fall": True})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_videos must be zero or greater"):
        runtime_compare.read_manifest(manifest, max_videos=-1)


def test_run_comparison_reuses_first_models_pose_feature_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    video_path = project_root / "data" / "fall.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"scene_id": "fall", "video_path": "data/fall.mp4", "is_fall": True}
        )
        + "\n",
        encoding="utf-8",
    )
    feature_cache_dir = tmp_path / "features"
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if "--output-feature-json" in command:
            output_index = command.index("--output-feature-json") + 1
            feature_path = Path(command[output_index])
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            feature_path.write_text(
                json.dumps(
                    {
                        "frames_seen": 60,
                        "frames_with_pose": 30,
                        "frame_records": [],
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            "prediction: [1]\n"
            "fall_probability: 0.9\n"
            "threshold: 0.5\n"
            "frames_with_pose: 30\n",
            "",
        )

    monkeypatch.setattr(runtime_compare.subprocess, "run", fake_run)

    runtime_compare.run_comparison(
        manifest=manifest,
        baseline_model=Path("models/baseline.pt"),
        candidate_model=Path("models/candidate.pt"),
        pose_model=Path("models/pose.pt"),
        output_json=tmp_path / "results.json",
        output_csv=tmp_path / "results.csv",
        python_executable="python",
        project_root=project_root,
        container_project_root=Path("/app"),
        device="cpu",
        feature_cache_dir=feature_cache_dir,
    )

    assert "--video" in commands[0]
    assert "--output-feature-json" in commands[0]
    assert "--feature-json" not in commands[0]
    assert "--feature-json" in commands[1]
    assert "--video" not in commands[1]
