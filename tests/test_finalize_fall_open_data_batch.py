import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path("scripts/ops/finalize_fall_open_data_batch.py")
WATCH_SCRIPT = Path("scripts/ops/run_unattended_fall_batch_followup.sh")


def _write_fixture(tmp_path: Path, *, processed: int = 4, no_result: bool = False) -> tuple[Path, Path]:
    manifest_rows = [
        {"scene_id": "front_c1", "camera": 1, "scene_category": "전면낙상", "is_fall": True},
        {"scene_id": "front_c2", "camera": 2, "scene_category": "전면낙상", "is_fall": True},
        {"scene_id": "normal_c1", "camera": 1, "scene_category": "비낙상", "is_fall": False},
        {"scene_id": "normal_c2", "camera": 2, "scene_category": "비낙상", "is_fall": False},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )

    videos = []
    baseline_predictions = [1, 0, 0, 0]
    candidate_predictions = [1, 1, 1, 0]
    for index, row in enumerate(manifest_rows[:processed]):
        baseline_prediction = baseline_predictions[index]
        candidate_prediction = candidate_predictions[index]
        if no_result and index == processed - 1:
            candidate_prediction = None
        is_fall = row["is_fall"]
        videos.append(
            {
                "scene_id": row["scene_id"],
                "video_path": f"/app/{row['scene_id']}.mp4",
                "is_fall": is_fall,
                "baseline": {
                    "prediction": baseline_prediction,
                    "fall_probability": float(baseline_prediction),
                    "frames_with_pose": 30,
                    "error": None,
                },
                "candidate": {
                    "prediction": candidate_prediction,
                    "fall_probability": None if candidate_prediction is None else float(candidate_prediction),
                    "frames_with_pose": 0 if candidate_prediction is None else 30,
                    "error": "failed" if candidate_prediction is None else None,
                },
            }
        )
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"videos": videos}, ensure_ascii=False), encoding="utf-8")
    return manifest, comparison


def _run_finalizer(tmp_path: Path, manifest: Path, comparison: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--comparison",
            str(comparison),
            "--report",
            str(tmp_path / "report.md"),
            "--status",
            str(tmp_path / "status.json"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )


def test_completed_batch_writes_grouped_report_and_status(tmp_path: Path) -> None:
    manifest, comparison = _write_fixture(tmp_path)

    completed = _run_finalizer(tmp_path, manifest, comparison)

    assert completed.returncode == 0, completed.stderr
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["processed"] == 4
    assert status["expected"] == 4
    assert status["no_result"] == {"baseline": 0, "candidate": 0}
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## 전체 결과" in report
    assert "## 카메라별 결과" in report
    assert "## 장면 방향별 결과" in report
    assert "C1" in report
    assert "전면낙상" in report
    assert "운영 모델 자동 교체는 수행하지 않았다" in report


def test_incomplete_batch_records_progress_without_writing_final_report(tmp_path: Path) -> None:
    manifest, comparison = _write_fixture(tmp_path, processed=2)

    completed = _run_finalizer(tmp_path, manifest, comparison)

    assert completed.returncode == 2
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "incomplete"
    assert status["processed"] == 2
    assert status["expected"] == 4
    assert status["progress_percent"] == 50.0
    assert not (tmp_path / "report.md").exists()


def test_completed_batch_with_no_result_is_marked_for_review(tmp_path: Path) -> None:
    manifest, comparison = _write_fixture(tmp_path, no_result=True)

    completed = _run_finalizer(tmp_path, manifest, comparison)

    assert completed.returncode == 0, completed.stderr
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed_with_issues"
    assert status["no_result"] == {"baseline": 0, "candidate": 1}
    assert "failed" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_unattended_runner_finalizes_an_already_completed_batch(tmp_path: Path) -> None:
    manifest, comparison = _write_fixture(tmp_path)
    report = tmp_path / "unattended_report.md"
    status = tmp_path / "unattended_status.json"
    log = tmp_path / "unattended.log"
    lock_dir = tmp_path / "unattended.lock"
    environment = {
        "PATH": "/usr/bin:/bin",
        "CCTV_PROJECT_ROOT": str(Path(__file__).resolve().parents[1]),
        "CCTV_FALL_MANIFEST": str(manifest),
        "CCTV_FALL_COMPARISON": str(comparison),
        "CCTV_FALL_REPORT": str(report),
        "CCTV_FALL_STATUS": str(status),
        "CCTV_FALL_LOG": str(log),
        "CCTV_FALL_LOCK_DIR": str(lock_dir),
        "CCTV_FALL_PROCESS_PATTERN": "process-that-does-not-exist-for-test",
        "CCTV_FALL_POLL_SECONDS": "0",
        "CCTV_FALL_MAX_WAIT_SECONDS": "5",
        "CCTV_FALL_RESTART_LIMIT": "0",
    }

    completed = subprocess.run(
        ["bash", str(WATCH_SCRIPT)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(status.read_text(encoding="utf-8"))["status"] == "complete"
    assert report.exists()
    assert "batch finalization complete" in log.read_text(encoding="utf-8")
    assert not lock_dir.exists()


def test_unattended_runner_defaults_outputs_to_container_mounted_data(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    manifest, comparison = _write_fixture(fixture_dir)
    batch_dir = tmp_path / "data" / "fall_eval" / "open_data_test_20260914"
    batch_dir.mkdir(parents=True)
    manifest.replace(batch_dir / "balanced320_c1_c8.jsonl")
    comparison.replace(batch_dir / "balanced320_c1_c8_comparison.json")
    environment = {
        "PATH": "/usr/bin:/bin",
        "CCTV_PROJECT_ROOT": str(tmp_path),
        "CCTV_FALL_FINALIZER": str(
            Path(__file__).resolve().parents[1]
            / "scripts/ops/finalize_fall_open_data_batch.py"
        ),
        "CCTV_FALL_PROCESS_PATTERN": "process-that-does-not-exist-for-default-test",
        "CCTV_FALL_POLL_SECONDS": "0",
        "CCTV_FALL_MAX_WAIT_SECONDS": "5",
        "CCTV_FALL_RESTART_LIMIT": "0",
    }

    completed = subprocess.run(
        ["bash", str(WATCH_SCRIPT)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (batch_dir / "FALL_OPEN_DATA_C1_C8_FINAL_20260914.md").exists()
    assert (batch_dir / "unattended_followup.log").exists()


def test_unattended_runner_exits_and_releases_lock_on_termination(tmp_path: Path) -> None:
    manifest, comparison = _write_fixture(tmp_path, processed=2)
    lock_dir = tmp_path / "unattended.lock"
    environment = {
        "PATH": "/usr/bin:/bin",
        "CCTV_PROJECT_ROOT": str(Path(__file__).resolve().parents[1]),
        "CCTV_FALL_MANIFEST": str(manifest),
        "CCTV_FALL_COMPARISON": str(comparison),
        "CCTV_FALL_REPORT": str(tmp_path / "report.md"),
        "CCTV_FALL_STATUS": str(tmp_path / "status.json"),
        "CCTV_FALL_LOG": str(tmp_path / "unattended.log"),
        "CCTV_FALL_LOCK_DIR": str(lock_dir),
        "CCTV_FALL_PROCESS_PATTERN": "pytest",
        "CCTV_FALL_POLL_SECONDS": "60",
        "CCTV_FALL_MAX_WAIT_SECONDS": "120",
        "CCTV_FALL_RESTART_LIMIT": "0",
    }
    process = subprocess.Popen(
        ["bash", str(WATCH_SCRIPT)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 3
        while not lock_dir.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert lock_dir.exists()

        process.terminate()
        returncode = process.wait(timeout=3)

        assert returncode == 130
        assert not lock_dir.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
