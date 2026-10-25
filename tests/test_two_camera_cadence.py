"""추론 주기 비교에서 미완료 재생과 실제 낙상 누락을 구분한다."""

import importlib
import json
from unittest.mock import Mock

import pytest


def module():
    return importlib.import_module("scripts.ops.two_camera_cadence")


def capture(*, camera="replay_1", alarm=None, complete=True):
    return {
        "camera_id": camera,
        "all_frames_processed": complete,
        "fps": 24,
        "source": [
            {"frame_index": i, "pts": i / 24, "monotonic": 100 + i / 24}
            for i in range(240)
        ],
        "processed": [
            {"frame_index": i, "pts": i / 24, "monotonic": 100.01 + i / 24}
            for i in range(240)
        ],
        "stats": {
            "frames_processed": 240,
            "frames_dropped": 0,
            "primary_camera_stats": {
                camera: {
                    "frames_seen": 240,
                    "frames_inferred": 80,
                    "frames_inference_unknown": 0,
                }
            },
        },
        "events": []
        if alarm is None
        else [
            {
                "stage": "publish_queue",
                "accepted": True,
                "frame_index": alarm,
                "monotonic": 100.01 + alarm / 24,
            }
        ],
    }


def test_24fps_maps_to_exact_requested_cadence():
    assert module().conditions() == [(12, 1), (8, 2), (6, 3)]


def test_onset_seconds_survive_resampling():
    assert (
        module().onset_seconds({"is_fall": True, "fall_start_frame": 285}, "60/1")
        == 4.75
    )
    with pytest.raises(ValueError):
        module().onset_seconds({"is_fall": True, "fall_start_frame": None}, "60/1")


def test_resampling_command_keeps_duration_and_audio_out():
    command = module().video_command("input.mp4", "output.mp4")
    assert "setpts" not in " ".join(command)
    assert "fps=24" in command[command.index("-vf") + 1]
    assert "-an" in command and "-n" in command


def test_fall_miss_and_normal_false_alarm_are_separate():
    assert module().score_capture(capture(), True, 2.0, 8)["verdict"] == "FN"
    assert module().score_capture(capture(alarm=100), False, None, 8)["verdict"] == "FP"
    assert module().score_capture(capture(), False, None, 8)["verdict"] == "TN"


def test_alarm_delay_uses_source_seconds_and_wall_clock():
    result = module().score_capture(capture(alarm=120), True, 2.0, 8)
    assert result["verdict"] == "TP"
    assert result["pts_delay_seconds"] == pytest.approx(3)
    assert result["wall_delay_seconds"] == pytest.approx(3.01)
    assert result["inference_hz"] == pytest.approx(8, rel=0.02)


def test_incomplete_or_wrong_camera_capture_is_invalid_not_a_miss():
    assert (
        module().score_capture(capture(complete=False), True, 2, 8)["verdict"]
        == "INVALID"
    )
    row = capture()
    row["stats"]["primary_camera_stats"] = {"wrong_camera": {}}
    assert module().score_capture(row, True, 2, 8)["verdict"] == "INVALID"


def test_early_and_late_alarms_are_reported_without_hiding_them():
    assert module().score_capture(capture(alarm=12), True, 2, 8)["early_count"] == 1
    late = module().score_capture(capture(alarm=220), True, 2, 8)
    assert late["verdict"] == "TP" and late["over_5s"]


def test_two_channels_must_really_overlap():
    a, b = capture(), capture(camera="replay_2")
    assert module().overlap_ratio(a, b) == pytest.approx(1)
    for row in b["processed"]:
        row["monotonic"] += 20
    assert module().overlap_ratio(a, b) == 0


def test_processed_gap_is_not_reported_as_good_throughput():
    row = capture()
    for frame in row["processed"]:
        frame["monotonic"] = 100 + frame["pts"] * 2
    result = module().score_capture(row, True, 2, 8)
    assert not result["throughput_pass"]
    assert result["lag_growth_seconds"] > 9


def operating_sample(at=100, frames=1000, inferred=500):
    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")
    return {
        "at": at,
        "values": {
            runner.FRAME: frames,
            runner.INFER: inferred,
            runner.AGE: 0.1,
            **dict.fromkeys(runner.ERRORS, 0),
        },
        "memory_available_kib": 20 * 1024**2,
        "temperatures_c": [50, 51],
    }


def test_operating_guard_accepts_30fps_15hz_and_rejects_degradation():
    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")
    before, after = operating_sample(), operating_sample(105, 1150, 575)
    runner.guard(before, after)
    assert after["window"]["fps"] == 30
    assert after["window"]["primary_hz"] == 15
    with pytest.raises(RuntimeError, match="처리율"):
        runner.guard(before, operating_sample(105, 1100, 550))
    with pytest.raises(RuntimeError, match="초기화"):
        runner.guard(before, operating_sample(105, 1, 1))
    after["values"][runner.ERRORS[0]] = 1
    with pytest.raises(RuntimeError, match="오류"):
        runner.guard(before, after)
    after["values"][runner.FRAME] = float("nan")
    with pytest.raises(RuntimeError, match="수치"):
        runner.guard(before, after)


def test_unknown_or_missing_inference_count_cannot_pass():
    row = capture()
    row["stats"]["primary_camera_stats"]["replay_1"].pop("frames_inferred")
    assert module().score_capture(row, True, 2, 8)["verdict"] == "INVALID"


def test_partial_runs_do_not_produce_a_zero_miss_rate():
    rows = [{"hz": hz, "verdict": "NOT_RUN"} for hz in (12, 8, 6) for _ in range(20)]
    summaries = module().summarize(rows)
    assert all(
        not row["complete"] and row["miss_rate"] is None and not row["quality_pass"]
        for row in summaries
    )


@pytest.mark.parametrize("fault", ["camera", "frame_count", "execution", "overlap"])
def test_results_require_assigned_inputs_and_successful_concurrent_execution(
    tmp_path, fault
):
    m = module()
    m.write(
        tmp_path / "plan.json",
        {
            "scope": "test",
            "dataset": "test",
            "cases": [{"hz": 8, "pair": 0, "directory": str(tmp_path)}],
        },
    )
    m.write(tmp_path / "execution.json", {"valid_execution": fault != "execution"})
    for slot in (1, 2):
        folder = tmp_path / f"replay_{slot}"
        (folder / "output").mkdir(parents=True)
        m.write(
            folder / "manifest.json",
            {
                "camera_id": f"replay_{slot}",
                "expected_frames": 241 if fault == "frame_count" else 240,
                "scene_id": str(slot),
                "is_fall": True,
                "onset_seconds": 2,
            },
        )
        row = capture(camera="wrong" if fault == "camera" else f"replay_{slot}")
        if fault == "overlap" and slot == 2:
            for frame in row["processed"]:
                frame["monotonic"] += 20
        m.write(folder / "output/capture.json", row)
    m.evaluate(tmp_path)
    assert all(
        row["verdict"] == "INVALID"
        for row in m.read(tmp_path / "comparison.json")["rows"]
    )


def test_inactive_jetson_thermal_sensor_does_not_mask_active_sensor(monkeypatch):
    import errno
    from pathlib import Path
    from types import SimpleNamespace

    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")

    def inactive():
        raise OSError(errno.ENODATA, "no data")

    active = SimpleNamespace(read_text=lambda: "50000")
    monkeypatch.setattr(
        Path, "glob", lambda *args: [SimpleNamespace(read_text=inactive), active]
    )
    monkeypatch.setattr(Path, "read_text", lambda *args: "MemAvailable: 20000000 kB\n")
    raw = "\n".join(
        f"{key} {value}"
        for key, value in {
            runner.FRAME: 1000,
            runner.INFER: 500,
            runner.AGE: 0.1,
            **dict.fromkeys(runner.ERRORS, 0),
        }.items()
    )
    monkeypatch.setattr(
        runner.urllib.request,
        "urlopen",
        lambda *args, **kwargs: SimpleNamespace(read=lambda: raw.encode()),
    )
    assert runner.metrics()["values"][runner.FRAME] == 1000
    active.read_text = lambda: "81000"
    sample = runner.metrics()
    assert sample["temperatures_c"] == [81]
    assert sample["memory_available_kib"] == 20000000
    with pytest.raises(RuntimeError, match="80도"):
        runner.guard(None, sample)


def test_summary_pass_requires_complete_quality_and_throughput():
    m = module()
    rows = [
        {
            "hz": 8,
            **m.score_capture(
                capture(alarm=120 if i % 2 else None),
                bool(i % 2),
                2 if i % 2 else None,
                8,
            ),
        }
        for i in range(20)
    ]
    summary = m.summarize(rows)[1]
    assert (
        summary["complete"] and summary["quality_pass"] and summary["throughput_pass"]
    )
    assert summary["miss_rate"] == summary["false_alarm_clip_rate"] == 0


def test_pilot_limits_both_conversion_and_execution_to_one_pair():
    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")
    plan = {
        "cases": [
            {"pair": pair, "hz": hz, "scenes": [f"fall{pair}", f"normal{pair}"]}
            for pair in range(10)
            for hz in (12, 8, 6)
        ],
        "scenes": [
            {"scene_id": f"{kind}{pair}", "is_fall": kind == "fall"}
            for pair in range(10)
            for kind in ("fall", "normal")
        ],
    }
    pilot = runner.select_pilot(plan)
    assert len(pilot["cases"]) == 3 and len(pilot["scenes"]) == 2
    assert {s["scene_id"] for s in pilot["scenes"]} == {"fall0", "normal0"}
    assert len(plan["cases"]) == 30 and len(plan["scenes"]) == 20
    plan["cases"][0]["scenes"] = ["fall1", "normal1"]
    with pytest.raises(ValueError, match="동일한"):
        runner.select_pilot(plan)


def test_pilot_slot_rejects_unassigned_expired_and_overlong_windows(monkeypatch):
    from datetime import datetime, timedelta, timezone

    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")
    monkeypatch.setattr(runner, "WORK_DEADLINE", None)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 1000)
    start = datetime.now(timezone.utc) - timedelta(seconds=1)
    end = start + timedelta(minutes=15)
    runner.configure_slot(start.isoformat(), end.isoformat())
    assert runner.WORK_DEADLINE == pytest.approx(1599, abs=2)
    for a, b in [
        (start + timedelta(minutes=1), end),
        (start, end + timedelta(seconds=1)),
        (start - timedelta(minutes=16), end - timedelta(minutes=16)),
        (start, start + timedelta(minutes=4)),
    ]:
        with pytest.raises(ValueError):
            runner.configure_slot(a.isoformat(), b.isoformat())
    with pytest.raises(ValueError, match="시간대"):
        runner.configure_slot(
            start.replace(tzinfo=None).isoformat(), end.replace(tzinfo=None).isoformat()
        )


def test_cleanup_cutoff_stops_work_before_any_metrics_request(monkeypatch):
    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")
    monkeypatch.setattr(runner, "WORK_DEADLINE", 100)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 101)
    monkeypatch.setattr(
        runner.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("cutoff must stop before I/O"),
    )
    with pytest.raises(RuntimeError, match="정리 구간"):
        runner.metrics()


@pytest.mark.parametrize("action", ["execute", "pilot"])
def test_pilot_cli_requires_explicit_slot_and_rejects_full_sweep(monkeypatch, action):
    import sys

    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")
    monkeypatch.setattr(sys, "argv", ["run_two_camera_cadence.py", action])
    monkeypatch.setattr(
        runner, "metrics", lambda: pytest.fail("no precheck before slot validation")
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("force_kill", [False, True])
def test_conversion_saves_failed_window_before_stopping_its_process(
    monkeypatch, tmp_path, force_kill
):
    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")
    before, after = operating_sample(), operating_sample(105, 1100, 550)
    after.update(memory_available_kib=7 * 1024**2, temperatures_c=[81])
    monkeypatch.setattr(runner, "metrics", Mock(side_effect=[before, after]))
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    proc = Mock(returncode=None)
    proc.poll.return_value = None
    saved = tmp_path / "fall.guard.json"

    def terminate():
        # 정리 도중 오류가 생겨도 직전 실패 표본은 이미 디스크에 남아 있어야 한다.
        record = json.loads(saved.read_text())
        assert not record["samples"][-1]["guard_passed"]
        assert record["samples"][-1]["window"]["fps"] == 20
        if not force_kill:
            proc.returncode = -15

    def kill():
        proc.returncode = -9

    proc.terminate.side_effect = terminate
    proc.kill.side_effect = kill
    if force_kill:
        proc.wait.side_effect = [runner.subprocess.TimeoutExpired("mock", 5), -9]
    monkeypatch.setattr(runner.subprocess, "Popen", Mock(return_value=proc))
    with pytest.raises(RuntimeError, match="처리율"):
        runner.guarded_process(["mock-convert"], tmp_path / "fall.log", 120)
    record = json.loads(saved.read_text())
    assert record["process_started"] and not record["passed"]
    assert record["returncode"] == (-9 if force_kill else -15)
    first, failed = record["samples"]
    assert first["at"] == 100 and failed["at"] == 105
    assert first["values"][runner.FRAME] == 1000
    assert failed["values"][runner.FRAME] == 1100
    assert failed["window"]["duration_seconds"] == 5
    assert failed["window"]["primary_hz"] == 10
    assert failed["memory_available_kib"] == 7 * 1024**2
    assert failed["max_temperature_c"] == 81
    assert failed["observed_at"].endswith("+09:00")
    assert "8GiB" in record["reason"] and "80도" in record["reason"]
    proc.terminate.assert_called_once()
    assert proc.kill.call_count == int(force_kill)


@pytest.mark.parametrize(
    "resource, reason",
    [
        ({"memory_available_kib": 7 * 1024**2}, "8GiB"),
        ({"temperatures_c": [80]}, "80도"),
        ({"temperatures_c": []}, "온도 센서"),
    ],
)
def test_conversion_records_initial_resource_failure_without_starting(
    monkeypatch, tmp_path, resource, reason
):
    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")
    sample = {**operating_sample(), **resource}
    monkeypatch.setattr(runner, "metrics", Mock(return_value=sample))
    spawn = Mock(side_effect=AssertionError("보호 실패 뒤 프로세스 실행 금지"))
    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match=reason):
        runner.guarded_process(["mock-convert"], tmp_path / "fall.log", 120)
    record = json.loads((tmp_path / "fall.guard.json").read_text())
    assert not record["process_started"] and not record["passed"]
    assert record["samples"][0]["window"] is None
    assert reason in record["samples"][0]["reason"]
    assert record["samples"][0]["values"] == sample["values"]
    spawn.assert_not_called()
    assert not (tmp_path / "fall.log").exists()


def test_metric_fetch_failure_is_unknown_not_zero_and_preserves_prior_record(
    monkeypatch, tmp_path
):
    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")
    monkeypatch.setattr(
        runner, "metrics", Mock(side_effect=OSError("metrics unavailable"))
    )
    spawn = Mock(side_effect=AssertionError("계측 불가 상태에서 실행 금지"))
    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="metrics unavailable"):
        runner.guarded_process(["mock-convert"], tmp_path / "fall.log", 120)
    path = tmp_path / "fall.guard.json"
    original = path.read_bytes()
    record = json.loads(original)
    assert record["samples"][0]["window"] is None
    assert "values" not in record["samples"][0]
    assert not record["process_started"]
    with pytest.raises(RuntimeError, match="덮어쓰지"):
        runner.guarded_process(["mock-convert"], tmp_path / "fall.log", 120)
    assert path.read_bytes() == original
    spawn.assert_not_called()


def test_preflight_persists_failed_rate_window(monkeypatch, tmp_path):
    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")
    monkeypatch.setattr(
        runner,
        "metrics",
        Mock(side_effect=[operating_sample(), operating_sample(105, 1100, 550)]),
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="처리율"):
        runner.check(tmp_path, seconds=5)
    record = json.loads((tmp_path / "operating_check.json").read_text())
    assert not record["passed"]
    assert record["samples"][-1]["window"]["fps"] == 20
    assert record["samples"][-1]["window"]["primary_hz"] == 10


def test_case_initial_guard_failure_is_saved_without_collectors_or_ai(
    monkeypatch, tmp_path
):
    runner = importlib.import_module("scripts.ops.run_two_camera_cadence")
    (tmp_path / "coordination").mkdir()
    (tmp_path / "compose.json").write_text(
        json.dumps(
            {
                "services": {
                    "one": {
                        "container_name": "test-only",
                        "network_mode": "none",
                        "pull_policy": "never",
                        "volumes": [],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(runner, "snapshot", lambda: {"operating": {"id": "unchanged"}})
    sample = {**operating_sample(), "temperatures_c": [81]}
    monkeypatch.setattr(runner, "metrics", Mock(return_value=sample))
    spawn = Mock(side_effect=AssertionError("보호 실패 뒤 수집기·AI 실행 금지"))
    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    command = Mock(side_effect=AssertionError("보호 실패 뒤 컨테이너 생성 금지"))
    monkeypatch.setattr(runner, "command", command)
    cleanup = Mock(return_value=Mock(returncode=1))
    monkeypatch.setattr(runner.subprocess, "run", cleanup)
    with pytest.raises(RuntimeError, match="시험 중단"):
        runner.run_case({"directory": str(tmp_path), "hz": 12, "pair": 0}, tmp_path)
    record = json.loads((tmp_path / "execution.json").read_text())
    assert not record["valid_execution"] and record["operating_unchanged"]
    assert record["samples"][0]["max_temperature_c"] == 81
    assert "80도" in record["errors"][0]
    spawn.assert_not_called()
    command.assert_not_called()
    assert cleanup.call_args.args[0] == ["docker", "inspect", "test-only"]
