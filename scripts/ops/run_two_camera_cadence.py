"""장비 사용 조율 후 실행하는 2채널 주기 시험. 운영 서비스는 중지하지 않는다."""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import math
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

if __package__:
    from .two_camera_cadence import (
        DEFAULT_OUTPUT,
        FPS,
        digest,
        evaluate,
        read,
        video_command,
        write,
    )
else:
    from two_camera_cadence import (
        DEFAULT_OUTPUT,
        FPS,
        digest,
        evaluate,
        read,
        video_command,
        write,
    )

FRAME = "cctv_processor_frames_processed_total"
INFER = 'cctv_camera_primary_frames_inferred_total{camera_id="camera_1"}'
AGE = 'cctv_camera_primary_last_frame_age_seconds{camera_id="camera_1"}'
ERRORS = (
    "cctv_processor_source_error_identified_total",
    "cctv_processor_source_error_unidentified_total",
    "cctv_processor_pipeline_restart_executed_total",
    "cctv_processor_frames_dropped_total",
)


WORK_DEADLINE = None
CLEANUP_RESERVE_SECONDS = 300


def select_pilot(plan):
    # 첫 배정은 낙상/정상 한 쌍의 세 주기로 제한하며 전체 계획으로 확대하지 않는다.
    cases = [case for case in plan["cases"] if case["pair"] == 0]
    if len(cases) != 3 or {case["hz"] for case in cases} != {12, 8, 6}:
        raise ValueError("첫 쌍의 12/8/6Hz 세 조건이 필요함")
    ids = cases[0]["scenes"]
    if (
        len(ids) != 2
        or len(set(ids)) != 2
        or any(case["scenes"] != ids for case in cases)
    ):
        raise ValueError("세 조건의 동일한 영상 한 쌍을 확인할 수 없음")
    scenes = [scene for scene in plan["scenes"] if scene["scene_id"] in ids]
    if len(scenes) != 2 or sum(bool(scene["is_fall"]) for scene in scenes) != 1:
        raise ValueError("낙상 1개·정상 1개만 배정 가능")
    return {**plan, "cases": cases, "scenes": scenes}


def configure_slot(start_at, end_at):
    global WORK_DEADLINE
    start, end = datetime.fromisoformat(start_at), datetime.fromisoformat(end_at)
    if start.utcoffset() is None or end.utcoffset() is None:
        raise ValueError("배정 시각에 명시적인 시간대가 필요함")
    now = datetime.now(start.tzinfo)
    if not 0 < (end - start).total_seconds() <= 900 or not start <= now < end:
        raise ValueError("현재 시각을 포함하는 최대 15분 명시 배정이 필요함")
    remaining = (end - now).total_seconds()
    if remaining <= CLEANUP_RESERVE_SECONDS:
        raise ValueError("자체 정리 여유 시간이 부족해 시험을 시작하지 않음")
    # Docker 조회·로그·삭제가 지연돼도 종료 구간을 남기고 신규 작업부터 중단한다.
    WORK_DEADLINE = time.monotonic() + remaining - CLEANUP_RESERVE_SECONDS
    return (end - timedelta(seconds=CLEANUP_RESERVE_SECONDS)).isoformat()


def ensure_time():
    if WORK_DEADLINE is not None and time.monotonic() >= WORK_DEADLINE:
        raise RuntimeError("배정 시간의 자체 정리 구간 도달: 추가 작업 중단")


def command(*args):
    return subprocess.check_output(args, text=True, timeout=25).strip()


def metrics():
    ensure_time()
    raw = (
        urllib.request.urlopen("http://127.0.0.1:8770/metrics", timeout=5)
        .read()
        .decode()
    )
    values = dict(
        line.rsplit(" ", 1)
        for line in raw.splitlines()
        if line and not line.startswith("#")
    )
    memory = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    temperatures = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temperatures.append(float(path.read_text()) / 1000)
        except OSError as exc:
            # Jetson의 비활성 열 센서는 ENODATA를 반환하므로 활성 센서만 확인한다.
            if exc.errno != errno.ENODATA:
                raise
    # 보호 판정보다 수집을 먼저 끝내야 실패한 구간의 자원 상태도 보존할 수 있다.
    return {
        "at": time.monotonic(),
        "values": {k: float(values[k]) for k in (FRAME, INFER, AGE, *ERRORS)},
        "memory_available_kib": int(memory["MemAvailable"].split()[0]),
        "temperatures_c": temperatures,
    }


def guard(before, after):
    # 판정에 사용한 차분을 같은 표본에 남겨 예외가 발생해도 원시값과 대조한다.
    after["window"] = None
    b = after["values"]
    if not math.isfinite(after["at"]) or not all(math.isfinite(v) for v in b.values()):
        raise RuntimeError("운영 계측에 유효하지 않은 수치")
    reasons = []
    if before is not None:
        a = before["values"]
        duration = after["at"] - before["at"]
        if not math.isfinite(duration) or not all(math.isfinite(v) for v in a.values()):
            raise RuntimeError("운영 계측에 유효하지 않은 수치")
        if duration <= 0 or any(b[k] < a[k] for k in (FRAME, INFER, *ERRORS)):
            raise RuntimeError("운영 계측 시계·카운터 초기화")
        fps = (b[FRAME] - a[FRAME]) / duration
        hz = (b[INFER] - a[INFER]) / duration
        after["window"] = {
            "duration_seconds": duration,
            "frames_delta": b[FRAME] - a[FRAME],
            "inferred_delta": b[INFER] - a[INFER],
            "fps": fps,
            "primary_hz": hz,
            "error_deltas": {k: b[k] - a[k] for k in ERRORS},
        }
        if fps < 27 or hz < 13.5:
            reasons.append(
                f"운영 처리율 미달: {fps:.6f} FPS / {hz:.6f} Hz (최소 27 / 13.5)"
            )
        if any(b[k] > a[k] for k in ERRORS):
            reasons.append("운영 오류·누락·재시작 증가")
    if b[AGE] > 1:
        reasons.append(f"운영 프레임 정체: {b[AGE]:.6f}초")
    memory = after["memory_available_kib"]
    temperatures = after["temperatures_c"]
    after["max_temperature_c"] = max(temperatures) if temperatures else None
    if not math.isfinite(memory) or not all(math.isfinite(t) for t in temperatures):
        reasons.append("자원 계측에 유효하지 않은 수치")
    if not temperatures:
        reasons.append("활성 온도 센서를 확인할 수 없음")
    if memory < 8 * 1024 * 1024:
        reasons.append(f"가용 메모리 8GiB 미만: {memory / 1024**2:.6f} GiB")
    if temperatures and max(temperatures) >= 80:
        reasons.append(f"온도 80도 이상: {max(temperatures):.3f}도")
    if reasons:
        raise RuntimeError("; ".join(reasons))


def observe(record, path):
    # 사전점검·변환·추론 모두 예외를 전달하기 전에 동일한 형식으로 표본을 저장한다.
    sample = {
        "observed_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "guard_passed": False,
        "window": None,
    }
    try:
        sample.update(metrics())
        guard(record["samples"][-1] if record["samples"] else None, sample)
        sample["guard_passed"] = True
    except BaseException as exc:
        sample["reason"] = str(exc) or type(exc).__name__
        raise
    finally:
        record["samples"].append(sample)
        write(path, record)
    return sample


def snapshot():
    ids = command("docker", "ps", "-aq").split()
    if not ids:
        raise RuntimeError("운영 컨테이너를 확인할 수 없음")
    data = json.loads(command("docker", "inspect", *ids))
    return {
        c["Name"]: {
            "id": c["Id"],
            "image": c["Image"],
            "started": c["State"]["StartedAt"],
            "status": c["State"]["Status"],
            "restarts": c["RestartCount"],
        }
        for c in data
    }


def check(output, seconds=30):
    path = output / "operating_check.json"
    record = {"passed": False, "samples": []}
    try:
        observe(record, path)
        for _ in range(seconds // 5):
            time.sleep(5)
            observe(record, path)
        record["passed"] = True
    except BaseException as exc:
        record["reason"] = str(exc) or type(exc).__name__
        raise
    finally:
        write(path, record)
    return record["samples"][-1]


def guarded_process(args, log_path, timeout):
    path = log_path.with_suffix(".guard.json")
    if path.exists():
        raise RuntimeError(f"이전 변환 계측은 덮어쓰지 않음: {path}")
    record = {"passed": False, "process_started": False, "samples": []}
    proc = None
    try:
        observe(record, path)
        with log_path.open("x") as log:
            proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            record["process_started"] = True
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                time.sleep(5)
                observe(record, path)
                if time.monotonic() > deadline:
                    raise RuntimeError("개인 변환 작업 시간 초과")
            if proc.returncode:
                raise RuntimeError(f"작업 실패: {log_path}")
        record["passed"] = True
    except BaseException as exc:
        record["reason"] = str(exc) or type(exc).__name__
        raise
    finally:
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        finally:
            record["returncode"] = proc.returncode if proc is not None else None
            write(path, record)


def convert(plan, output):
    record_path = output / "converted.json"
    converted = read(record_path) if record_path.exists() else {}
    for scene in plan["scenes"]:
        ensure_time()
        scene_id = scene["scene_id"]
        destination = Path(scene["derived_video"])
        if (
            digest(scene["source_video"]) != scene["source_sha256"]
            or digest(scene["label_path"]) != scene["label_sha256"]
        ):
            raise RuntimeError("입력 영상 또는 정답 파일이 고정 기준에서 변경됨")
        if destination.exists():
            if (
                scene_id not in converted
                or digest(destination) != converted[scene_id]["sha256"]
            ):
                raise RuntimeError("기존 변환 출력의 해시를 검증할 수 없음")
            continue
        temporary = destination.with_suffix(".partial.mp4")
        if temporary.exists():
            raise RuntimeError(f"이전 실패 출력 확인 필요: {temporary}")
        guarded_process(
            video_command(scene["source_video"], temporary),
            output / "videos" / f"{scene_id}.log",
            120,
        )
        probe = json.loads(
            command(
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(temporary),
            )
        )["streams"][0]
        if (probe["width"], probe["height"], probe["avg_frame_rate"]) != (
            1920,
            1080,
            "24/1",
        ):
            raise RuntimeError("24FPS·1080p 변환 규격 불일치")
        if (
            int(probe["nb_frames"]) != scene["expected_frames"]
            or abs(float(probe["duration"]) - scene["duration_seconds"]) > 1 / FPS
        ):
            raise RuntimeError("변환이 영상 길이·정답 시간축을 바꿈")
        temporary.rename(destination)
        converted[scene_id] = {"sha256": digest(destination), "probe": probe}
        write(record_path, converted)
        print("converted", scene_id, flush=True)


def run_case(case, output):
    ensure_time()
    folder = Path(case["directory"])
    compose = folder / "compose.json"
    spec = read(compose)["services"]
    project = f"cctv-cadence24-{case['hz']}-{case['pair']:02d}"
    before = snapshot()
    names = [v["container_name"] for v in spec.values()]
    if any("/" + name in before for name in names):
        raise RuntimeError("이전 시험 컨테이너가 남아 있음")
    if any((folder / f"replay_{i}/output/capture.json").exists() for i in (1, 2)):
        raise RuntimeError("기존 시험 결과는 덮어쓰지 않음")
    if any((folder / "coordination").iterdir()):
        raise RuntimeError("이전 동시 재생 표식이 남아 있음")
    for service in spec.values():
        assert service["network_mode"] == "none" and service["pull_policy"] == "never"
        assert (
            not service.get("ports")
            and not service.get("devices")
            and not service.get("privileged")
        )
        assert all(
            v["read_only"] or Path(v["source"]).is_relative_to(folder)
            for v in service["volumes"]
        )
    result = {
        "case": case,
        "valid_execution": False,
        "samples": [],
        "errors": [],
        "cleanup_errors": [],
        "before": before,
    }
    starters = []
    ids = []
    stats_log = (folder / "tegrastats.log").open("w")
    collector = None
    try:
        observe(result, folder / "execution.json")
        collector = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=stats_log,
            stderr=subprocess.STDOUT,
        )
        command(
            "docker",
            "compose",
            "-f",
            str(compose),
            "-p",
            project,
            "create",
            "--no-build",
        )
        for name in names:
            item = json.loads(command("docker", "inspect", name))[0]
            assert item["Config"]["Labels"]["com.docker.compose.project"] == project
            ids.append(item["Id"])
        for container_id in ids:
            ensure_time()
            starters.append(
                subprocess.Popen(
                    ["docker", "start", container_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            time.sleep(5)
            observe(result, folder / "execution.json")
            if any(p.poll() is None for p in starters):
                continue
            if any(p.returncode for p in starters):
                raise RuntimeError("시험 컨테이너 시작 실패")
            states = [
                c["State"] for c in json.loads(command("docker", "inspect", *ids))
            ]
            if any(
                not c["Running"] and (c["ExitCode"] or c["OOMKilled"]) for c in states
            ):
                raise RuntimeError("시험 컨테이너 오류 또는 OOM")
            if all(not c["Running"] for c in states):
                result["valid_execution"] = True
                break
        else:
            raise RuntimeError("시험 시간 초과")
    except Exception as exc:
        result["errors"].append(str(exc))
    finally:
        if collector is not None:
            collector.terminate()
            try:
                collector.wait(timeout=5)
            except subprocess.TimeoutExpired:
                collector.kill()
                collector.wait()
        stats_log.close()
        # 생성 도중 실패해도 이름·프로젝트가 일치하는 이번 시험 자원만 정리한다.
        for name in names:
            try:
                inspect = subprocess.run(
                    ["docker", "inspect", name],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if inspect.returncode:
                    continue
                item = json.loads(inspect.stdout)[0]
                if (
                    item["Config"].get("Labels", {}).get("com.docker.compose.project")
                    != project
                ):
                    raise RuntimeError("컨테이너 소유권 불일치")
                try:
                    logs = subprocess.run(
                        ["docker", "logs", item["Id"]],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    (folder / f"{name}.log").write_text(logs.stdout + logs.stderr)
                finally:
                    subprocess.run(
                        ["docker", "rm", "-f", item["Id"]],
                        check=True,
                        capture_output=True,
                        timeout=30,
                    )
            except Exception as exc:
                result["cleanup_errors"].append(str(exc))
        for process in starters:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        try:
            result["after"] = snapshot()
            result["operating_unchanged"] = result["after"] == before
        except Exception as exc:
            result["operating_unchanged"] = False
            result["cleanup_errors"].append(str(exc))
        result["valid_execution"] = (
            result["valid_execution"]
            and not result["errors"]
            and not result["cleanup_errors"]
            and result["operating_unchanged"]
        )
        write(folder / "execution.json", result)
    if (
        result["errors"]
        or result["cleanup_errors"]
        or not result["operating_unchanged"]
    ):
        raise RuntimeError(f"시험 중단·정리 결과 확인: {folder / 'execution.json'}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "pilot"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--start-at", help="통합 담당이 확정한 시작 ISO 시각(+09:00 포함)"
    )
    parser.add_argument(
        "--end-at", help="통합 담당이 확정한 종료 ISO 시각(+09:00 포함)"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "pilot":
        if not args.start_at or not args.end_at:
            parser.error(
                "pilot에는 통합 담당이 확정한 --start-at 및 --end-at이 필요합니다"
            )
        cutoff = configure_slot(args.start_at, args.end_at)
    plan = read(output / "plan.json")
    for relative, expected in plan["fixed_sha256"].items():
        if digest(output / relative) != expected:
            raise RuntimeError(f"고정 시험 파일 변경: {relative}")
    if digest(plan["engine_path"]) != plan["model_sha256"]:
        raise RuntimeError("엔진 파일 변경")
    if args.action == "check":
        check(output)
        print("operating precheck passed; resource coordination still required")
        return
    pilot = select_pilot(plan)
    record_path = (
        output / f"pilot_{datetime.now(ZoneInfo('Asia/Seoul')).strftime('%H%M%S')}.json"
    )
    record = {
        "start_at": args.start_at,
        "end_at": args.end_at,
        "work_cutoff": cutoff,
        "cases": pilot["cases"],
        "scene_ids": [s["scene_id"] for s in pilot["scenes"]],
        "status": "STARTED",
        "completed_conditions": [],
    }
    write(record_path, record)
    try:
        check(output)
        # 보조 잠금은 통합 담당의 명시 배정을 대신하지 않는다.
        with Path("/tmp/cctv-jetson-a-gpu-benchmark.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            convert(pilot, output)
            for case in pilot["cases"]:
                check(output, 10)
                run_case(case, output)
                evaluate(output)
                rows = [
                    row
                    for row in read(output / "comparison.json")["rows"]
                    if row["hz"] == case["hz"] and row["pair"] == case["pair"]
                ]
                if any(row["verdict"] in ("INVALID", "NOT_RUN") for row in rows):
                    raise RuntimeError(
                        "이번 2채널 재생·계측이 유효하지 않음: comparison.json 확인"
                    )
                record["completed_conditions"].append(case["hz"])
                write(record_path, record)
                print("completed", case["hz"], case["pair"], flush=True)
        record["status"] = "FINISHED"
    except BaseException as exc:
        record.update(status="STOPPED", reason=str(exc) or type(exc).__name__)
        raise
    finally:
        record["finished_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        write(record_path, record)
        evaluate(output)


if __name__ == "__main__":
    main()
