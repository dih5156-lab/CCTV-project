"""제어기·감독자·Agent·영상 접수 모형을 별도 프로세스로 연결한 격리 시험."""

import hashlib
import json
import os
import select
import shutil
import signal
import socket
import sqlite3
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.services.edge_handoff import EdgeHandoff
from src.services.edge_node_agent import RegistryClient
from src.services.edge_result_dds import node_component
from tests.test_edge_orchestration_remote import certificates as certificates
from tests.test_edge_result_mqtt_integration import isolated_mqtt as isolated_mqtt
from tests.test_edge_service import private_json
from tests.test_mqtt_durable_integration import wait_until


@pytest.fixture
def result_broker(request):
    if os.environ.get("CCTV_EDGE_DDS_FULL_PIPELINE") == "1":
        return None
    return request.getfixturevalue("isolated_mqtt")


@pytest.fixture
def tmp_path(tmp_path_factory):
    # 장시간 시험은 --basetemp의 격리 디스크를 따르고 인증 파일은 종료 시 제거한다.
    root = tmp_path_factory.mktemp("edge-full-pipeline")
    root.chmod(0o700)
    try:
        yield root
    finally:
        shutil.rmtree(root)


@pytest.mark.parametrize(
    "handoff_mode", ["fencing", "manual-handoff", "automatic-handoff"]
)
def test_two_process_nodes_actual_intake_mqtt_ack_and_final_fence(
    tmp_path, tmp_path_factory, certificates, result_broker, caplog, handoff_mode
):
    manual_handoff = handoff_mode != "fencing"
    automatic_handoff = handoff_mode == "automatic-handoff"
    isolated_mqtt = result_broker
    use_dds = isolated_mqtt is None
    dds_keys = Path(os.environ["EDGE_DDS_TEST_KEYSTORES"]) if use_dds else None
    domain_id = int(os.environ.get("EDGE_DDS_TEST_DOMAIN", "73"))
    repository = Path(__file__).resolve().parents[1]
    cert, key = certificates[0]
    processes = []
    dds_containers = []
    agents, agent_arguments, assignments = {}, {}, {}
    relays, relay_arguments = {}, {}
    # 명시한 장시간 시험에서만 같은 프로세스를 유지하며 장애와 복구를 관측한다.
    soak_seconds = int(os.environ.get("CCTV_EDGE_SOAK_SECONDS", "0"))
    assert soak_seconds == 0 or 120 <= soak_seconds <= 7200
    evidence_path = tmp_path_factory.getbasetemp() / "edge-soak.json"
    if soak_seconds:
        evidence_path.touch(mode=0o600)
    soak = {
        "requested_seconds": soak_seconds,
        "transport": "dds" if use_dds else "mqtt",
        "manual_handoff": manual_handoff,
        "automatic_handoff": automatic_handoff,
        "samples": [],
        "faults": [],
        "request_max_seconds": 0,
    }
    cli = [sys.executable, "-m", "runners.run_edge_orchestration"]
    node_tokens = {
        node: uuid.uuid4().hex + uuid.uuid4().hex for node in ("jetson-a", "jetson-b")
    }
    roles = {
        name: uuid.uuid4().hex + uuid.uuid4().hex
        for name in ("reader_token", "control_token", "consumer_token")
    }
    credentials = private_json(
        tmp_path / "controller-secrets.json", {"node_tokens": node_tokens, **roles}
    )
    collector_config = private_json(
        tmp_path / "collector.json",
        {"domain_id": domain_id}
        if use_dds
        else {
            "mqtt": isolated_mqtt.settings("collector"),
            "client_id": "full-pipeline-" + uuid.uuid4().hex,
        },
    )
    dds_image = os.environ.get("CCTV_EDGE_DDS_CONTAINER_IMAGE") if use_dds else None
    udp_profile = tmp_path / "dds-loopback.xml"
    if dds_image:
        # 별도 IPC 환경의 송수신은 공유 메모리 대신 loopback UDP만 사용한다.
        udp_profile.write_text("""<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
<transport_descriptors><transport_descriptor><transport_id>isolated_udp</transport_id><type>UDPv4</type>
<interfaceWhiteList><address>127.0.0.1</address></interfaceWhiteList></transport_descriptor></transport_descriptors>
<participant profile_name="isolated_edge" is_default_profile="true"><rtps><userTransports>
<transport_id>isolated_udp</transport_id></userTransports><useBuiltinTransports>false</useBuiltinTransports>
</rtps></participant></profiles>""")

    def launch(name, arguments, endpoint=False, *, dds_role=None):
        log = (tmp_path / (name + ".stderr")).open("w")
        command = cli + arguments
        if dds_role and dds_image:
            container_name = "cctv-edge-dds-full-" + uuid.uuid4().hex[:12]
            dds_containers.append(container_name)
            mounts = [
                (repository / "src", "ro"),
                (repository / "runners", "ro"),
                (dds_keys / dds_role, "ro"),
                (cert, "ro"),
                (udp_profile, "ro"),
            ]
            if arguments[0] == "controller":
                state = Path(arguments[arguments.index("--state-dir") + 1])
                state.mkdir(mode=0o700)
                mounts += [
                    (state, "rw"),
                    (credentials, "ro"),
                    (collector_config, "ro"),
                    (key, "ro"),
                    (repository / "web/edge-viewer.html", "ro"),
                    (repository / "web/edge-viewer.js", "ro"),
                ]
            else:
                config = Path(arguments[arguments.index("--config") + 1])
                context = Path(arguments[arguments.index("--context-db") + 1])
                mounts += [(config, "ro"), (context.parent, "rw")]
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "--label",
                "cctv.purpose=edge-dds-full",
                "--network",
                "host",
                "--ipc",
                "private",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--cpus",
                "0.5",
                "--memory",
                "512m",
                "--pids-limit",
                "128",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=32m",
                "-e",
                "HOME=/tmp",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                "-e",
                "PYTHONPATH=" + str(repository),
                "-e",
                "ROS_SECURITY_ENABLE=true",
                "-e",
                "ROS_SECURITY_STRATEGY=Enforce",
                "-e",
                "RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
                "-e",
                # Humble의 localhost 옵션은 SHM도 추가하므로 XML의 UDP 제한만 사용한다.
                "ROS_LOCALHOST_ONLY=0",
                "-e",
                "ROS_SECURITY_KEYSTORE=" + str(dds_keys / dds_role),
                "-e",
                "FASTRTPS_DEFAULT_PROFILES_FILE=" + str(udp_profile),
                "-w",
                str(repository),
            ]
            for path, mode in mounts:
                command += ["-v", f"{path}:{path}:{mode}"]
            command += [
                dds_image,
                "python3",
                "-m",
                "runners.run_edge_orchestration",
            ] + arguments
        try:
            process = subprocess.Popen(
                command,
                cwd=repository,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                env=dict(os.environ, ROS_SECURITY_KEYSTORE=str(dds_keys / dds_role))
                if dds_role
                else None,
            )
        finally:
            log.close()
        processes.append(process)
        if not endpoint:
            return process
        readable, _, _ = select.select([process.stdout], [], [], 20)
        assert readable, name + " endpoint timeout"
        line = process.stdout.readline()
        assert line, name + " exited before endpoint"
        return json.loads(line)

    def request(client, method, path, body=None):
        started = time.monotonic()
        code, data = client.request(method, path, body)
        soak["request_max_seconds"] = max(
            soak["request_max_seconds"], time.monotonic() - started
        )
        assert code == 200, (code, data)
        return data

    def ready_nodes():
        return {
            item["node_id"]: item
            for item in request(reader, "GET", "/v1/nodes")["nodes"]
        }

    try:
        endpoint = launch(
            "controller",
            [
                "controller",
                "--bind",
                "127.0.0.1",
                "--state-dir",
                str(tmp_path / "controller"),
                "--credentials",
                str(credentials),
                "--cert",
                str(cert),
                "--key",
                str(key),
                "--enable-ownership",
                "--result-dds" if use_dds else "--result-mqtt",
                str(collector_config),
            ],
            True,
            dds_role="controller" if use_dds else None,
        )
        reader = RegistryClient(
            endpoint["controller"], roles["reader_token"], ca_file=cert
        )
        control = RegistryClient(
            endpoint["controller"], roles["control_token"], ca_file=cert
        )
        consumer = RegistryClient(
            endpoint["controller"], roles["consumer_token"], ca_file=cert
        )
        for asset in ("/viewer", "/viewer.js"):
            with urllib.request.urlopen(
                endpoint["controller"] + asset,
                context=ssl.create_default_context(cafile=str(cert)),
                timeout=5,
            ) as response:
                assert response.status == 200 and response.read()
                assert response.headers["Cache-Control"] == "no-store"
        nodes = {}
        for node, token in node_tokens.items():
            root = tmp_path / node
            root.mkdir(mode=0o700)
            model = root / "test.engine"
            model.write_bytes(b"fixture model; no GPU inference")
            model.chmod(0o600)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            environment = private_json(
                root / "env.json",
                {
                    "FIXTURE_MODE_PATH": str(root / "mode"),
                    "FIXTURE_EMIT_EVENTS": "1",
                    # 호스트 AI 모형은 ROS 설치 여부와 무관하게 저장소를 가져와야 한다.
                    "PYTHONPATH": str(repository)
                    + (
                        os.pathsep + os.environ["PYTHONPATH"]
                        if os.environ.get("PYTHONPATH")
                        else ""
                    ),
                },
            )
            manifest = private_json(
                root / "manifest.json",
                {
                    "manifest_id": "lab-v1",
                    "service_id": "cctv-ai",
                    "camera_id": "camera_1",
                    "argv": [
                        sys.executable,
                        str(repository / "tests/fixtures/edge_ai_process.py"),
                    ],
                    "cwd": str(root),
                    "environment_file": str(environment),
                    "metrics_url": f"http://127.0.0.1:{port}/metrics",
                    "ready_timeout_seconds": 20,
                    "stop_timeout_seconds": 1,
                    "stale_after_seconds": 3,
                    "models": [
                        {
                            "model_id": name,
                            "metric_name": name,
                            "path": str(model),
                            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                            "allow_empty": name == "helmet",
                        }
                        for name in ("primary", "helmet")
                    ],
                },
            )
            node_roles = {
                name: uuid.uuid4().hex + uuid.uuid4().hex
                for name in ("reader_token", "control_token")
            }
            node_credentials = private_json(root / "credentials.json", node_roles)
            transport = private_json(
                root / "transport.json",
                {
                    "controller_url": endpoint["controller"],
                    "controller_ca_file": str(cert),
                    "node_token": token,
                    **(
                        {"domain_id": domain_id}
                        if use_dds
                        else {"mqtt": isolated_mqtt.settings(node)}
                    ),
                },
            )
            service = launch(
                node + "-supervisor",
                [
                    "service",
                    "--node-id",
                    node,
                    "--manifest",
                    str(manifest),
                    "--state-dir",
                    str(root / "state"),
                    "--credentials",
                    str(node_credentials),
                    "--enable-results",
                    *(
                        ["--result-sidecar"]
                        if use_dds
                        else ["--result-transport", str(transport)]
                    ),
                ],
                True,
            )
            if use_dds:
                relay_arguments[node] = [
                    "dds-relay",
                    "--node-id",
                    node,
                    "--config",
                    str(transport),
                    "--context-db",
                    str(root / "state/results-context.sqlite3"),
                ]
                relays[node] = launch(
                    node + "-dds", relay_arguments[node], dds_role=node_component(node)
                )
            read_token = root / "read.token"
            read_token.write_text(node_roles["reader_token"])
            read_token.chmod(0o600)
            token_file = root / "node.token"
            token_file.write_text(token)
            token_file.chmod(0o600)
            agent_arguments[node] = [
                "agent",
                "--node-id",
                node,
                "--node-ip",
                "127.0.0.1",
                "--interval",
                "0.2",
                "--controller",
                endpoint["controller"],
                "--ca-file",
                str(cert),
                "--token-file",
                str(token_file),
                "--service-url",
                service["service"],
                "--service-token-file",
                str(read_token),
            ]
            agents[node] = launch(node + "-agent", agent_arguments[node])
            nodes[node] = (
                service,
                RegistryClient(service["service"], node_roles["control_token"]),
            )
        wait_until(
            lambda: (
                len(ready_nodes()) == 2
                and all(
                    node["report"] and node["report"].get("service")
                    for node in ready_nodes().values()
                )
            ),
            timeout=20,
        )

        def switch(node):
            previous = request(reader, "GET", "/v1/cameras/camera_1")["active"]
            service, client = nodes[node]
            assignment = request(
                control,
                "POST",
                "/v1/cameras/camera_1/prepare",
                {
                    "request_id": str(uuid.uuid4()),
                    "controller_boot_id": endpoint["controller_boot_id"],
                    "node_id": node,
                    "supervisor_boot_id": service["boot_id"],
                    "manifest_id": "lab-v1",
                    "expected_assignment_id": previous["assignment_id"]
                    if previous
                    else None,
                    "expected_epoch": previous["epoch"] if previous else 0,
                },
            )
            request(
                client,
                "PUT",
                "/v1/assignments/camera_1",
                {
                    "command_id": str(uuid.uuid4()),
                    "assignment_id": assignment["assignment_id"],
                    "epoch": assignment["epoch"],
                    "desired_state": "running",
                    "manifest_id": "lab-v1",
                    "expected_boot_id": service["boot_id"],
                },
            )
            wait_until(
                lambda: ready_nodes()[node]["report"]["service"]["ready"], timeout=20
            )
            wait_until(
                lambda: request(reader, "GET", "/v1/cameras/camera_1")["pending"][
                    "sample_id"
                ],
                timeout=20,
            )
            current = ready_nodes()[node]["report"]["service"]
            barrier = request(
                client,
                "PUT",
                "/v1/assignments/camera_1/activate",
                {
                    "command_id": str(uuid.uuid4()),
                    "assignment_id": assignment["assignment_id"],
                    "epoch": assignment["epoch"],
                    "expected_boot_id": service["boot_id"],
                    "session_id": current["session_id"],
                    "expected_activation_id": None,
                    "activation_id": assignment["activation_id"],
                },
            )
            wait_until(
                lambda: (
                    ready_nodes()[node]["report"]["service"]["activation"] == barrier
                )
            )
            assignments[node] = request(
                control,
                "POST",
                f"/v1/assignments/{assignment['assignment_id']}/activate",
                {
                    "request_id": str(uuid.uuid4()),
                    "controller_boot_id": endpoint["controller_boot_id"],
                    "session_id": current["session_id"],
                    **barrier,
                },
            )
            return assignments[node]

        def pending(node):
            return [
                item
                for item in request(consumer, "GET", "/v1/results/pending")["results"]
                if item["result"]["edge"]["node_id"] == node
            ]

        def soak_phase(node, duration, *, broker_failure=False):
            if not duration:
                return
            started = time.monotonic()
            agent_stopped = broker_stopped = None
            agent_restarted = broker_restarted = False
            saw_stale = False
            consumed = set()
            last_delivery = started
            while time.monotonic() - started < duration:
                elapsed = time.monotonic() - started
                if agent_stopped is None and elapsed >= duration / 4:
                    process = agents[node]
                    process.terminate()
                    process.wait(timeout=8)
                    process.stdout.close()
                    processes.remove(process)
                    agent_stopped = time.monotonic()
                    soak["faults"].append(
                        {"node": node, "kind": "agent_stop", "at": time.time()}
                    )
                if (
                    agent_stopped is not None
                    and not agent_restarted
                    and time.monotonic() - agent_stopped >= 21
                ):
                    agents[node] = launch(
                        node + "-agent-restarted", agent_arguments[node]
                    )
                    agent_restarted = True
                    soak["faults"].append(
                        {"node": node, "kind": "agent_restart", "at": time.time()}
                    )
                if (
                    broker_failure
                    and broker_stopped is None
                    and elapsed >= duration * 0.65
                ):
                    if use_dds:
                        process = relays[node]
                        process.terminate()
                        process.wait(timeout=10)
                        process.stdout.close()
                        processes.remove(process)
                    else:
                        subprocess.run(
                            ["docker", "stop", "-t", "3", isolated_mqtt.name],
                            check=True,
                            capture_output=True,
                            timeout=20,
                        )
                    broker_stopped = time.monotonic()
                    soak["faults"].append(
                        {
                            "kind": "dds_relay_stop" if use_dds else "broker_stop",
                            "at": time.time(),
                        }
                    )
                if (
                    broker_stopped is not None
                    and not broker_restarted
                    and time.monotonic() - broker_stopped >= 12
                ):
                    if use_dds:
                        relays[node] = launch(
                            node + "-dds-restarted",
                            relay_arguments[node],
                            dds_role=node_component(node),
                        )
                    else:
                        subprocess.run(
                            ["docker", "start", isolated_mqtt.name],
                            check=True,
                            capture_output=True,
                            timeout=20,
                        )
                    broker_restarted = True
                    soak["faults"].append(
                        {
                            "kind": "dds_relay_restart"
                            if use_dds
                            else "broker_restart",
                            "at": time.time(),
                        }
                    )
                observed = ready_nodes()
                service_status = (observed[node]["report"] or {}).get("service") or {}
                assert service_status.get("status") not in ("error", "stopped"), (
                    service_status
                )
                if agent_stopped is not None and not agent_restarted:
                    saw_stale |= observed[node]["freshness"] == "stale"
                other = "jetson-b" if node == "jetson-a" else "jetson-a"
                assert observed[other]["freshness"] == "online"
                if manual_handoff:
                    assert observed[other]["report"]["service"]["status"] == "stopped"
                active = request(reader, "GET", "/v1/cameras/camera_1")["active"]
                assert active["node_id"] == node
                for item in pending(node):
                    result = request(
                        consumer,
                        "POST",
                        f"/v1/results/{item['event_id']}/consume",
                        {"sha256": item["sha256"]},
                    )
                    if result["consumed"]:
                        assert item["event_id"] not in consumed
                        consumed.add(item["event_id"])
                        last_delivery = time.monotonic()
                # 보고만 살아 있고 AI/전달이 멎은 경우를 장시간 합격으로 잘못 판정하지 않는다.
                assert time.monotonic() - last_delivery < 45, (
                    "active result delivery stalled"
                )
                node_samples = {}
                for key, value in observed.items():
                    report = value["report"] or {}
                    service = report.get("service") or {}
                    node_samples[key] = {
                        "freshness": value["freshness"],
                        "sequence": report.get("sequence"),
                        "boot_id": report.get("boot_id"),
                        "service_status": service.get("status"),
                        "service_ready": service.get("ready"),
                        "service_error": service.get("last_error"),
                    }
                    # 관측 실패를 대기0건으로 바꾸지 않으며 읽기 시간이 시험 부하에 미친 영향도 남긴다.
                    sampled = time.monotonic()
                    try:
                        path = tmp_path / key / "state/results-context.sqlite3"
                        with sqlite3.connect(
                            path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1
                        ) as database:
                            count, oldest = database.execute(
                                "SELECT COUNT(*),MIN(json_extract(body,'$.edge.captured_at')) FROM node_outbox WHERE state='pending'"
                            ).fetchone()
                        node_samples[key]["outbox_pending"] = count
                        node_samples[key]["oldest_pending_captured_at"] = oldest
                        path = tmp_path / key / "state/video-ingest.sqlite3"
                        if path.exists():
                            with sqlite3.connect(
                                path.resolve().as_uri() + "?mode=ro",
                                uri=True,
                                timeout=0.1,
                            ) as database:
                                node_samples[key]["intake_pending"] = database.execute(
                                    "SELECT COUNT(*) FROM event_ingest_state WHERE stage='pending'"
                                ).fetchone()[0]
                    except sqlite3.Error as error:
                        node_samples[key]["queue_observation_error"] = type(
                            error
                        ).__name__
                    node_samples[key]["queue_observation_seconds"] = (
                        time.monotonic() - sampled
                    )
                soak["samples"].append(
                    {
                        "at": time.time(),
                        "owner": node,
                        "nodes": node_samples,
                        "consumed": len(consumed),
                    }
                )
                evidence_path.write_text(json.dumps(soak))
                time.sleep(1)
            assert saw_stale and agent_restarted and consumed
            if broker_failure:
                assert broker_restarted
            wait_until(lambda: ready_nodes()[node]["freshness"] == "online", timeout=15)
            soak.setdefault("phases", []).append(
                {
                    "owner": node,
                    "seconds": time.monotonic() - started,
                    "consumed": len(consumed),
                    "stale_observed": saw_stale,
                }
            )

        def finish_soak():
            # 입력을 멈춘 뒤 전수 대조하여 '일부 ACK 성공'을 무손실 판정으로 확대하지 않는다.
            running = {
                node
                for node, entry in ready_nodes().items()
                if entry["report"]["service"]["status"] != "stopped"
            }
            for node in running:
                (tmp_path / node / "mode").write_text("drain")

            def intake_drained():
                for node in running:
                    if (tmp_path / node / "mode").read_text() != "drained":
                        return False
                    with sqlite3.connect(
                        tmp_path / node / "state/video-ingest.sqlite3"
                    ) as intake:
                        if intake.execute(
                            "SELECT COUNT(*) FROM event_ingest_state WHERE stage='pending'"
                        ).fetchone()[0]:
                            return False
                return True

            wait_until(intake_drained, timeout=45)
            for node, assignment in assignments.items():
                service, client = nodes[node]
                request(
                    client,
                    "PUT",
                    "/v1/assignments/camera_1",
                    {
                        "command_id": str(uuid.uuid4()),
                        "assignment_id": assignment["assignment_id"],
                        "epoch": assignment["epoch"],
                        "desired_state": "stopped",
                        "manifest_id": "lab-v1",
                        "expected_boot_id": service["boot_id"],
                    },
                )

            def drained():
                for node in nodes:
                    with sqlite3.connect(
                        tmp_path / node / "state/results-context.sqlite3"
                    ) as database:
                        if database.execute(
                            "SELECT COUNT(*) FROM node_outbox WHERE state!='committed'"
                        ).fetchone()[0]:
                            return False
                return True

            wait_until(
                lambda: all(
                    value["report"]["service"]["status"] == "stopped"
                    for value in ready_nodes().values()
                ),
                timeout=15,
            )
            wait_until(drained, timeout=45)
            with sqlite3.connect(
                tmp_path / "controller/ownership/ownership.sqlite3"
            ) as central:
                assert central.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                soak["reconciliation"] = {}
                for node in nodes:
                    with sqlite3.connect(
                        tmp_path / node / "state/results-context.sqlite3"
                    ) as database:
                        local = dict(
                            database.execute(
                                "SELECT event_id,sha256 FROM node_outbox"
                            ).fetchall()
                        )
                    remote = dict(
                        central.execute(
                            "SELECT event_id,sha256 FROM results WHERE node_id=?",
                            (node,),
                        ).fetchall()
                    )
                    assert local and local == remote
                    with sqlite3.connect(
                        tmp_path / node / "state/video-ingest.sqlite3"
                    ) as intake:
                        assert (
                            intake.execute("PRAGMA integrity_check").fetchone()[0]
                            == "ok"
                        )
                        accepted = {
                            row[0]
                            for row in intake.execute(
                                "SELECT source_record_id FROM canonical_records"
                            )
                        }
                        assert accepted <= local.keys()
                        raw_states = dict(
                            intake.execute(
                                "SELECT stage,COUNT(*) FROM event_ingest_state GROUP BY stage"
                            ).fetchall()
                        )
                        if node in running or not manual_handoff:
                            assert raw_states == {"accepted": len(accepted)}
                        else:
                            # 중지된 AI의 접수 복구는 재시작 때 이어지므로 저장 잔여를 숨기지 않는다.
                            assert set(raw_states) <= {"accepted", "pending"}
                            assert raw_states.get("accepted", 0) == len(accepted)
                    soak["reconciliation"][node] = {
                        "node_originals": len(local),
                        "central_originals": len(remote),
                        "missing_or_changed": 0,
                        "accepted_video_records": len(accepted),
                        "video_raw_states": raw_states,
                    }
                soak["central_decisions"] = central.execute(
                    "SELECT node_id,kind,decision,reason,COUNT(*) FROM results GROUP BY node_id,kind,decision,reason"
                ).fetchall()
            retained = sum(
                item["video_raw_states"].get("pending", 0)
                for item in soak["reconciliation"].values()
            )
            soak["retained_intake_pending"] = retained
            soak["status"] = "PASS_WITH_RETAINED_INTAKE" if retained else "PASS"

        if manual_handoff:
            # 같은 격리 장비 구성으로 제품 CLI가 A→B→A를 연결하는지 확인한다.
            handoff_config = private_json(
                tmp_path / "handoff.json",
                {
                    "controller": {
                        "url": endpoint["controller"],
                        "ca_file": str(cert),
                        "credentials_file": str(credentials),
                    },
                    "nodes": {
                        node: {
                            "url": value[0]["service"],
                            "ca_file": None,
                            "credentials_file": str(
                                tmp_path / node / "credentials.json"
                            ),
                        }
                        for node, value in nodes.items()
                    },
                },
            )

            def automatic_switch():
                # 운영 GPU 부하 대신 인증 보고만 통제하고 실제 감독자/접수 경로는 유지한다.
                for process in agents.values():
                    process.terminate()
                    process.wait(timeout=8)
                    process.stdout.close()
                    processes.remove(process)
                observed = ready_nodes()
                sequences = {
                    name: value["last_sequence"] for name, value in observed.items()
                }
                reporter_clients = {
                    name: RegistryClient(
                        endpoint["controller"], node_tokens[name], ca_file=str(cert)
                    )
                    for name in nodes
                }
                service_readers = {
                    name: RegistryClient(
                        value[0]["service"],
                        json.loads((tmp_path / name / "credentials.json").read_text())[
                            "reader_token"
                        ],
                    )
                    for name, value in nodes.items()
                }
                phase = {"cpu": 95}
                reporter_stop = threading.Event()
                report_errors = []

                def report_resources():
                    try:
                        while not reporter_stop.is_set():
                            for name in nodes:
                                service = request(
                                    service_readers[name], "GET", "/v1/node/status"
                                )["service"]
                                sequences[name] += 1
                                request(
                                    reporter_clients[name],
                                    "POST",
                                    f"/v1/nodes/{name}/reports",
                                    {
                                        "boot_id": observed[name]["boot_id"],
                                        "sequence": sequences[name],
                                        "sampled_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "simulation": False,
                                        "resources": {
                                            "cpu_percent": phase["cpu"]
                                            if name == "jetson-a"
                                            else 10,
                                            "gpu_percent": 10,
                                            "memory_percent": 20,
                                            "network_rx_bytes_per_second": 0,
                                            "network_tx_bytes_per_second": 0,
                                        },
                                        "service": service,
                                    },
                                )
                            reporter_stop.wait(0.25)
                    except Exception as error:
                        report_errors.append(f"{type(error).__name__}: {error}")

                policy = private_json(
                    tmp_path / "automatic-policy.json",
                    {
                        "camera_id": "camera_1",
                        "manifests": {name: "lab-v1" for name in nodes},
                        "overload_percent": {
                            "cpu_percent": 85,
                            "gpu_percent": 90,
                            "memory_percent": 85,
                        },
                        "available_percent": {
                            "cpu_percent": 60,
                            "gpu_percent": 60,
                            "memory_percent": 70,
                        },
                        "sustain_seconds": 15,
                        "min_samples": 3,
                        "max_report_age_seconds": 5,
                        "cooldown_seconds": 5,
                        "poll_seconds": 0.2,
                        "handoff_timeout_seconds": 45,
                    },
                )
                state = tmp_path / "automatic-state"
                arguments = [
                    "auto",
                    "--config",
                    str(handoff_config),
                    "--policy",
                    str(policy),
                    "--state-dir",
                    str(state),
                ]
                reporter = threading.Thread(target=report_resources, daemon=True)
                reporter.start()
                worker = None

                def decision():
                    assert not report_errors, report_errors
                    path = state / "decisions.sqlite3"
                    if not path.exists():
                        return None
                    with sqlite3.connect(
                        path.resolve().as_uri() + "?mode=ro", uri=True
                    ) as connection:
                        if not connection.execute(
                            "SELECT 1 FROM sqlite_master WHERE name='decisions'"
                        ).fetchone():
                            return None
                        row = connection.execute(
                            "SELECT body FROM decisions ORDER BY rowid DESC LIMIT 1"
                        ).fetchone()
                    return json.loads(row[0]) if row else None

                try:
                    worker = launch("automatic-initial", arguments)
                    time.sleep(3)
                    phase["cpu"] = 10
                    time.sleep(3)
                    assert decision() is None
                    assert (
                        request(reader, "GET", "/v1/cameras/camera_1")["active"][
                            "node_id"
                        ]
                        == "jetson-a"
                    )
                    (tmp_path / "jetson-b/mode").write_text("cpu")
                    phase["cpu"] = 95
                    high_started = time.monotonic()

                    def started_target():
                        item = decision()
                        if not item:
                            return False
                        service = request(
                            service_readers["jetson-b"], "GET", "/v1/node/status"
                        )["service"]
                        return (
                            service["session_id"]
                            # 미준비 상태를 의도했으므로 준비 완료 대신 생성된 실행만 기다린다.
                            if service["status"] in {"starting", "unhealthy", "running"}
                            else False
                        )

                    wait_until(started_target, timeout=35)
                    before = decision()
                    session = started_target()
                    worker.kill()
                    worker.communicate(timeout=10)
                    (tmp_path / "jetson-b/mode").unlink()
                    worker = launch("automatic-resumed", arguments)
                    wait_until(
                        lambda: decision().get("status") == "complete", timeout=60
                    )
                    completed = decision()
                    elapsed = time.monotonic() - high_started
                    assert completed["decision_id"] == before["decision_id"]
                    assert (
                        request(service_readers["jetson-b"], "GET", "/v1/node/status")[
                            "service"
                        ]["session_id"]
                        == session
                    )
                    phase["cpu"] = 10
                    time.sleep(7)
                    assert decision()["decision_id"] == completed["decision_id"]
                    assert (
                        request(reader, "GET", "/v1/cameras/camera_1")["active"][
                            "node_id"
                        ]
                        == "jetson-b"
                    )
                    with sqlite3.connect(state / "decisions.sqlite3") as connection:
                        assert (
                            connection.execute(
                                "SELECT COUNT(*) FROM decisions"
                            ).fetchone()[0]
                            == 1
                        )
                    report = {
                        "status": "PASS",
                        "resources": "controlled_authenticated_reports",
                        "ai": "fixture_no_gpu_inference",
                        "short_spike_did_not_move": True,
                        "resumed_same_decision_and_ai_session": True,
                        "healthy_failback_after_cooldown": False,
                        "high_report_to_completion_seconds": elapsed,
                        "decision": completed,
                        "policy": json.loads(policy.read_text()),
                    }
                    (tmp_path_factory.getbasetemp() / "edge-automatic.json").write_text(
                        json.dumps(report, indent=2)
                    )
                    return completed["outcome"]
                finally:
                    if worker is not None and worker.poll() is None:
                        worker.terminate()
                        try:
                            worker.communicate(timeout=15)
                        except subprocess.TimeoutExpired:
                            worker.kill()
                            worker.communicate(timeout=5)
                    reporter_stop.set()
                    reporter.join(timeout=15)
                    assert not reporter.is_alive()
                    for name in nodes:
                        agents[name] = launch(
                            name + "-agent-after-auto", agent_arguments[name]
                        )
                    wait_until(
                        lambda: all(
                            v["freshness"] == "online" for v in ready_nodes().values()
                        ),
                        timeout=15,
                    )
                    assert not report_errors, report_errors

            transitions = []
            view_checks = []
            previous_owner = None
            for number, target in enumerate(("jetson-a", "jetson-b", "jetson-a")):
                started = time.monotonic()
                command = [
                    "handoff",
                    "--config",
                    str(handoff_config),
                    "--state-dir",
                    str(tmp_path / f"handoff-{number}"),
                    "--camera-id",
                    "camera_1",
                    "--target-node",
                    target,
                    "--manifest-id",
                    "lab-v1",
                    "--timeout",
                    "45",
                ]
                if automatic_handoff and number == 1:
                    outcome = automatic_switch()
                else:
                    worker = launch(f"handoff-{number}", command)
                    output, _ = worker.communicate(timeout=65)
                    outcome = json.loads(output)
                    assert worker.returncode == 0, outcome
                assert outcome["status"] == "complete" and outcome["proof"]
                current = request(reader, "GET", "/v1/cameras/camera_1/progress")
                assert current["active"]["node_id"] == target
                assignments[target] = current["active"]

                def view_ready():
                    snapshot = request(reader, "GET", "/v1/cameras/camera_1/view")
                    if snapshot["inference_ready"]:
                        assert (
                            snapshot["active"]["assignment_id"]
                            == current["active"]["assignment_id"]
                        )
                        assert snapshot["active"]["node_id"] == target
                        assert snapshot["media_verified"] is False
                        view_checks.append(
                            {
                                "node_id": target,
                                "epoch": snapshot["active"]["epoch"],
                                "status": snapshot["status"],
                                "history_count": len(snapshot["history"]),
                                "result_kind": snapshot["latest_result"]["kind"],
                                "media_verified": False,
                            }
                        )
                        return True
                    return False

                # 실제 Agent/DDS가 갱신한 담당·실행·저장 증거를 조회 API까지 연결한다.
                wait_until(view_ready, timeout=15)
                if previous_owner:
                    wait_until(
                        lambda: (
                            ready_nodes()[previous_owner]["report"]["service"]["status"]
                            == "stopped"
                        )
                    )
                    assert outcome["previous_stop"] == "stopped"
                transitions.append(
                    {
                        "target": target,
                        "elapsed_seconds": time.monotonic() - started,
                        "outcome": outcome,
                    }
                )
                previous_owner = target
                if number < 2:
                    soak_phase(target, soak_seconds / 2, broker_failure=number == 1)

            # 완료 기록의 재실행은 새 배정/재시작을 만들지 않는다.
            repeated = launch("handoff-repeat", command)
            output, _ = repeated.communicate(timeout=10)
            assert repeated.returncode == 0
            assert (
                json.loads(output)["operation_id"]
                == transitions[-1]["outcome"]["operation_id"]
            )
            unchanged = (
                "assignment_id",
                "epoch",
                "session_id",
                "activation_id",
                "supervisor_boot_id",
            )
            assert all(
                request(reader, "GET", "/v1/cameras/camera_1")["active"][key]
                == assignments["jetson-a"][key]
                for key in unchanged
            )

            # 준비 실패는 기존 A를 유지하고 자기 후보만 정리해야 한다.
            (tmp_path / "jetson-b" / "mode").write_text("cpu")
            failure_command = command.copy()
            failure_command[failure_command.index("--state-dir") + 1] = str(
                tmp_path / "handoff-failed"
            )
            failure_command[failure_command.index("--target-node") + 1] = "jetson-b"
            started = time.monotonic()
            failed = launch("handoff-failed", failure_command)
            output, _ = failed.communicate(timeout=65)
            failure = json.loads(output)
            assert failed.returncode == 2 and failure["status"] == "aborted", failure
            assert failure["candidate_stop"] == "stopped"
            current = request(reader, "GET", "/v1/cameras/camera_1")
            # 정상 표본의 ID/시각은 계속 갱신되므로 담당/실행 경계만 비교한다.
            assert all(
                current["active"][key] == assignments["jetson-a"][key]
                for key in unchanged
            )
            assert current["pending"] is None
            assert ready_nodes()["jetson-a"]["report"]["service"]["ready"]
            failure_seconds = time.monotonic() - started

            # 실행기만 중단한 뒤 같은 기록으로 재개하여 B 시작 명령/세션을 재사용한다.
            resume_command = failure_command.copy()
            journal_dir = tmp_path / "handoff-resume"
            resume_command[resume_command.index("--state-dir") + 1] = str(journal_dir)
            interrupted = launch("handoff-interrupted", resume_command)

            def start_recorded():
                path = journal_dir / "handoff.sqlite3"
                if not path.exists():
                    return False
                with sqlite3.connect(
                    path.resolve().as_uri() + "?mode=ro", uri=True
                ) as connection:
                    # 파일 생성과 스키마 commit 사이에는 아직 작업 기록을 조회할 수 없다.
                    if not connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='operation'"
                    ).fetchone():
                        return False
                    row = connection.execute("SELECT body FROM operation").fetchone()
                if row is None:
                    return False
                return json.loads(row[0])["requests"].get("start", {}).get("response")

            wait_until(start_recorded, timeout=15)
            interrupted.terminate()
            output, _ = interrupted.communicate(timeout=10)
            interruption = json.loads(output)
            assert (
                interrupted.returncode == 2
                and interruption["last_error"] == "interrupted"
            )
            wait_until(
                lambda: ready_nodes()["jetson-b"]["report"]["service"]["session_id"]
            )
            session = ready_nodes()["jetson-b"]["report"]["service"]["session_id"]
            (tmp_path / "jetson-b" / "mode").unlink()
            resumed = launch("handoff-resumed", resume_command)
            output, _ = resumed.communicate(timeout=65)
            recovery = json.loads(output)
            assert resumed.returncode == 0 and recovery["status"] == "complete", (
                recovery
            )
            assert recovery["operation_id"] == interruption["operation_id"]
            assert (
                ready_nodes()["jetson-b"]["report"]["service"]["session_id"] == session
            )
            assert recovery["previous_stop"] == "stopped"
            assignments["jetson-b"] = request(reader, "GET", "/v1/cameras/camera_1")[
                "active"
            ]

            # 중앙 활성화 직후 이 시험이 만든 AI만 종료해 새 세대 복귀를 확인한다.
            rollback_task = EdgeHandoff.from_file(
                handoff_config, tmp_path / "handoff-rollback"
            )
            original_send = rollback_task._send
            crashed_pid = None
            held_receipts = []

            def crash_after_activation(name, peer, method, path, body):
                nonlocal crashed_pid
                answer = original_send(name, peer, method, path, body)
                if name == "activate_controller" and answer["node_id"] == "jetson-a":
                    mode = tmp_path / "jetson-a/mode"
                    mode.write_text("hold_publish")
                    wait_until(lambda: (tmp_path / "jetson-a/publish-paused").exists())

                    def retained_before_crash():
                        with sqlite3.connect(
                            tmp_path / "jetson-a/state/video-ingest.sqlite3"
                        ) as database:
                            rows = database.execute(
                                "SELECT r.ingest_id,r.raw_sha256,r.raw_payload FROM ingest_receipts r "
                                "JOIN event_ingest_state s USING(ingest_id) WHERE s.stage='pending'"
                            ).fetchall()
                        held_receipts[:] = [
                            {
                                "ingest_id": row[0],
                                "raw_sha256": row[1],
                                "stamp": json.loads(row[2])["payload"]["metadata"][
                                    "edge_context"
                                ],
                            }
                            for row in rows
                            if json.loads(row[2])["payload"]["metadata"][
                                "edge_context"
                            ]["edge"]["assignment_id"]
                            == answer["assignment_id"]
                        ]
                        return bool(held_receipts)

                    wait_until(retained_before_crash)
                    with sqlite3.connect(
                        tmp_path / "jetson-a/state/results-context.sqlite3"
                    ) as database:
                        for receipt in held_receipts:
                            assert (
                                database.execute(
                                    "SELECT 1 FROM node_outbox WHERE event_id=?",
                                    (receipt["stamp"]["event_id"],),
                                ).fetchone()
                                is None
                            )
                    matches = []
                    for entry in Path("/proc").iterdir():
                        if not entry.name.isdigit():
                            continue
                        try:
                            if (entry / "cwd").resolve() != tmp_path / "jetson-a":
                                continue
                            arguments = (entry / "cmdline").read_bytes().split(b"\0")
                            if (
                                os.fsencode(
                                    repository / "tests/fixtures/edge_ai_process.py"
                                )
                                not in arguments
                            ):
                                continue
                            parent = next(
                                int(line.split()[1])
                                for line in (entry / "status").read_text().splitlines()
                                if line.startswith("PPid:")
                            )
                            assert parent in {process.pid for process in processes}
                            matches.append(int(entry.name))
                        except (FileNotFoundError, PermissionError, ProcessLookupError):
                            continue
                    assert len(matches) == 1 and crashed_pid is None
                    crashed_pid = matches[0]
                    os.kill(crashed_pid, signal.SIGKILL)
                    mode.unlink()
                    wait_until(
                        lambda: (
                            ready_nodes()["jetson-a"]["report"]["service"]["status"]
                            == "error"
                        ),
                        timeout=10,
                    )
                return answer

            rollback_task._send = crash_after_activation
            started = time.monotonic()
            try:
                rollback = rollback_task.run(
                    "camera_1", "jetson-a", "lab-v1", timeout=45
                )
            finally:
                rollback_task.close()
            rollback_seconds = time.monotonic() - started
            assert rollback["status"] == "rolled_back" and rollback["proof"], rollback
            assert rollback["previous_stop"] == "stopped"
            current = request(reader, "GET", "/v1/cameras/camera_1/progress")
            assert current["active"]["node_id"] == "jetson-b"
            assert current["active"]["epoch"] > assignments["jetson-b"]["epoch"]
            assert current["latest_result"]
            assignments["jetson-a"] = rollback["rollback"]["failed_assignment"]
            assignments["jetson-b"] = current["active"]

            # 새 입력을 멈춘 B에서 A로 다시 인계해, 크래시 전 원본의 복구를 전수 대조한다.
            (tmp_path / "jetson-b/mode").write_text("drain")

            def b_intake_drained():
                if (tmp_path / "jetson-b/mode").read_text() != "drained":
                    return False
                with sqlite3.connect(
                    tmp_path / "jetson-b/state/video-ingest.sqlite3"
                ) as database:
                    return (
                        database.execute(
                            "SELECT COUNT(*) FROM event_ingest_state WHERE stage='pending'"
                        ).fetchone()[0]
                        == 0
                    )

            wait_until(b_intake_drained, timeout=30)
            restart_command = command.copy()
            restart_command[restart_command.index("--state-dir") + 1] = str(
                tmp_path / "handoff-intake-restart"
            )
            restarted = launch("handoff-intake-restart", restart_command)
            output, _ = restarted.communicate(timeout=65)
            intake_restart = json.loads(output)
            assert (
                restarted.returncode == 0 and intake_restart["status"] == "complete"
            ), intake_restart
            assignments["jetson-a"] = request(reader, "GET", "/v1/cameras/camera_1")[
                "active"
            ]
            assert (
                assignments["jetson-a"]["assignment_id"]
                != rollback["rollback"]["failed_assignment"]["assignment_id"]
            )

            def retained_recovered():
                with (
                    sqlite3.connect(
                        tmp_path / "jetson-a/state/video-ingest.sqlite3"
                    ) as intake,
                    sqlite3.connect(
                        tmp_path / "jetson-a/state/results-context.sqlite3"
                    ) as local,
                    sqlite3.connect(
                        tmp_path / "controller/ownership/ownership.sqlite3"
                    ) as central,
                ):
                    for receipt in held_receipts:
                        row = intake.execute(
                            "SELECT r.raw_sha256,s.stage FROM ingest_receipts r JOIN event_ingest_state s USING(ingest_id) WHERE r.ingest_id=?",
                            (receipt["ingest_id"],),
                        ).fetchone()
                        assert row and row[0] == receipt["raw_sha256"]
                        queued = local.execute(
                            "SELECT sha256,state FROM node_outbox WHERE event_id=?",
                            (receipt["stamp"]["event_id"],),
                        ).fetchone()
                        stored = central.execute(
                            "SELECT sha256,body,decision FROM results WHERE event_id=?",
                            (receipt["stamp"]["event_id"],),
                        ).fetchone()
                        if (
                            row[1] != "accepted"
                            or not queued
                            or queued[1] != "committed"
                            or not stored
                        ):
                            return False
                        assert queued[0] == stored[0]
                        assert json.loads(stored[1])["edge"] == receipt["stamp"]["edge"]
                        assert stored[2] == "rejected"
                return True

            wait_until(retained_recovered, timeout=30)
            wait_until(
                lambda: (
                    ready_nodes()["jetson-b"]["report"]["service"]["status"]
                    == "stopped"
                )
            )
            finish_soak()
            assert soak["retained_intake_pending"] == 0

            with sqlite3.connect(
                tmp_path / "controller/ownership/ownership.sqlite3"
            ) as connection:
                rows = connection.execute(
                    "SELECT body,received_at,decision,reason FROM results"
                ).fetchall()
                delays = {}
                for body, received_at, decision, reason in rows:
                    item = json.loads(body)
                    captured_at = datetime.fromisoformat(
                        item["edge"]["captured_at"].replace("Z", "+00:00")
                    ).timestamp()
                    label = ":".join(
                        (
                            item["edge"]["node_id"],
                            item["kind"],
                            decision,
                            reason or "none",
                        )
                    )
                    delays.setdefault(label, []).append(received_at - captured_at)
                timing = {}
                for label, values in delays.items():
                    values.sort()
                    timing[label] = {
                        "count": len(values),
                        "min": values[0],
                        "max": values[-1],
                        "p50": values[(len(values) - 1) // 2],
                        "p95": values[int((len(values) - 1) * 0.95)],
                    }
                assert (
                    connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                )
            report = tmp_path_factory.getbasetemp() / "edge-handoff.json"
            report.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "ai": "fixture_no_gpu_inference",
                        "transitions": transitions,
                        "view_checks": view_checks,
                        "capture_to_receive_seconds": timing,
                        "target_failure": {
                            "elapsed_seconds": failure_seconds,
                            "outcome": failure,
                        },
                        "runner_resume": {
                            "interruption": interruption,
                            "recovery": recovery,
                            "same_ai_session": True,
                        },
                        "active_process_failure": {
                            "elapsed_seconds": rollback_seconds,
                            "terminated_fixture_pid": crashed_pid,
                            "outcome": rollback,
                        },
                        "retained_intake_restart": {
                            "held_receipts": held_receipts,
                            "same_raw_hash_event_id_and_original_assignment": True,
                            "central_rejected_old_assignment": True,
                            "outcome": intake_restart,
                        },
                        "final_reconciliation": soak["reconciliation"],
                        "retained_intake_pending": soak["retained_intake_pending"],
                    },
                    indent=2,
                )
            )
            return

        switch("jetson-a")
        soak_phase("jetson-a", soak_seconds / 2)
        wait_until(lambda: pending("jetson-a"), timeout=20)
        prior = pending("jetson-a")[0]
        switch("jetson-b")
        wait_until(lambda: pending("jetson-b"), timeout=20)
        current = pending("jetson-b")[0]
        assert current["result"]["event"]["type"] == "person"
        assert (
            current["result"]["event"]["observation"]["source_record_id"]
            == current["event_id"]
        )
        assert not request(
            consumer,
            "POST",
            f"/v1/results/{prior['event_id']}/consume",
            {"sha256": prior["sha256"]},
        )["consumed"]
        assert request(
            consumer,
            "POST",
            f"/v1/results/{current['event_id']}/consume",
            {"sha256": current["sha256"]},
        )["consumed"]
        assert request(
            consumer,
            "POST",
            f"/v1/results/{current['event_id']}/consume",
            {"sha256": current["sha256"]},
        )["duplicate"]
        soak_phase("jetson-b", soak_seconds / 2, broker_failure=True)

        def rejected_a():
            with sqlite3.connect(
                tmp_path / "controller/ownership/ownership.sqlite3"
            ) as connection:
                return connection.execute(
                    "SELECT COUNT(*) FROM results WHERE node_id='jetson-a' AND reason='not_current_owner'"
                ).fetchone()[0]

        wait_until(rejected_a, timeout=15)
        assert all(process.poll() is None for process in processes)
        for node in nodes:

            def acknowledged():
                with sqlite3.connect(
                    tmp_path / node / "state/results-context.sqlite3"
                ) as connection:
                    return (
                        connection.execute(
                            "SELECT COUNT(*) FROM node_outbox WHERE state='committed'"
                        ).fetchone()[0]
                        > 0
                    )

            wait_until(acknowledged)
            with sqlite3.connect(
                tmp_path / node / "state/results-context.sqlite3"
            ) as connection:
                assert (
                    connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                )
        if soak_seconds:
            finish_soak()
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()
        for name in dds_containers:
            # Docker 클라이언트만 강제 종료된 경우에도 이 시험이 만든 컨테이너만 정리한다.
            info = subprocess.run(
                ["docker", "inspect", name], capture_output=True, text=True, timeout=10
            )
            if info.returncode == 0:
                assert (
                    json.loads(info.stdout)[0]["Config"]["Labels"].get("cctv.purpose")
                    == "edge-dds-full"
                )
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    check=True,
                    capture_output=True,
                    timeout=20,
                )
        if soak_seconds:
            # 복구된 일시 오류도 합격 결과에서 숨기지 않고 종류별 횟수를 보존한다.
            soak["diagnostics"] = {
                "driver_storage_read_retries": sum(
                    record.getMessage() == "retrying storage read once"
                    for record in caplog.records
                ),
                "processes": {
                    log.stem: {
                        "database_locked": log.read_text().count(
                            "edge storage unavailable: database_locked"
                        ),
                        "storage_failures": log.read_text().count(
                            "edge storage unavailable:"
                        ),
                        "storage_read_retries": log.read_text().count(
                            "retrying storage read once"
                        ),
                        "metrics_timeouts": log.read_text().count("TimeoutError"),
                    }
                    for log in tmp_path.glob("*.stderr")
                },
            }
            evidence_path.write_text(json.dumps(soak, indent=2))
        for database in tmp_path.rglob("ownership.sqlite3"):
            with sqlite3.connect(database) as connection:
                print(
                    "central results",
                    connection.execute(
                        "SELECT node_id,kind,decision,reason,COUNT(*) FROM results GROUP BY node_id,kind,decision,reason"
                    ).fetchall(),
                )
                if soak_seconds:
                    # 실패한 실행도 개별 만료 시점과 배정을 남겨 다음 조사에서 원인을 좁힌다.
                    timing_path = (
                        tmp_path_factory.getbasetemp() / "edge-result-timing.jsonl"
                    )
                    with timing_path.open("w") as stream:
                        for (
                            event_id,
                            body,
                            received_at,
                            decision,
                            reason,
                            sha256,
                        ) in connection.execute(
                            "SELECT event_id,body,received_at,decision,reason,sha256 FROM results ORDER BY received_at,event_id"
                        ):
                            payload = json.loads(body)
                            edge = payload["edge"]
                            captured = datetime.fromisoformat(
                                edge["captured_at"].replace("Z", "+00:00")
                            ).timestamp()
                            expires = datetime.fromisoformat(
                                edge["expires_at"].replace("Z", "+00:00")
                            ).timestamp()
                            stream.write(
                                json.dumps(
                                    {
                                        "event_id": event_id,
                                        "sha256": sha256,
                                        "node_id": edge["node_id"],
                                        "assignment_id": edge["assignment_id"],
                                        "session_id": edge["session_id"],
                                        "kind": payload["kind"],
                                        "captured_at": captured,
                                        "received_at": received_at,
                                        "expires_at": expires,
                                        "capture_to_receive_seconds": received_at
                                        - captured,
                                        "decision": decision,
                                        "reason": reason,
                                    }
                                )
                                + "\n"
                            )
        for database in tmp_path.rglob("results-context.sqlite3"):
            with sqlite3.connect(database) as connection:
                print(
                    "node queue",
                    database.parent.parent.name,
                    connection.execute(
                        "SELECT state,last_error,COUNT(*) FROM node_outbox GROUP BY state,last_error"
                    ).fetchall(),
                )
                if soak_seconds:
                    # 실패 후 비밀 DB를 지워도 원본 대조가 가능하도록 ID/해시만 보존한다.
                    queue_path = (
                        tmp_path_factory.getbasetemp() / "edge-outbox-final.jsonl"
                    )
                    with queue_path.open("a") as stream:
                        for event_id, sha256, state in connection.execute(
                            "SELECT event_id,sha256,state FROM node_outbox ORDER BY rowid"
                        ):
                            stream.write(
                                json.dumps(
                                    {
                                        "node_id": database.parent.parent.name,
                                        "event_id": event_id,
                                        "sha256": sha256,
                                        "state": state,
                                    }
                                )
                                + "\n"
                            )
        for log in tmp_path.rglob("ai-*.log"):
            print("fixture log", log.name, "\n".join(log.read_text().splitlines()[-8:]))
