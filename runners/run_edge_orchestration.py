"""별도 프로세스로 HTTPS 제어기·Jetson Agent·상태 조회를 실행한다."""

# 노드의 운영 Python을 교체하지 않고 별도 Agent를 실행하기 위해 지연 평가한다.
from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import ssl
import stat
import threading
import time
from pathlib import Path

from src.services.edge_node_agent import JetsonNodeAgent, RegistryClient


def private_file(path: str | Path) -> Path:
    """비밀 파일 권한은 확인만 하고 운영 파일의 권한을 자동 변경하지 않는다."""
    path = Path(path)
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode) or mode & 0o077:
        raise ValueError(
            "secret files must be regular files with owner-only permissions"
        )
    return path


def read_token(path: str | Path) -> str:
    token = private_file(path).read_text().strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token):
        raise ValueError("token file must contain a 32-128 character URL-safe token")
    return token


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="운영과 분리된 Jetson HTTPS 자원 보고")
    commands = parser.add_subparsers(dest="command", required=True)
    controller = commands.add_parser("controller", help="독립 HTTPS 제어기")
    controller.add_argument("--bind", required=True, help="제어기 자신의 IPv4 주소")
    controller.add_argument("--port", type=int, default=0, help="기본값은 임의 빈 포트")
    controller.add_argument("--state-dir", type=Path, required=True)
    controller.add_argument("--credentials", type=Path, required=True)
    controller.add_argument("--cert", type=Path, required=True)
    controller.add_argument("--key", type=Path, required=True)
    controller.add_argument("--stale-after", type=float, default=10)
    controller.add_argument("--allow-service-status", action="store_true")
    controller.add_argument("--enable-ownership", action="store_true")
    controller.add_argument(
        "--preview-config",
        type=Path,
        help="선택형 담당 영상 미리보기 끝점(소유자 전용 JSON)",
    )
    result_input = controller.add_mutually_exclusive_group()
    result_input.add_argument(
        "--result-mqtt", type=Path, help="격리 MQTT 수신 설정(소유자 전용 JSON)"
    )
    result_input.add_argument(
        "--result-dds", type=Path, help="격리 SROS2 수신 설정(소유자 전용 JSON)"
    )
    agent = commands.add_parser("agent", help="이 Jetson의 자원과 선택한 AI 상태 보고")
    agent.add_argument("--node-id", required=True)
    agent.add_argument(
        "--node-ip", required=True, help="등록할 Jetson 자신의 IPv4 주소"
    )
    agent.add_argument("--interval", type=float, default=1)
    agent.add_argument("--reports", type=int, help="생략하면 중지 요청까지 주기 보고")
    agent.add_argument("--service-url", help="별도 감독자의 http(s)://주소:포트")
    agent.add_argument("--service-token-file", type=Path)
    agent.add_argument("--service-ca-file", type=Path)
    agent.add_argument("--services-config", type=Path, help="카메라별 감독자 접속 설정(단일 service 설정과 배타적)")
    status = commands.add_parser("status", help="별도 조회 토큰으로 두 노드 상태 조회")
    for client in (agent, status):
        client.add_argument(
            "--controller", required=True, help="https://제어기주소:포트"
        )
        client.add_argument("--ca-file", type=Path, required=True)
        client.add_argument("--token-file", type=Path, required=True)
    service = commands.add_parser(
        "service", help="허용된 시험 AI만 실행/중지하는 감독자"
    )
    service.add_argument("--node-id", required=True)
    service.add_argument("--manifest", type=Path, required=True)
    service.add_argument("--state-dir", type=Path, required=True)
    service.add_argument("--credentials", type=Path, required=True)
    service.add_argument("--bind", default="127.0.0.1")
    service.add_argument("--port", type=int, default=0)
    service.add_argument("--cert", type=Path)
    service.add_argument("--key", type=Path)
    service.add_argument("--enable-results", action="store_true")
    result_output = service.add_mutually_exclusive_group()
    result_output.add_argument(
        "--result-transport",
        type=Path,
        help="중앙 저장 ACK·격리 MQTT 설정(소유자 전용 JSON)",
    )
    result_output.add_argument(
        "--result-sidecar",
        action="store_true",
        help="별도 DDS 송신기를 위한 준비 표본 생성",
    )
    dds = commands.add_parser(
        "dds-relay", help="별도 ROS 환경에서 영속 결과를 DDS로 전송"
    )
    dds.add_argument("--config", type=Path, required=True)
    dds.add_argument("--context-db", type=Path, required=True)
    dds.add_argument("--node-id", required=True)
    service_status = commands.add_parser(
        "service-status", help="실측 AI 준비 상태 조회"
    )
    assign = commands.add_parser("assign", help="허용된 manifest의 희망 실행 상태 명령")
    assign.add_argument("--camera-id", required=True)
    assign.add_argument("--command-file", type=Path, required=True)
    for client in (service_status, assign):
        client.add_argument("--node-url", required=True)
        client.add_argument("--token-file", type=Path, required=True)
        client.add_argument("--ca-file", type=Path)
    handoff = commands.add_parser(
        "handoff", help="결과 도착 확인 후 이전 AI를 중지하는 수동 인계"
    )
    handoff.add_argument("--config", type=Path, required=True)
    handoff.add_argument("--state-dir", type=Path, required=True)
    handoff.add_argument("--camera-id", required=True)
    handoff.add_argument("--target-node", required=True)
    handoff.add_argument("--manifest-id", required=True)
    handoff.add_argument("--timeout", type=float, default=60)
    automatic = commands.add_parser(
        "auto", help="명시한 자원 정책으로 기존 인계를 자동 실행"
    )
    automatic.add_argument(
        "--config", type=Path, required=True, help="기존 인계 접속 설정"
    )
    automatic.add_argument(
        "--policy", type=Path, required=True, help="소유자 전용 정책 JSON"
    )
    automatic.add_argument("--state-dir", type=Path, required=True)
    return parser


def run_controller(args, stop: threading.Event) -> None:
    # Python 3.8 Agent가 사용하지 않는 기존 제어기/공통 API 모듈을 불러오지 않는다.
    from src.services.edge_orchestrator import NodeRegistry, create_registry_server

    if (args.result_mqtt or args.result_dds) and not args.enable_ownership:
        raise ValueError("result transport requires ownership control")
    if getattr(args, "preview_config", None) and not args.enable_ownership:
        raise ValueError("preview requires ownership control")
    credentials = json.loads(private_file(args.credentials).read_text())
    expected = {"node_tokens", "reader_token"}
    if args.enable_ownership:
        expected.update(("control_token", "consumer_token"))
    if not isinstance(credentials, dict) or set(credentials) != expected:
        raise ValueError("controller credentials do not match enabled roles")
    if not isinstance(credentials["node_tokens"], dict):
        raise ValueError("node_tokens must be an object")
    # 인증서/비밀키는 시험 전용 경로에서 명시적으로 읽으며 전역 TLS 설정은 건드리지 않는다.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(args.cert), str(private_file(args.key)))
    registry = NodeRegistry(
        args.state_dir / "nodes.sqlite3",
        stale_after=args.stale_after,
        allow_real_resources=True,
        allow_remote_nodes=True,
        allow_real_services=args.allow_service_status or args.enable_ownership,
    )
    ownership = None
    collector = None
    try:
        options = dict(host=args.bind, port=args.port, tls_context=context)
        if args.enable_ownership:
            from src.services.edge_controller_api import create_controller_server
            from src.services.edge_ownership import EdgeOwnershipStore

            ownership = EdgeOwnershipStore(
                args.state_dir / "ownership", registry, stale_after=args.stale_after
            )
            preview = None
            if getattr(args, "preview_config", None):
                from src.services.edge_preview import EdgePreviewRelay

                preview = EdgePreviewRelay.from_file(args.preview_config, ownership)
            if args.result_mqtt:
                from src.services.edge_result_transport import EdgeResultCollector

                collector = EdgeResultCollector.from_file(
                    args.result_mqtt, ownership, credentials["node_tokens"]
                )
            elif args.result_dds:
                from src.services.edge_result_dds import EdgeDdsCollector

                collector = EdgeDdsCollector.from_file(
                    args.result_dds, ownership, credentials["node_tokens"]
                )
            server = create_controller_server(
                registry,
                ownership,
                credentials["node_tokens"],
                credentials["reader_token"],
                credentials["control_token"],
                credentials["consumer_token"],
                preview_relay=preview,
                **options,
            )
        else:
            server = create_registry_server(
                registry,
                credentials["node_tokens"],
                credentials["reader_token"],
                **options,
            )
        with server:
            if collector is not None:
                collector.start()
            server.timeout = 0.2
            server.daemon_threads = False
            endpoint = {"controller": f"https://{args.bind}:{server.server_port}"}
            if ownership is not None:
                endpoint["controller_boot_id"] = ownership.boot_id
            print(json.dumps(endpoint), flush=True)
            while not stop.is_set():
                server.handle_request()
                if args.result_dds:
                    collector.check()
    finally:
        try:
            if collector is not None:
                collector.close()
        finally:
            if ownership is not None:
                ownership.close()


def run_service(args, stop):
    """감독자와 AI는 명시한 격리 경로에서만 실행하며 종료 시 소유한 자식만 정리한다."""
    from src.services.edge_service import (
        EdgeServiceSupervisor,
        ServiceManifest,
        create_service_server,
    )

    if (args.result_transport or args.result_sidecar) and not args.enable_results:
        raise ValueError("result transport requires result capture")
    credentials = json.loads(private_file(args.credentials).read_text())
    if not isinstance(credentials, dict) or set(credentials) != {
        "control_token",
        "reader_token",
    }:
        raise ValueError(
            "service credentials require control_token and reader_token only"
        )
    context = None
    if bool(args.cert) != bool(args.key):
        raise ValueError("both service certificate and key are required")
    if args.cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(args.cert), str(private_file(args.key)))
    supervisor = EdgeServiceSupervisor(
        args.node_id,
        ServiceManifest(args.manifest),
        args.state_dir,
        enable_results=args.enable_results,
    )
    relay = None
    try:
        if args.result_transport:
            from src.services.edge_result_transport import EdgeResultRelay

            relay = EdgeResultRelay.from_file(
                args.result_transport,
                supervisor.result_context,
                args.node_id,
                observation=supervisor.capture_observation,
            )
            relay.start()
        with create_service_server(
            supervisor,
            credentials["control_token"],
            credentials["reader_token"],
            host=args.bind,
            port=args.port,
            tls_context=context,
            allow_remote=True,
        ) as server:
            server.timeout = 0.2
            server.daemon_threads = False
            scheme = "https" if context else "http"
            print(
                json.dumps(
                    {
                        "service": f"{scheme}://{args.bind}:{server.server_port}",
                        "boot_id": supervisor.boot_id,
                    }
                ),
                flush=True,
            )
            next_observation = time.monotonic()
            while not stop.is_set():
                server.handle_request()
                supervisor.tick()
                if args.result_sidecar and time.monotonic() >= next_observation:
                    # DDS 통신은 별도 Python에서 실행하되 실제 AI 표본 생성은 감독자가 맡는다.
                    next_observation = time.monotonic() + 1
                    try:
                        supervisor.capture_observation()
                    except Exception as error:
                        logging.getLogger(__name__).warning(
                            "준비 표본 생성 대기: %s", type(error).__name__
                        )
    finally:
        try:
            if relay is not None:
                relay.close()
        finally:
            # 통신 종료 실패가 자신이 실행한 AI의 종료와 잠금 해제를 막지 않게 한다.
            supervisor.close()


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    stop = threading.Event()
    previous = {}
    try:
        # SIGTERM도 자신의 대기/소켓만 정리한다. 다른 프로세스나 서비스는 제어하지 않는다.
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: stop.set())
        if args.command == "controller":
            run_controller(args, stop)
            return
        if args.command == "service":
            run_service(args, stop)
            return
        if args.command == "dds-relay":
            from src.services.edge_result_context import EdgeResultContext
            from src.services.edge_result_dds import run_dds_relay

            context = EdgeResultContext(args.context_db)
            try:
                run_dds_relay(args.config, context, args.node_id, stop)
            finally:
                context.close()
            return
        if args.command == "handoff":
            # 같은 격리 작업 경로로 재실행하면 저장된 명령을 재대조한다.
            from src.services.edge_handoff import EdgeHandoff

            task = EdgeHandoff.from_file(args.config, args.state_dir)
            try:
                outcome = task.run(
                    args.camera_id,
                    args.target_node,
                    args.manifest_id,
                    timeout=args.timeout,
                    stop=stop,
                )
            finally:
                task.close()
            print(json.dumps(outcome, ensure_ascii=False, indent=2))
            if outcome["status"] != "complete":
                parser.exit(2)
            return
        if args.command == "auto":
            from src.services.edge_auto_handoff import EdgeAutoHandoff

            task = EdgeAutoHandoff.from_file(args.config, args.policy, args.state_dir)
            try:
                task.run(stop)
            finally:
                task.close()
            return
        if args.command in ("service-status", "assign"):
            client = RegistryClient(
                args.node_url, read_token(args.token_file), ca_file=args.ca_file
            )
            if args.command == "service-status":
                code, result = client.request("GET", "/v1/node/status")
            else:
                from src.services.edge_orchestrator import _identifier

                camera_id = _identifier(args.camera_id)
                payload = json.loads(private_file(args.command_file).read_text())
                code, result = client.request(
                    "PUT", f"/v1/assignments/{camera_id}", payload
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if code != 200:
                raise RuntimeError(f"service request failed: HTTP {code}")
            return
        client = RegistryClient(
            args.controller, read_token(args.token_file), ca_file=args.ca_file
        )
        if args.command == "status":
            code, result = client.request("GET", "/v1/nodes")
            if code != 200:
                raise RuntimeError(f"status request failed: HTTP {code}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            if (
                bool(args.service_url) != bool(args.service_token_file)
                or args.service_ca_file
                and not args.service_url
            ):
                raise ValueError("service URL and read token must be supplied together")
            service_clients = None
            if args.services_config:
                if args.service_url or args.service_token_file or args.service_ca_file:
                    raise ValueError("single and multiple service settings are mutually exclusive")
                from src.services.edge_orchestrator import _fields, _identifier
                from src.services.edge_service import _private_json

                settings = _private_json(args.services_config)
                if not isinstance(settings, dict) or not 1 <= len(settings) <= 32:
                    raise ValueError("1-32 camera endpoints are required")
                service_clients = {}
                for camera, item in settings.items():
                    _identifier(camera)
                    _fields(item, {"url", "ca_file", "token_file"})
                    service_clients[camera] = RegistryClient(
                        item["url"], read_token(item["token_file"]), ca_file=item["ca_file"], timeout=1
                    )
            service_client = (
                RegistryClient(
                    args.service_url,
                    read_token(args.service_token_file),
                    ca_file=args.service_ca_file,
                )
                if args.service_url
                else None
            )
            agent = JetsonNodeAgent(
                client,
                args.node_id,
                node_ip=args.node_ip,
                interval=args.interval,
                service_client=service_client,
                service_clients=service_clients,
            )
            agent.run(stop, report_limit=args.reports)
            print(
                json.dumps(
                    {
                        "node_id": args.node_id,
                        "last_sequence": agent.sequence,
                        "last_error": agent.last_error,
                    }
                )
            )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
