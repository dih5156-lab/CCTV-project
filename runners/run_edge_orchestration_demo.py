"""모형 2노드 또는 로컬 호스트 실측을 격리 제어기에 보고한 뒤 종료한다."""

import argparse
import json
import math
import secrets
import tempfile
import threading
from pathlib import Path

from src.services.edge_node_agent import JetsonNodeAgent, MockNodeAgent, RegistryClient
from src.services.edge_orchestrator import NodeRegistry, create_registry_server


def run_demo(
    *, interval: float = 1, reports: int = 3, real_resources: bool = False
) -> dict:
    if not math.isfinite(interval) or not 0 < interval <= 60:
        raise ValueError("interval must be within (0, 60]")
    if type(reports) is not int or not 1 <= reports <= 100:
        raise ValueError("reports must be within [1, 100]")
    if type(real_resources) is not bool:
        raise ValueError("real_resources must be boolean")
    # 운영 경로/설정은 받지 않으며 DB와 인증키를 실행마다 새로 만든다.
    with tempfile.TemporaryDirectory(prefix="cctv-edge-node-demo-") as temporary:
        registry = NodeRegistry(
            Path(temporary) / "nodes.sqlite3",
            stale_after=max(10, interval * 3),
            allow_real_resources=real_resources,
        )
        # 실측은 현재 호스트 하나이며 같은 장비를 가짜 2노드로 복제하지 않는다.
        node_ids = (
            ("local-jetson",) if real_resources else ("mock-jetson-a", "mock-jetson-b")
        )
        tokens = {node_id: secrets.token_urlsafe(32) for node_id in node_ids}
        reader_token = secrets.token_urlsafe(32)
        server = create_registry_server(registry, tokens, reader_token)
        server_thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}
        )
        endpoint = f"http://127.0.0.1:{server.server_port}"
        stop = threading.Event()
        agents = [
            JetsonNodeAgent(RegistryClient(endpoint, token), node_id, interval=interval)
            if real_resources
            else MockNodeAgent(
                RegistryClient(endpoint, token),
                node_id,
                interval=interval,
                active=index == 0,
            )
            for index, (node_id, token) in enumerate(tokens.items())
        ]
        workers = [
            threading.Thread(
                target=agent.run, args=(stop,), kwargs={"report_limit": reports}
            )
            for agent in agents
        ]
        server_thread.start()
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=reports * interval + 10)
            if any(worker.is_alive() for worker in workers):
                raise RuntimeError("reporting did not finish within the deadline")
            code, snapshot = RegistryClient(endpoint, reader_token).request(
                "GET", "/v1/nodes"
            )
            if (
                code != 200
                or len(snapshot["nodes"]) != len(node_ids)
                or any(
                    node["report_count"] != reports or node["freshness"] != "online"
                    for node in snapshot["nodes"]
                )
            ):
                raise RuntimeError("node reports did not match the expected result")
            return snapshot
        finally:
            stop.set()
            for worker in workers:
                if worker.ident is not None:
                    worker.join()
            server.shutdown()
            server_thread.join()
            server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="모형 2노드 또는 로컬 자원 실측의 격리 등록/주기 보고"
    )
    parser.add_argument(
        "--interval", type=float, default=1, help="보고 간격(초), 기본 1"
    )
    parser.add_argument(
        "--reports", type=int, default=3, help="노드당 성공 보고 횟수, 기본 3"
    )
    parser.add_argument(
        "--real-resources",
        action="store_true",
        help="현재 호스트 1대의 /proc·Orin GPU sysfs를 읽기 전용으로 수집",
    )
    args = parser.parse_args()
    try:
        snapshot = run_demo(
            interval=args.interval,
            reports=args.reports,
            real_resources=args.real_resources,
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
