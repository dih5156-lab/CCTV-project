#!/usr/bin/env python3
"""Compose 내부 서비스 포트의 host_ip 제한을 읽기 전용으로 검사한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

DEFAULT_SERVICES = ("cctv-alert-api", "cctv-action-layer", "cctv-ai-engine")


class _ComposeLoader(yaml.SafeLoader):
    """Docker Compose 확장 태그를 값 구조로만 읽는 Loader."""


def _construct_unknown_tag(loader: _ComposeLoader, node: yaml.Node) -> Any:
    if isinstance(node, ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, MappingNode):
        return loader.construct_mapping(node)
    return None


_ComposeLoader.add_constructor(None, _construct_unknown_tag)


def _port_exposure(port: Any) -> str:
    if isinstance(port, dict):
        host_ip = port.get("host_ip")
        if host_ip is None:
            return "missing_host_ip"
        value = str(host_ip)
        if value in {"127.0.0.1", "::1"} or "127.0.0.1" in value:
            return "localhost"
        if "${" in value:
            return "configurable"
        return "external"
    if isinstance(port, str):
        value = port.strip()
        if value.startswith("127.0.0.1:") or value.startswith("[::1]:"):
            return "localhost"
        if value.startswith("${") or ":${" in value:
            return "configurable"
        return "missing_host_ip"
    return "unknown"


def check_compose(text: str, services: tuple[str, ...] = DEFAULT_SERVICES) -> dict[str, Any]:
    """대상 서비스별 포트 노출 상태와 위험 항목을 반환한다."""
    try:
        compose = yaml.load(text, Loader=_ComposeLoader) or {}
    except yaml.YAMLError as exc:
        return {"passed": False, "errors": [f"invalid compose YAML: {exc}"]}
    service_map = compose.get("services") if isinstance(compose, dict) else None
    if not isinstance(service_map, dict):
        return {"passed": False, "errors": ["compose services must be an object"]}

    findings: list[dict[str, str]] = []
    for service_name in services:
        service = service_map.get(service_name)
        if not isinstance(service, dict):
            continue
        ports = service.get("ports") or []
        for index, port in enumerate(ports):
            exposure = _port_exposure(port)
            if exposure in {"missing_host_ip", "external"}:
                findings.append(
                    {"service": service_name, "port_index": str(index), "exposure": exposure}
                )
    return {"passed": not findings, "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose", type=Path, default=Path("docker-compose.jetson.yml"))
    parser.add_argument("--warn-only", action="store_true", help="위험 항목이 있어도 종료 코드는 0")
    args = parser.parse_args()
    try:
        result = check_compose(args.compose.read_text(encoding="utf-8"))
    except OSError as exc:
        result = {"passed": False, "errors": [str(exc)]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed") or args.warn_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
