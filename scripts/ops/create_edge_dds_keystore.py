"""새 격리 폴더에만 DDS 시험 인증서와 노드별 최소 토픽 권한을 만든다."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from src.services.edge_result_dds import node_component, result_topic


def sign_policy(path, certificate, key):
    subprocess.run(
        [
            "openssl",
            "smime",
            "-sign",
            "-text",
            "-in",
            str(path),
            "-out",
            str(path.with_suffix(".p7s")),
            "-signer",
            str(certificate),
            "-inkey",
            str(key),
        ],
        check=True,
    )


def create_keystores(output, node_ids, domain_id):
    """CA 비밀키는 authority에 남기고 각 역할 폴더에는 자신의 서명된 자격만 복사한다."""
    if type(domain_id) is not int or not 0 <= domain_id <= 232:
        raise ValueError("invalid domain_id")
    nodes = {node_id: node_component(node_id) for node_id in node_ids}
    if not nodes or len(nodes) != len(node_ids):
        raise ValueError("explicit unique node IDs required")
    output = Path(output).resolve()
    output.mkdir(mode=0o700)
    authority = output / "authority"
    old_umask = os.umask(0o077)
    try:
        subprocess.run(
            ["ros2", "security", "create_keystore", str(authority)], check=True
        )
        governance_path = authority / "enclaves/governance.xml"
        governance = ET.parse(governance_path)
        # 허용 토픽뿐 아니라 거버넌스의 도메인도 시험 도메인에 맞춰 서명한다.
        domains = governance.find(".//domain_rule/domains")
        assert domains is not None
        domains.clear()
        ET.SubElement(domains, "id").text = str(domain_id)
        governance.write(governance_path, encoding="utf-8", xml_declaration=True)
        sign_policy(
            governance_path,
            authority / "public/permissions_ca.cert.pem",
            authority / "private/permissions_ca.key.pem",
        )
        roles = {"controller": ([], [result_topic(node) for node in nodes])}
        roles.update(
            {component: ([result_topic(node)], []) for node, component in nodes.items()}
        )
        for role, (publish, subscribe) in roles.items():
            enclave = "/soma/" + role
            subprocess.run(
                ["ros2", "security", "create_enclave", str(authority), enclave],
                check=True,
            )
            source = authority / "enclaves" / enclave.lstrip("/")
            permissions = ET.parse(source / "permissions.xml")
            grant = permissions.find(".//grant")
            assert grant is not None
            for rule in list(grant):
                if rule.tag in {"allow_rule", "deny_rule", "default"}:
                    grant.remove(rule)
            rule = ET.SubElement(grant, "allow_rule")
            ET.SubElement(ET.SubElement(rule, "domains"), "id").text = str(domain_id)
            for direction, topics in (("publish", publish), ("subscribe", subscribe)):
                element = ET.SubElement(ET.SubElement(rule, direction), "topics")
                # ROS 발견/기본 파라미터 외에는 이 역할의 정확한 결과 토픽만 허용한다.
                for topic in ["ros_discovery_info", "rt/parameter_events"] + [
                    "rt" + topic for topic in topics
                ]:
                    ET.SubElement(element, "topic").text = topic
            ET.SubElement(grant, "default").text = "DENY"
            permissions.write(
                source / "permissions.xml", encoding="utf-8", xml_declaration=True
            )
            sign_policy(
                source / "permissions.xml",
                source / "permissions_ca.cert.pem",
                authority / "private/permissions_ca.key.pem",
            )
            for name in [
                "discovery_protection_kind",
                "liveliness_protection_kind",
                "metadata_protection_kind",
                "data_protection_kind",
            ]:
                values = list(governance.iter(name))
                assert values and all(item.text == "ENCRYPT" for item in values)
            # 기본 정책은 메시지/메타데이터 암호화와 RTPS 서명을 함께 사용한다.
            assert governance.findtext(".//rtps_protection_kind") in {"SIGN", "ENCRYPT"}
            assert (
                governance.findtext(".//allow_unauthenticated_participants") == "false"
            )
            assert governance.findtext(".//enable_join_access_control") == "true"
            for name in ("enable_read_access_control", "enable_write_access_control"):
                values = list(governance.iter(name))
                assert values and all(item.text == "true" for item in values)
            destination = output / "roles" / role / "enclaves" / enclave.lstrip("/")
            destination.parent.mkdir(parents=True, mode=0o700)
            shutil.copytree(source, destination, symlinks=False)
            # 역할별 배포물에 CA 개인키나 다른 노드 키가 포함되지 않도록 구조를 고정한다.
            assert not (output / "roles" / role / "private").exists()
        return output / "roles"
    finally:
        os.umask(old_umask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-id", action="append", required=True)
    parser.add_argument("--domain-id", type=int, required=True)
    args = parser.parse_args()
    print(create_keystores(args.output, args.node_id, args.domain_id))


if __name__ == "__main__":
    main()
