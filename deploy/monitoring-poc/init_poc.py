"""Create laptop-only credentials once; never print or archive server secrets."""

import argparse
import ipaddress
import json
import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path


def initialize(root, host, *, lan=False):
    root = Path(root).resolve()
    host = str(ipaddress.IPv4Address(host))
    binding = json.loads((root / "binding.json").read_text())
    # Validate before using the topic in Mosquitto's line-oriented ACL.
    from src.monitoring_contract import MonitoringBinding
    from src.record_delivery import record_topic

    topic = record_topic(MonitoringBinding(**binding).destination)
    if any(char.isspace() for char in topic):
        raise ValueError("Whitespace is not allowed in a POC topic")
    if any((root / name).exists() for name in ("state", "client", ".env")):
        raise ValueError("Already initialized: preserve existing state/client/.env")
    for command in ("openssl", "docker"):
        if not shutil.which(command):
            raise ValueError(f"Required command missing: {command}")
    os.umask(0o077)
    # Stage all outputs so a failed command does not replace an existing identity.
    with tempfile.TemporaryDirectory(prefix=".init-", dir=root) as temporary:
        work = Path(temporary)
        state, client = work / "state", work / "client"
        private = state / "private"
        private.mkdir(parents=True)
        client.mkdir()
        (state / "data").mkdir()
        api_token, sender_password, receiver_password = (
            secrets.token_urlsafe(32) for _ in range(3)
        )
        for name, value in {
            "api-token": api_token,
            "receiver-password": receiver_password,
        }.items():
            (private / name).write_text(value + "\n")

        def openssl(*args):
            subprocess.run(
                ["openssl", *args], cwd=private, check=True, capture_output=True
            )

        openssl(
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-sha256",
            "-days",
            "365",
            "-keyout",
            "ca.key",
            "-out",
            "ca.crt",
            "-subj",
            "/CN=CCTV Monitoring POC CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
        )
        openssl(
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            "server.key",
            "-out",
            "server.csr",
            "-subj",
            "/CN=CCTV Monitoring POC",
        )
        (private / "extensions.cnf").write_text(
            "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
            f"extendedKeyUsage=serverAuth\nsubjectAltName=DNS:broker,DNS:localhost,IP:127.0.0.1,IP:{host}\n"
        )
        openssl(
            "x509",
            "-req",
            "-in",
            "server.csr",
            "-CA",
            "ca.crt",
            "-CAkey",
            "ca.key",
            "-CAcreateserial",
            "-out",
            "server.crt",
            "-days",
            "365",
            "-sha256",
            "-extfile",
            "extensions.cnf",
        )
        (private / "passwords").write_text(
            f"sender:{sender_password}\nreceiver:{receiver_password}\n"
        )
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{private}:/credentials",
                "eclipse-mosquitto:2.0",
                "mosquitto_passwd",
                "-U",
                "/credentials/passwords",
            ],
            check=True,
        )
        (state / "broker.conf").write_text(
            "listener 8883\nallow_anonymous false\npersistence false\nlog_dest stdout\n"
            "cafile /state/ca.crt\ncertfile /state/server.crt\nkeyfile /state/server.key\n"
            "tls_version tlsv1.2\npassword_file /state/passwords\nacl_file /state/acl\n"
        )
        (state / "acl").write_text(
            f"user sender\ntopic write {topic}\nuser receiver\ntopic read {topic}\n"
        )
        shutil.copyfile(private / "ca.crt", client / "ca.crt")
        (client / "api-token").write_text(api_token + "\n")
        (client / "sender-password").write_text(sender_password + "\n")
        (client / "config.json").write_text(
            json.dumps(
                {
                    "binding": binding,
                    "mqtt_host": host,
                    "mqtt_port": 18883,
                    "https_url": f"https://{host}:18443",
                    "mqtt_username": "sender",
                },
                indent=2,
            )
            + "\n"
        )
        for folder in (private, client):
            for path in folder.iterdir():
                path.chmod(0o600)
        env = f"POC_UID={os.getuid()}\nPOC_GID={os.getgid()}\nPOC_BIND_IP={'0.0.0.0' if lan else '127.0.0.1'}\n"
        (work / ".env").write_text(env)
        for name in ("state", "client", ".env"):
            (work / name).rename(root / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Windows Wi-Fi IPv4 address")
    parser.add_argument(
        "--lan", action="store_true", help="Publish the two TLS ports on LAN"
    )
    args = parser.parse_args()
    initialize(Path(__file__).resolve().parent, args.host, lan=args.lan)
    print(
        "Initialized. Start: docker compose up -d --build. Copy only client/ to Jetson."
    )
