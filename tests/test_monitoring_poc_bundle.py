import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from runners.run_monitoring_remote_poc import build_workers, seed_demo
from scripts.ops.build_monitoring_poc_bundle import ROOT, build
from src.monitoring_delivery import MonitoringDeliveryStore


def test_bundle_contains_only_manifested_sources_and_no_database_or_credentials(
    tmp_path,
):
    result = build(tmp_path / "source.db", tmp_path / "bundle.tar.gz")
    unpack = tmp_path / "unpacked"
    with tarfile.open(result["archive"]) as archive:
        archive.extractall(unpack, filter="data")
    root = unpack / "cctv-monitoring-poc"
    manifest = json.loads((root / "MANIFEST.json").read_text())
    assert set(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    ) == set(manifest) | {"MANIFEST.json"}
    for name, digest in manifest.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        assert not any(
            part in {"private", "state", "client", ".env"} for part in Path(name).parts
        )
        assert not name.endswith((".key", ".db"))
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, '.'); import receiver; import init_poc; import src.protocols.monitoring_local",
        ],
        cwd=root,
        check=True,
    )
    assert json.loads((root / "binding.json").read_text()) == result["binding"]


def test_bundle_refuses_to_overwrite_source_identity(tmp_path):
    source = tmp_path / "source.db"
    source.write_text("must survive")
    with pytest.raises(ValueError, match="preserved"):
        build(source, tmp_path / "bundle.tar.gz")
    assert source.read_text() == "must survive"


def test_demo_repeat_has_two_records_and_no_physical_delivery_jobs(tmp_path):
    source = MonitoringDeliveryStore(tmp_path / "source.db")
    seed_demo(source)
    seed_demo(source)
    with sqlite3.connect(source.path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM canonical_records").fetchone()[0]
            == 2
        )
        assert (
            connection.execute("SELECT count(*) FROM delivery_jobs").fetchone()[0] == 0
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM canonical_records WHERE source_namespace='tlv_mqtt'"
            ).fetchone()[0]
            == 1
        )


def test_sender_rejects_a_different_poc_identity_before_network(tmp_path):
    result = build(tmp_path / "source.db", tmp_path / "bundle.tar.gz")
    client = tmp_path / "client"
    client.mkdir()
    (client / "config.json").write_text(
        json.dumps({"binding": {**result["binding"], "origin_id": "wrong"}})
    )
    with pytest.raises(ValueError, match="marker"):
        build_workers(tmp_path / "source.db", client)


def test_initializer_preserves_existing_credentials(tmp_path):
    result = build(tmp_path / "source.db", tmp_path / "bundle.tar.gz")
    (tmp_path / "binding.json").write_text(json.dumps(result["binding"]))
    (tmp_path / "state").mkdir()
    secret = tmp_path / "state/keep"
    secret.write_text("unchanged")
    spec = importlib.util.spec_from_file_location(
        "poc_init", ROOT / "deploy/monitoring-poc/init_poc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="Already initialized"):
        module.initialize(tmp_path, "127.0.0.1")
    assert secret.read_text() == "unchanged"
