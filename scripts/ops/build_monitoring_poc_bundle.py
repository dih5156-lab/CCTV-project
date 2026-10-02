"""Build a source-only portable receiver archive and a separate Jetson POC DB."""

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.monitoring_contract import MonitoringBinding
from src.monitoring_delivery import MonitoringDeliveryStore

SOURCES = (
    "src/__init__.py",
    "src/ingest_store.py",
    "src/event_ingest_store.py",
    "src/record_delivery.py",
    "src/monitoring_contract.py",
    "src/monitoring_receiver.py",
    "src/monitoring_api.py",
    "src/protocols/__init__.py",
    "src/protocols/monitoring_local.py",
)
TEMPLATES = (
    "Dockerfile",
    "compose.yaml",
    "requirements.txt",
    "receiver.py",
    "init_poc.py",
    ".dockerignore",
    "README.md",
)


def build(source_db, output):
    import io

    source_db, output = Path(source_db).resolve(), Path(output).resolve()
    if (
        source_db.exists()
        or output.exists()
        or source_db.with_suffix(".poc.json").exists()
    ):
        raise ValueError(
            "Use new POC source DB and archive paths; existing files are preserved"
        )
    files = {name: (ROOT / name).read_bytes() for name in SOURCES}
    files.update(
        {
            name: (ROOT / "deploy/monitoring-poc" / name).read_bytes()
            for name in TEMPLATES
        }
    )
    store = MonitoringDeliveryStore(source_db)
    name = "laptop-poc-" + uuid4().hex[:12]
    binding = MonitoringBinding(
        store.origin_id, name, "record-mqtt:cctv/records/v1/" + name
    )
    files["binding.json"] = (json.dumps(binding.as_dict(), indent=2) + "\n").encode()
    files["MANIFEST.json"] = json.dumps(
        {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        indent=2,
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo("cctv-monitoring-poc/" + name)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    source_db.with_suffix(".poc.json").write_text(
        json.dumps(binding.as_dict(), indent=2) + "\n"
    )
    return {
        "archive": str(output),
        "source_db": str(source_db),
        "binding": binding.as_dict(),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_db, args.output), indent=2))
