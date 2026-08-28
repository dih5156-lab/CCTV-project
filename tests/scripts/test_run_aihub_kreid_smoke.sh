#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="${PROJECT_ROOT}/scripts/train/run_aihub_kreid_smoke.sh"
TEST_ROOT=$(mktemp -d "${PROJECT_ROOT}/.tmp_aihub_train_test.XXXXXX")
trap 'rm -rf "$TEST_ROOT"' EXIT

DATASET_DIR="${TEST_ROOT}/dataset"
TRAIN_ROOT="${TEST_ROOT}/train_root"
OUTPUT_ROOT="${TEST_ROOT}/output"
NUMPY_ROOT="${TEST_ROOT}/numpy1"
FAKE_PYTHON="${TEST_ROOT}/fake_python.py"
CALLS_FILE="${TEST_ROOT}/calls.jsonl"

mkdir -p "$DATASET_DIR/Training" "$DATASET_DIR/Validation" \
    "$TRAIN_ROOT/configs/pedes_baseline" "$NUMPY_ROOT"

TEST_ROOT="$TEST_ROOT" .venv/bin/python - <<'PY'
import os
import zipfile
from pathlib import Path

root = Path(os.environ["TEST_ROOT"])
archives = {
    root / "dataset/Training/[라벨]Training_labels.zip": {"H00001.xml": "<xml/>"},
    root / "dataset/Training/Training_source.zip": {"H00001.png": "img"},
    root / "dataset/Validation/[라벨]Validation_labels.zip": {"H00002.xml": "<xml/>"},
    root / "dataset/Validation/Validation_source.zip": {"H00002.png": "img"},
}
for path, files in archives.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)

(root / "train_root/train.py").write_text("print('stub')\n", encoding="utf-8")
(root / "train_root/configs/pedes_baseline/aihub_kreid_smoke.yaml").write_text(
    "NAME: stub\nTRAIN:\n  MAX_EPOCH: 1\n",
    encoding="utf-8",
)
PY

cat > "$FAKE_PYTHON" <<'PY'
#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


def parse_flag(args, name):
    for index, value in enumerate(args):
        if value == name:
            return args[index + 1]
    raise SystemExit(f"missing flag: {name}")


def record(payload):
    path = Path(os.environ["CALLS_FILE"])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


argv = sys.argv[1:]
target = argv[0]
if target.endswith("prepare_aihub_kreid_appearance.py"):
    output_dir = Path(parse_flag(argv, "--output-dir"))
    (output_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images/validation").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.csv").write_text(
        "image_path,person_id,split,gender,upper_clothes,upper_color,lower_clothes,lower_color,bag,hat\n"
        "images/train/H00001.png,H00001,train,female,long_sleeve,black,long_pants,blue,yes,no\n",
        encoding="utf-8",
    )
    record({
        "step": "prepare",
        "train_label": parse_flag(argv, "--train-label-zip"),
        "train_source": parse_flag(argv, "--train-source-zip"),
        "validation_label": parse_flag(argv, "--validation-label-zip"),
        "validation_source": parse_flag(argv, "--validation-source-zip"),
        "output_dir": str(output_dir),
    })
elif target.endswith("build_rethinking_par_dataset.py"):
    output_pkl = Path(parse_flag(argv, "--output-pkl"))
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    output_pkl.write_bytes(b"PKL")
    record({
        "step": "build",
        "manifest": parse_flag(argv, "--manifest"),
        "image_root": parse_flag(argv, "--image-root"),
        "output_pkl": str(output_pkl),
    })
elif target == "train.py":
    record({
        "step": "train",
        "cfg": parse_flag(argv, "--cfg"),
        "cwd": os.getcwd(),
        "par_data_root": os.environ.get("PAR_DATA_ROOT"),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    })
else:
    raise SystemExit(f"unexpected target: {target}")
PY
chmod +x "$FAKE_PYTHON"

CALLS_FILE="$CALLS_FILE" \
PYTHON="$FAKE_PYTHON" \
TRAIN_ROOT="$TRAIN_ROOT" \
AIHUB_KREID_DIR="$DATASET_DIR" \
AIHUB_KREID_TRAIN_ROOT="$OUTPUT_ROOT" \
NUMPY_ROOT="$NUMPY_ROOT" \
AIHUB_FORCE_REBUILD=1 \
"$SCRIPT" >/dev/null

TEST_ROOT="$TEST_ROOT" CALLS_FILE="$CALLS_FILE" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["TEST_ROOT"])
calls = [json.loads(line) for line in Path(os.environ["CALLS_FILE"]).read_text(encoding="utf-8").splitlines()]
assert [call["step"] for call in calls] == ["prepare", "build", "train"], calls
assert Path(calls[0]["output_dir"]) == root / "output"
assert Path(calls[1]["output_pkl"]) == root / "output/RAP2/dataset_all.pkl"
assert calls[2]["cfg"] == "configs/pedes_baseline/aihub_kreid_smoke.yaml"
assert calls[2]["par_data_root"] == str(root / "output")
assert str(root / "numpy1") in calls[2]["pythonpath"]
assert (root / "output/manifest.csv").is_file()
assert (root / "output/RAP2/dataset_all.pkl").is_file()
PY

printf 'PASS: AI-Hub smoke runner prepares manifest, builds pkl, and starts training.\n'