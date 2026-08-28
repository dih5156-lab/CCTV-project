import os
import subprocess
from pathlib import Path


def test_get_pkl_rootpath_accepts_environment_override(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    code = "from tools.function import get_pkl_rootpath; print(get_pkl_rootpath('RAP2', False))"
    environment = os.environ.copy()
    environment["PAR_DATA_ROOT"] = str(tmp_path)

    result = subprocess.run(
        [str(project_root / ".venv/bin/python"), "-c", code],
        cwd=project_root / "Rethinking_of_PAR",
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(tmp_path / "RAP2/dataset_all.pkl")
