import os
import subprocess
from pathlib import Path


def test_rethinking_train_imports_without_mmcv():
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / ".training_env/numpy1")
    result = subprocess.run(
        [str(project_root / ".venv/bin/python"), "-c", "import train"],
        cwd=project_root / "Rethinking_of_PAR",
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_rethinking_train_skips_tensorboard_visualizer_when_writer_is_disabled():
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "Rethinking_of_PAR/train.py").read_text(encoding="utf-8")

    assert "if args.local_rank == 0 and tb_writer is not None:" in source
