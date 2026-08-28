import subprocess
from pathlib import Path


def test_aihub_kreid_smoke_config_loads():
    project_root = Path(__file__).resolve().parents[1]
    code = "from configs import cfg,update_config; from types import SimpleNamespace; update_config(cfg,SimpleNamespace(cfg='configs/pedes_baseline/aihub_kreid_smoke.yaml')); print(cfg.NAME,cfg.TRAIN.MAX_EPOCH,cfg.DATASET.LABEL,cfg.DATASET.TRAIN_SPLIT,cfg.DATASET.VAL_SPLIT)"
    result = subprocess.run(
        [str(project_root / ".venv/bin/python"), "-c", code],
        cwd=project_root / "Rethinking_of_PAR",
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "aihub_kreid.resnet50.smoke1 1 all train val"


def test_aihub_kreid_native11_smoke_config_loads():
    project_root = Path(__file__).resolve().parents[1]
    code = "from configs import cfg,update_config; from types import SimpleNamespace; update_config(cfg,SimpleNamespace(cfg='configs/pedes_baseline/aihub_kreid_native11_smoke.yaml')); print(cfg.NAME,cfg.TRAIN.MAX_EPOCH,cfg.DATASET.LABEL,cfg.DATASET.TRAIN_SPLIT,cfg.DATASET.VAL_SPLIT)"
    result = subprocess.run(
        [str(project_root / ".venv/bin/python"), "-c", code],
        cwd=project_root / "Rethinking_of_PAR",
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "aihub_kreid.resnet50.native11.smoke1 1 all train val"
