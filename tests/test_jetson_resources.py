"""실장비 부하 없이 커널 표본의 단위·누락·복구 경계를 검증한다."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.services.jetson_resources import JetsonResourceCollector


@pytest.fixture
def collector(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "stat").write_text("cpu 100 10 20 800 20 10 30 10 80 9\n")
    (proc / "meminfo").write_text(
        "MemTotal: 10000 kB\nMemAvailable: 7500 kB\nMemFree: 100 kB\n"
    )
    gpu = tmp_path / "sys/devices/platform/17000000.gpu/load"
    gpu.parent.mkdir(parents=True)
    gpu.write_text("357\n")
    return JetsonResourceCollector(proc_root=proc, sys_root=tmp_path / "sys")


def test_units_warmup_guest_exclusion_and_available_memory(collector):
    first = collector.sample()
    assert first == {
        "cpu_percent": None,
        "memory_percent": 25.0,
        "gpu_percent": 35.7,
        "network_rx_bytes_per_second": None,
        "network_tx_bytes_per_second": None,
    }
    (collector.proc_root / "stat").write_text("cpu 130 10 30 850 30 10 30 10 110 9\n")
    assert collector.sample()["cpu_percent"] == 40.0


@pytest.mark.parametrize(
    "value",
    [
        "",
        "cpu 1 2",
        "cpu0 1 2 3 4 5 6 7 8",
        "cpu x 2 3 4 5 6 7 8",
        "cpu -1 2 3 4 5 6 7 8",
    ],
)
def test_bad_cpu_does_not_hide_memory_or_gpu(collector, value):
    collector.sample()
    (collector.proc_root / "stat").write_text(value)
    sample = collector.sample()
    assert sample["cpu_percent"] is None
    assert sample["memory_percent"] == 25 and sample["gpu_percent"] == 35.7
    (collector.proc_root / "stat").write_text("cpu 1 2 3 4 5 6 7 8\n")
    assert collector.sample()["cpu_percent"] is None
    (collector.proc_root / "stat").write_text("cpu 2 2 3 5 5 6 7 8\n")
    assert collector.sample()["cpu_percent"] == 50


def test_cpu_no_progress_and_reset_recover(collector):
    collector.sample()
    assert collector.sample()["cpu_percent"] is None
    (collector.proc_root / "stat").write_text("cpu 1 2 3 4 5 6 7 8\n")
    assert collector.sample()["cpu_percent"] is None
    (collector.proc_root / "stat").write_text("cpu 11 2 3 14 5 6 7 8\n")
    assert collector.sample()["cpu_percent"] == 50


@pytest.mark.parametrize(
    "text",
    [
        "",
        "MemTotal: 10000 kB\n",
        "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
        "MemTotal: 10 kB\nMemAvailable: 11 kB\n",
        "MemTotal: 10 kB\nMemAvailable: -1 kB\n",
        "MemTotal: 10 MB\nMemAvailable: 1 MB\n",
        "MemTotal: invalid kB\nMemAvailable: 1 kB\n",
    ],
)
def test_invalid_memory_is_unknown(collector, text):
    (collector.proc_root / "meminfo").write_text(text)
    assert collector.sample()["memory_percent"] is None


@pytest.mark.parametrize(
    "value,expected",
    [("0", 0), ("1000", 100), ("-1", None), ("1001", None), ("nan", None), ("", None)],
)
def test_gpu_permille_boundaries_and_no_cached_success(collector, value, expected):
    assert collector.sample()["gpu_percent"] == 35.7
    (collector.sys_root / "devices/platform/17000000.gpu/load").write_text(value)
    assert collector.sample()["gpu_percent"] == expected


@pytest.mark.parametrize(
    "path",
    [
        "devices/platform/bus@0/17000000.gpu/load",
        "devices/platform/17000000.ga10b/load",
        "devices/platform/bus@0/17000000.ga10b/load",
    ],
)
def test_orin_kernel_path_fallback(collector, path):
    (collector.sys_root / "devices/platform/17000000.gpu/load").unlink()
    target = collector.sys_root / path
    target.parent.mkdir(parents=True)
    target.write_text("123\n")
    assert collector.sample()["gpu_percent"] == 12.3


def test_permission_failure_and_missing_files_are_unknown(collector):
    with patch.object(Path, "open", side_effect=PermissionError):
        assert all(value is None for value in collector.sample().values())
    for path in [
        collector.proc_root / "stat",
        collector.proc_root / "meminfo",
        collector.sys_root / "devices/platform/17000000.gpu/load",
    ]:
        path.unlink()
    assert all(value is None for value in collector.sample().values())


def test_permission_failure_on_gpu_preserves_other_resources(collector):
    original = Path.open

    def deny_gpu(path, *args, **kwargs):
        if path.name == "load":
            raise PermissionError
        return original(path, *args, **kwargs)

    with patch.object(Path, "open", deny_gpu):
        sample = collector.sample()
    assert sample["gpu_percent"] is None and sample["memory_percent"] == 25
