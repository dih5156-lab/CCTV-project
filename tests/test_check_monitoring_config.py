import importlib.util
import sys
from pathlib import Path


def _load_script_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_monitoring_config = _load_script_module(
    "check_monitoring_config",
    "scripts/check_monitoring_config.py",
)


def test_parse_prometheus_jobs_reads_targets_and_metrics_paths():
    jobs = check_monitoring_config.parse_prometheus_jobs(
        """
scrape_configs:
  - job_name: "cctv-action-layer"
    static_configs:
      - targets: ["cctv-action-layer:8080"]
    metrics_path: /metrics
"""
    )

    assert jobs["cctv-action-layer"]["targets"] == ["cctv-action-layer:8080"]
    assert jobs["cctv-action-layer"]["metrics_path"] == "/metrics"


def test_collect_dashboard_expressions_reads_panel_targets():
    expressions = check_monitoring_config.collect_dashboard_expressions(
        {
            "panels": [
                {"targets": [{"expr": "up{job=\"cctv-action-layer\"}"}]},
                {"targets": [{"expr": "cctv_pending_events"}]},
            ]
        }
    )

    assert 'up{job="cctv-action-layer"}' in expressions
    assert "cctv_pending_events" in expressions


def test_current_monitoring_config_is_consistent():
    assert check_monitoring_config.find_monitoring_config_issues() == []
