from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_public_demo_nginx_conf_matches_template() -> None:
    assert _read("config/nginx/public-demo.conf") == _read(
        "config/nginx/public-demo.conf.template"
    )


def test_compose_files_mount_public_demo_template() -> None:
    template_mount = "source: ./config/nginx/public-demo.conf.template"

    assert template_mount in _read("docker-compose.yml")
    assert template_mount in _read("docker-compose.jetson.yml")
