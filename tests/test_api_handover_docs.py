from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_api_handover_uses_compose_port() -> None:
    handover = (PROJECT_ROOT / "docs/API_HANDOVER.md").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker-compose.jetson.yml").read_text(encoding="utf-8")

    assert '"9000"' in compose
    assert "cctv-public-api" in handover
    assert "http://localhost:9000" in handover
    assert "http://localhost:8000/api/v1" not in handover


def test_alert_api_port_is_not_used_for_public_api_examples() -> None:
    handover = (PROJECT_ROOT / "docs/API_HANDOVER.md").read_text(encoding="utf-8")

    assert "http://localhost:8000/api/" not in handover
