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


def test_face_api_accepts_encoded_face_photos_up_to_ten_megabytes() -> None:
    config = _read("config/nginx/public-demo.conf.template")
    face_location = config.split("location /face-api/ {", 1)[1].split("\n    }", 1)[0]

    assert "client_max_body_size 10m;" in face_location


def test_public_demo_rejects_oversized_face_photo_before_upload() -> None:
    html = _read("web/public-demo.html")

    assert "const MAX_FACE_IMAGE_BYTES = 6 * 1024 * 1024;" in html
    assert "file.size > MAX_FACE_IMAGE_BYTES" in html
    assert "사진은 6MB 이하로 선택하세요" in html


def test_face_registration_does_not_hide_api_errors_behind_fallback() -> None:
    html = _read("web/public-demo.html")
    register_face = html.split("async function registerFace()", 1)[1].split(
        "async function deleteFace", 1
    )[0]

    assert "catch (error)" in register_face
    assert "if (!(error instanceof TypeError)) throw error;" in register_face
