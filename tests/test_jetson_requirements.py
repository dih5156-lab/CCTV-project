from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_jetson_onnxruntime_wheel_is_limited_to_aarch64_python310():
    requirements = (PROJECT_ROOT / "requirements/jetson.txt").read_text(
        encoding="utf-8"
    )

    onnxruntime_line = next(
        line
        for line in requirements.splitlines()
        if line.startswith("onnxruntime-gpu @ ")
    )

    assert 'platform_machine == "aarch64"' in onnxruntime_line
    assert 'python_version == "3.10"' in onnxruntime_line
