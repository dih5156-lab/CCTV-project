import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ops/prepare_jetson_display.sh"


def test_display_preparation_retries_authentication_until_it_succeeds(tmp_path):
    # 소켓/쿠키 파일이 있어도 아직 인증이 안 되는 로그인 구간을 재현한다.
    for name, body in {
        "xhost": 'n=0; [ ! -f "$DISPLAY_TEST_STATE" ] || read -r n < "$DISPLAY_TEST_STATE"; n=$((n+1)); echo "$n" > "$DISPLAY_TEST_STATE"; [ "$n" -ge 3 ]',
        "timeout": 'shift; exec "$@"',
        "sleep": "exit 0",
    }.items():
        p = tmp_path / name
        p.write_text("#!/bin/sh\n" + body + "\n")
        p.chmod(0o755)
    state = tmp_path / "attempts"
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)], capture_output=True, text=True,
        env={**os.environ, "PATH": str(tmp_path) + ":/usr/bin:/bin", "DISPLAY": ":1", "DISPLAY_TEST_STATE": str(state)},
    )
    assert result.returncode == 0, result.stderr
    assert state.read_text().strip() == "3"


def test_display_preparation_does_not_claim_success_when_access_never_arrives(tmp_path):
    for name, body in {"xhost": "exit 1", "timeout": 'shift; exec "$@"', "sleep": "exit 0"}.items():
        p = tmp_path / name
        p.write_text("#!/bin/sh\n" + body + "\n")
        p.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)], capture_output=True, text=True,
        env={**os.environ, "PATH": str(tmp_path) + ":/usr/bin:/bin", "DISPLAY": ":1"},
    )
    assert result.returncode == 1
    assert "X11" in result.stderr
