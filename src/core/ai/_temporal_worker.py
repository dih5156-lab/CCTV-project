"""TCN 모델 재사용용 단일 자식 프로세스. 실패한 CUDA 프로세스는 재사용하지 않는다."""

from __future__ import annotations

import json
import os
import select
import subprocess
import time
from pathlib import Path
from threading import Lock


class TemporalWorker:
    def __init__(self) -> None:
        self._process = None
        self._key = None
        self._request_id = 0
        self._lock = Lock()

    def _stop(self) -> None:
        process, self._process = self._process, None
        self._key = None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)
            process.stdin.close()
            process.stdout.close()

    def close(self) -> None:
        with self._lock:
            self._stop()

    def run(self, command: list[str], *, timeout: float) -> subprocess.CompletedProcess:
        started = time.monotonic()
        # 동시에 들어온 요청도 대기 시간을 제한하고, 다른 요청의 응답을 가져가지 않는다.
        if not self._lock.acquire(timeout=max(timeout, 0.0)):
            raise subprocess.TimeoutExpired(command, timeout)
        try:
            base = list(command)
            index = base.index("--video")
            video = base[index + 1]
            del base[index:index + 2]
            # 같은 경로의 체크포인트가 교체돼도 이전 모델을 계속 사용하지 않는다.
            paths = [base[1], base[base.index("--model") + 1],
                     base[base.index("--pose-model") + 1]]
            fingerprints = tuple((str(p), Path(p).stat().st_mtime_ns, Path(p).stat().st_size)
                                 for p in paths)
            key = (tuple(base), fingerprints)
            if self._process is None or self._process.poll() is not None or key != self._key:
                self._stop()
                # 1회 실행과 같은 작업 경로/모듈 탐색 경로를 사용한다.
                project_root = Path(__file__).resolve().parents[3]
                environment = dict(os.environ, PYTHONPATH=str(project_root))
                self._process = subprocess.Popen(
                    [*base, "--worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=0,
                    cwd=project_root, env=environment,
                )
                self._key = key
            self._request_id += 1
            request_id = self._request_id
            message = json.dumps({"request_id": request_id, "video": video}).encode() + b"\n"
            if len(message) > 4096:
                raise ValueError("TCN worker request too large")
            self._process.stdin.write(message)
            pending = b""
            received = 0
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0 or not select.select([self._process.stdout], [], [], remaining)[0]:
                    raise subprocess.TimeoutExpired(command, timeout)
                chunk = os.read(self._process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("TCN worker exited without a result")
                pending += chunk
                received += len(chunk)
                if received > 1024 * 1024:
                    raise RuntimeError("TCN worker output limit exceeded")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if not line.startswith(b"TCN_RESULT "):
                        continue
                    result = json.loads(line[len(b"TCN_RESULT "):])
                    if result.get("request_id") != request_id:
                        raise RuntimeError("TCN worker response ID mismatch")
                    if "error" in result:
                        raise RuntimeError("TCN worker inference failed: " + str(result["error"]))
                    output = result.get("stdout")
                    if not isinstance(output, str):
                        raise RuntimeError("TCN worker result has no stdout")
                    return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        except BaseException:
            # 시간 초과·파싱 오류·CUDA 예외 후에는 자식의 상태를 신뢰하지 않는다.
            self._stop()
            raise
        finally:
            self._lock.release()
