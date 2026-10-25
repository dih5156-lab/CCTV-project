"""프로세스 실행이나 장치 설정 변경 없이 Jetson 호스트 자원을 읽는다."""

# 구형 JetPack의 Python에서도 최신 타입 표기를 실행 시 평가하지 않는다.
from __future__ import annotations

from pathlib import Path


class JetsonResourceCollector:
    """CPU 구간 사용률·가용 메모리 기반 사용률·Orin GPU 사용률을 수집한다."""

    def __init__(
        self, *, proc_root: Path = Path("/proc"), sys_root: Path = Path("/sys")
    ):
        self.proc_root = Path(proc_root)
        self.sys_root = Path(sys_root)
        self.previous_cpu = None

    def _cpu_percent(self) -> float | None:
        try:
            with (self.proc_root / "stat").open() as stream:
                fields = stream.readline(4096).split()
            if fields[0] != "cpu" or len(fields) < 9:
                raise ValueError("invalid aggregate CPU counters")
            # guest는 user/nice에 포함되므로 앞의 8개 카운터만 합산한다.
            current = tuple(int(value) for value in fields[1:9])
            if any(value < 0 for value in current):
                raise ValueError("negative CPU counter")
        except (OSError, ValueError, IndexError, UnicodeError):
            self.previous_cpu = None
            return None
        previous, self.previous_cpu = self.previous_cpu, current
        if previous is None:
            return None
        deltas = [now - before for now, before in zip(current, previous)]
        total = sum(deltas)
        # 첫 표본·카운터 역행·무진행은 0%로 오인하지 않고 다음 구간부터 복구한다.
        if min(deltas) < 0 or total <= 0:
            return None
        return round(100 * (total - deltas[3] - deltas[4]) / total, 2)

    def _memory_percent(self) -> float | None:
        try:
            values = {}
            for line in (self.proc_root / "meminfo").read_text().splitlines():
                fields = line.split()
                if fields and fields[0] in ("MemTotal:", "MemAvailable:"):
                    if len(fields) != 3 or fields[2] != "kB":
                        raise ValueError("invalid memory counter")
                    values[fields[0]] = int(fields[1])
            total, available = values["MemTotal:"], values["MemAvailable:"]
            if total <= 0 or not 0 <= available <= total:
                return None
            # 회수 가능한 캐시를 모두 사용 중으로 세지 않도록 MemAvailable을 쓴다.
            return round(100 * (total - available) / total, 2)
        except (OSError, ValueError, KeyError, UnicodeError):
            return None

    def _gpu_percent(self) -> float | None:
        # Orin의 JetPack/커널별 읽기 경로만 시도하며 권한이나 클럭은 바꾸지 않는다.
        for relative in (
            "devices/platform/17000000.gpu/load",
            "devices/platform/bus@0/17000000.gpu/load",
            "devices/platform/17000000.ga10b/load",
            "devices/platform/bus@0/17000000.ga10b/load",
        ):
            try:
                value = int((self.sys_root / relative).read_text().strip())
                if 0 <= value <= 1000:
                    return value / 10
            except (OSError, ValueError, UnicodeError):
                continue
        return None

    def sample(self) -> dict:
        """미수집·읽기 실패는 null로 남겨 가짜 정상값과 이전 값을 보고하지 않는다."""
        return {
            "cpu_percent": self._cpu_percent(),
            "memory_percent": self._memory_percent(),
            "gpu_percent": self._gpu_percent(),
            "network_rx_bytes_per_second": None,
            "network_tx_bytes_per_second": None,
        }
