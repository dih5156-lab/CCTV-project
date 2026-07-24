"""PyTorch 학습을 시작하기 전에 Jetson CUDA 동작을 검증한다."""

from __future__ import annotations

import json
import platform
import sys
from typing import Any


def collect_cuda_status() -> tuple[bool, dict[str, Any]]:
    """PyTorch 정보와 실제 CUDA 텐서 연산 결과를 반환한다."""
    status: dict[str, Any] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
    }

    try:
        import torch
    except Exception as exc:
        status["error"] = f"torch import 실패: {type(exc).__name__}: {exc}"
        return False, status

    status.update(
        {
            "torch": getattr(torch, "__version__", "unknown"),
            "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    )
    if not status["cuda_available"]:
        status["error"] = "PyTorch가 CUDA 장치를 사용할 수 없음"
        return False, status

    try:
        device = torch.device("cuda:0")
        value = (torch.ones(4, device=device) * 2).sum().item()
        torch.cuda.synchronize(device)
        status["cuda_device"] = torch.cuda.get_device_name(device)
        status["cuda_tensor_result"] = value
    except Exception as exc:
        status["error"] = f"CUDA 텐서 연산 실패: {type(exc).__name__}: {exc}"
        return False, status

    return value == 8.0, status


def main() -> int:
    """점검 결과를 JSON으로 출력하고 실패 시 1을 반환한다."""
    ok, status = collect_cuda_status()
    status["ok"] = ok
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
