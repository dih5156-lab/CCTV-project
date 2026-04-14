"""동적 성능 조율기 — _AdaptiveGovernor.

processor.py에서 분리. VideoProcessor에서만 사용한다.
"""

from __future__ import annotations

import logging
from threading import Event
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..config import AppConfig
    from .processor import ProcessorStats

logger = logging.getLogger(__name__)


class _AdaptiveGovernor:
    """추론 지연 기반 동적 성능 조율기.

    매 CHECK_SEC 초마다 직전 윈도우의 실제 추론 평균 시간(ms)을 측정하여
    frame_skip 과 YOLO imgsz 를 자동으로 올리거나 내린다.

    조정 우선순위:
      1. 느릴 때(UPPER_MS 초과):  frame_skip 증가 → 그래도 느리면 imgsz 낮춤
      2. 여유 있을 때(LOWER_MS 미만): frame_skip 감소 → 최소이면 imgsz 높임

    device-aware:
      GPU(Jetson): UPPER=25ms / LOWER=8ms,  imgsz=[320, 416, 640]
      CPU:         UPPER=180ms/ LOWER=60ms, imgsz=[160, 224, 320]
    """

    _IMGSZ_STEPS_CPU: List[int] = [160, 224, 320]
    _IMGSZ_STEPS_GPU: List[int] = [320, 416, 640]
    _FRAME_SKIP_MIN = 1
    _FRAME_SKIP_MAX = 16
    CHECK_SEC       = 5.0

    def __init__(
        self,
        config: "AppConfig",
        stats: "ProcessorStats",
        device: str = "cpu",
    ) -> None:
        self._config    = config
        self._stats     = stats
        self._prev_total: float = 0.0
        self._prev_count: int   = 0
        self._trt_locked: bool  = False

        _is_gpu = device.lower().startswith("cuda")
        self.UPPER_MS     = 25.0 if _is_gpu else 180.0
        self.LOWER_MS     =  8.0 if _is_gpu else  60.0
        self._IMGSZ_STEPS = self._IMGSZ_STEPS_GPU if _is_gpu else self._IMGSZ_STEPS_CPU

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _window_avg_ms(self) -> Optional[float]:
        """직전 CHECK_SEC 구간의 평균 추론 시간(ms). 데이터 없으면 None."""
        curr_total = self._stats.total_inference_time
        curr_count = self._stats.inference_count
        delta_cnt  = curr_count - self._prev_count
        if delta_cnt <= 0:
            self._prev_total = curr_total
            self._prev_count = curr_count
            return None
        avg_ms = ((curr_total - self._prev_total) / delta_cnt) * 1000
        self._prev_total = curr_total
        self._prev_count = curr_count
        return avg_ms

    def lock_imgsz(self) -> None:
        """TRT .engine 파일 사용 시 imgsz 자동 조정을 비활성화한다."""
        if not self._trt_locked:
            self._trt_locked = True
            logger.info(
                "AdaptiveGovernor: TRT engine 감지 → imgsz 자동 조정 비활성화"
                " (engine 컴파일 시 고정된 imgsz 사용)"
            )

    def _imgsz_idx(self, key: str) -> int:
        """현재 imgsz 값에서 가장 가까운 _IMGSZ_STEPS 인덱스 반환."""
        from .ai._constants import _MODEL_IMGSZ
        cur = _MODEL_IMGSZ.get(key, 320)
        return min(
            range(len(self._IMGSZ_STEPS)),
            key=lambda i: abs(self._IMGSZ_STEPS[i] - cur),
        )

    def _set_imgsz(self, idx: int) -> None:
        """pose + helmet imgsz 를 동시에 설정한다."""
        from .ai._constants import _MODEL_IMGSZ, _IMGSZ_LOCK
        clamped = max(0, min(idx, len(self._IMGSZ_STEPS) - 1))
        new_val = self._IMGSZ_STEPS[clamped]
        with _IMGSZ_LOCK:
            _MODEL_IMGSZ["pose"]   = new_val
            _MODEL_IMGSZ["helmet"] = new_val

    @property
    def _skip(self) -> int:
        return self._config.processing.frame_skip

    @_skip.setter
    def _skip(self, val: int) -> None:
        self._config.processing.frame_skip = int(
            max(self._FRAME_SKIP_MIN, min(self._FRAME_SKIP_MAX, val))
        )

    # ── 1회 조정 ──────────────────────────────────────────────────────

    def step(self) -> None:
        """한 주기의 측정·조정을 수행한다."""
        avg_ms = self._window_avg_ms()
        if avg_ms is None:
            return

        skip     = self._skip
        pose_idx = self._imgsz_idx("pose")

        if avg_ms > self.UPPER_MS:
            if skip < self._FRAME_SKIP_MAX:
                self._skip = skip + 2
                logger.info(
                    "🔧 AdaptiveGovernor: 추론 %.0fms > %.0fms → frame_skip %d → %d",
                    avg_ms, self.UPPER_MS, skip, self._skip,
                )
            elif pose_idx > 0 and not self._trt_locked:
                self._set_imgsz(pose_idx - 1)
                from .ai._constants import _MODEL_IMGSZ
                logger.info(
                    "🔧 AdaptiveGovernor: 추론 %.0fms, frame_skip 최대 → imgsz 낮춤 → pose=%d",
                    avg_ms, _MODEL_IMGSZ["pose"],
                )

        elif avg_ms < self.LOWER_MS:
            if skip > self._FRAME_SKIP_MIN:
                self._skip = skip - 1
                logger.info(
                    "🔧 AdaptiveGovernor: 추론 %.0fms < %.0fms → frame_skip %d → %d (품질 회복)",
                    avg_ms, self.LOWER_MS, skip, self._skip,
                )
            elif pose_idx < len(self._IMGSZ_STEPS) - 1 and not self._trt_locked:
                self._set_imgsz(pose_idx + 1)
                from .ai._constants import _MODEL_IMGSZ
                logger.info(
                    "🔧 AdaptiveGovernor: 추론 %.0fms, frame_skip 최소 → imgsz 높임 → pose=%d (품질 회복)",
                    avg_ms, _MODEL_IMGSZ["pose"],
                )
        else:
            logger.debug(
                "AdaptiveGovernor: 추론 %.0fms ✓ (skip=%d, imgsz=%d)",
                avg_ms, skip, self._IMGSZ_STEPS[pose_idx],
            )

    # ── 백그라운드 루프 ───────────────────────────────────────────────

    def run(self, stop_event: Event) -> None:
        """stop_event 신호까지 CHECK_SEC 주기로 step()을 반복 실행한다."""
        while not stop_event.wait(timeout=self.CHECK_SEC):
            try:
                self.step()
            except Exception as exc:
                logger.warning("AdaptiveGovernor 오류: %s", exc)
