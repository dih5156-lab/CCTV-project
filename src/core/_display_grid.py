"""다중 카메라 통합 그리드 디스플레이."""

from __future__ import annotations

import logging
import time
from threading import Event, Lock
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..utils.visualizer import draw_events
from ..utils.zone_drawer import GridLayout, ZoneDrawer
from .events import DetectionEvent

logger = logging.getLogger(__name__)


class _DisplayGrid:
    """다중 카메라 프레임을 하나의 그리드 화면으로 합친다."""

    WIDTH = 1280
    HEIGHT = 720
    MAX_FPS = 20

    def __init__(self, get_fps) -> None:
        self._get_fps = get_fps
        self._frames: Dict[str, Any] = {}
        self._lock = Lock()
        self.window_name = "CCTV Multi-Camera View"
        self._drawer: Optional[ZoneDrawer] = None

    def set_drawer(self, drawer: ZoneDrawer) -> None:
        """run_worker 시작 전에 ZoneDrawer를 등록한다."""
        self._drawer = drawer

    def update_frame(
        self, camera_id: str, frame: Any, events: List[DetectionEvent]
    ) -> None:
        """추론 스레드에서 최신 프레임과 이벤트를 갱신한다."""
        if frame is None:
            return
        with self._lock:
            self._frames[camera_id] = (frame, list(events))

    def build_grid(self) -> Optional[Any]:
        """현재 프레임들을 그리드 이미지로 합성한다."""
        with self._lock:
            if not self._frames:
                return None
            raw_items = [
                (cam_id, frame.copy(), list(evts))
                for cam_id, (frame, evts) in self._frames.items()
                if frame is not None
            ]

        count = len(raw_items)
        if count == 0:
            return None

        cols = max(1, int(count ** 0.5) + (1 if count > 1 else 0))
        rows = (count + cols - 1) // cols
        tile_w = self.WIDTH // cols
        tile_h = self.HEIGHT // rows

        if self._drawer is not None:
            self._drawer.set_layout(
                GridLayout(
                    camera_ids=[cam_id for cam_id, _, _ in raw_items],
                    cols=cols,
                    rows=rows,
                    tile_w=tile_w,
                    tile_h=tile_h,
                    orig_sizes={
                        cam_id: (frame.shape[1], frame.shape[0])
                        for cam_id, frame, _ in raw_items
                    },
                )
            )

        resized: List[Any] = []
        for cam_id, frame, events in raw_items:
            annotated = draw_events(frame, events)
            cv2.putText(
                annotated,
                f"[{cam_id}] {len(events)}",
                (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            resized.append(cv2.resize(annotated, (tile_w, tile_h)))

        black = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        grid_rows = []
        for row_index in range(rows):
            row = [
                resized[row_index * cols + col_index]
                if row_index * cols + col_index < count
                else black
                for col_index in range(cols)
            ]
            grid_rows.append(cv2.hconcat(row))

        grid = cv2.vconcat(grid_rows)
        cv2.putText(
            grid,
            f"FPS: {self._get_fps():.1f} | Cams: {count}",
            (8, grid.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if self._drawer is not None:
            grid = self._drawer.overlay(grid)
        return grid

    def run_worker(self, stop_event: Event, is_running) -> None:
        """메인 스레드 또는 전용 스레드에서 디스플레이 루프를 실행한다."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.WIDTH, self.HEIGHT)
        if self._drawer is not None:
            cv2.setMouseCallback(self.window_name, self._drawer.mouse_callback)

        interval = 1.0 / self.MAX_FPS
        last_render = 0.0
        while is_running() and not stop_event.is_set():
            try:
                now = time.monotonic()
                elapsed = now - last_render
                if elapsed < interval:
                    wait_ms = max(1, int((interval - elapsed) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == 0xFF:
                        continue
                    if self._drawer is not None and self._drawer.handle_key(key):
                        continue
                    if key == ord("q"):
                        logger.info("'q' 입력 감지 - 중지합니다")
                        stop_event.set()
                        break
                    continue

                grid = self.build_grid()
                last_render = time.monotonic()
                if grid is not None:
                    cv2.imshow(self.window_name, np.ascontiguousarray(grid))
                    grid = None

                key = cv2.waitKey(1) & 0xFF
                if key != 0xFF:
                    if self._drawer is not None and self._drawer.handle_key(key):
                        pass
                    elif key == ord("q"):
                        logger.info("'q' 입력 감지 - 중지합니다")
                        stop_event.set()
                        break
            except Exception as exc:
                logger.error("디스플레이 워커 오류: %s", exc)
                time.sleep(0.1)
