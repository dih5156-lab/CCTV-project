"""zone_drawer.py - OpenCV 창에서 마우스로 위험구역 폴리곤을 그리는 도우미.

_DisplayGrid 의 run_worker 루프와 연동되어 동작한다.
VideoProcessor 인스턴스를 받아 완성된 구역을 즉시 저장한다.

Controls::
    d key     -> toggle drawing mode ON / OFF
    left-click  -> add point  (first click selects the camera tile)
    right-click -> undo last point
    c / Enter -> finish & save polygon  (min 3 points)
    z key     -> undo last point (keyboard)
    ESC       -> cancel current polygon
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GridLayout — build_grid() 결과 레이아웃 정보
# ---------------------------------------------------------------------------


@dataclass
class GridLayout:
    """그리드 이미지 내 카메라 타일 배치 정보."""

    camera_ids: List[str]
    cols: int
    rows: int
    tile_w: int
    tile_h: int
    orig_sizes: Dict[str, Tuple[int, int]]   # camera_id → (orig_w, orig_h)


# ---------------------------------------------------------------------------
# ZoneDrawer
# ---------------------------------------------------------------------------


class ZoneDrawer:
    """OpenCV 창에서 마우스로 위험구역 폴리곤을 그리는 인터랙티브 도우미."""

    MIN_POINTS = 3

    def __init__(self, processor: Any, cameras_json_path: str) -> None:
        """
        매개변수:
            processor: VideoProcessor 인스턴스
            cameras_json_path: 구역을 저장할 cameras.json 경로
        """
        self._processor = processor
        self._cameras_json_path = cameras_json_path

        self._lock = threading.Lock()
        self._drawing: bool = False
        self._points: List[Tuple[int, int]] = []
        self._active_camera: Optional[str] = None
        self._layout: Optional[GridLayout] = None
        self._hover: Optional[Tuple[int, int]] = None
        self._zone_counter: int = self._calc_initial_counter()
        # (camera_id, zone_id) 또는 None — hover 중인 저장된 zone
        self._hovered_zone: Optional[Tuple[str, str]] = None

    # ------------------------------------------------------------------
    # 레이아웃 업데이트 (매 프레임 _DisplayGrid 에서 호출)
    # ------------------------------------------------------------------

    def _calc_initial_counter(self) -> int:
        """기존 cameras.json zone id에서 최대 번호를 찾아 counter를 초기화한다."""
        import json, re
        from pathlib import Path
        max_num = 0
        try:
            cameras = json.loads(Path(self._cameras_json_path).read_text(encoding='utf-8'))
            for cam in cameras:
                for zone in cam.get('zones', []):
                    m = re.search(r'(\d+)$', zone.get('id', ''))
                    if m:
                        max_num = max(max_num, int(m.group(1)))
        except Exception:
            pass
        return max_num + 1

    def set_layout(self, layout: GridLayout) -> None:
        with self._lock:
            self._layout = layout

    # ------------------------------------------------------------------
    # 마우스 콜백
    # ------------------------------------------------------------------

    def mouse_callback(self, event: int, x: int, y: int,
                       flags: int, param: Any) -> None:  # noqa: N802
        with self._lock:
            if event == cv2.EVENT_MOUSEMOVE:
                self._hover = (x, y)
                if not self._drawing:
                    # 드로잉 모드 OFF → 저장된 zone hover 탐지
                    self._hovered_zone = self._hit_test_zone(x, y)
                return

            if not self._drawing or self._layout is None:
                return

            if event == cv2.EVENT_LBUTTONDOWN:
                cam_id = self._hit_test(x, y)
                if cam_id is None:
                    return
                # 첫 점: 카메라 확정
                if not self._points:
                    self._active_camera = cam_id
                elif cam_id != self._active_camera:
                    return  # 다른 카메라 셀 무시
                self._points.append((x, y))
                logger.debug("point added [%s] (%d, %d)", cam_id, x, y)

            elif event == cv2.EVENT_RBUTTONDOWN:
                if self._points:
                    self._points.pop()
                    if not self._points:
                        self._active_camera = None

    # ------------------------------------------------------------------
    # 키 입력 처리 (run_worker 에서 호출)
    # ------------------------------------------------------------------

    def handle_key(self, key: int) -> bool:
        """True 반환이면 해당 키를 소비한 것(caller 는 추가 처리 불필요)."""
        if key == ord("d"):
            with self._lock:
                self._drawing = not self._drawing
                if not self._drawing:
                    self._reset()
                    self._hovered_zone = None
            state = "ON" if self._drawing else "OFF"
            logger.info(
                "Zone drawing mode %s  (left=add point, right=undo, c=finish, ESC=cancel)", state
            )
            return True

        if key in (ord("c"), 13):          # 'c' 또는 Enter
            self._complete_zone()
            return True

        if key == 27:                       # ESC
            with self._lock:
                self._reset()
            logger.info("Zone drawing cancelled")
            return True

        if key == ord("z"):
            with self._lock:
                if self._points:
                    self._points.pop()
                    if not self._points:
                        self._active_camera = None
            return True

        if key == ord("x"):
            self._delete_hovered_zone()
            return True

        return False

    # ------------------------------------------------------------------
    # 오버레이 (run_worker 에서 build_grid 결과 위에 호출)
    # ------------------------------------------------------------------

    def overlay(self, grid: np.ndarray) -> np.ndarray:
        """그리드 이미지 위에 현재 그리기 상태와 저장된 zone을 렌더링한다."""
        with self._lock:
            drawing = self._drawing
            points = list(self._points)
            active_cam = self._active_camera
            hover = self._hover
            layout = self._layout
            hovered_zone = self._hovered_zone

        h, w = grid.shape[:2]

        # ── 저장된 zone polygon 렌더링 ─────────────────────────────────────
        zm = self._processor.zone_manager
        if layout is not None and zm is not None:
            for idx, cam_id in enumerate(layout.camera_ids):
                if cam_id not in zm.zones:
                    continue
                col = idx % layout.cols
                row = idx // layout.cols
                orig_w, orig_h = layout.orig_sizes.get(cam_id, (layout.tile_w, layout.tile_h))
                scale_x = layout.tile_w / orig_w
                scale_y = layout.tile_h / orig_h
                ox = col * layout.tile_w
                oy = row * layout.tile_h
                for zone in zm.zones[cam_id].values():
                    pts = zone.polygon.astype(float)
                    grid_pts = np.array(
                        [[int(ox + p[0] * scale_x), int(oy + p[1] * scale_y)] for p in pts],
                        dtype=np.int32,
                    )
                    is_hovered = (hovered_zone == (cam_id, zone.zone_id))
                    color      = (0, 140, 255) if is_hovered else (0, 255, 0)
                    thickness  = 3            if is_hovered else 2
                    overlay_fc = (0, 100, 200) if is_hovered else (0, 200, 0)
                    overlay_img = grid.copy()
                    cv2.fillPoly(overlay_img, [grid_pts], overlay_fc)
                    cv2.addWeighted(overlay_img, 0.18, grid, 0.82, 0, grid)
                    cv2.polylines(grid, [grid_pts], True, color, thickness, cv2.LINE_AA)
                    if len(grid_pts) > 0:
                        tx, ty = grid_pts[0]
                        label = f"{zone.name}  [x=delete]" if is_hovered else zone.name
                        cv2.putText(
                            grid, label,
                            (tx + 4, ty - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                        )

        if drawing:
            cam_part = (
                f" | {active_cam}" if active_cam
                else " | click to select camera"
            )
            pt_part = f" ({len(points)} pts)" if points else ""
            msg = (
                f"[DRAWING ON]  d=exit  left=add  right=undo"
                f"  c=finish  z=undo  ESC=cancel{cam_part}{pt_part}"
            )
            cv2.rectangle(grid, (0, 0), (w, 26), (30, 30, 30), -1)
            cv2.putText(
                grid, msg, (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA,
            )
        else:
            hint = "[ d ] Draw zone  |  Hover over a zone + [ x ] to delete"
            cv2.putText(
                grid, hint,
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 160), 1, cv2.LINE_AA,
            )

        if not points:
            return grid

        pts_arr = np.array(points, dtype=np.int32)

        # 선분
        for i in range(1, len(pts_arr)):
            cv2.line(grid, tuple(pts_arr[i - 1]), tuple(pts_arr[i]),
                     (0, 255, 255), 2, cv2.LINE_AA)

        # 미리보기선 (hover)
        if hover and drawing:
            cv2.line(grid, tuple(pts_arr[-1]), hover,
                     (0, 200, 255), 1, cv2.LINE_AA)

        # 닫힘 미리보기 (3점 이상)
        if len(pts_arr) >= self.MIN_POINTS:
            cv2.polylines(grid, [pts_arr], True, (0, 220, 255), 1)

        # 점들
        for i, pt in enumerate(points):
            color = (0, 0, 255) if i == 0 else (0, 255, 255)
            cv2.circle(grid, pt, 6, color, -1)
            cv2.circle(grid, pt, 6, (255, 255, 255), 1)

        return grid

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _hit_test(self, gx: int, gy: int) -> Optional[str]:
        """그리드 좌표가 어느 카메라 타일에 속하는지 반환 (lock 내부에서 호출)."""
        layout = self._layout
        if layout is None:
            return None
        col = gx // layout.tile_w
        row = gy // layout.tile_h
        idx = row * layout.cols + col
        if 0 <= idx < len(layout.camera_ids):
            return layout.camera_ids[idx]
        return None

    def _grid_to_cam(self, gx: int, gy: int, camera_id: str,
                     layout: GridLayout) -> List[int]:
        """그리드 픽셀 좌표 → 원본 카메라 프레임 좌표로 변환."""
        col = gx // layout.tile_w
        row = gy // layout.tile_h
        local_x = gx - col * layout.tile_w
        local_y = gy - row * layout.tile_h
        orig_w, orig_h = layout.orig_sizes.get(
            camera_id, (layout.tile_w, layout.tile_h)
        )
        return [
            int(local_x * orig_w / layout.tile_w),
            int(local_y * orig_h / layout.tile_h),
        ]

    def _complete_zone(self) -> None:
        """현재 폴리곤을 카메라 좌표로 변환하고 update_zones 로 저장한다."""
        with self._lock:
            if len(self._points) < self.MIN_POINTS or self._active_camera is None:
                logger.warning(
                    "Cannot finish: %d points (min %d) or no camera selected",
                    len(self._points), self.MIN_POINTS,
                )
                return
            points = list(self._points)
            camera_id = self._active_camera
            layout = self._layout
            zone_num = self._zone_counter
            self._reset()

        if layout is None:
            return

        cam_pts = [
            self._grid_to_cam(gx, gy, camera_id, layout)
            for gx, gy in points
        ]

        proc = self._processor

        # zone_manager 가 비활성화 상태면 온디맨드로 초기화
        if proc.zone_manager is None:
            try:
                from ..utils.zone_detection import ZoneManager
                proc.zone_manager = ZoneManager(proc.config.zones_config)
                logger.info("zone_manager initialized on-demand")
            except Exception as exc:
                logger.error("zone_manager init failed: %s", exc)
                return

        # 기존 구역 수집
        existing: List[Dict] = []
        if camera_id in proc.zone_manager.zones:
            existing = [z.to_dict() for z in proc.zone_manager.zones[camera_id].values()]

        new_zone: Dict = {
            "id": f"zone_{zone_num}",
            "name": f"Zone {zone_num}",
            "polygon": cam_pts,
        }
        all_zones = existing + [new_zone]

        ok = proc.update_zones(camera_id, all_zones, self._cameras_json_path)
        if ok:
            logger.info("[%s] zone saved: zone_%d (%d points)", camera_id, zone_num, len(cam_pts))
            with self._lock:
                self._zone_counter = zone_num + 1
        else:
            logger.warning("[%s] zone save failed", camera_id)

    def _hit_test_zone(self, gx: int, gy: int) -> Optional[Tuple[str, str]]:
        """그리드 좌표 (gx, gy) 가 어느 저장된 zone polygon 안에 있는지 반환.

        lock 없이 호출 가능 (layout/zone_manager 는 읽기 전용 접근).
        반환: (camera_id, zone_id) 또는 None.
        """
        layout = self._layout
        zm = self._processor.zone_manager
        if layout is None or zm is None:
            return None
        for idx, cam_id in enumerate(layout.camera_ids):
            if cam_id not in zm.zones:
                continue
            col = idx % layout.cols
            row = idx // layout.cols
            orig_w, orig_h = layout.orig_sizes.get(cam_id, (layout.tile_w, layout.tile_h))
            scale_x = layout.tile_w / orig_w
            scale_y = layout.tile_h / orig_h
            ox = col * layout.tile_w
            oy = row * layout.tile_h
            for zone in zm.zones[cam_id].values():
                grid_pts = np.array(
                    [[int(ox + p[0] * scale_x), int(oy + p[1] * scale_y)]
                     for p in zone.polygon.astype(float)],
                    dtype=np.int32,
                )
                if cv2.pointPolygonTest(grid_pts, (float(gx), float(gy)), False) >= 0:
                    return (cam_id, zone.zone_id)
        return None

    def _delete_hovered_zone(self) -> None:
        """현재 hover 중인 zone을 삭제하고 cameras.json에 저장한다."""
        with self._lock:
            target = self._hovered_zone
        if target is None:
            return
        cam_id, zone_id = target
        proc = self._processor
        zm = proc.zone_manager
        if zm is None or cam_id not in zm.zones or zone_id not in zm.zones[cam_id]:
            return
        remaining = [
            z.to_dict()
            for z in zm.zones[cam_id].values()
            if z.zone_id != zone_id
        ]
        ok = proc.update_zones(cam_id, remaining, self._cameras_json_path)
        if ok:
            logger.info("[%s] zone 삭제됨: %s", cam_id, zone_id)
            with self._lock:
                self._hovered_zone = None
        else:
            logger.warning("[%s] zone 삭제 실패: %s", cam_id, zone_id)

    def _reset(self) -> None:
        """내부 상태 초기화 (lock 내부에서 호출)."""
        self._points = []
        self._active_camera = None
        self._hover = None


__all__ = ["ZoneDrawer", "GridLayout"]
