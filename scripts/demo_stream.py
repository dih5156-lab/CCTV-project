"""MJPEG 스트리밍 서버 데모 스크립트.

실제 카메라 없이 테스트 프레임(색상 바 + 타임스탬프)을 생성해
브라우저에서 스트리밍 확인이 가능합니다.

실행:
    python scripts/demo_stream.py

브라우저에서 열기:
    http://localhost:8769/cameras          카메라 목록 JSON
    http://localhost:8769/stream/demo      MJPEG 스트림
    http://localhost:8769/stream/demo2     MJPEG 스트림 #2
"""

from __future__ import annotations

import sys
import time
import threading
import webbrowser
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("cv2 / numpy 가 필요합니다: pip install opencv-python numpy")

from src.services.stream_api import start_stream_api_server


# ---------------------------------------------------------------------------
# 더미 카메라 / 프로세서 구현
# ---------------------------------------------------------------------------

class _DummyCamera:
    connected = True


class DemoProcessor:
    """테스트용 프레임을 실시간으로 생성하는 더미 프로세서."""

    def __init__(self) -> None:
        self.cameras = {
            "demo": _DummyCamera(),
            "demo2": _DummyCamera(),
        }
        self._lock = threading.Lock()
        self._frames: dict[str, object] = {}
        self._running = True
        self._thread = threading.Thread(target=self._generate, daemon=True)
        self._thread.start()

    def get_camera_frame(self, camera_id: str):
        with self._lock:
            return self._frames.get(camera_id)

    def _generate(self) -> None:
        """카메라별 컬러 바 + 타임스탬프 프레임을 ~30fps 로 생성한다."""
        colors = {
            "demo":  (60,  180, 75),   # 초록
            "demo2": (66,  135, 245),  # 파랑
        }
        while self._running:
            t = time.time()
            ts = time.strftime("%Y-%m-%d %H:%M:%S") + f".{int((t % 1) * 100):02d}"
            for cam_id, color in colors.items():
                h, w = 480, 640
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                # 그라디언트 배경
                for x in range(w):
                    ratio = x / w
                    frame[:, x] = [
                        int(color[0] * (1 - ratio) + 30 * ratio),
                        int(color[1] * (1 - ratio) + 30 * ratio),
                        int(color[2] * (1 - ratio) + 30 * ratio),
                    ]
                # 카메라 이름
                cv2.putText(
                    frame, f"Camera: {cam_id}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
                )
                # 타임스탬프
                cv2.putText(
                    frame, ts, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 1,
                )
                # 이동하는 원 (애니메이션)
                cx = int((w // 2) + (w // 3) * np.sin(t * 1.5 + (0 if cam_id == "demo" else 3.14)))
                cy = int((h // 2) + (h // 4) * np.cos(t * 1.0))
                cv2.circle(frame, (cx, cy), 40, (255, 255, 100), -1)

                with self._lock:
                    self._frames[cam_id] = frame
            time.sleep(1 / 30)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

PORT = 8769

if __name__ == "__main__":
    proc = DemoProcessor()
    start_stream_api_server(proc, port=PORT)

    print()
    print("=" * 55)
    print(f"  MJPEG 스트리밍 서버 시작됨 (포트 {PORT})")
    print("=" * 55)
    print(f"  카메라 목록: http://localhost:{PORT}/cameras")
    print(f"  스트림 #1:   http://localhost:{PORT}/stream/demo")
    print(f"  스트림 #2:   http://localhost:{PORT}/stream/demo2")
    print()
    print("  HTML 페이지:  http://localhost:{PORT}/view  (아래 내용 참고)")
    print("  종료: Ctrl+C")
    print("=" * 55)

    # 3초 후 브라우저 자동 열기
    def _open():
        time.sleep(2)
        webbrowser.open(f"http://localhost:{PORT}/stream/demo")

    threading.Thread(target=_open, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n종료합니다.")
        proc.stop()
