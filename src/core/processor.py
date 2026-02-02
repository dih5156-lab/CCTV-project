"""
processor.py - 실시간 CCTV 객체 감지 프로세서
다중 카메라 처리, RTSP 재연결, 이벤트 필터링 및 서버 전송
"""

import logging
import time
import cv2
import os
import numpy as np

from dataclasses import dataclass, field, asdict
from threading import Thread, Lock, Event
from queue import Queue, Empty
from typing import Dict, List, Union, Tuple, Any, Optional

from ..config import AppConfig
from .ai_analysis import AIAnalyzer
from ..utils.visualizer import draw_events
from ..services.server_comm import send_event
from ..utils.camera_input import RTSPCamera
from ..utils.zone_detection import ZoneManager, ZoneEvent
from ..utils.dataset_collector import DatasetCollector
from ..utils.geometry import calculate_iou
from .events import DetectionEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EventRecord:
    """추적 및 중복 제거를 위한 이벤트 레코드"""
    event_type: str
    object_id: int
    bbox: Dict
    confidence: float
    camera_id: str
    timestamp: float = field(default_factory=time.time)
    last_sent_time: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProcessorStats:
    """처리 통계 추적기"""
    frames_processed: int = 0
    frames_dropped: int = 0
    events_detected: int = 0
    events_sent: int = 0
    events_filtered: int = 0
    events_failed: int = 0
    inference_errors: int = 0
    camera_errors: int = 0
    start_time: float = field(default_factory=time.time)
    camera_count: int = 0
    total_inference_time: float = 0.0
    inference_count: int = 0

    def get_fps(self) -> float:
        elapsed = time.time() - self.start_time
        return self.frames_processed / elapsed if elapsed > 0 else 0
    
    def get_avg_inference_time(self) -> float:
        """평균 추론 시간 (ms)"""
        if self.inference_count == 0:
            return 0.0
        return (self.total_inference_time / self.inference_count) * 1000

    def to_dict(self) -> Dict:
        stats = asdict(self)
        stats["fps"] = round(self.get_fps(), 2)
        stats["uptime_seconds"] = round(time.time() - self.start_time, 2)
        stats["avg_inference_ms"] = round(self.get_avg_inference_time(), 2)
        return stats

class VideoProcessor:
    """AI 추론을 사용한 다중 카메라 비디오 처리 파이프라인"""
    def __init__(self, config: AppConfig):
        self.config = config
        
        self.analyzer = AIAnalyzer(
            helmet_model_path=config.models.helmet_model,
            pose_model_path=config.models.pose_model,
            confidence_threshold=config.detection.pose_confidence,
            device=config.detection.device,
            fall_angle_threshold=config.detection.fall_angle_threshold,
            fall_height_ratio=config.detection.fall_height_ratio
        )
        self.analyzer.helmet_threshold = config.detection.helmet_confidence
        self.analyzer.pose_threshold = config.detection.pose_confidence
        
        self.cameras: Dict[str, RTSPCamera] = {}
        self.camera_threads: Dict[str, Thread] = {}
        self.event_queue = Queue(maxsize=config.events.queue_max_size)
        self.last_events: Dict[Tuple[str, str, int], float] = {}
        self._event_lock = Lock()
        self.active_tracks: Dict[str, Dict[int, Tuple[float, DetectionEvent]]] = {}
        self.track_timeout = 0.5
        self.track_iou_threshold = 0.5
        self.stats = ProcessorStats()
        self.running = False
        self.stop_event = Event()
        self.sender_thread = None
        self.cleanup_thread = None
        self.display_thread = None
        self.cleanup_interval = config.events.cleanup_interval
        
        if hasattr(config, 'save_dataset') and config.save_dataset:
            os.makedirs(config.dataset_dir, exist_ok=True)
        
        self.zone_manager = None
        if config.zone_detection:
            try:
                self.zone_manager = ZoneManager(config.zones_config)
                logger.info("Zone detection enabled")
            except Exception as e:
                logger.warning(f"Zone loading failed: {e}")
        
        self.dataset_collector = None
        if config.collect_dataset:
            try:
                self.dataset_collector = DatasetCollector(
                    output_dir=config.dataset_dir,
                    format='yolo'
                )
                logger.info("Dataset collection enabled")
            except Exception as e:
                logger.warning(f"Dataset collector init failed: {e}")
        
        # 다중 카메라 통합 디스플레이
        self.camera_frames: Dict[str, Any] = {}
        self.frame_lock = Lock()
        self.unified_window = "CCTV Multi-Camera View"
        
        # 누적 판정 방식: 최근 N개의 추론 결과를 저장
        self.detection_history: Dict[Tuple[str, int], list] = {}  # (camera_id, object_id) -> [결과, 결과, ...]
        self.history_max_size = config.processing.detection_history_size
        self.violation_threshold = config.processing.violation_threshold
        self.cumulative_enabled = config.processing.cumulative_detection_enabled
                       
    
    def _cleanup_old_events(self, max_age_hours: Optional[int] = None) -> int:
        """보존 기간이 지난 오래된 이벤트 레코드 제거"""
        if max_age_hours is None:
            max_age_hours = self.config.events.event_retention_hours
        current_time = time.time()
        cutoff = current_time - (max_age_hours * 3600)
        before_count = len(self.last_events)
        
        self.last_events = {
            k: v for k, v in self.last_events.items() 
            if v > cutoff
        }
        
        return before_count - len(self.last_events)            
                
    def add_camera(self, camera_id: str, source: Union[str, int]) -> bool:
        """처리 파이프라인에 카메라 추가"""
        if camera_id in self.cameras:
            logger.warning(f"[{camera_id}] Camera already registered")
            return False

        camera = RTSPCamera(camera_id, source, self.config)
        if camera.connect():
            self.cameras[camera_id] = camera
            self.stats.camera_count = len(self.cameras)
            logger.info(f"Camera added: {camera_id}")
            
            if self.zone_manager:
                try:
                    self.zone_manager.load_zones(camera_id)
                except Exception as e:
                    logger.warning(f"[{camera_id}] Zone loading failed: {e}")
            
            return True
        else:
            logger.error(f"Camera connection failed: {camera_id}")
            return False

    def remove_camera(self, camera_id: str):
        """처리 파이프라인에서 카메라 제거"""
        if camera_id in self.cameras:
            self.cameras[camera_id].release()
            del self.cameras[camera_id]
            if camera_id in self.active_tracks:
                del self.active_tracks[camera_id]
            self.stats.camera_count = len(self.cameras)
            logger.info(f"Camera removed: {camera_id}")

    def _should_send_event(self, camera_id: str, event_type: str, object_id: int) -> bool:
        """중복 전송 방지를 위한 이벤트 디바운싱 확인"""
        if not self.config.events.debounce_enabled:
            return True

        key = (camera_id, event_type, object_id)
        now = time.time()

        with self._event_lock:
            last_time = self.last_events.get(key, 0)
            if now - last_time >= self.config.events.debounce_seconds:
                self.last_events[key] = now
                return True
            else:
                self.stats.events_filtered += 1
                return False
    
    def _apply_cumulative_detection(self, events: List[DetectionEvent], camera_id: str) -> List[DetectionEvent]:
        """
        누적 판정 방식: 최근 N번의 추론 결과 중 임계값 이상이 위반이면 경고
        목적: 일시적인 고개 움직임이나 모델 오류로 인한 오탐 필터링
        """
        if not self.cumulative_enabled:
            return events
        
        filtered_events = []
        
        for event in events:
            if event.object_id is None:
                # ID가 없으면 그대로 추가
                filtered_events.append(event)
                continue
            
            key = (camera_id, event.object_id)
            
            # 히스토리 초기화
            if key not in self.detection_history:
                self.detection_history[key] = []
            
            # 이벤트 추가 (True: 위반, False: 정상)
            is_violation = event.event_type.value in ["no_helmet", "fall"]  # 위반 이벤트인지 확인
            self.detection_history[key].append(is_violation)
            
            # 히스토리 크기 제한
            if len(self.detection_history[key]) > self.history_max_size:
                self.detection_history[key].pop(0)
            
            # 누적 판정: 최근 결과 중 위반 횟수 계산
            violation_count = sum(self.detection_history[key])
            
            # 임계값 이상이면 경고 발생
            if violation_count >= self.violation_threshold:
                filtered_events.append(event)
                logger.info(
                    f"[{camera_id}] 객체 {event.object_id}: "
                    f"누적 판정 결과 위반 ({violation_count}/{len(self.detection_history[key])}) "
                    f"-> {event.event_type.value}"
                )
            else:
                # 아직 임계값에 도달하지 않음 (불필요한 이벤트 전송 방지)
                logger.debug(
                    f"[{camera_id}] 객체 {event.object_id}: "
                    f"누적 판정 진행 중 ({violation_count}/{len(self.detection_history[key])}) "
                    f"- 아직 경고 아님"
                )
        
        return filtered_events
    
    def _run_ai_inference(self, frame: Any, frame_count: int) -> List[DetectionEvent]:
        """프레임에 대한 AI 추론 실행"""
        start_time = time.time()
        
        events = self.analyzer.run_inference(
            frame, 
            use_helmet=True, 
            use_pose=True, 
            check_compliance=True
        )
        
        inference_time = time.time() - start_time
        self.stats.total_inference_time += inference_time
        self.stats.inference_count += 1
        
        return events
    
    def _apply_tracking(
        self, 
        events: List[DetectionEvent], 
        camera_id: str
    ) -> List[DetectionEvent]:
        """추적 관리: 중복 제거 및 만료된 트랙 정리"""
        current_time = time.time()
        
        if camera_id not in self.active_tracks:
            self.active_tracks[camera_id] = {}
        
        current_track_ids = set()
        filtered_events = []
        
        for event in events:
            if event.object_id is None:
                filtered_events.append(event)
                continue
            
            track_id = event.object_id
            current_track_ids.add(track_id)
            should_add = True
            to_remove = []
            
            for existing_id, (last_seen, existing_event) in self.active_tracks[camera_id].items():
                if existing_id == track_id:
                    continue
                
                iou = calculate_iou(event, existing_event)
                if iou > self.track_iou_threshold:
                    to_remove.append(existing_id)
            
            for old_id in to_remove:
                del self.active_tracks[camera_id][old_id]
            
            if should_add:
                self.active_tracks[camera_id][track_id] = (current_time, event)
                filtered_events.append(event)
        
        expired_ids = []
        for track_id, (last_seen, _) in self.active_tracks[camera_id].items():
            if track_id not in current_track_ids:
                if current_time - last_seen > self.track_timeout:
                    expired_ids.append(track_id)
        
        for track_id in expired_ids:
            del self.active_tracks[camera_id][track_id]
        
        return filtered_events
    
    def _collect_dataset(
        self, 
        frame: Any, 
        events: List[DetectionEvent], 
        camera_id: str
    ) -> None:
        """데이터셋 수집 및 저장"""
        if not self.dataset_collector:
            return
        
        try:
            self.dataset_collector.save_frame(frame, events, camera_id=camera_id)
        except IOError as e:
            logger.error(f"[{camera_id}] Dataset file save failed: {e}")
        except Exception as e:
            logger.warning(f"[{camera_id}] Dataset save error: {e}")
    
    def _check_danger_zones(
        self, 
        camera_id: str, 
        events: List[DetectionEvent], 
        frame: Any
    ) -> Tuple[List[ZoneEvent], Any]:
        """위험 구역 침입 감지"""
        zone_events = []
        if not self.zone_manager:
            return zone_events, frame
        
        try:
            zone_events = self.zone_manager.check_zones(camera_id, events, frame.shape[:2])
            frame = self.zone_manager.draw_zones(frame, camera_id)
        except Exception as e:
            logger.warning(f"[{camera_id}] Zone detection error: {e}")
        
        return zone_events, frame
    
    def _queue_events(
        self, 
        camera_id: str, 
        events: List[DetectionEvent], 
        zone_events: List[ZoneEvent]
    ) -> None:
        """디바운싱과 함께 이벤트를 큐에 추가"""
        for event in events:
            event_id = event.object_id if event.object_id is not None else 0
            
            if self._should_send_event(
                camera_id,
                event.event_type.value,
                event_id
            ):
                event_data = event.to_dict()
                event_data["camera_id"] = camera_id
                self.event_queue.put(event_data)
                self.stats.events_detected += 1
        
        for zone_event in zone_events:
            zone_event_data = zone_event.to_dict()
            self.event_queue.put(zone_event_data)
            self.stats.events_detected += 1
    
    def _update_camera_frame(
        self, 
        camera_id: str, 
        frame: Any, 
        events: List[DetectionEvent]
    ) -> None:
        """공유 프레임 버퍼에서 카메라 프레임 업데이트"""
        if not self.config.display or frame is None:
            return
        
        frame = draw_events(frame, events)
        
        cv2.putText(
            frame,
            f"[{camera_id}] Objects: {len(events)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        with self.frame_lock:
            self.camera_frames[camera_id] = frame.copy()
    
    def _create_grid_display(self) -> Optional[Any]:
        """모든 카메라 프레임으로부터 통합 그리드 디스플레이 생성"""
        with self.frame_lock:
            if not self.camera_frames:
                return None
            
            frames_list = list(self.camera_frames.items())
            num_cameras = len(frames_list)
            
            if num_cameras == 0:
                return None
            
            # 그리드 레이아웃 계산 (행 x 열)
            cols = int(num_cameras ** 0.5) + (1 if num_cameras > 1 else 0)
            rows = (num_cameras + cols - 1) // cols
            
            # FHD 해상도 기준으로 각 카메라 프레임 크기 계산
            total_width = 1920
            total_height = 1080
            target_width = total_width // cols
            target_height = total_height // rows
            
            resized_frames = []
            for cam_id, frame in frames_list:
                if frame is not None:
                    resized = cv2.resize(frame, (target_width, target_height))
                    resized_frames.append((cam_id, resized))
            
            if not resized_frames:
                return None
            
            # 빈 그리드 생성
            grid_rows = []
            for row_idx in range(rows):
                row_frames = []
                for col_idx in range(cols):
                    frame_idx = row_idx * cols + col_idx
                    if frame_idx < len(resized_frames):
                        row_frames.append(resized_frames[frame_idx][1])
                    else:
                        # 빈 슬롯을 검은색 프레임으로 채우기
                        black_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                        row_frames.append(black_frame)
                
                if row_frames:
                    row_img = cv2.hconcat(row_frames)
                    grid_rows.append(row_img)
            
            if not grid_rows:
                return None
            
            # 모든 행 연결
            grid = cv2.vconcat(grid_rows)
            
            # 전역 통계 추가
            cv2.putText(
                grid,
                f"FPS: {self.stats.get_fps():.1f} | Cameras: {num_cameras}",
                (10, grid.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            return grid

    def _process_camera(self, camera_id: str, camera: RTSPCamera) -> None:
        """
        카메라별 처리 메인 루프
        
        프레임 획득 → AI 추론 → 추적 → 데이터 수집 → 구역 탐지 → 이벤트 전송 → 화면 표시
        
        Args:
            camera_id: 카메라 ID
            camera: RTSPCamera 인스턴스
        """
        frame_count = 0
        last_events = []  # 이전 프레임 결과 캐싱
        
        while self.running and not self.stop_event.is_set():
            ret, frame = camera.get_frame()
            if not ret or frame is None:
                time.sleep(self.config.processing.camera_reconnect_delay)
                continue

            frame_count += 1
            self.stats.frames_processed += 1

            try:
                # 1. AI 추론 (프레임 스킵으로 성능 향상)
                frame_skip = self.config.processing.frame_skip
                if frame_count % frame_skip == 1 or not last_events:  # frame_skip마다 추론
                    events = self._run_ai_inference(frame, frame_count)
                    last_events = events  # 결과 캐싱
                else:
                    events = last_events  # 이전 결과 재사용
                
                # 2. 데이터셋 수집용 백업
                events_for_dataset = events.copy()
                
                # 3. 객체 추적
                events = self._apply_tracking(events, camera_id)
                
                # 4. 누적 판정 방식 적용 (오탐 필터링)
                events = self._apply_cumulative_detection(events, camera_id)
                
                # 5. 데이터셋 수집
                self._collect_dataset(frame, events_for_dataset, camera_id)
                
                # 6. 위험 구역 탐지
                zone_events, frame = self._check_danger_zones(camera_id, events, frame)
                
                # 7. 이벤트 큐에 추가
                self._queue_events(camera_id, events, zone_events)
                
                # 7. 화면 표시용 프레임 업데이트
                self._update_camera_frame(camera_id, frame, events)

            except ValueError as e:
                logger.error(f"[{camera_id}] 데이터 처리 오류: {e}")
                self.stats.frames_dropped += 1
                self.stats.inference_errors += 1
            except RuntimeError as e:
                logger.error(f"[{camera_id}] 모델 실행 오류: {e}")
                self.stats.frames_dropped += 1
                self.stats.inference_errors += 1
            except Exception as e:
                import traceback
                logger.error(f"[{camera_id}] 예상치 못한 오류: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                self.stats.frames_dropped += 1
                self.stats.inference_errors += 1
                
                # 연속 에러 경고
                if self.stats.inference_errors % 10 == 0:
                    logger.warning(f"🚨 [{camera_id}] 추론 오류 {self.stats.inference_errors}회 발생")

            # FPS 제어
            time.sleep(1.0 / self.config.detection.target_fps)        
    

    def _send_events_worker(self):
        """이벤트 전송 워커"""
        consecutive_failures = 0
        
        while self.running and not self.stop_event.is_set():
            try:
                event_data = self.event_queue.get(timeout=1.0)
                
                try:
                    result = send_event(event_data)
                    if result:
                        self.stats.events_sent += 1
                        consecutive_failures = 0
                        logger.info(f"Event sent: {event_data.get('camera_id')} - {event_data.get('type')}")
                    else:
                        self.stats.events_failed += 1
                        consecutive_failures += 1
                        logger.warning(f"Event send failed: {event_data}")
                        
                        if consecutive_failures >= self.config.processing.consecutive_failure_threshold:
                            logger.error(f"Consecutive send failures: {consecutive_failures} times - Check server status")
                            
                except Exception as e:
                    logger.error(f"Send error: {e}")
                    self.stats.events_failed += 1
                    consecutive_failures += 1
                    
            except Empty:
                pass
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _cleanup_worker(self):
        """주기적 메모리 정리 워커"""
        while self.running and not self.stop_event.is_set():
            try:
                if self.stop_event.wait(timeout=self.cleanup_interval):
                    break
                
                logger.info("Memory cleanup started...")
                
                removed = self._cleanup_old_events()
                
                if removed > 0:
                    logger.info(f"  - last_events: {removed} cleaned (remaining: {len(self.last_events)})")
                
                queue_size = self.event_queue.qsize()
                queue_max = self.config.events.queue_max_size
                if queue_size > queue_max * self.config.processing.queue_warning_threshold:
                    logger.warning(f"Event queue saturation: {queue_size}/{queue_max}")
                
                logger.info("Memory cleanup completed")
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
    
    def _display_worker(self):
        """통합 디스플레이 워커 - 모든 카메라를 하나의 그리드 창에 표시"""
        if not self.config.display:
            return
        
        cv2.namedWindow(self.unified_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.unified_window, 1920, 1080)
        
        while self.running and not self.stop_event.is_set():
            try:
                grid_frame = self._create_grid_display()
                
                if grid_frame is not None:
                    cv2.imshow(self.unified_window, grid_frame)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    logger.info("User pressed 'q' - stopping")
                    self.running = False
                    break
                    
            except Exception as e:
                logger.error(f"Display worker error: {e}")
                time.sleep(0.1)

    def start(self) -> None:
        """비디오 프로세서 시작"""
        if self.running:
            logger.warning("Already running")
            return

        if not self.cameras:
            logger.error("No cameras registered")
            return

        # 기존 OpenCV 창 모두 닫기
        cv2.destroyAllWindows()
        time.sleep(0.1)

        self.running = True
        self.stop_event.clear()
        self.stats.start_time = time.time()

        for camera_id, camera in self.cameras.items():
            thread = Thread(
                target=self._process_camera,
                args=(camera_id, camera),
                daemon=True,
                name=f"Camera-{camera_id}"
            )
            self.camera_threads[camera_id] = thread
            thread.start()

        self.sender_thread = Thread(
            target=self._send_events_worker,
            daemon=True,
            name="EventSender"
        )
        self.sender_thread.start()
        
        self.cleanup_thread = Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="MemoryCleanup"
        )
        self.cleanup_thread.start()
        
        # 통합 디스플레이 스레드 시작
        if self.config.display:
            self.display_thread = Thread(
                target=self._display_worker,
                daemon=True,
                name="UnifiedDisplay"
            )
            self.display_thread.start()

        logger.info(f"Processor started ({len(self.cameras)} cameras)")

    def stop(self) -> None:
        """비디오 프로세서 중지"""
        logger.info("Stopping processor...")
        self.running = False
        self.stop_event.set()

        timeout = self.config.processing.thread_join_timeout
        for camera_id, thread in self.camera_threads.items():
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"[{camera_id}] Thread termination timeout")

        if self.sender_thread and self.sender_thread.is_alive():
            self.sender_thread.join(timeout=timeout)
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=timeout)
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=timeout)

        for camera in self.cameras.values():
            camera.release()

        cv2.destroyAllWindows()
        logger.info("Processor stopped")

    def get_stats(self) -> Dict:
        """통계 가져오기"""
        return self.stats.to_dict()

    def print_stats(self):
        """통계 출력"""
        stats = self.get_stats()
        logger.info(
            f"\n{'='*70}\n"
            f"Processing Statistics\n"
            f"{'='*70}\n"
            f"Frames: {stats['frames_processed']} | Dropped: {stats['frames_dropped']} | FPS: {stats['fps']}\n"
            f"Events: Detected {stats['events_detected']} | Sent {stats['events_sent']} | "
            f"Filtered {stats['events_filtered']} | Failed {stats['events_failed']}\n"
            f"Errors: Inference {stats['inference_errors']} | Camera {stats['camera_errors']}\n"
            f"Performance: Avg inference {stats['avg_inference_ms']:.1f}ms\n"
            f"Cameras: {stats['camera_count']} | Uptime: {stats['uptime_seconds']}s\n"
            f"{'='*70}\n"
        )




