"""
processor.py - Real-time CCTV Object Detection Processor
Multi-camera processing, RTSP reconnection, event filtering and server transmission
"""

import logging
import time
import cv2
import os

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
    """Event record for tracking and deduplication."""
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
    """Processing statistics tracker."""
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
        """Average inference time (ms)"""
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
    """Multi-camera video processing pipeline with AI inference."""
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
                       
    
    def _cleanup_old_events(self, max_age_hours: Optional[int] = None) -> int:
        """Remove old event records beyond retention period."""
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
        """Add camera to processing pipeline."""
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
        """Remove camera from processing pipeline."""
        if camera_id in self.cameras:
            self.cameras[camera_id].release()
            del self.cameras[camera_id]
            if camera_id in self.active_tracks:
                del self.active_tracks[camera_id]
            self.stats.camera_count = len(self.cameras)
            logger.info(f"Camera removed: {camera_id}")

    def _should_send_event(self, camera_id: str, event_type: str, object_id: int) -> bool:
        """Check event debouncing to prevent duplicate sends."""
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
    
    def _run_ai_inference(self, frame: Any, frame_count: int) -> List[DetectionEvent]:
        """Run AI inference on frame."""
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
        """Track management: deduplication and expired track cleanup"""
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
        """Collect and save dataset"""
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
        """Danger zone intrusion detection"""
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
        """Add events to queue with debouncing"""
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
    
    def _display_frame(
        self, 
        camera_id: str, 
        frame: Any, 
        events: List[DetectionEvent]
    ) -> bool:
        """Display frame on OpenCV window"""
        if not self.config.display or frame is None:
            return True
        
        frame = draw_events(frame, events)
        
        cv2.putText(
            frame,
            f"[{camera_id}] Objects: {len(events)} | FPS: {self.stats.get_fps():.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.imshow(f"Camera: {camera_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

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
                
                # 4. 데이터셋 수집
                self._collect_dataset(frame, events_for_dataset, camera_id)
                
                # 5. 위험 구역 탐지
                zone_events, frame = self._check_danger_zones(camera_id, events, frame)
                
                # 6. 이벤트 큐에 추가
                self._queue_events(camera_id, events, zone_events)
                
                # 7. 화면 표시
                if not self._display_frame(camera_id, frame, events):
                    self.running = False

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
        """Event transmission worker"""
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
        """Periodic memory cleanup worker"""
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

    def start(self) -> None:
        """Start video processor"""
        if self.running:
            logger.warning("Already running")
            return

        if not self.cameras:
            logger.error("No cameras registered")
            return

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

        logger.info(f"Processor started ({len(self.cameras)} cameras)")

    def stop(self) -> None:
        """Stop video processor"""
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

        for camera in self.cameras.values():
            camera.release()

        cv2.destroyAllWindows()
        logger.info("Processor stopped")

    def get_stats(self) -> Dict:
        """Get statistics"""
        return self.stats.to_dict()

    def print_stats(self):
        """Print statistics"""
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




