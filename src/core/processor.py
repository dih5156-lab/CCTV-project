"""
processor.py - 실시간 CCTV 객체 탐지 프로세서
설명: 멀티 카메라 동시 처리, RTSP 재연결, 이벤트 필터링 및 서버 전송
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
from .events import DetectionEvent

# 로깅 설정
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
    events_filtered: int = 0  # 디바운싱으로 필터된 이벤트
    events_failed: int = 0
    inference_errors: int = 0  # 추론 오류 카운터
    camera_errors: int = 0  # 카메라 오류 카운터
    start_time: float = field(default_factory=time.time)
    camera_count: int = 0
    total_inference_time: float = 0.0  # 총 추론 시간
    inference_count: int = 0  # 추론 횟수

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
    """Multi-camera video processing pipeline with AI inference."""
    def __init__(self, config: AppConfig):
        self.config = config
        
        # 멀티 모델 AI 분석기 (헬멧 + Pose)
        self.analyzer = AIAnalyzer(
            helmet_model_path=config.models.helmet_model,
            pose_model_path=config.models.pose_model,
            confidence_threshold=config.detection.pose_confidence,  # 기본 threshold
            device=config.detection.device,
            fall_angle_threshold=config.detection.fall_angle_threshold,
            fall_height_ratio=config.detection.fall_height_ratio
        )
        # 각 모델별 threshold 설정
        self.analyzer.helmet_threshold = config.detection.helmet_confidence
        self.analyzer.pose_threshold = config.detection.pose_confidence
        
        # 모델 자동 로딩 (ai_analysis.py에서 처리)
        
        # 카메라 관리
        self.cameras: Dict[str, RTSPCamera] = {}
        self.camera_threads: Dict[str, Thread] = {}
        
        # 이벤트 관리
        self.event_queue = Queue(maxsize=config.events.queue_max_size)
        self.last_events: Dict[Tuple[str, str, int], float] = {}  # (camera_id, type, object_id) -> timestamp
        self._event_lock = Lock()
        
        # Track 관리 (클래스 변경 시 중복 제거용)
        self.active_tracks: Dict[str, Dict[int, Tuple[float, DetectionEvent]]] = {}  # camera_id -> {track_id: (last_seen, event)}
        self.track_timeout = 0.5  # track 만료 시간 (초)
        self.track_iou_threshold = 0.5  # 중복 판단 IoU 임계값
          
        # 통계
        self.stats = ProcessorStats()
        
        # 제어
        self.running = False
        self.stop_event = Event()
        
        # 서버 전송 스레드
        self.sender_thread = None
        
        # 클린업 스레드
        self.cleanup_thread = None
        self.cleanup_interval = config.events.cleanup_interval
        
        # 데이터셋 저장 (deprecated)
        if hasattr(config, 'save_dataset') and config.save_dataset:
            os.makedirs(config.dataset_dir, exist_ok=True)
        
        # 위험 구역 관리
        self.zone_manager = None
        if config.zone_detection:
            try:
                self.zone_manager = ZoneManager(config.zones_config)
                logger.info("✅ 위험 구역 탐지 활성화")
            except Exception as e:
                logger.warning(f"⚠️ 위험 구역 로드 실패: {e}")
        
        # 데이터셋 수집기
        self.dataset_collector = None
        if config.collect_dataset:
            try:
                self.dataset_collector = DatasetCollector(
                    output_dir=config.dataset_dir,
                    format='yolo'
                )
                logger.info("✅ 데이터셋 수집 활성화")
            except Exception as e:
                logger.warning(f"⚠️ 데이터셋 수집 초기화 실패: {e}")
                       
    
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
            logger.warning(f"[{camera_id}] 이미 등록된 카메라입니다")
            return False

        camera = RTSPCamera(camera_id, source, self.config)
        if camera.connect():
            self.cameras[camera_id] = camera
            self.stats.camera_count = len(self.cameras)
            logger.info(f"✅ 카메라 추가: {camera_id}")
            
            # 위험 구역 로드
            if self.zone_manager:
                try:
                    self.zone_manager.load_zones(camera_id)
                except Exception as e:
                    logger.warning(f"[{camera_id}] 위험 구역 로드 실패: {e}")
            
            return True
        else:
            logger.error(f"❌ 카메라 연결 실패: {camera_id}")
            return False

    def remove_camera(self, camera_id: str):
        """Remove camera from processing pipeline."""
        if camera_id in self.cameras:
            self.cameras[camera_id].release()
            del self.cameras[camera_id]
            if camera_id in self.active_tracks:
                del self.active_tracks[camera_id]
            self.stats.camera_count = len(self.cameras)
            logger.info(f"카메라 제거: {camera_id}")

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
        # 추론 시간 측정
        start_time = time.time()
        
        # 헬멧 + Pose 모델 추론 (사람 + 낙상 동시 감지)
        events = self.analyzer.run_inference(
            frame, 
            use_helmet=True, 
            use_pose=True, 
            check_compliance=True
        )
        
        # 추론 시간 기록
        inference_time = time.time() - start_time
        self.stats.total_inference_time += inference_time
        self.stats.inference_count += 1
        self.stats.inference_count = frame_count
        
        return events
    
    def _apply_tracking(
        self, 
        events: List[DetectionEvent], 
        camera_id: str
    ) -> List[DetectionEvent]:
        """
        Track 관리: 중복 제거 및 만료된 track 정리
        
        - 같은 위치(IoU 높음)에 다른 클래스의 객체가 있으면 최신 것만 유지
        - 일정 시간 보이지 않은 track은 제거
        """
        current_time = time.time()
        
        # 카메라별 active tracks 초기화
        if camera_id not in self.active_tracks:
            self.active_tracks[camera_id] = {}
        
        # 현재 프레임에서 탐지된 track ID 집합
        current_track_ids = set()
        filtered_events = []
        
        for event in events:
            if event.object_id is None:
                filtered_events.append(event)
                continue
            
            track_id = event.object_id
            current_track_ids.add(track_id)
            
            # 같은 위치에 다른 track ID가 있는지 확인 (IoU > 0.5)
            should_add = True
            to_remove = []
            
            for existing_id, (last_seen, existing_event) in self.active_tracks[camera_id].items():
                if existing_id == track_id:
                    continue
                
                # IoU 계산
                iou = self._calculate_iou(event, existing_event)
                if iou > self.track_iou_threshold:  # 설정된 임계값 이상 겹치면 중복으로 판단
                    # 최신 것(현재 프레임)을 유지하고 이전 것 제거
                    to_remove.append(existing_id)
            
            # 중복 제거
            for old_id in to_remove:
                del self.active_tracks[camera_id][old_id]
            
            if should_add:
                self.active_tracks[camera_id][track_id] = (current_time, event)
                filtered_events.append(event)
        
        # 만료된 track 제거 (현재 프레임에 없고 일정 시간 지난 것)
        expired_ids = []
        for track_id, (last_seen, _) in self.active_tracks[camera_id].items():
            if track_id not in current_track_ids:
                if current_time - last_seen > self.track_timeout:
                    expired_ids.append(track_id)
        
        for track_id in expired_ids:
            del self.active_tracks[camera_id][track_id]
        
        return filtered_events
    
    def _calculate_iou(self, event1: DetectionEvent, event2: DetectionEvent) -> float:
        """두 이벤트의 IoU 계산"""
        x1_min = event1.x
        y1_min = event1.y
        x1_max = event1.x + event1.width
        y1_max = event1.y + event1.height
        
        x2_min = event2.x
        y2_min = event2.y
        x2_max = event2.x + event2.width
        y2_max = event2.y + event2.height
        
        # 교집합 영역
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 합집합 영역
        area1 = event1.width * event1.height
        area2 = event2.width * event2.height
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _collect_dataset(
        self, 
        frame: Any, 
        events: List[DetectionEvent], 
        camera_id: str
    ) -> None:
        """
        데이터셋 수집 및 저장
        
        Args:
            frame: OpenCV 이미지 프레임
            events: 감지된 이벤트 리스트
            camera_id: 카메라 ID
        """
        if not self.dataset_collector:
            return
        
        try:
            self.dataset_collector.save_frame(frame, events, camera_id=camera_id)
        except IOError as e:
            logger.error(f"[{camera_id}] 데이터셋 파일 저장 실패: {e}")
        except Exception as e:
            logger.warning(f"[{camera_id}] 데이터셋 저장 오류: {e}")
    
    def _check_danger_zones(
        self, 
        camera_id: str, 
        events: List[DetectionEvent], 
        frame: Any
    ) -> Tuple[List[ZoneEvent], Any]:
        """
        위험 구역 침입 탐지
        
        Args:
            camera_id: 카메라 ID
            events: 감지된 이벤트 리스트
            frame: OpenCV 이미지 프레임
            
        Returns:
            (위험 구역 이벤트 리스트, 구역이 그려진 프레임)
        """
        zone_events = []
        if not self.zone_manager:
            return zone_events, frame
        
        try:
            zone_events = self.zone_manager.check_zones(camera_id, events, frame.shape[:2])
            # 구역 그리기
            frame = self.zone_manager.draw_zones(frame, camera_id)
        except Exception as e:
            logger.warning(f"[{camera_id}] 위험 구역 탐지 오류: {e}")
        
        return zone_events, frame
    
    def _queue_events(
        self, 
        camera_id: str, 
        events: List[DetectionEvent], 
        zone_events: List[ZoneEvent]
    ) -> None:
        """
        이벤트 큐에 추가 (디바운싱 적용)
        
        Args:
            camera_id: 카메라 ID
            events: 감지된 이벤트 리스트
            zone_events: 위험 구역 이벤트 리스트
        """
        # 객체 탐지 이벤트 처리
        for event in events:
            # object_id가 None인 경우는 이제 발생하지 않지만, 안전장치로 유지
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
        
        # 위험 구역 이벤트 처리
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
        """
        프레임 화면 표시 (OpenCV 윈도우)
        
        Args:
            camera_id: 카메라 ID
            frame: OpenCV 이미지 프레임
            events: 감지된 이벤트 리스트
            
        Returns:
            계속 실행 여부 (False면 종료)
        """
        if not self.config.display or frame is None:
            return True
        
        # 바운딩 박스 그리기
        frame = draw_events(frame, events)
        
        # 카메라 정보 표시
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
            return False  # 종료 신호
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
                frame_skip = 2  # 기본값
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
        """이벤트 전송 워커"""
        consecutive_failures = 0  # 연속 실패 카운터
        
        while self.running and not self.stop_event.is_set():
            try:
                event_data = self.event_queue.get(timeout=1.0)
                
                try:
                    # config에서 자동으로 server URL 가져옴
                    result = send_event(event_data)
                    if result:
                        self.stats.events_sent += 1
                        consecutive_failures = 0  # 성공하면 리셋
                        logger.info(f"✅ 이벤트 전송: {event_data.get('camera_id')} - {event_data.get('type')}")
                    else:
                        self.stats.events_failed += 1
                        consecutive_failures += 1
                        logger.warning(f"⚠️ 이벤트 전송 실패: {event_data}")
                        
                        # 연속 실패 시 경고
                        if consecutive_failures >= self.config.processing.consecutive_failure_threshold:
                            logger.error(f"🚨 서버 전송 연속 {consecutive_failures}회 실패 - 서버 상태 확인 필요")
                            
                except Exception as e:
                    logger.error(f"❌ 전송 오류: {e}")
                    self.stats.events_failed += 1
                    consecutive_failures += 1
                    
            except Empty:
                pass
            except Exception as e:
                logger.error(f"워커 오류: {e}")
    
    def _cleanup_worker(self):
        """주기적 메모리 정리 워커"""
        while self.running and not self.stop_event.is_set():
            try:
                # cleanup_interval 초 대기
                if self.stop_event.wait(timeout=self.cleanup_interval):
                    break  # stop 신호 받으면 종료
                
                logger.info("🧹 메모리 정리 시작...")
                
                # 1. 오래된 이벤트 기록 정리
                removed = self._cleanup_old_events()
                
                if removed > 0:
                    logger.info(f"  - last_events: {removed}개 정리 (남은: {len(self.last_events)}개)")
                
                # 2. 이벤트 큐 크기 체크
                queue_size = self.event_queue.qsize()
                queue_max = self.config.events.queue_max_size
                if queue_size > queue_max * self.config.processing.queue_warning_threshold:
                    logger.warning(f"⚠️ 이벤트 큐 포화 상태: {queue_size}/{queue_max}")
                
                logger.info("✅ 메모리 정리 완료")
                
            except Exception as e:
                logger.error(f"❌ 정리 워커 오류: {e}")

    def start(self) -> None:
        """
        비디오 프로세서 시작
        
        카메라 스레드, 이벤트 전송 스레드, 메모리 정리 스레드를 시작합니다.
        """
        if self.running:
            logger.warning("이미 실행 중입니다")
            return

        if not self.cameras:
            logger.error("등록된 카메라가 없습니다")
            return

        self.running = True
        self.stop_event.clear()
        self.stats.start_time = time.time()

        # 카메라 스레드 시작
        for camera_id, camera in self.cameras.items():
            thread = Thread(
                target=self._process_camera,
                args=(camera_id, camera),
                daemon=True,
                name=f"Camera-{camera_id}"
            )
            self.camera_threads[camera_id] = thread
            thread.start()

        # 이벤트 전송 스레드 시작
        self.sender_thread = Thread(
            target=self._send_events_worker,
            daemon=True,
            name="EventSender"
        )
        self.sender_thread.start()
        
        # 클린업 스레드 시작
        self.cleanup_thread = Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="MemoryCleanup"
        )
        self.cleanup_thread.start()

        logger.info(f"✅ 프로세서 시작 (카메라 {len(self.cameras)}개)")

    def stop(self) -> None:
        """
        비디오 프로세서 정지
        
        모든 스레드를 안전하게 종료하고 리소스를 해제합니다.
        """
        logger.info("프로세서 정지 중...")
        self.running = False
        self.stop_event.set()

        # 모든 스레드 대기
        timeout = self.config.processing.thread_join_timeout
        for camera_id, thread in self.camera_threads.items():
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"[{camera_id}] 스레드 종료 시간 초과")

        if self.sender_thread and self.sender_thread.is_alive():
            self.sender_thread.join(timeout=timeout)
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=timeout)

        # 카메라 해제
        for camera in self.cameras.values():
            camera.release()

        cv2.destroyAllWindows()
        logger.info("✅ 프로세서 정지 완료")

    def get_stats(self) -> Dict:
        """통계 조회"""
        return self.stats.to_dict()

    def print_stats(self):
        """통계 출력"""
        stats = self.get_stats()
        logger.info(
            f"\n{'='*70}\n"
            f"📊 처리 통계\n"
            f"{'='*70}\n"
            f"프레임: {stats['frames_processed']} | 드롭: {stats['frames_dropped']} | FPS: {stats['fps']}\n"
            f"이벤트: 감지 {stats['events_detected']} | 전송 {stats['events_sent']} | "
            f"필터됨 {stats['events_filtered']} | 실패 {stats['events_failed']}\n"
            f"오류: 추론 {stats['inference_errors']} | 카메라 {stats['camera_errors']}\n"
            f"성능: 평균 추론 {stats['avg_inference_ms']:.1f}ms\n"
            f"카메라: {stats['camera_count']}개 | 가동시간: {stats['uptime_seconds']}초\n"
            f"{'='*70}\n"
        )




