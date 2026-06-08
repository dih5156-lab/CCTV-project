"""
service/event_service.py
========================
Go 원본: aiot-tlv-parser/pkg/service/event_service.go

이벤트(알림) 데이터를 비동기 큐로 처리하는 서비스입니다.
Go의 goroutine + channel → Python의 threading + queue.Queue 로 변환되었습니다.

동작 원리:
  1. 이벤트 발생 시 add_notification_to_queue() 로 큐에 추가
  2. 1초마다 _queue_processor 가 queue.Queue 에서 최대 200건 꺼냄
  3. 배치 DB INSERT 실행
  4. 36초마다 applicationIDs 테이블 갱신
"""

import logging
import queue
import threading
import uuid
from typing import Dict, List, Tuple

from database.connection import DB
from database.models import Notification, QueryResult
from database.queries import QueryService

logger = logging.getLogger(__name__)

_QUEUE_SIZE = 5000     # Go: make(chan database.Notification, 5000)
_BATCH_SIZE = 200      # Go: batchSize: 200
_UPDATE_INTERVAL = 36  # Go: time.NewTicker(36 * time.Second)
_PROCESS_INTERVAL = 1  # Go: time.NewTicker(1 * time.Second)


class EventService:
    """
    이벤트/알림 처리 서비스
    Go: type EventService struct { db *database.DB; queryService *database.QueryService; applicationIDs map[string][]string; ... }
    """

    def __init__(self, db: DB):
        """
        Go: func NewEventService(db *database.DB) *EventService
        """
        self._db = db
        self._query_service = QueryService(db)
        self._application_ids: Dict[str, List[str]] = {}
        self._lock = threading.RLock()    # Go: mu sync.RWMutex

        # 알림 큐 (Go: notificationQueue chan database.Notification)
        self._notification_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_SIZE)

        self._stop_event = threading.Event()

        # 초기 실행 (Go: service.setUserIDByAppEUI())
        self._set_user_id_by_app_eui()

        # 주기적 업데이트 스레드 시작 (Go: go service.startApplicationIDsUpdate())
        self._update_thread = threading.Thread(
            target=self._start_application_ids_update,
            daemon=True,
            name="EventService-AppIDUpdater",
        )
        self._update_thread.start()

        # 큐 처리 스레드 시작 (Go: go service.startQueueProcessor())
        self._queue_thread = threading.Thread(
            target=self._start_queue_processor,
            daemon=True,
            name="EventService-QueueProcessor",
        )
        self._queue_thread.start()

    def start_application_ids_update(self) -> None:
        """
        Go: func (e *EventService) StartApplicationIdsUpdate()
        외부에서 명시적으로 업데이트를 시작할 때 사용합니다.
        """
        self._start_application_ids_update()

    def _start_application_ids_update(self) -> None:
        """
        주기적 applicationIDs 업데이트 루프
        Go: func (e *EventService) startApplicationIDsUpdate()
        Go의 for-select { case <-ticker.C } → threading.Event.wait(timeout) 로 변환
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_UPDATE_INTERVAL)
            if not self._stop_event.is_set():
                logger.info("Updating application_ids...")
                self._set_user_id_by_app_eui()

    def _set_user_id_by_app_eui(self) -> None:
        """
        DB에서 앱EUI → 사용자ID 목록 매핑 로드
        Go: func (e *EventService) setUserIDByAppEUI()
        """
        try:
            result: QueryResult = self._query_service.get_user_id_by_app_eui()
        except Exception as e:
            logger.error(f"Failed to get user ID by app EUI: {e}")
            return

        with self._lock:
            self._application_ids = {}
            for row in result.rows:
                app_id = row.get("application_id")
                user_ids = row.get("user_ids")
                if app_id and user_ids:
                    # PostgreSQL ARRAY_AGG → Python list
                    if isinstance(user_ids, list):
                        self._application_ids[app_id] = user_ids
                    elif isinstance(user_ids, str):
                        self._application_ids[app_id] = [user_ids]

        logger.info(f"Updated application IDs: {len(self._application_ids)} applications")

    def add_notification_to_queue(self, notification: Notification) -> None:
        """
        알림을 큐에 추가
        Go: func (e *EventService) AddNotificationToQueue(notification database.Notification)

        큐가 가득 차면 드롭 (Go: default 브랜치에서 로그)
        """
        try:
            self._notification_queue.put_nowait(notification)
        except queue.Full:
            logger.warning(
                f"Warning: notification queue is full, dropping notification for appEUI: {notification.app_eui}"
            )

    def _start_queue_processor(self) -> None:
        """
        주기적 큐 처리 루프
        Go: func (e *EventService) startQueueProcessor()
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_PROCESS_INTERVAL)
            if not self._stop_event.is_set():
                self._process_queue_batch()

    def _process_queue_batch(self) -> None:
        """
        큐에서 배치 크기만큼 꺼내 DB INSERT
        Go: func (e *EventService) processQueueBatch()
        """
        queue_len = self._notification_queue.qsize()
        if queue_len == 0:
            return

        batch_size = min(queue_len, _BATCH_SIZE)
        notifications = []
        for _ in range(batch_size):
            try:
                notifications.append(self._notification_queue.get_nowait())
            except queue.Empty:
                break

        if notifications:
            try:
                self._insert_notifications_batch(notifications)
                logger.info(
                    f"Successfully inserted {len(notifications)} notifications "
                    f"(queue: {self._notification_queue.qsize()} remaining)"
                )
            except Exception as e:
                logger.error(f"Failed to insert notification batch: {e}")

    def _insert_notifications_batch(self, notifications: List[Notification]) -> None:
        """
        알림 데이터 배치 DB INSERT
        Go: func (e *EventService) insertNotificationsBatch(notifications []database.Notification) error
        Go: e.db.NewInsert().Model(&notifications).Exec(context.Background())
        """
        if not notifications:
            return

        import dataclasses
        rows = [dataclasses.asdict(n) for n in notifications]

        if rows:
            columns = list(rows[0].keys())
            placeholders = ", ".join(["%s"] * len(columns))
            col_names = ", ".join(columns)
            query = f"INSERT INTO notifications ({col_names}) VALUES ({placeholders})"
            values_list = [tuple(r[c] for c in columns) for r in rows]

            with self._db._get_conn() as conn:
                with conn.cursor() as cur:
                    import psycopg2.extras
                    psycopg2.extras.execute_batch(cur, query, values_list)
                    conn.commit()

    def generate_message_id(self) -> str:
        """
        UUID 기반 메시지 ID 생성
        Go: func (e *EventService) GenerateMessageID() string
        Go: return uuid.New().String()
        """
        return str(uuid.uuid4())

    def get_application_ids(self) -> Dict[str, List[str]]:
        """
        현재 applicationIDs 복사본 반환
        Go: func (e *EventService) GetApplicationIDs() map[string][]string
        """
        with self._lock:
            return {k: list(v) for k, v in self._application_ids.items()}

    def get_user_ids_by_app_eui(self, app_eui: str) -> Tuple[List[str], bool]:
        """
        앱 EUI로 사용자 ID 목록 조회
        Go: func (e *EventService) GetUserIDsByAppEUI(appEUI string) ([]string, bool)

        Returns:
            (user_ids, found) 튜플
        """
        with self._lock:
            user_ids = self._application_ids.get(app_eui)
        if user_ids is None:
            return [], False
        return user_ids, True

    def close(self) -> None:
        """
        서비스 종료 (스레드 중지)
        Go: ctx.cancel() 대응
        """
        self._stop_event.set()
        logger.info("EventService closed")
