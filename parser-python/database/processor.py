"""
processor.py
============
Go 원본: aiot-tlv-parser/pkg/database/processor.go

센서 데이터를 메모리 큐에 누적한 후 주기적으로 배치 INSERT 하는 모듈입니다.
Go의 goroutine + ticker → Python의 threading.Timer 루프로 변환되었습니다.

동작 원리:
  1. add_data() 로 각 테이블별 리스트에 데이터 추가
  2. interval 마다 _start_bulk_processor()가 깨어나
  3. 각 리스트에서 최대 1000건씩 꺼내 배치 INSERT 수행
"""

import logging
import threading
from datetime import timedelta
from typing import Any, List

from database.connection import DB, bulk_insert
from database.models import (
    T3,
    T34950,
    T34952,
    T34954,
    T34955,
    T34956,
    T34957,
    T34958,
    SensorData,
)

logger = logging.getLogger(__name__)

_BATCH_SIZE = 1000  # Go: batchSize := 1000


class DataProcessor:
    """
    배치 데이터 처리기
    Go: type DataProcessor struct { db *bun.DB; interval time.Duration; t3 []T3; ... }

    각 테이블별 Python 리스트가 Go의 슬라이스 큐에 대응합니다.
    threading.Lock 이 Go의 암묵적 goroutine 분리를 보완합니다.
    """

    def __init__(self, db: DB, interval: timedelta):
        """
        Go: func NewDataProcessor(db *bun.DB, interval time.Duration) *DataProcessor

        Args:
            db        : DB 인스턴스
            interval  : 배치 처리 주기 (Go: 3초 고정이었으나 인자로 받음)
        """
        self._db = db
        self._interval = interval.total_seconds()

        # 각 테이블별 슬라이스 큐 (Go: t3 []T3 등)
        self._t3: List[T3] = []
        self._t34950: List[T34950] = []
        self._t34952: List[T34952] = []
        self._t34954: List[T34954] = []
        self._t34955: List[T34955] = []
        self._t34956: List[T34956] = []
        self._t34957: List[T34957] = []
        self._t34958: List[T34958] = []
        self._sensor_data: List[SensorData] = []

        self._lock = threading.Lock()  # Go에서는 goroutine 분리로 보호됨
        self._stop_event = threading.Event()

        # Go: go dp.startBulkProcessor()
        self._worker_thread = threading.Thread(
            target=self._start_bulk_processor,
            daemon=True,
            name="DataProcessor-BulkWorker",
        )
        self._worker_thread.start()

    def add_data(self, data: Any) -> None:
        """
        데이터를 해당 테이블 큐에 추가
        Go: func (dp *DataProcessor) AddData(data interface{}) error

        타입별 switch-case → Python isinstance() 체크로 변환
        """
        with self._lock:
            if isinstance(data, T3):
                self._t3.append(data)
            elif isinstance(data, T34950):
                self._t34950.append(data)
            elif isinstance(data, T34952):
                self._t34952.append(data)
            elif isinstance(data, T34954):
                self._t34954.append(data)
            elif isinstance(data, T34955):
                self._t34955.append(data)
            elif isinstance(data, T34956):
                self._t34956.append(data)
            elif isinstance(data, T34957):
                self._t34957.append(data)
            elif isinstance(data, T34958):
                self._t34958.append(data)
            elif isinstance(data, SensorData):
                self._sensor_data.append(data)
            else:
                raise TypeError(f"unsupported data type: {type(data)}")

    def _start_bulk_processor(self) -> None:
        """
        주기적 배치 처리 워커 (백그라운드 스레드 실행)
        Go: func (dp *DataProcessor) startBulkProcessor()

        Go의 ticker.C 채널 수신 → threading.Event.wait(timeout) 으로 변환
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._interval)
            if not self._stop_event.is_set():
                self._process_all_tables()

    def _process_all_tables(self) -> None:
        """
        모든 테이블 큐에서 배치 처리 실행
        Go: func (dp *DataProcessor) processAllTables()
        """
        with self._lock:
            # 각 리스트에서 앞부분을 추출 (Go: batch := (*data)[:batchSize])
            batches = {
                "t3":          (self._t3[:_BATCH_SIZE],       "_t3"),
                "t34950":      (self._t34950[:_BATCH_SIZE],   "_t34950"),
                "t34952":      (self._t34952[:_BATCH_SIZE],   "_t34952"),
                "t34954":      (self._t34954[:_BATCH_SIZE],   "_t34954"),
                "t34955":      (self._t34955[:_BATCH_SIZE],   "_t34955"),
                "t34956":      (self._t34956[:_BATCH_SIZE],   "_t34956"),
                "t34957":      (self._t34957[:_BATCH_SIZE],   "_t34957"),
                "t34958":      (self._t34958[:_BATCH_SIZE],   "_t34958"),
                "sensor_data": (self._sensor_data[:_BATCH_SIZE], "_sensor_data"),
            }

            # 원본 리스트에서 처리된 부분 제거 (Go: *data = (*data)[batchSize:])
            for table_name, (batch, attr) in batches.items():
                if batch:
                    current = getattr(self, attr)
                    setattr(self, attr, current[len(batch):])

        # 락 해제 후 DB INSERT 수행 (Go와 동일한 구조)
        for table_name, (batch, _) in batches.items():
            self._process_table_batch(batch, table_name)

    def _process_table_batch(self, batch: list, table_name: str) -> None:
        """
        단일 테이블 배치 INSERT 실행
        Go: func (dp *DataProcessor) processTableBatch(data interface{}, tableName string)
        """
        if not batch:
            return
        try:
            bulk_insert(self._db, table_name, batch)
        except Exception as e:
            logger.error(f"Error inserting {table_name} batch: {e}")

    def close(self) -> None:
        """
        배치 처리 워커 스레드 종료
        Go: ctx.cancel() → <-ctx.Done() 패턴에 대응
        """
        self._stop_event.set()
        self._worker_thread.join(timeout=10)
        logger.info("DataProcessor closed")
