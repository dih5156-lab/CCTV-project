"""
batch/manager.py
================
Go 원본: aiot-tlv-parser/pkg/batch/manager.go

배치 작업들을 관리하는 매니저 모듈입니다.
Go의 goroutine + WaitGroup → Python의 threading.Thread로 변환되었습니다.
"""

import logging
import threading
import time
from typing import List, Dict, Optional

from batch.devices_batch import DeviceScheduler, SchedulerConfig, DeviceUpdateCallback

logger = logging.getLogger(__name__)


class BatchManager:
    """
    배치 작업 매니저
    Go: type BatchManager struct { jobs []BatchJob; ctx context.Context; cancel context.CancelFunc; ... }

    현재 지원하는 배치 작업:
      - DeviceScheduler: 디바이스 목록 주기적 갱신
    """

    def __init__(self, cfg, device_update_callback: DeviceUpdateCallback):
        """
        Go: func NewBatchManager(cfg config.BatchConfig, deviceUpdateCallback DeviceUpdateCallback) *BatchManager

        Args:
            cfg                   : BatchConfig 인스턴스
            device_update_callback: 디바이스 목록 갱신 시 호출할 콜백
        """
        self._cfg = cfg
        self._device_update_callback = device_update_callback
        self._jobs: List[DeviceScheduler] = []
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def init(self) -> None:
        """
        배치 작업 초기화 및 시작
        Go: func (bm *BatchManager) Init() error

        application_ids 가 있으면 DeviceScheduler 를 생성합니다.
        """
        logger.info("Initializing batch jobs...")
        jobs: List[DeviceScheduler] = []

        # Go: if len(bm.cfg.ApplicationIds) > 0 { jobs = append(jobs, NewDeviceScheduler(...)) }
        if self._cfg.application_ids:
            scheduler = DeviceScheduler(
                config=SchedulerConfig(
                    api_url=self._cfg.device_api_url,
                    interval=self._cfg.interval,
                    max_retries=self._cfg.max_retries,
                    enabled=self._cfg.enabled,
                    application_ids=self._cfg.application_ids,
                    token=self._cfg.token,
                    skip_tls_verify=self._cfg.skip_tls_verify,
                ),
                callback=self._device_update_callback,
            )
            jobs.append(scheduler)

        self._jobs = jobs

        # 모든 배치 작업 시작 (Go: go func(batchJob BatchJob) { ... }(job))
        for job in self._jobs:
            def run_job(j=job):
                logger.info(f"Starting batch job: {j.get_name()}")
                try:
                    j.start()
                except Exception as e:
                    logger.error(f"Failed to start job '{j.get_name()}': {e}")
                    return

                # 종료 신호 대기 (Go: <-bm.ctx.Done())
                self._stop_event.wait()
                j.stop()
                logger.info(f"Stopped batch job: {j.get_name()}")

            t = threading.Thread(target=run_job, daemon=True, name=f"BatchJob-{job.get_name()}")
            t.start()
            self._threads.append(t)

        logger.info(f"Initialized {len(self._jobs)} batch jobs")

    def stop_all(self) -> None:
        """
        모든 배치 작업 종료
        Go: func (bm *BatchManager) StopAll()

        Go의 bm.cancel() + bm.wg.Wait() → Event.set() + Thread.join() 으로 변환
        30초 타임아웃 적용
        """
        logger.info("Stopping all batch jobs...")
        self._stop_event.set()

        # 최대 30초 대기 (Go: time.After(30 * time.Second))
        deadline = time.time() + 30
        for t in self._threads:
            remaining = deadline - time.time()
            if remaining > 0:
                t.join(timeout=remaining)

        if any(t.is_alive() for t in self._threads):
            logger.warning("Timeout waiting for batch jobs to stop")
        else:
            logger.info("All batch jobs stopped successfully")

    def get_status(self) -> Dict[str, bool]:
        """
        모든 배치 작업 실행 상태 반환
        Go: func (bm *BatchManager) GetStatus() map[string]bool
        """
        return {job.get_name(): job.is_running() for job in self._jobs}
