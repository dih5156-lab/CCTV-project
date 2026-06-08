"""HTTP 멀티-플랫폼 이벤트 포워더

S-PARK_SP / D_HUB / CITY_SP 등 다수의 외부 플랫폼으로
CCTV 이벤트를 동시 전송하고 실패 시 재시도한다.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Thread
from typing import Dict, List, Optional

import requests

from ..time_utils import now_kst_iso
from .http_outbox import HttpEventOutbox

logger = logging.getLogger(__name__)

_RETRY_MAX_ATTEMPTS = 3      # 최대 재시도 횟수
_RETRY_BACKOFF_BASE = 2.0    # 지수 백오프 기반 (초)
_RETRY_QUEUE_MAX   = 500     # 재시도 큐 최대 크기


# ===========================================================================
# 전송 대상 설정
# ===========================================================================

@dataclass
class HttpEventTarget:
    """단일 HTTP 전송 대상 설정.

    속성:
        name:    플랫폼 식별자 (예: "S-PARK_SP", "D_HUB")
        url:     전송 엔드포인트 URL
        headers: 추가 HTTP 헤더 (기본 Content-Type: application/json)
        timeout: 요청 타임아웃 (초)
    """
    name: str
    url: str
    headers: Dict = field(default_factory=lambda: {"Content-Type": "application/json"})
    timeout: int = 5


# ===========================================================================
# 포워더
# ===========================================================================

class HttpEventForwarder:
    """다중 HTTP 엔드포인트로 이벤트를 전송하는 포워더.

    실패 시 재시도 큐에 넣어 백그라운드 워커가 지수 백오프로 재전송한다.
    """

    def __init__(
        self,
        targets: Optional[List[HttpEventTarget]] = None,
        outbox_db_path: Optional[str] = None,
    ):
        self._targets: List[HttpEventTarget] = list(targets or [])
        self._retry_queue: Queue = Queue(maxsize=_RETRY_QUEUE_MAX)
        self._running = False
        self._worker: Optional[Thread] = None
        self._outbox = None
        resolved_outbox = outbox_db_path or os.environ.get("ACTION_HTTP_OUTBOX_DB")
        if resolved_outbox:
            self._outbox = HttpEventOutbox(resolved_outbox)

    @property
    def has_targets(self) -> bool:
        """등록된 HTTP 전송 대상이 하나 이상 있으면 True."""
        return bool(self._targets)

    @property
    def target_count(self) -> int:
        """등록된 HTTP 전송 대상 수."""
        return len(self._targets)

    @property
    def retry_queue_size(self) -> int:
        """대기 중인 재시도 이벤트 수."""
        return self._retry_queue.qsize()

    @property
    def outbox_pending_count(self) -> int:
        """영속 HTTP outbox pending 수."""
        return self._outbox.pending_count() if self._outbox else 0

    def target_at(self, index: int) -> HttpEventTarget:
        """테스트/운영 점검용 대상 조회. 내부 리스트는 노출하지 않는다."""
        return self._targets[index]

    # ------------------------------------------------------------------

    def add_target(self, target: HttpEventTarget) -> None:
        """전송 대상을 추가한다."""
        self._targets.append(target)
        logger.info("[Forwarder] 대상 추가: %s (%s)", target.name, target.url)

    def start(self) -> None:
        """재시도 워커를 시작한다."""
        self._running = True
        self._worker = Thread(
            target=self._retry_worker, daemon=True, name="HttpRetryWorker"
        )
        self._worker.start()

    def stop(self) -> None:
        """재시도 워커를 중지한다."""
        self._running = False

    # ------------------------------------------------------------------

    def forward(self, topic: str, payload: Dict) -> None:
        """등록된 모든 대상으로 이벤트를 전송한다 (비블로킹)."""
        for target in self._targets:
            self._send(target, topic, payload, attempt=1)

    @staticmethod
    def _build_body(topic: str, payload: Dict) -> Dict:
        return {
            "topic": topic,
            "event": payload,
            "source": "edge-ai",
            "timestamp": now_kst_iso(),
        }

    def _send(
        self,
        target: HttpEventTarget,
        topic: str,
        payload: Dict,
        attempt: int,
        outbox_row_id: int = 0,
    ) -> None:
        body = self._build_body(topic, payload)
        if self._outbox is not None and not outbox_row_id:
            outbox_row_id = self._outbox.save_pending(target.name, target.url, body)
        try:
            resp = requests.post(
                target.url,
                json=body,
                headers=target.headers,
                timeout=target.timeout,
            )
            if self._is_success_status(resp.status_code):
                if self._outbox is not None and outbox_row_id:
                    self._outbox.mark_sent(outbox_row_id)
                logger.debug("[%s] 전송 성공: %s", target.name, resp.status_code)
                return
            last_error = f"http status {resp.status_code}: {resp.text[:200]}"
            logger.warning(
                "[%s] 전송 실패 (%s): %s",
                target.name, resp.status_code, resp.text[:200],
            )
        except requests.exceptions.Timeout as exc:
            last_error = str(exc)
            logger.error("[%s] 전송 타임아웃: %s", target.name, exc)
        except requests.exceptions.ConnectionError as exc:
            last_error = str(exc)
            logger.error("[%s] 연결 오류: %s", target.name, exc)
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            logger.error("[%s] HTTP 오류: %s", target.name, exc)

        if self._outbox is not None and outbox_row_id:
            self._outbox.mark_retry_failed(outbox_row_id, last_error)
        # 재시도 큐 등록
        self._enqueue_retry(target, topic, payload, attempt)

    @staticmethod
    def _is_success_status(status_code: int) -> bool:
        return status_code in (200, 201, 202)

    def _enqueue_retry(
        self,
        target: HttpEventTarget,
        topic: str,
        payload: Dict,
        attempt: int,
    ) -> None:
        if attempt >= _RETRY_MAX_ATTEMPTS:
            return
        try:
            self._retry_queue.put_nowait((target, topic, payload, attempt + 1))
        except Full:
            logger.warning("[%s] 재시도 큐 가득 참 - 드롭", target.name)

    def _retry_worker(self) -> None:
        """실패한 전송을 지수 백오프로 재시도하는 백그라운드 워커."""
        while self._running:
            try:
                item = self._retry_queue.get(timeout=1.0)
                target, topic, payload, attempt = item
                delay = _RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.debug(
                    "[%s] 재시도 대기 %.1fs (시도 %s/%s)",
                    target.name, delay, attempt, _RETRY_MAX_ATTEMPTS,
                )
                time.sleep(delay)
                self._send(target, topic, payload, attempt)
            except Empty:
                self.replay_pending_once()
                continue
            except Exception as exc:
                logger.error("[Forwarder] 재시도 워커 오류: %s", exc)

    def _target_for_outbox_row(self, row: dict) -> HttpEventTarget:
        for target in self._targets:
            if target.name == row["target_name"] and target.url == row["target_url"]:
                return target
        return HttpEventTarget(name=row["target_name"], url=row["target_url"])

    def replay_pending_once(self, *, limit: int = 100) -> int:
        """영속 outbox pending 항목을 한 번 재전송한다."""
        if self._outbox is None:
            return 0

        sent = 0
        for row in self._outbox.get_pending(limit=limit, max_retry=_RETRY_MAX_ATTEMPTS):
            target = self._target_for_outbox_row(row)
            before = self._outbox.pending_count()
            self._send(
                target,
                str(row["body"].get("topic", "")),
                row["body"].get("event", {}),
                attempt=min(row["retry_count"] + 1, _RETRY_MAX_ATTEMPTS),
                outbox_row_id=row["id"],
            )
            if self._outbox.pending_count() < before:
                sent += 1
        return sent
