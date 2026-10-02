"""EdgeX request_id 중복 실행을 막는 프로세스 내 결과 캐시."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Generic, TypeVar

ResultType = TypeVar("ResultType")


class RequestIdempotencyCache(Generic[ResultType]):
    """최근 처리 결과를 request_id별로 보관해 중복 물리 동작을 막는다."""

    def __init__(self, max_entries: int = 4096) -> None:
        """캐시 최대 크기를 설정한다."""
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, ResultType] = OrderedDict()
        self._lock = Lock()

    def get(self, request_id: str) -> ResultType | None:
        """기존 request_id 결과를 최신 항목으로 갱신해 반환한다."""
        with self._lock:
            result = self._entries.get(request_id)
            if result is not None:
                self._entries.move_to_end(request_id)
            return result

    def put(self, request_id: str, result: ResultType) -> None:
        """처리 결과를 저장하고 오래된 결과부터 제거한다."""
        with self._lock:
            self._entries[request_id] = result
            self._entries.move_to_end(request_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
