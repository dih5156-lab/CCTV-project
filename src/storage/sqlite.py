"""SQLite 공통 연결/스키마 헬퍼.

서비스별 저장소가 직접 sqlite3 연결 옵션과 PRAGMA를 반복하지 않도록
가벼운 래퍼만 제공한다. 장기 실행 프로세스에서는 `connect()`를 한 번 열어
재사용하고, API 요청처럼 짧은 작업은 `session()` 컨텍스트를 사용한다.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, Optional, Sequence

_DEFAULT_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA busy_timeout=30000",
)


class SQLiteDatabase:
    """프로젝트 표준 SQLite 연결 팩토리."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        timeout: float = 30.0,
        pragmas: Optional[Sequence[str]] = None,
    ) -> None:
        self.path = Path(db_path)
        self.timeout = timeout
        self.pragmas = tuple(pragmas or _DEFAULT_PRAGMAS)
        self._schema_lock = threading.Lock()

    def connect(self, *, check_same_thread: bool = True) -> sqlite3.Connection:
        """표준 옵션이 적용된 SQLite 연결을 반환한다."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.path),
            timeout=self.timeout,
            check_same_thread=check_same_thread,
        )
        conn.row_factory = sqlite3.Row
        self.apply_pragmas(conn)
        return conn

    def apply_pragmas(self, conn: sqlite3.Connection) -> None:
        for pragma in self.pragmas:
            cursor = conn.execute(pragma)
            cursor.close()

    def initialize(
        self,
        schema: str,
        *,
        migrations: Iterable[str] = (),
        check_same_thread: bool = True,
    ) -> sqlite3.Connection:
        """스키마/마이그레이션 적용 후 연결을 반환한다."""
        conn = self.connect(check_same_thread=check_same_thread)
        with self._schema_lock:
            conn.executescript(schema)
            for statement in migrations:
                cursor = conn.execute(statement)
                cursor.close()
            conn.commit()
        return conn

    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        """짧은 작업용 연결 컨텍스트."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
