"""
connection.py
=============
Go 원본: aiot-tlv-parser/pkg/database/connection.go

PostgreSQL 데이터베이스 연결 풀(Connection Pool) 초기화 및 쿼리 실행 모듈입니다.
Go의 Bun ORM + database/sql → Python의 psycopg2 + connection pool 로 변환되었습니다.

사용 라이브러리:
  psycopg2        : PostgreSQL 드라이버 (Go: lib/pq)
  psycopg2.pool   : 커넥션 풀 관리 (Go: sql.SetMaxOpenConns 등)
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

try:
    import psycopg2
    import psycopg2.pool
    import psycopg2.extras
except ImportError:
    psycopg2 = None  # 테스트 환경에서도 import 가능하게 처리

logger = logging.getLogger(__name__)


_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS user_applicationids (
        user_id TEXT NOT NULL,
        application_ids TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[]
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notifications (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT,
        app_eui TEXT,
        dev_eui TEXT,
        device_id TEXT,
        object_id TEXT,
        message_id TEXT,
        created_at TIMESTAMPTZ DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS t3 (
        id BIGSERIAL PRIMARY KEY,
        app_eui TEXT,
        dev_eui TEXT,
        device_id TEXT,
        payload TEXT,
        channel INTEGER,
        frequency BIGINT,
        received_at TIMESTAMPTZ,
        manufacturer TEXT,
        model_number TEXT,
        firmware_version TEXT,
        reboot BOOLEAN,
        factory_reset BOOLEAN,
        battery_level INTEGER,
        error_code INTEGER,
        reset_error_code INTEGER,
        supported_binding_and_modes TEXT,
        hardware_version TEXT,
        battery_status INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS t34950 (
        id BIGSERIAL PRIMARY KEY,
        app_eui TEXT,
        dev_eui TEXT,
        device_id TEXT,
        created_at TIMESTAMPTZ,
        payload TEXT,
        channel INTEGER,
        frequency BIGINT,
        received_at TIMESTAMPTZ,
        water_level DOUBLE PRECISION,
        flow_velocity DOUBLE PRECISION,
        rain_fall DOUBLE PRECISION,
        reporting_period INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS t34952 (
        id BIGSERIAL PRIMARY KEY,
        app_eui TEXT,
        dev_eui TEXT,
        device_id TEXT,
        created_at TIMESTAMPTZ,
        payload TEXT,
        channel INTEGER,
        frequency BIGINT,
        received_at TIMESTAMPTZ,
        flood_level DOUBLE PRECISION,
        reporting_period INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS t34954 (
        id BIGSERIAL PRIMARY KEY,
        app_eui TEXT,
        dev_eui TEXT,
        device_id TEXT,
        created_at TIMESTAMPTZ,
        payload TEXT,
        channel INTEGER,
        frequency BIGINT,
        received_at TIMESTAMPTZ,
        temperature DOUBLE PRECISION,
        humidity DOUBLE PRECISION,
        reporting_period INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS t34955 (
        id BIGSERIAL PRIMARY KEY,
        app_eui TEXT,
        dev_eui TEXT,
        device_id TEXT,
        created_at TIMESTAMPTZ,
        payload TEXT,
        channel INTEGER,
        frequency BIGINT,
        received_at TIMESTAMPTZ,
        angle_x DOUBLE PRECISION,
        angle_y DOUBLE PRECISION,
        reporting_angle_threshold DOUBLE PRECISION,
        relative_angle_value_reset DOUBLE PRECISION,
        reporting_period INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS t34956 (
        id BIGSERIAL PRIMARY KEY,
        app_eui TEXT,
        dev_eui TEXT,
        device_id TEXT,
        created_at TIMESTAMPTZ,
        payload TEXT,
        channel INTEGER,
        frequency BIGINT,
        received_at TIMESTAMPTZ,
        fire_alarm BOOLEAN,
        reporting_period INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS t34957 (
        id BIGSERIAL PRIMARY KEY,
        app_eui TEXT,
        dev_eui TEXT,
        device_id TEXT,
        created_at TIMESTAMPTZ,
        payload TEXT,
        channel INTEGER,
        frequency BIGINT,
        received_at TIMESTAMPTZ,
        temperature DOUBLE PRECISION,
        angle_x DOUBLE PRECISION,
        angle_y DOUBLE PRECISION,
        event_code BOOLEAN
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS t34958 (
        id BIGSERIAL PRIMARY KEY,
        app_eui TEXT,
        dev_eui TEXT,
        device_id TEXT,
        created_at TIMESTAMPTZ,
        payload TEXT,
        channel INTEGER,
        frequency BIGINT,
        received_at TIMESTAMPTZ,
        acc_x DOUBLE PRECISION,
        acc_y DOUBLE PRECISION,
        acc_z DOUBLE PRECISION,
        gyro_x DOUBLE PRECISION,
        gyro_y DOUBLE PRECISION,
        gyro_z DOUBLE PRECISION,
        angle_x DOUBLE PRECISION,
        angle_y DOUBLE PRECISION,
        event_code BOOLEAN
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sensor_data (
        id BIGSERIAL PRIMARY KEY,
        app_eui TEXT,
        dev_eui TEXT,
        device_id TEXT,
        created_at TIMESTAMPTZ,
        payload TEXT,
        channel INTEGER,
        frequency BIGINT,
        received_at TIMESTAMPTZ,
        object_id TEXT,
        payload_tlv JSONB,
        is_event BOOLEAN
    )
    """,
]


def _ensure_schema(pool) -> None:
    """Create the minimal AIoT parser schema when running on a fresh Jetson DB."""
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            for statement in _SCHEMA_STATEMENTS:
                cur.execute(statement)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


class DB:
    """
    데이터베이스 연결 풀 래퍼 클래스
    Go: type DB struct { *bun.DB; Debug bool }

    Go의 *bun.DB 를 wrapping하는 방식과 동일하게,
    Python에서는 psycopg2 커넥션 풀을 내부적으로 보유합니다.
    """

    def __init__(self, pool, debug: bool = False):
        """
        Go: Init() 함수에서 생성된 DB 반환값에 대응
        직접 생성하지 말고 init() 함수를 사용하세요.
        """
        self._pool = pool
        self.debug = debug

    @contextmanager
    def _get_conn(self):
        """
        커넥션 풀에서 연결을 빌려오는 컨텍스트 매니저
        Go: defer db.Close() 패턴의 Python 대응
        """
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def close(self):
        """
        커넥션 풀 전체 종료
        Go: func (db *DB) Close() error
        """
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed")

    def execute_query(self, query: str, args: Tuple = ()) -> List[dict]:
        """
        SELECT 쿼리 실행 후 dict 리스트 반환
        Go: func (db *DB) ExecuteQuery(query string, args ...interface{}) (*sql.Rows, error)
        """
        if self.debug:
            logger.debug(f"Executing query: {query}")
            logger.debug(f"With parameters: {args}")

        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, args)
                rows = cur.fetchall()
                return [dict(row) for row in rows]

    def execute_query_row(self, query: str, args: Tuple = ()) -> Optional[dict]:
        """
        단일 행 반환 쿼리 실행
        Go: func (db *DB) ExecuteQueryRow(query string, args ...interface{}) *sql.Row
        """
        if self.debug:
            logger.debug(f"Executing query row: {query}")

        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, args)
                row = cur.fetchone()
                return dict(row) if row else None

    def execute_insert(self, query: str, args: Tuple = ()) -> int:
        """
        INSERT 쿼리 실행 후 영향받은 행 수 반환
        Go: func (db *DB) ExecuteInsert(query string, args ...interface{}) (sql.Result, error)
        """
        if self.debug:
            logger.debug(f"Executing insert: {query}")
            logger.debug(f"With parameters: {args}")

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, args)
                conn.commit()
                return cur.rowcount

    def execute_in_transaction(self, fn) -> None:
        """
        트랜잭션 내에서 함수 실행 (오류 시 자동 롤백)
        Go: func (db *DB) ExecuteInTransaction(fn func(bun.Tx) error) error
        """
        with self._get_conn() as conn:
            try:
                fn(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def health_check(self) -> None:
        """
        데이터베이스 연결 상태 확인
        Go: func (db *DB) HealthCheck() error
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

    def get_stats(self) -> dict:
        """
        커넥션 풀 통계 반환
        Go: func (db *DB) GetStats() sql.DBStats
        """
        pool = self._pool
        return {
            "min_conns": getattr(pool, "minconn", 0),
            "max_conns": getattr(pool, "maxconn", 0),
        }

    def new_insert(self):
        """
        Bun ORM의 NewInsert() 에 대응하는 헬퍼
        실제 삽입은 bulk_insert() 를 사용하세요.
        Go: db.NewInsert().Model(&batch).Exec(ctx)
        """
        return _BulkInserter(self)


class _BulkInserter:
    """
    Go의 db.NewInsert().Model(&batch).Exec(ctx) 체이닝 패턴을 모방한 헬퍼 클래스.
    실제 사용은 processor.py 에서 이루어집니다.
    """
    def __init__(self, db: DB):
        self._db = db
        self._table = None
        self._records = None

    def table(self, table_name: str) -> "_BulkInserter":
        self._table = table_name
        return self

    def records(self, data: list) -> "_BulkInserter":
        self._records = data
        return self

    def exec(self) -> int:
        if not self._table or not self._records:
            return 0
        return bulk_insert(self._db, self._table, self._records)


def bulk_insert(db: DB, table_name: str, records: list) -> int:
    """
    여러 레코드를 한 번에 INSERT (배치 삽입)
    Go: db.NewInsert().Model(&batch).Exec(dp.ctx)

    Args:
        db         : DB 인스턴스
        table_name : 대상 테이블명
        records    : dataclass 인스턴스 리스트

    Returns:
        삽입된 행 수
    """
    if not records:
        return 0

    def to_flat_dict(rec) -> dict:
        """
        dataclass → DB 삽입용 flat dict 변환.
        - sensor_data(DefaultSensorData) 중첩 필드는 최상위로 평탄화
        - dict 타입 필드(payload_tlv 등)는 JSON 문자열로 직렬화
        - datetime 타입은 그대로 유지 (psycopg2가 처리)
        """
        import dataclasses as dc
        import json as _json

        raw = dc.asdict(rec) if dc.is_dataclass(rec) else (rec if isinstance(rec, dict) else vars(rec))

        flat = {}
        for k, v in raw.items():
            if k == "sensor_data" and isinstance(v, dict):
                # DefaultSensorData 필드를 최상위로 평탄화
                flat.update(v)
            elif isinstance(v, dict):
                # payload_tlv 등 → JSON 문자열
                flat[k] = _json.dumps(v)
            else:
                flat[k] = v
        return flat

    rows = [to_flat_dict(rec) for rec in records]

    if not rows:
        return 0

    columns = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    query = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    values_list = [tuple(row[col] for col in columns) for row in rows]

    with db._get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, query, values_list)
            conn.commit()
            return cur.rowcount


# ──────────────────────────────────────────────
# init() 함수 : DB 초기화 진입점
# Go: func Init(cfg config.DatabaseConfig) (*DB, error)
# ──────────────────────────────────────────────

def init(cfg) -> DB:
    """
    DatabaseConfig 를 받아 DB 커넥션 풀을 초기화합니다.
    Go: func Init(cfg config.DatabaseConfig) (*DB, error)

    Args:
        cfg: DatabaseConfig 인스턴스

    Returns:
        DB: 초기화된 DB 인스턴스
    """
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Run: pip install psycopg2-binary")

    dsn = (
        f"host={cfg.host} "
        f"port={cfg.port} "
        f"user={cfg.user} "
        f"password={cfg.password} "
        f"dbname={cfg.database} "
        f"sslmode=disable "
        f"connect_timeout={int(cfg.connect_timeout.total_seconds())}"
    )

    # 커넥션 풀 생성
    # Go: sqldb.SetMaxOpenConns / SetMaxIdleConns 대응
    pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=max(1, cfg.max_connections // 2),
        maxconn=cfg.max_connections,
        dsn=dsn,
    )

    # 연결 테스트 (Go: sqldb.Ping())
    test_conn = pool.getconn()
    pool.putconn(test_conn)
    _ensure_schema(pool)

    logger.info(f"Database connected successfully to {cfg.host}:{cfg.port}/{cfg.database}")

    return DB(pool=pool, debug=cfg.debug)
