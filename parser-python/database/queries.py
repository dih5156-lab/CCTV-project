"""
queries.py
==========
Go 원본: aiot-tlv-parser/pkg/database/queries.go

센서 데이터 조회 쿼리 서비스 모듈입니다.
Go의 pq.Array → Python의 psycopg2 배열 처리로 변환되었습니다.
"""

import logging
from typing import Optional

from database.models import QueryResult
from database.connection import DB

logger = logging.getLogger(__name__)


class QueryService:
    """
    센서 데이터 조회 서비스
    Go: type QueryService struct { db *DB }
    """

    def __init__(self, db: DB):
        """
        Go: func NewQueryService(db *DB) *QueryService
        """
        self._db = db

    def get_user_id_by_app_eui(self) -> QueryResult:
        """
        앱 EUI별 사용자 ID 목록 조회
        Go: func (qs *QueryService) GetUserIDByAppEUI() (*QueryResult, error)

        SQL:
            SELECT
                unnested_application_id AS application_id,
                ARRAY_AGG(user_id) AS user_ids
            FROM
                user_applicationids,
                UNNEST(application_ids) AS unnested_application_id
            GROUP BY
                unnested_application_id

        Returns:
            QueryResult: application_id, user_ids(list) 를 포함하는 행 목록
        """
        query = """
            SELECT
                unnested_application_id AS application_id,
                ARRAY_AGG(user_id) AS user_ids
            FROM
                user_applicationids,
                UNNEST(application_ids) AS unnested_application_id
            GROUP BY
                unnested_application_id
        """

        try:
            rows = self._db.execute_query(query)
            return QueryResult(rows=rows)
        except Exception as e:
            logger.error(f"Failed to get user ID by app EUI: {e}")
            raise
