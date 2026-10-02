"""Persist one sensor packet and its idempotency receipt in one PostgreSQL commit."""

import hashlib
import json
from dataclasses import asdict, fields
from datetime import datetime

from psycopg2 import sql
from psycopg2.extras import Json

from database import models

RECORD_TYPES = {
    "t3": models.T3,
    "t34950": models.T34950,
    "t34952": models.T34952,
    "t34954": models.T34954,
    "t34955": models.T34955,
    "t34956": models.T34956,
    "t34957": models.T34957,
    "t34958": models.T34958,
    "sensor_data": models.SensorData,
}


class ProjectionConflict(ValueError):
    pass


def encode_records(records: list) -> str:
    result = {}
    for record in records:
        table = next(
            name for name, model in RECORD_TYPES.items() if type(record) is model
        )
        row = asdict(record)
        row.update(row.pop("sensor_data", {}))
        result.setdefault(table, []).append(row)

    def encode(value):
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError(f"Unsupported projection value: {type(value).__name__}")

    return json.dumps(result, default=encode, sort_keys=True, allow_nan=False)


def ensure_projection_schema(db) -> None:
    def create(connection):
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sensor_projection_receipts (
                    ingest_id TEXT NOT NULL,
                    projection_version TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    committed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (ingest_id, projection_version)
                )
            """
            )

    db.execute_in_transaction(create)


def write_sensor_projection(db, ingest_id: str, projection_json: str) -> None:
    records = json.loads(projection_json)
    if (
        not isinstance(records, dict)
        or "sensor_data" not in records
        or len(records) != 2
    ):
        raise ValueError(
            "Projection must contain one typed row and one sensor_data row"
        )
    for table, rows in records.items():
        if table not in RECORD_TYPES or not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("Unsupported sensor projection table or row count")
        allowed = {field.name for field in fields(RECORD_TYPES[table])} - {
            "sensor_data"
        }
        if table != "t3":
            allowed.update(field.name for field in fields(models.DefaultSensorData))
        if not isinstance(rows[0], dict) or not rows[0] or not set(rows[0]) <= allowed:
            raise ValueError("Unsupported sensor projection columns")
    fingerprint = hashlib.sha256(projection_json.encode()).hexdigest()

    def write(connection):
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL lock_timeout = '2s'")
            cursor.execute("SET LOCAL statement_timeout = '5s'")
            cursor.execute(
                "INSERT INTO sensor_projection_receipts "
                "(ingest_id, projection_version, payload_sha256) VALUES (%s, 'sensor-v1', %s) "
                "ON CONFLICT DO NOTHING RETURNING ingest_id",
                (ingest_id, fingerprint),
            )
            if cursor.fetchone() is None:
                cursor.execute(
                    "SELECT payload_sha256 FROM sensor_projection_receipts "
                    "WHERE ingest_id=%s AND projection_version='sensor-v1'",
                    (ingest_id,),
                )
                if cursor.fetchone()[0] != fingerprint:
                    raise ProjectionConflict(
                        "Same ingest ID has a different stored projection"
                    )
                return
            for table, rows in records.items():
                row = rows[0]
                query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(table),
                    sql.SQL(", ").join(map(sql.Identifier, row)),
                    sql.SQL(", ").join(sql.Placeholder() for _ in row),
                )
                cursor.execute(
                    query,
                    [
                        Json(value) if isinstance(value, dict) else value
                        for value in row.values()
                    ],
                )

    db.execute_in_transaction(write)
