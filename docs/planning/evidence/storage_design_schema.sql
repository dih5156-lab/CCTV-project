-- P1.2 design prototype only. Not a production migration.
PRAGMA foreign_keys = ON;
CREATE TABLE ingest_receipts (
    ingest_id TEXT PRIMARY KEY NOT NULL,
    source_namespace TEXT NOT NULL,
    device_id TEXT,
    received_at TEXT NOT NULL,
    raw_format TEXT NOT NULL CHECK (raw_format IN ('json', 'bytes_base64', 'error_repr')),
    raw_payload TEXT NOT NULL,
    raw_sha256 TEXT NOT NULL CHECK (length(raw_sha256) = 64),
    mode TEXT NOT NULL CHECK (mode IN ('live', 'replay', 'import'))
);
CREATE TABLE canonical_records (
    source_namespace TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    data_kind TEXT NOT NULL CHECK (data_kind IN ('observation', 'event')),
    device_id TEXT,
    observed_at TEXT,
    processed_at TEXT,
    source_sha256 TEXT NOT NULL,
    canonical_json TEXT NOT NULL,
    first_ingest_id TEXT NOT NULL REFERENCES ingest_receipts(ingest_id),
    PRIMARY KEY (source_namespace, source_record_id)
);
CREATE TABLE ingest_revisions (
    ingest_id TEXT NOT NULL REFERENCES ingest_receipts(ingest_id),
    revision INTEGER NOT NULL CHECK (revision > 0),
    contract_version TEXT NOT NULL,
    normalizer_version TEXT NOT NULL,
    validation_status TEXT NOT NULL CHECK (
        validation_status IN ('valid', 'invalid', 'unsupported', 'duplicate', 'conflict')
    ),
    normalized_payload TEXT,
    error_json TEXT,
    record_namespace TEXT,
    record_id TEXT,
    PRIMARY KEY (ingest_id, revision),
    CHECK ((record_namespace IS NULL) = (record_id IS NULL)),
    FOREIGN KEY (record_namespace, record_id)
        REFERENCES canonical_records(source_namespace, source_record_id)
);
CREATE TABLE delivery_jobs (
    job_id TEXT PRIMARY KEY NOT NULL,
    source_namespace TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    destination TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'inflight', 'delivered', 'expired', 'quarantined')
    ),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_attempt_at TEXT,
    lease_until TEXT,
    payload_json TEXT NOT NULL,
    UNIQUE (source_namespace, source_record_id, destination),
    FOREIGN KEY (source_namespace, source_record_id)
        REFERENCES canonical_records(source_namespace, source_record_id)
);
CREATE TABLE legacy_import_map (
    source_store TEXT NOT NULL,
    table_name TEXT NOT NULL,
    legacy_key TEXT NOT NULL,
    ingest_id TEXT NOT NULL REFERENCES ingest_receipts(ingest_id),
    PRIMARY KEY (source_store, table_name, legacy_key)
);
CREATE INDEX receipt_device_time ON ingest_receipts(device_id, received_at, ingest_id);
CREATE INDEX canonical_device_time ON canonical_records(device_id, observed_at, source_record_id);
CREATE INDEX delivery_pending ON delivery_jobs(status, next_attempt_at, job_id);
