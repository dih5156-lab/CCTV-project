"""대기 이벤트와 WAL 커밋이 실제 백업에서 빠지지 않는지 확인한다."""

import os
import sqlite3
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/ops/backup_runtime_databases.sh'
DATABASES = (
    'appearances.db', 'action_events.db', 'event_reviews.db', 'commercial_faces.db',
    'mqtt_event_outbox.db', 'action_http_outbox.db', 'event_outbox.db',
    'edgex_command_results.db',
)


def run_backup(tmp_path, *, prune='0'):
    result = subprocess.run(
        ['bash', str(SCRIPT)], capture_output=True, text=True, timeout=30,
        env={**os.environ, 'CCTV_RUNTIME_DATA_DIR': str(tmp_path / 'data'),
             'CCTV_RUNTIME_BACKUP_DIR': str(tmp_path / 'backups'),
             'CCTV_RUNTIME_BACKUP_PRUNE': prune},
    )
    assert result.returncode == 0, result.stderr
    return sorted(p for p in (tmp_path / 'backups').iterdir() if p.name != 'older')[-1]


def test_backup_preserves_all_runtime_queues_and_committed_wal(tmp_path):
    runtime = tmp_path / 'data/runtime'
    runtime.mkdir(parents=True)
    connections = []
    try:
        for name in DATABASES:
            connection = sqlite3.connect(runtime / name)
            connections.append(connection)
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA wal_autocheckpoint=0')
            connection.execute('CREATE TABLE pending(event_id TEXT PRIMARY KEY, payload TEXT)')
            connection.execute('INSERT INTO pending VALUES (?, ?)', ('event-1', name))
            connection.commit()
        # 쓰는 연결을 열어 두어 DB 본체 복사로는 빠질 수 있는 WAL 커밋을 시험한다.
        backup = run_backup(tmp_path)
        for name in DATABASES:
            assert (backup / name).is_file(), f'missing durable queue: {name}'
            with sqlite3.connect(f'file:{backup / name}?mode=ro', uri=True) as restored:
                assert restored.execute('PRAGMA integrity_check').fetchone() == ('ok',)
                assert restored.execute('SELECT * FROM pending').fetchall() == [('event-1', name)]
    finally:
        for connection in connections:
            connection.close()


def test_rehearsal_can_preserve_existing_backups(tmp_path):
    old = tmp_path / 'backups/older'
    old.mkdir(parents=True)
    sentinel = old / 'evidence.txt'
    sentinel.write_text('keep this earlier backup')
    past = time.time() - 30 * 86400
    os.utime(old, (past, past))
    run_backup(tmp_path, prune='0')
    assert sentinel.read_text() == 'keep this earlier backup'


def test_new_backup_is_private_even_with_permissive_parent_umask(tmp_path):
    runtime = tmp_path / 'data/runtime'
    runtime.mkdir(parents=True)
    with sqlite3.connect(runtime / 'appearances.db') as connection:
        connection.execute('CREATE TABLE sample(id INTEGER)')
    previous = os.umask(0o022)
    try:
        backup = run_backup(tmp_path)
    finally:
        os.umask(previous)
    assert backup.stat().st_mode & 0o077 == 0
    assert (backup / 'appearances.db').stat().st_mode & 0o077 == 0


def test_corrupt_source_does_not_leave_a_completed_backup(tmp_path):
    runtime = tmp_path / 'data/runtime'
    runtime.mkdir(parents=True)
    (runtime / 'appearances.db').write_bytes(b'not a SQLite database')
    result = subprocess.run(
        ['bash', str(SCRIPT)], capture_output=True, text=True, timeout=30,
        env={**os.environ, 'CCTV_RUNTIME_DATA_DIR': str(tmp_path / 'data'),
             'CCTV_RUNTIME_BACKUP_DIR': str(tmp_path / 'backups'),
             'CCTV_RUNTIME_BACKUP_PRUNE': '0'},
    )
    assert result.returncode != 0
    assert not list((tmp_path / 'backups').rglob('*.db'))
    assert not list((tmp_path / 'backups').rglob('*.partial'))


def test_enabled_retention_still_removes_expired_fixture_backup(tmp_path):
    old = tmp_path / 'backups/older'
    old.mkdir(parents=True)
    (old / 'evidence.txt').write_text('expired fixture')
    past = time.time() - 30 * 86400
    os.utime(old, (past, past))
    run_backup(tmp_path, prune='1')
    assert not old.exists()
