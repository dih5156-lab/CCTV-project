"""대기 이벤트와 WAL 커밋이 실제 백업에서 빠지지 않는지 확인한다."""

import os
import sqlite3
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.test_device_request_durability import CASES, request

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/ops/backup_runtime_databases.sh'
DATABASES = (
    'appearances.db', 'action_events.db', 'action_mqtt_intake.db', 'event_reviews.db', 'commercial_faces.db',
    'mqtt_event_outbox.db', 'action_http_outbox.db', 'event_outbox.db',
    'edgex_command_results.db',
)


def run_backup(tmp_path, *, prune='0', adapter_dir=None):
    result = subprocess.run(
        ['bash', str(SCRIPT)], capture_output=True, text=True, timeout=30,
        env={**os.environ, 'CCTV_RUNTIME_DATA_DIR': str(tmp_path / 'data'),
             'CCTV_RUNTIME_BACKUP_DIR': str(tmp_path / 'backups'),
             'CCTV_RUNTIME_BACKUP_PRUNE': prune,
             'CCTV_EDGEX_ADAPTER_DATA_DIR': str(adapter_dir) if adapter_dir else ''},
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


@pytest.mark.parametrize('device', ['signboard', 'speaker', 'siren'])
def test_backup_preserves_device_execution_history_and_unconfirmed_claim(tmp_path, device):
    # 완료 이력뿐 아니라 실행 여부 미확정 선점도 복구해야 재출력을 막을 수 있다.
    source = tmp_path / 'data/device_requests' / f'{device}.sqlite3'
    source.parent.mkdir(parents=True)
    connection = sqlite3.connect(source)
    try:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA wal_autocheckpoint=0')
        connection.execute('CREATE TABLE device_requests(request_id TEXT PRIMARY KEY, result_json TEXT)')
        rows = [('completed', '{"status":"acknowledged"}'), ('unconfirmed', None)]
        connection.executemany('INSERT INTO device_requests VALUES (?, ?)', rows)
        connection.commit()
        backup = run_backup(tmp_path) / source.name
        assert backup.is_file(), f'missing device execution history: {device}'
        with sqlite3.connect(f'file:{backup}?mode=ro', uri=True) as restored:
            assert restored.execute('PRAGMA integrity_check').fetchone() == ('ok',)
            assert restored.execute('SELECT * FROM device_requests ORDER BY request_id').fetchall() == rows
        assert backup.stat().st_mode & 0o077 == 0
    finally:
        connection.close()


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


def test_adapter_backup_keeps_separate_outboxes_and_committed_wal(tmp_path):
    # 전용 볼륨의 동명 DB가 기존 Outbox를 덮어쓰지 않고 WAL 대기 기록도 보존해야 한다.
    adapter = tmp_path / 'adapter data'
    adapter.mkdir()
    runtime = tmp_path / 'data/runtime'
    runtime.mkdir(parents=True)
    sources = [
        (runtime / 'event_outbox.db', Path('event_outbox.db')),
        (adapter / 'event_outbox.db', Path('edgex-adapter/event_outbox.db')),
        (adapter / 'aiot_commands.db', Path('edgex-adapter/aiot_commands.db')),
    ]
    connections = []
    try:
        for source, destination in sources:
            connection = sqlite3.connect(source)
            connections.append(connection)
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA wal_autocheckpoint=0')
            connection.execute('CREATE TABLE pending(id INTEGER PRIMARY KEY, payload TEXT)')
            connection.execute('INSERT INTO pending VALUES (1, ?)', (str(destination),))
            connection.commit()
        backup = run_backup(tmp_path, adapter_dir=adapter)
        assert (backup / 'edgex-adapter').stat().st_mode & 0o077 == 0
        for connection, (_, relative) in zip(connections, sources):
            destination = backup / relative
            assert destination.is_file(), f'missing adapter snapshot: {relative}'
            with sqlite3.connect(destination.resolve().as_uri() + '?mode=ro', uri=True) as restored:
                assert restored.execute('PRAGMA integrity_check').fetchone() == ('ok',)
                assert restored.execute('SELECT * FROM pending').fetchall() == [(1, str(relative))]
            assert destination.stat().st_mode & 0o077 == 0
            assert connection.execute('SELECT * FROM pending').fetchall() == [(1, str(relative))]
    finally:
        for connection in connections:
            connection.close()


@pytest.mark.parametrize("kind,service_class,config,action,method", CASES)
def test_restored_device_history_blocks_duplicates_and_keeps_later_writes(
    tmp_path, kind, service_class, config, action, method,
):
    source = tmp_path / "data/device_requests" / f"{kind}.sqlite3"
    source.parent.mkdir(parents=True)
    original_device = MagicMock()
    original = service_class(device_id="device-1", config=config(), device=original_device,
                             request_db_path=source)
    completed_request = request(kind, action)
    completed = original.execute_request(completed_request)
    assert completed.status == "acknowledged"
    held_request = request(kind, action, request_id=f"held:{kind}:device-1", event_id="held")
    # 출력 직후 결과 저장이 실패한 상태도 백업·이행 후 재출력되면 안 된다.
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TRIGGER fail_result BEFORE UPDATE ON device_requests "
                           "BEGIN SELECT RAISE(ABORT, '결과 저장 실패'); END")
    assert original.execute_request(held_request).status == "uncertain"
    with sqlite3.connect(source) as connection:
        connection.execute("DROP TRIGGER fail_result")
        original_rows = connection.execute("SELECT * FROM device_requests ORDER BY request_id").fetchall()
    snapshot = run_backup(tmp_path) / source.name
    snapshot_bytes = snapshot.read_bytes()
    restored_path = tmp_path / "restored/device_requests" / source.name
    restored_path.parent.mkdir(parents=True)
    # 운영 DB와 분리된 새 위치에 런북의 SQLite 복원 절차를 적용한다.
    with sqlite3.connect(snapshot.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("VACUUM INTO ?", (str(restored_path),))
    restored_path.chmod(0o600)
    with sqlite3.connect(restored_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("SELECT * FROM device_requests ORDER BY request_id").fetchall() == original_rows
    restored_device = MagicMock()
    options = dict(device_id="device-1", config=config(), device=restored_device,
                   request_db_path=restored_path)
    restored = service_class(**options)
    assert restored.execute_request(completed_request) == completed
    assert restored.execute_request(held_request).error_code == "execution_unconfirmed"
    getattr(restored_device, method).assert_not_called()
    later_request = request(kind, action, request_id=f"later:{kind}:device-1", event_id="later")
    later_result = restored.execute_request(later_request)
    assert later_result.status == "acknowledged"
    # 이행 뒤 기록을 과거 백업으로 덮지 않고 유지하면 재시작 후에도 중복 실행이 차단된다.
    reopened = service_class(**options)
    assert reopened.execute_request(later_request) == later_result
    assert reopened.execute_request(held_request).error_code == "execution_unconfirmed"
    getattr(restored_device, method).assert_called_once()
    assert snapshot.read_bytes() == snapshot_bytes
    with sqlite3.connect(source) as connection:
        assert connection.execute("SELECT * FROM device_requests ORDER BY request_id").fetchall() == original_rows


@pytest.mark.parametrize('missing', ['event_outbox.db', 'aiot_commands.db'])
def test_configured_adapter_requires_both_databases_before_backup(tmp_path, missing):
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    for name in ['event_outbox.db', 'aiot_commands.db']:
        if name != missing:
            with sqlite3.connect(adapter / name) as connection:
                connection.execute('CREATE TABLE sample(id INTEGER)')
    result = subprocess.run(
        ['bash', str(SCRIPT)], capture_output=True, text=True, timeout=30,
        env={**os.environ, 'CCTV_RUNTIME_DATA_DIR': str(tmp_path / 'data'),
             'CCTV_RUNTIME_BACKUP_DIR': str(tmp_path / 'backups'),
             'CCTV_RUNTIME_BACKUP_PRUNE': '0', 'CCTV_EDGEX_ADAPTER_DATA_DIR': str(adapter)},
    )
    assert result.returncode != 0
    assert missing in result.stderr
    assert not (tmp_path / 'backups').exists()
