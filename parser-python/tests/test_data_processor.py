"""Sensor batch retention while the process stays alive; no real DB or worker."""

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import processor as module  # noqa: E402
from database.models import T34954, SensorData  # noqa: E402


@pytest.fixture
def processor(monkeypatch):
    with monkeypatch.context() as creation:
        creation.setattr(module.threading.Thread, "start", lambda self: None)
        return module.DataProcessor(object(), timedelta(seconds=3))


def test_failed_batch_is_retained_and_retried_before_new_records(
    processor, monkeypatch
):
    first, second = T34954(temperature=1), T34954(temperature=2)
    processor.add_data(first)
    insert = Mock(side_effect=RuntimeError("DB unavailable"))
    monkeypatch.setattr(module, "bulk_insert", insert)
    processor._process_all_tables()
    processor.add_data(second)
    assert processor._t34954 == [first, second]

    insert.reset_mock(side_effect=True)
    processor._process_all_tables()
    insert.assert_called_once_with(processor._db, "t34954", [first, second])
    assert processor._t34954 == []


def test_one_failed_table_does_not_requeue_successful_table(processor, monkeypatch):
    temperature, raw = T34954(), SensorData()
    processor.add_data(temperature)
    processor.add_data(raw)

    def insert(_db, table, _batch):
        if table == "t34954":
            raise RuntimeError("typed table failed")
        return 1

    monkeypatch.setattr(module, "bulk_insert", insert)
    processor._process_all_tables()
    assert processor._t34954 == [temperature]
    assert processor._sensor_data == []


def test_batch_limit_and_zero_rowcount_still_acknowledge_success(
    processor, monkeypatch
):
    records = [T34954(temperature=value) for value in range(module._BATCH_SIZE + 1)]
    for record in records:
        processor.add_data(record)
    insert = Mock(return_value=0)
    monkeypatch.setattr(module, "bulk_insert", insert)
    processor._process_all_tables()
    assert insert.call_args.args[2] == records[: module._BATCH_SIZE]
    assert processor._t34954 == records[module._BATCH_SIZE :]
    processor._process_all_tables()
    assert processor._t34954 == []


@pytest.mark.parametrize("fail", [False, True])
def test_input_can_append_during_insert_without_being_removed(
    processor, monkeypatch, fail
):
    original, arriving = T34954(temperature=1), T34954(temperature=2)
    processor.add_data(original)

    def insert(_db, _table, _batch):
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(processor.add_data, arriving).result(timeout=2)
        if fail:
            raise RuntimeError("insert failed after input arrived")
        return 1

    monkeypatch.setattr(module, "bulk_insert", insert)
    processor._process_all_tables()
    assert processor._t34954 == ([original, arriving] if fail else [arriving])


def test_overlapping_flushes_do_not_insert_the_same_batch_twice(processor, monkeypatch):
    processor.add_data(T34954())
    insert_entered, allow_commit, second_started = Event(), Event(), Event()
    batches = []

    def insert(_db, _table, batch):
        batches.append(batch)
        insert_entered.set()
        assert allow_commit.wait(timeout=3)
        return len(batch)

    def flush_again():
        second_started.set()
        processor._process_all_tables()

    monkeypatch.setattr(module, "bulk_insert", insert)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(processor._process_all_tables)
        try:
            assert insert_entered.wait(timeout=2)
            second = executor.submit(flush_again)
            assert second_started.wait(timeout=2)
        finally:
            allow_commit.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert len(batches) == 1
    assert processor._t34954 == []
