"""프로젝트 공통 KST 시간 유틸리티 테스트."""

from __future__ import annotations

import pytest

from src.time_utils import coerce_timestamp_seconds, now_kst_iso, timestamp_to_kst_iso


def test_now_kst_iso_has_korean_timezone_offset() -> None:
    assert now_kst_iso().endswith("+09:00")


def test_timestamp_to_kst_iso_converts_epoch_to_korean_timezone() -> None:
    assert timestamp_to_kst_iso(0) == "1970-01-01T09:00:00+09:00"


def test_coerce_timestamp_seconds_accepts_seconds_milliseconds_and_iso() -> None:
    assert coerce_timestamp_seconds(1700000000.5) == pytest.approx(1700000000.5)
    assert coerce_timestamp_seconds(1700000000500) == pytest.approx(1700000000.5)
    assert coerce_timestamp_seconds("2026-05-06T01:55:27.452483Z") == pytest.approx(
        1778032527.452483
    )


def test_coerce_timestamp_seconds_uses_fallback() -> None:
    assert coerce_timestamp_seconds(None, "1700000000") == pytest.approx(1700000000.0)
    assert coerce_timestamp_seconds("", 1700000000500) == pytest.approx(1700000000.5)
    assert coerce_timestamp_seconds("bad", "2026-05-06T01:55:27.452483+00:00") == pytest.approx(
        1778032527.452483
    )


def test_coerce_timestamp_seconds_rejects_non_finite_numbers() -> None:
    assert coerce_timestamp_seconds(float("nan")) == 0.0
    assert coerce_timestamp_seconds(float("inf")) == 0.0
    assert coerce_timestamp_seconds(float("nan"), 1700000000) == pytest.approx(1700000000.0)
