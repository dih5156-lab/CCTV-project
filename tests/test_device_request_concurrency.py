"""동시 재수신·응답 유실 재전송이 물리 출력을 중복시키지 않는지 검증한다."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.devices.signboard import SignboardConfig
from src.devices.siren import SensorConfig
from src.devices.speaker import SpeakerConfig
from src.edgex.dabit_device_service import DabitDeviceService
from src.edgex.request_idempotency import RequestIdempotencyCache
from src.edgex.siren_device_service import SirenDeviceService
from src.edgex.speaker_device_service import SpeakerDeviceService


@pytest.mark.parametrize("kind,service_class,config,action,method", [
    ("signboard", DabitDeviceService, SignboardConfig, "display", "display"),
    ("speaker", SpeakerDeviceService, SpeakerConfig, "play", "play"),
    ("siren", SirenDeviceService, SensorConfig, "trigger", "trigger"),
])
@pytest.mark.parametrize("success", [True, False])
def test_concurrent_request_and_lost_reply_execute_once(kind, service_class, config, action, method, success):
    entered, repeated, release, second_started = Event(), Event(), Event(), Event()

    def physical_call(*args, **kwargs):
        if entered.is_set():
            repeated.set()
        entered.set()
        assert release.wait(3)
        return success

    device = MagicMock()
    getattr(device, method).side_effect = physical_call
    service = service_class(device_id="device-1", config=config(), device=device)
    request = {"request_id": "same-request", "event_id": "event-1", "device": kind,
               "device_id": "device-1", "action": action, "payload": {"text": "경고"}}

    def duplicate():
        second_started.set()
        return service.execute_request(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.execute_request, request)
        try:
            assert entered.wait(2)
            second = pool.submit(duplicate)
            assert second_started.wait(2)
            assert not repeated.wait(0.15), "같은 요청이 첫 실행 완료 전에 다시 장치를 호출함"
        finally:
            release.set()
        assert first.result(timeout=2) == second.result(timeout=2)
    # 첫 응답을 잃어 같은 ID로 재전송해도 성공 결과만 재사용한다.
    getattr(device, method).assert_called_once()
    assert first.result().status == ("acknowledged" if success else "failed")
    if success:
        assert service.execute_request(request) == first.result()
        getattr(device, method).assert_called_once()
    else:
        # 동시에 기다리던 실패는 공유하지만 완료 후 명시적인 복구 요청은 막지 않는다.
        getattr(device, method).side_effect = None
        getattr(device, method).return_value = True
        assert service.execute_request(request).status == "acknowledged"
        assert getattr(device, method).call_count == 2


def test_siren_exception_reports_selected_device():
    first, second = MagicMock(), MagicMock()
    second.trigger.side_effect = RuntimeError("통신 실패")
    service = SirenDeviceService(device_id="siren-1", config=SensorConfig(), device=first,
                                 devices={"siren-1": first, "siren-2": second})
    result = service.execute_request({"request_id": "event-1:siren:siren-2", "event_id": "event-1",
        "device": "siren", "device_id": "siren-2", "action": "trigger", "payload": {}})
    assert result.status == "failed"
    assert result.device_id == "siren-2"


def test_slow_request_does_not_block_other_request_and_exception_releases_slot():
    cache = RequestIdempotencyCache()
    entered, release = Event(), Event()

    def fail():
        entered.set()
        assert release.wait(3)
        raise RuntimeError("실행 예외")

    success = SimpleNamespace(status="acknowledged")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.run, "slow", fail)
        try:
            assert entered.wait(2)
            assert pool.submit(cache.run, "other", lambda: success).result(timeout=1) == success
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="실행 예외"):
            first.result(timeout=2)
    assert cache.run("slow", lambda: success) == success


def test_success_cache_remains_bounded():
    cache = RequestIdempotencyCache(max_entries=2)
    execute = MagicMock(return_value=SimpleNamespace(status="acknowledged"))
    for request in ["one", "two", "two", "three", "three", "one"]:
        cache.run(request, execute)
    assert execute.call_count == 4
