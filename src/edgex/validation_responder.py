"""EdgeX Core Metadata의 Device validation 요청에 응답하는 공통 모듈."""

from __future__ import annotations

import json
import logging
import os
from threading import Event, Thread
from typing import Any

import redis

logger = logging.getLogger(__name__)


def build_validation_response(envelope: dict[str, Any], request_topic: str) -> dict[str, Any]:
    """Metadata validation 요청에 대한 성공 응답 payload를 만든다."""
    return {
        "apiVersion": "",
        "receivedTopic": request_topic,
        "correlationID": envelope.get("correlationID", ""),
        "requestID": envelope.get("requestID") or envelope.get("requestId", ""),
        "errorCode": 0,
        "payload": "",
        "contentType": "application/json",
    }


def start_validation_responder(service_name: str, stop_event: Event) -> Thread:
    """Redis 메시지 버스에서 validation 요청을 받아 양식에 맞게 응답한다."""
    thread = Thread(
        target=_run_validation_responder,
        args=(service_name, stop_event),
        daemon=True,
        name=f"{service_name}-validation",
    )
    thread.start()
    return thread


def _run_validation_responder(service_name: str, stop_event: Event) -> None:
    """지정한 Device Service의 validation 요청 수신 루프를 실행한다."""
    client = None
    pubsub = None
    try:
        client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "edgex-redis"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        client.ping()
        pubsub = client.pubsub(ignore_subscribe_messages=True)
        request_channels = (
            f"edgex.{service_name}.validate.device",
            f"edgex/{service_name}/validate/device",
        )
        pubsub.subscribe(*request_channels)
        logger.info("EdgeX validation 응답기 시작: %s", service_name)
        while not stop_event.is_set():
            message = pubsub.get_message(timeout=1.0)
            if not message or message.get("type") != "message":
                continue
            try:
                envelope = json.loads(message.get("data") or "{}")
            except json.JSONDecodeError:
                continue
            response = build_validation_response(envelope, str(message.get("channel", "")))
            request_id = response["requestID"]
            if not request_id:
                continue
            body = json.dumps(response, ensure_ascii=False)
            client.publish(f"edgex.response.{service_name}.{request_id}", body)
            client.publish(f"edgex/response/{service_name}/{request_id}", body)
    except Exception as exc:
        logger.error("EdgeX validation 응답기 오류(%s): %s", service_name, exc)
    finally:
        if pubsub is not None:
            pubsub.close()
        if client is not None:
            client.close()
