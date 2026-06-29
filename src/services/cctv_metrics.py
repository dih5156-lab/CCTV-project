"""CCTV Action Bridge — Prometheus 메트릭 레지스트리.

Action Bridge 서비스의 주요 지표를 Prometheus Counter/Gauge로 정의한다.
별도 CollectorRegistry를 사용해 기본 레지스트리와의 충돌을 방지한다.

import 위치:
    - src/services/action_bridge.py             (이벤트·가동 카운터 증감)
    - src/services/_action_bridge_support.py    (장치 명령 카운터 증감)
    - src/protocols/rest.py                     (GET /metrics 라우트)
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge

# 독립 레지스트리 — 기본 REGISTRY와 충돌 없이 재임포트 가능
REGISTRY: CollectorRegistry = CollectorRegistry()

# ── 이벤트 수신 ──────────────────────────────────────────────────────────────

mqtt_events_received: Counter = Counter(
    "cctv_mqtt_events_received_total",
    "수신된 MQTT 메시지 총 수",
    ["topic_prefix"],
    registry=REGISTRY,
)
"""topic_prefix: MQTT 토픽의 첫 두 세그먼트 (예: 'cctv/rules', 'aiot/rules')."""

events_handled: Counter = Counter(
    "cctv_events_handled_total",
    "처리된 이벤트 총 수",
    ["mode"],  # auto | manual
    registry=REGISTRY,
)

# ── 장치 명령 ────────────────────────────────────────────────────────────────

device_commands: Counter = Counter(
    "cctv_device_commands_total",
    "장치 명령 발송 총 수",
    ["device", "status"],  # device: speaker|signboard|siren, status: ok|skip|error
    registry=REGISTRY,
)

rest_events_dropped: Counter = Counter(
    "cctv_rest_events_dropped_total",
    "REST action queue 포화로 거부된 이벤트 총 수",
    ["reason"],  # queue_full
    registry=REGISTRY,
)

# ── 상태 게이지 ──────────────────────────────────────────────────────────────

pending_events: Gauge = Gauge(
    "cctv_pending_events",
    "수동 승인 대기 중인 이벤트 수",
    registry=REGISTRY,
)

action_bridge_up: Gauge = Gauge(
    "cctv_action_bridge_up",
    "Action Bridge 가동 여부 (1=정상, 0=비정상)",
    registry=REGISTRY,
)

rest_action_queue_depth: Gauge = Gauge(
    "cctv_rest_action_queue_depth",
    "REST action worker 대기 큐에 남아 있는 이벤트 수",
    registry=REGISTRY,
)
