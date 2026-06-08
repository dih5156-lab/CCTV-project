"""API 레이어 공통 환경변수 설정.

모든 v1 엔드포인트는 이 모듈에서 설정값을 참조한다.
"""

from __future__ import annotations

import os
from pathlib import Path

ACTION_LAYER_URL: str = os.environ.get("ACTION_LAYER_URL", "http://cctv-action-layer:8080")
ALERT_API_URL: str = os.environ.get("ALERT_API_URL", "http://cctv-alert-api:8000")
AI_ENGINE_ZONE_API_URL: str = os.environ.get("AI_ENGINE_ZONE_API_URL", "http://cctv-ai-engine:8765")
AI_ENGINE_MODEL_API_URL: str = os.environ.get("AI_ENGINE_MODEL_API_URL", "http://cctv-ai-engine:8766")
AI_ENGINE_FACE_API_URL: str = os.environ.get("AI_ENGINE_FACE_API_URL", "http://cctv-ai-engine:8767")
AI_ENGINE_STREAM_API_URL: str = os.environ.get("AI_ENGINE_STREAM_API_URL", "http://cctv-ai-engine:8769")
ALERT_LOG_PATH: Path = Path(os.environ.get("ALERT_LOG_PATH", "/app/data/logs/alert_api_events.jsonl"))
SENSOR_LOG_PATH: Path = Path(os.environ.get("SENSOR_LOG_PATH", "/app/data/logs/sensor_readings.jsonl"))
SENSOR_DEVICE_MAP_PATH: Path = Path(os.environ.get("SENSOR_DEVICE_MAP_PATH", "/app/config/sensor_devices.json"))
ALERT_FALLBACK_LOG: Path = Path(os.environ.get("ALERT_FALLBACK_LOG", "/app/data/logs/public_api_fallback.jsonl"))
CAMERAS_JSON: Path = Path(os.environ.get("CAMERAS_JSON", "/app/cameras.json"))

# 내부 서비스 간 공유 시크릿 (X-Internal-Token 헤더)
# 미설정 시 헤더를 보내지 않음 (단일 컨테이너 / 개발 환경 허용)
INTERNAL_SERVICE_TOKEN: str | None = os.environ.get("INTERNAL_SERVICE_TOKEN") or None
