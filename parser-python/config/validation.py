"""
validation.py
=============
Go 원본: aiot-tlv-parser/pkg/config/validation.go

Config 설정값의 유효성을 검사하는 모듈입니다.
Go의 메서드 수신자(receiver) 방식 → Python 함수 방식으로 변환되었습니다.
"""

from urllib.parse import urlparse


# ──────────────────────────────────────────────
# 유효성 검사 함수들
# Go: func (c *Config) Validate() error
#     func (s *ServerConfig) Validate() error  등...
# ──────────────────────────────────────────────

def validate_server_config(server) -> None:
    """
    서버 설정 유효성 검사
    Go: func (s *ServerConfig) Validate() error
    """
    if not server.port:
        raise ValueError("server port is required")
    try:
        int(server.port)
    except ValueError:
        raise ValueError(f"invalid server port: {server.port}")


def validate_database_config(database) -> None:
    """
    데이터베이스 설정 유효성 검사
    Go: func (d *DatabaseConfig) Validate() error
    """
    if not database.host:
        raise ValueError("database host is required")
    if not database.user:
        raise ValueError("database user is required")
    if not database.database:
        raise ValueError("database name is required")
    if not (0 < database.port <= 65535):
        raise ValueError(f"invalid database port: {database.port}")
    if database.max_connections <= 0:
        raise ValueError(f"max connections must be positive: {database.max_connections}")
    if database.idle_timeout.total_seconds() < 0:
        raise ValueError(f"idle timeout must be non-negative: {database.idle_timeout}")
    if database.connect_timeout.total_seconds() < 0:
        raise ValueError(f"connect timeout must be non-negative: {database.connect_timeout}")


def validate_mqtt_config(mqtt_config, name: str) -> None:
    """
    단일 MQTT 브로커 설정 유효성 검사
    Go: func (m *MQTTConfig) Validate(name string) error
    """
    if not mqtt_config.host:
        raise ValueError(f"{name} MQTT host is required")
    if not (0 < mqtt_config.port <= 65535):
        raise ValueError(f"invalid {name} MQTT port: {mqtt_config.port}")


def validate_mqtt_configs(mqtt_configs) -> None:
    """
    전체 MQTT 브로커 설정 유효성 검사
    Go: func (m *MQTTConfigs) Validate() error
    """
    configs = {
        "Proxy":   mqtt_configs.proxy,
        "NsPark":  mqtt_configs.ns_park,
        "Lab":     mqtt_configs.lab,
        "LabTest": mqtt_configs.lab_test,
        "Local":   mqtt_configs.local,
    }
    for name, config in configs.items():
        validate_mqtt_config(config, name)


def validate_batch_config(batch) -> None:
    """
    배치 작업 설정 유효성 검사
    Go: func (b *BatchConfig) Validate() error
    """
    if batch.enabled and not batch.device_api_url:
        raise ValueError("batch device API URL is required when batch is enabled")
    if batch.max_retries < 0:
        raise ValueError(f"batch max retries must be non-negative: {batch.max_retries}")
    if batch.interval.total_seconds() <= 0:
        raise ValueError(f"batch interval must be positive: {batch.interval}")

    # API URL 형식 검사
    if batch.device_api_url:
        parsed = urlparse(batch.device_api_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"invalid batch device API URL: {batch.device_api_url}")


def validate_config(cfg) -> None:
    """
    전체 Config 유효성 검사 진입점
    Go: func (c *Config) Validate() error
    """
    try:
        validate_server_config(cfg.server)
    except ValueError as e:
        raise ValueError(f"server config validation failed: {e}")

    try:
        validate_database_config(cfg.database)
    except ValueError as e:
        raise ValueError(f"database config validation failed: {e}")

    try:
        validate_mqtt_configs(cfg.mqtt)
    except ValueError as e:
        raise ValueError(f"mqtt config validation failed: {e}")

    try:
        validate_batch_config(cfg.batch)
    except ValueError as e:
        raise ValueError(f"batch config validation failed: {e}")
