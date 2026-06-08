"""
config.py
==========
Go 원본: aiot-tlv-parser/pkg/config/config.go

애플리케이션 전체 설정값을 환경변수에서 읽어서 관리하는 모듈입니다.
Go의 struct → Python dataclass 로 변환되었습니다.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List

from config.validation import validate_config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 설정 데이터클래스 정의
# Go의 type XxxConfig struct { ... } 에 대응
# ──────────────────────────────────────────────

@dataclass
class ServerConfig:
    """
    서버 기본 설정
    Go: type ServerConfig struct { Port string; AppVersion string }
    """
    port: str = "3500"          # 환경변수 ROUTER
    app_version: str = "1.0.0"  # 환경변수 APP_VERSION


@dataclass
class DatabaseConfig:
    """
    PostgreSQL 데이터베이스 연결 설정
    Go: type DatabaseConfig struct { Host, User, Password, Database string; Port, MaxConnections int; ... }
    """
    host: str = "localhost"                         # 환경변수 DB_HOST
    port: int = 5432                                # 환경변수 DB_PORT
    user: str = "postgres"                          # 환경변수 DB_USER
    password: str = ""                              # 환경변수 DB_PW
    database: str = "aiot_sensor"                   # 환경변수 DB_NAME
    max_connections: int = 20                       # 환경변수 DB_MAX_CONNECTIONS
    idle_timeout: timedelta = timedelta(seconds=30) # 환경변수 DB_IDLE_TIMEOUT
    connect_timeout: timedelta = timedelta(seconds=2) # 환경변수 DB_CONNECT_TIMEOUT
    debug: bool = False                             # 환경변수 PGDEBUG


@dataclass
class MQTTConfig:
    """
    단일 MQTT 브로커 연결 설정
    Go: type MQTTConfig struct { Host string; Port int; Username, Password string }
    """
    host: str = "localhost"
    port: int = 1883
    username: str = ""
    password: str = ""


@dataclass
class MQTTConfigs:
    """
    모든 MQTT 브로커 설정 묶음
    Go: type MQTTConfigs struct { Proxy, NsPark, Lab, LabTest, Local MQTTConfig }

    브로커별 용도:
    - proxy   : 프록시 MQTT (PROXY_MQTT_*)
    - ns_park : 실제 LoRa 네트워크 서버 (NS_PARK_MQTT_*)
    - lab     : 랩 테스트 브로커 (LAB_MQTT_*)
    - lab_test: 추가 랩 테스트 브로커 (LAB_TEST_MQTT_*)
    - local   : 로컬 MQTT (LOCAL_MQTT_*)
    """
    proxy: MQTTConfig = field(default_factory=MQTTConfig)
    ns_park: MQTTConfig = field(default_factory=MQTTConfig)
    lab: MQTTConfig = field(default_factory=MQTTConfig)
    lab_test: MQTTConfig = field(default_factory=MQTTConfig)
    local: MQTTConfig = field(default_factory=MQTTConfig)


@dataclass
class BatchConfig:
    """
    배치 작업 설정 (디바이스 목록 주기적 갱신)
    Go: type BatchConfig struct { DeviceAPIURL string; Interval time.Duration; MaxRetries int; ... }
    """
    device_api_url: str = "http://localhost:3000/api/v1/devices"  # 환경변수 NC_API_RUI
    interval: timedelta = timedelta(hours=1)                      # 환경변수 BATCH_INTERVAL
    max_retries: int = 3                                          # 환경변수 BATCH_MAX_RETRIES
    enabled: bool = True                                          # 환경변수 BATCH_ENABLED
    application_ids: List[str] = field(default_factory=list)     # 환경변수 NC_APPLICATION_IDS (쉼표 구분)
    token: str = ""                                               # 환경변수 NC_PW
    skip_tls_verify: bool = False                                 # 환경변수 BATCH_SKIP_TLS_VERIFY


@dataclass
class Config:
    """
    최상위 설정 컨테이너
    Go: type Config struct { Server ServerConfig; Database DatabaseConfig; MQTT MQTTConfigs; Batch BatchConfig }
    """
    server: ServerConfig = field(default_factory=ServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    mqtt: MQTTConfigs = field(default_factory=MQTTConfigs)
    batch: BatchConfig = field(default_factory=BatchConfig)

    def print_config(self):
        """
        민감 정보(비밀번호)를 제외한 설정값 로그 출력
        Go: func (c *Config) PrintConfig()
        """
        logger.info("=== Configuration ===")
        logger.info(f"Server Port: {self.server.port}")
        logger.info(f"Database Host: {self.database.host}")
        logger.info(f"Database Port: {self.database.port}")
        logger.info(f"Database Name: {self.database.database}")
        logger.info(f"Database User: {self.database.user}")
        logger.info(f"Database Max Connections: {self.database.max_connections}")
        logger.info(f"Database Debug: {self.database.debug}")
        logger.info(f"MQTT Proxy: {self.mqtt.proxy.host}:{self.mqtt.proxy.port}")
        logger.info(f"MQTT NsPark: {self.mqtt.ns_park.host}:{self.mqtt.ns_park.port}")
        logger.info(f"MQTT Lab: {self.mqtt.lab.host}:{self.mqtt.lab.port}")
        logger.info(f"Batch Enabled: {self.batch.enabled}")
        logger.info(f"Batch Interval: {self.batch.interval}")
        logger.info(f"Batch Application IDs: {self.batch.application_ids}")


# ──────────────────────────────────────────────
# 환경변수 헬퍼 함수들
# Go: getEnvString / getEnvInt / getEnvBool / getEnvDuration / getEnvStringSlice
# ──────────────────────────────────────────────

def _get_env_string(key: str, default: str) -> str:
    """환경변수를 문자열로 읽기. Go: func getEnvString(key, defaultValue string) string"""
    value = os.getenv(key)
    return value if value else default


def _get_env_int(key: str, default: int) -> int:
    """환경변수를 정수로 읽기. Go: func getEnvInt(key string, defaultValue int) int"""
    value = os.getenv(key)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return default


def _get_env_bool(key: str, default: bool) -> bool:
    """환경변수를 bool로 읽기. Go: func getEnvBool(key string, defaultValue bool) bool"""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    return default


def _get_env_duration(key: str, default: timedelta) -> timedelta:
    """
    환경변수를 timedelta로 읽기.
    Go: func getEnvDuration(key string, defaultValue time.Duration) time.Duration
    지원 형식: "30s", "5m", "1h", "2h30m"
    """
    value = os.getenv(key)
    if not value:
        return default
    try:
        import re
        hours = int(re.search(r'(\d+)h', value).group(1)) if re.search(r'(\d+)h', value) else 0
        minutes = int(re.search(r'(\d+)m', value).group(1)) if re.search(r'(\d+)m', value) else 0
        seconds = int(re.search(r'(\d+)s', value).group(1)) if re.search(r'(\d+)s', value) else 0
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except Exception:
        return default


def _get_env_string_slice(key: str, default: List[str]) -> List[str]:
    """
    쉼표로 구분된 환경변수를 리스트로 읽기.
    Go: func getEnvStringSlice(key string, defaultValue []string) []string
    """
    value = os.getenv(key)
    if not value:
        return default
    parts = [p.strip().lower() for p in value.split(",")]
    return [p for p in parts if p]


def _get_mqtt_username(key: str) -> str:
    """Broker-specific username, falling back to the shared compose MQTT user."""
    return _get_env_string(key, _get_env_string("MQTT_USER", ""))


def _get_mqtt_password(key: str) -> str:
    """Broker-specific password, falling back to the shared compose MQTT password."""
    return _get_env_string(key, _get_env_string("MQTT_PASSWORD", ""))


# ──────────────────────────────────────────────
# Load 함수 (메인 진입점)
# Go: func Load() *Config
# ──────────────────────────────────────────────

def load() -> Config:
    """
    환경변수로부터 전체 설정을 로드합니다.
    Go: func Load() *Config

    Returns:
        Config: 모든 설정이 담긴 Config 인스턴스
    """
    cfg = Config(
        server=ServerConfig(
            port=_get_env_string("ROUTER", "3500"),
            app_version=_get_env_string("APP_VERSION", "1.0.0"),
        ),
        database=DatabaseConfig(
            host=_get_env_string("DB_HOST", "localhost"),
            port=_get_env_int("DB_PORT", 5432),
            user=_get_env_string("DB_USER", "postgres"),
            password=_get_env_string("DB_PW", ""),
            database=_get_env_string("DB_NAME", "aiot_sensor"),
            max_connections=_get_env_int("DB_MAX_CONNECTIONS", 20),
            idle_timeout=_get_env_duration("DB_IDLE_TIMEOUT", timedelta(seconds=30)),
            connect_timeout=_get_env_duration("DB_CONNECT_TIMEOUT", timedelta(seconds=2)),
            debug=_get_env_bool("PGDEBUG", False),
        ),
        mqtt=MQTTConfigs(
            proxy=MQTTConfig(
                host=_get_env_string("PROXY_MQTT_HOST", "localhost"),
                port=_get_env_int("PROXY_MQTT_PORT", 1883),
                username=_get_env_string("PROXY_MQTT_ID", ""),
                password=_get_env_string("PROXY_MQTT_PW", ""),
            ),
            ns_park=MQTTConfig(
                host=_get_env_string("NS_PARK_MQTT_HOST", "localhost"),
                port=_get_env_int("NS_PARK_MQTT_PORT", 1883),
                username=_get_mqtt_username("NS_PARK_MQTT_ID"),
                password=_get_mqtt_password("NC_PW"),
            ),
            lab=MQTTConfig(
                host=_get_env_string("LAB_MQTT_HOST", "localhost"),
                port=_get_env_int("LAB_MQTT_PORT", 1883),
                username=_get_mqtt_username("LAB_MQTT_ID"),
                password=_get_mqtt_password("NC_PW"),
            ),
            lab_test=MQTTConfig(
                host=_get_env_string("LAB_TEST_MQTT_HOST", "localhost"),
                port=_get_env_int("LAB_TEST_MQTT_PORT", 1883),
                username=_get_mqtt_username("LAB_TEST_MQTT_ID"),
                password=_get_mqtt_password("NC_PW"),
            ),
            local=MQTTConfig(
                host=_get_env_string("LOCAL_MQTT_HOST", "localhost"),
                port=_get_env_int("LOCAL_MQTT_PORT", 1883),
                username=_get_mqtt_username("LOCAL_MQTT_ID"),
                password=_get_mqtt_password("LOCAL_MQTT_PW"),
            ),
        ),
        batch=BatchConfig(
            device_api_url=_get_env_string("NC_API_RUI", "http://localhost:3000/api/v1/devices"),
            interval=_get_env_duration("BATCH_INTERVAL", timedelta(hours=1)),
            max_retries=_get_env_int("BATCH_MAX_RETRIES", 3),
            enabled=_get_env_bool("BATCH_ENABLED", True),
            application_ids=_get_env_string_slice("NC_APPLICATION_IDS", []),
            token=_get_env_string("NC_PW", ""),
            skip_tls_verify=_get_env_bool("BATCH_SKIP_TLS_VERIFY", False),
        ),
    )

    # 유효성 검사 (Go: cfg.Validate())
    validate_config(cfg)

    return cfg
