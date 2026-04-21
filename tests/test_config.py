"""
test_config.py — AppConfig / ModelPaths / DetectionConfig / EnvOverride 단위 테스트

전략: 파일시스템 패치와 환경변수 딕셔너리 주입으로 실제 파일 없이 설정 로직을 검증한다.
"""
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.config.config import (
    AppConfig,
    ModelPaths,
    DetectionConfig,
    ActionBridgeConfig,
    EdgeXConfig,
    ExternalIngestConfig,
    MqttConfig,
    _parse_bool,
    _identity,
    EnvOverride,
)


# ---------------------------------------------------------------------------
# 파서 함수
# ---------------------------------------------------------------------------

class TestParsers:
    @pytest.mark.parametrize("val,expected", [
        ("1", True), ("true", True), ("yes", True), ("on", True),
        ("True", True), ("YES", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("", False), ("anything", False),
    ])
    def test_parse_bool(self, val, expected):
        assert _parse_bool(val) is expected

    def test_identity_returns_unchanged(self):
        assert _identity("hello world") == "hello world"
        assert _identity("") == ""


# ---------------------------------------------------------------------------
# DetectionConfig 유효성 검증
# ---------------------------------------------------------------------------

class TestDetectionConfig:
    def test_defaults_valid(self):
        cfg = DetectionConfig()
        assert 0.0 <= cfg.helmet_confidence <= 1.0
        assert cfg.device in ("cpu", "cuda") or cfg.device.startswith("cuda:")

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="헬멧"):
            DetectionConfig(helmet_confidence=1.5)

    def test_invalid_person_confidence_raises(self):
        with pytest.raises(ValueError, match="사람"):
            DetectionConfig(person_confidence=-0.1)

    def test_invalid_iou_raises(self):
        with pytest.raises(ValueError, match="IoU"):
            DetectionConfig(iou_threshold=2.0)

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError, match="FPS"):
            DetectionConfig(target_fps=0)

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="장치"):
            DetectionConfig(device="gpu")

    def test_cuda_n_device_ok(self):
        cfg = DetectionConfig(device="cuda:0")
        assert cfg.device == "cuda:0"


# ---------------------------------------------------------------------------
# ModelPaths 경로 해석
# ---------------------------------------------------------------------------

class TestModelPaths:
    def test_resolve_path_finds_existing(self, tmp_path):
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake model")

        result = ModelPaths._resolve_path("test", [model_file])
        assert result == str(model_file)

    def test_resolve_path_fallback_when_none_exist(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.pt"
        fallback_str = "fallback.pt"

        result = ModelPaths._resolve_path("test", [nonexistent, fallback_str])
        assert result == fallback_str

    def test_resolve_path_prefers_first_existing(self, tmp_path):
        first = tmp_path / "first.pt"
        second = tmp_path / "second.pt"
        first.write_bytes(b"m1")
        second.write_bytes(b"m2")

        result = ModelPaths._resolve_path("test", [first, second])
        assert result == str(first)

    def test_resolve_path_returns_none_when_all_path_objects(self, tmp_path):
        nonexistent = tmp_path / "none.pt"
        # Path 객체만이고 존재하지 않으면 fallback=None (str이 없음)
        result = ModelPaths._resolve_path("test", [nonexistent])
        assert result is None


# ---------------------------------------------------------------------------
# AppConfig ENV 오버라이드
# ---------------------------------------------------------------------------

class TestAppConfigEnvOverrides:
    def test_mqtt_broker_override(self):
        env = {"MQTT_BROKER": "mqtt.example.com"}
        cfg = AppConfig.from_env(env)
        assert cfg.mqtt.broker == "mqtt.example.com"

    def test_mqtt_port_parsed_as_int(self):
        env = {"MQTT_PORT": "1884"}
        cfg = AppConfig.from_env(env)
        assert cfg.mqtt.port == 1884
        assert isinstance(cfg.mqtt.port, int)

    def test_display_flag_override_true(self):
        env = {"DISPLAY_ENABLED": "true"}
        cfg = AppConfig.from_env(env)
        assert cfg.display is True

    def test_display_flag_override_false(self):
        env = {"DISPLAY_ENABLED": "0"}
        cfg = AppConfig.from_env(env)
        assert cfg.display is False

    def test_zone_detection_override(self):
        env = {"ZONE_DETECTION_ENABLED": "yes"}
        cfg = AppConfig.from_env(env)
        assert cfg.zone_detection is True

    def test_device_override(self):
        env = {"DEVICE": "cuda:0"}
        cfg = AppConfig.from_env(env)
        assert cfg.detection.device == "cuda:0"

    def test_edgex_metadata_url_override(self):
        env = {"EDGEX_METADATA_URL": "http://edgex-host:59881"}
        cfg = AppConfig.from_env(env)
        assert cfg.edgex.metadata_url == "http://edgex-host:59881"

    def test_action_mqtt_broker_override(self):
        env = {"ACTION_MQTT_BROKER": "action-broker"}
        cfg = AppConfig.from_env(env)
        assert cfg.action.mqtt_broker == "action-broker"

    def test_external_mqtt_topics_override(self):
        env = {"EXTERNAL_MQTT_TOPICS": "factory/#, camera/1"}
        cfg = AppConfig.from_env(env)
        assert cfg.external_ingest.topics == ("factory/#", "camera/1")

    def test_external_mqtt_client_id_override(self):
        env = {"EXTERNAL_MQTT_CLIENT_ID": "my-fixed-client"}
        cfg = AppConfig.from_env(env)
        assert cfg.external_ingest.mqtt_client_id == "my-fixed-client"

    def test_external_republish_override(self):
        env = {"EXTERNAL_REPUBLISH_ENABLED": "true"}
        cfg = AppConfig.from_env(env)
        assert cfg.external_ingest.republish_enabled is True

    def test_no_env_returns_defaults(self):
        cfg = AppConfig.from_env({})
        assert cfg.mqtt.broker == "localhost"
        assert cfg.mqtt.port == 1883
        assert cfg.display is False

    def test_appearance_backend_override(self):
        env = {
            "APPEARANCE_BACKEND": "pphuman",
            "APPEARANCE_MODEL_PATH": "models/pphuman.onnx",
            "APPEARANCE_LABEL_MAP_PATH": "config/appearance_pphuman_labels.example.json",
            "APPEARANCE_RUNTIME": "auto",
            "APPEARANCE_SCORE_THRESHOLD": "0.6",
            "APPEARANCE_BBOX_EXPAND_RATIO": "0.2",
        }
        cfg = AppConfig.from_env(env)
        assert cfg.appearance.backend == "pphuman"
        assert cfg.appearance.model_path == "models/pphuman.onnx"
        assert cfg.appearance.label_map_path == "config/appearance_pphuman_labels.example.json"
        assert cfg.appearance.runtime == "auto"
        assert cfg.appearance.score_threshold == pytest.approx(0.6)
        assert cfg.appearance.bbox_expand_ratio == pytest.approx(0.2)

    def test_post_init_initializes_all_sub_configs(self):
        cfg = AppConfig()
        assert cfg.models is not None
        assert cfg.mqtt is not None
        assert cfg.camera is not None
        assert cfg.detection is not None
        assert cfg.events is not None
        assert cfg.processing is not None
        assert cfg.edgex is not None
        assert cfg.action is not None
        assert cfg.external_ingest is not None


# ---------------------------------------------------------------------------
# 기타 서브 데이터클래스
# ---------------------------------------------------------------------------

class TestSubConfigs:
    def test_action_bridge_config_defaults(self):
        cfg = ActionBridgeConfig()
        assert cfg.mqtt_port == 1883
        assert cfg.rest_port == 8080

    def test_edgex_config_defaults(self):
        cfg = EdgeXConfig()
        assert "59881" in cfg.metadata_url
        assert cfg.mqtt_port == 1883

    def test_mqtt_config_defaults(self):
        cfg = MqttConfig()
        assert cfg.qos == 0
        assert cfg.retain is False
        assert "cctv" in cfg.topic_prefix

    def test_external_ingest_config_defaults(self):
        cfg = ExternalIngestConfig()
        assert cfg.mqtt_port == 1883
        assert cfg.topics == ("#",)
        assert cfg.republish_enabled is False
