"""
test_speaker.py — SpeakerDevice / _IntermClient / 헬퍼 함수 단위 테스트

전략: requests.Session을 mock 처리하여 실제 스피커 없이 InterM HTTP API
      흐름과 비즈니스 로직을 검증한다.
"""
from unittest.mock import MagicMock, patch, call

import pytest

from src.devices.speaker import (
    SpeakerConfig,
    SpeakerDevice,
    SpeakerNetworkError,
    _IntermClient,
    _snap_beam_steering,
    _BEAM_STEERING_STEPS,
    build_tts_text,
    _param_create_tts,
    _param_play,
    _param_volume,
    _param_stop,
    _is_error_response,
)


# ---------------------------------------------------------------------------
# 헬퍼 함수 테스트
# ---------------------------------------------------------------------------

class TestBuildTtsText:
    def test_known_head(self):
        text = build_tts_text("head")
        assert "안전모" in text

    def test_known_fall_detected(self):
        text = build_tts_text("fall_detected")
        assert "낙상" in text

    def test_fall_variant(self):
        text = build_tts_text("fall_something")
        assert "낙상" in text

    def test_intrusion(self):
        text = build_tts_text("intrusion")
        assert "침입" in text

    def test_critical_severity(self):
        text = build_tts_text("unknown_event", severity="critical")
        assert "위험" in text

    def test_unknown_default(self):
        text = build_tts_text("xyz123")
        assert text  # 빈 문자열 아님


class TestSnapBeamSteering:
    def test_exact_value(self):
        assert _snap_beam_steering(0) == 0
        assert _snap_beam_steering(45) == 45

    def test_rounds_to_nearest(self):
        # 7은 5와 10 사이 → 5에 더 가까움
        assert _snap_beam_steering(7) == 5
        # 8은 5와 10의 중간 → 10에 더 가까움 (abs: 3 vs 2)
        assert _snap_beam_steering(8) == 10

    def test_out_of_range_clips_to_bound(self):
        # 100은 허용 목록에 없음 → 가장 가까운 60
        assert _snap_beam_steering(100) == 60
        assert _snap_beam_steering(-100) == -60


class TestParamBuilders:
    def _cfg(self) -> SpeakerConfig:
        return SpeakerConfig(host="192.168.0.1", username="admin", password="pass")

    def test_create_tts_structure(self):
        body = _param_create_tts(self._cfg(), title="경고", text="안전모 미착용")
        assert body["Title"] == "경고"
        assert body["Text"] == "안전모 미착용"
        assert "Language" in body
        assert "Option" in body

    def test_create_tts_with_chime(self):
        cfg = SpeakerConfig(chime_begin="chime.wav", chime_end="", chime_mix=True)
        body = _param_create_tts(cfg, title="T", text="X")
        assert "Chime" in body
        assert body["Chime"]["Begin"] == "chime.wav"

    def test_create_tts_no_chime_key(self):
        cfg = SpeakerConfig()
        body = _param_create_tts(cfg, title="T", text="X")
        assert "Chime" not in body

    def test_param_volume(self):
        body = _param_volume(50)
        assert body["Volume"] == 50
        assert body["ActionType"] == "Volume"

    def test_param_play(self):
        body = _param_play("abc123", loop_count=2)
        assert body["ActionType"] == "Play"
        assert body["Play"][0]["FileHash"] == "abc123"
        assert body["Play"][0]["FileLoopCount"] == 2

    def test_param_stop(self):
        body = _param_stop()
        assert body["ActionType"] == "PlayStop"


class TestIsErrorResponse:
    def test_no_error(self):
        assert _is_error_response({"result": {"Execute": "OK"}}) is False

    def test_error_code_present(self):
        assert _is_error_response({"result": {"Error": {"code": 500}}}) is True

    def test_error_code_1210_ignored(self):
        # 1210은 볼륨 관련 무시 코드
        assert _is_error_response({"result": {"Error": {"code": 1210}}}) is False

    def test_empty_result(self):
        assert _is_error_response({"result": {}}) is False


# ---------------------------------------------------------------------------
# SpeakerConfig 테스트
# ---------------------------------------------------------------------------

class TestSpeakerConfig:
    def test_is_configured_all_fields(self):
        cfg = SpeakerConfig(host="192.168.0.1", username="admin", password="pw")
        assert cfg.is_configured is True

    def test_is_configured_missing_host(self):
        cfg = SpeakerConfig(username="admin", password="pw")
        assert cfg.is_configured is False

    def test_is_configured_missing_username(self):
        cfg = SpeakerConfig(host="192.168.0.1", password="pw")
        assert cfg.is_configured is False

    def test_is_configured_missing_password(self):
        cfg = SpeakerConfig(host="192.168.0.1", username="admin")
        assert cfg.is_configured is False

    def test_defaults(self):
        cfg = SpeakerConfig()
        assert cfg.port == 80
        assert cfg.tts_language == "kor"
        assert cfg.tts_gender == "female"


# ---------------------------------------------------------------------------
# SpeakerDevice 테스트 (HTTP mock)
# ---------------------------------------------------------------------------

def _mock_response(json_data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


class TestSpeakerDevice:
    def _make_device(self, **kwargs) -> SpeakerDevice:
        cfg = SpeakerConfig(host="192.168.0.1", username="admin", password="pw", **kwargs)
        return SpeakerDevice(cfg)

    def test_get_client_returns_none_when_unconfigured(self):
        device = SpeakerDevice(SpeakerConfig())
        assert device._get_client() is None

    def test_get_client_creates_interm_client(self):
        device = self._make_device()
        client = device._get_client()
        assert isinstance(client, _IntermClient)

    def test_get_client_cached(self):
        device = self._make_device()
        c1 = device._get_client()
        c2 = device._get_client()
        assert c1 is c2

    def test_check_ok_execute_ok(self):
        assert SpeakerDevice._check_ok({"Execute": "OK"}) is True
        assert SpeakerDevice._check_ok({"result": {"Execute": "OK"}}) is True

    def test_check_ok_not_ok(self):
        assert SpeakerDevice._check_ok({"Execute": "FAIL"}) is False
        assert SpeakerDevice._check_ok({}) is False

    @patch("src.devices.speaker.requests.request")
    def test_get_file_hash_found(self, mock_request):
        mock_request.return_value = _mock_response({
            "result": {
                "FileList": {
                    "BGM": {
                        "Internal": [
                            {"FileName": "경고.mp3", "FileHash": "hash123"},
                        ]
                    }
                }
            }
        })
        device = self._make_device()
        h = device._get_client().get_file_hash("경고")
        assert h == "hash123"

    @patch("src.devices.speaker.requests.request")
    def test_get_file_hash_not_found(self, mock_request):
        mock_request.return_value = _mock_response({
            "result": {"FileList": {"BGM": {"Internal": []}}}
        })
        device = self._make_device()
        h = device._get_client().get_file_hash("없는파일")
        assert h is None

    @patch("src.devices.speaker.requests.request")
    def test_network_error_raises_speaker_network_error(self, mock_request):
        import requests as req_lib
        mock_request.side_effect = req_lib.exceptions.ConnectionError("연결 불가")
        device = self._make_device()
        with pytest.raises(SpeakerNetworkError):
            device._get_client().create_tts("경고", "안전모 미착용")

    @patch("src.devices.speaker.requests.request")
    def test_play_calls_volume_then_play(self, mock_request):
        """play() 흐름: get_file_hash → volume 설정 → play_file 호출 순서 확인."""
        # play() 내부에서 title = f"cctv_{event_type}" → "cctv_head"
        success_resp = _mock_response({"Execute": "OK", "result": {"Execute": "OK"}})
        file_status_resp = _mock_response({
            "result": {
                "FileList": {
                    "BGM": {"Internal": [{"FileName": "cctv_head.mp3", "FileHash": "h1"}]}
                }
            }
        })
        mock_request.side_effect = [
            file_status_resp,  # get_file_hash (GET /Audio/File/Status)
            success_resp,      # control_volume (POST /Audio/Output/PlayCtrl)
            success_resp,      # play_file (POST /Audio/Output/PlayCtrl)
        ]

        device = self._make_device()
        # cleanup_old_bgm_files 백그라운드 스레드 억제
        device._bgm_cleaned = True

        result = device.play("head")
        assert result is True
        assert mock_request.call_count == 3
