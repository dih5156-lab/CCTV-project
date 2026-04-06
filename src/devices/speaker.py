"""InterM 스피커 디바이스 컨트롤러

[참고] 인수인계 자료의 Control-Speaker 코드(api.py / communicate.py / control.py / parameter.py)
       를 CCTV 프로젝트 구조에 맞게 통합한 모듈이다.

동작 흐름 (start_broadcast):
  ① get_filehash  - 기존 BGM 파일 해시 확인
  ② (없으면) make_file
      a. POST /interm-api/TTS/Create    - TTS 텍스트 → 임시 TTS 파일 생성
      b. GET  /interm-api/TTS/Status    - TTS ID 조회
      c. POST /interm-api/TTS/ToBGM     - TTS → BGM 변환
      d. Timer(1s): POST /TTS/Remove    - 임시 TTS 정리
  ③ POST /interm-api/Audio/Output/PlayCtrl (Volume)
  ④ POST /interm-api/Audio/Output/PlayCtrl (Play)
"""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import requests
from requests.auth import HTTPDigestAuth

logger = logging.getLogger(__name__)


# ===========================================================================
# 설정 DTO
# ===========================================================================

@dataclass
class SpeakerConfig:
    """InterM 스피커 연결 및 TTS 설정.

    속성:
        host:             스피커 IP 주소 (예: "192.168.0.200")
        port:             HTTP 포트 (기본 80)
        username:         인증 사용자명
        password:         인증 비밀번호
        volume:           재생 볼륨 (0~200)
        tts_language:     TTS 언어 코드 ("kor", "eng" 등)
        tts_gender:       TTS 성별 ("female", "male")
        tts_pitch:        TTS 피치 (0~400, 기본 100)
        tts_speed:        TTS 속도 (0~400, 기본 100)
        tts_volume:       TTS 볼륨 (0~200)
        sentence_pause:   문장 사이 멈춤 (ms)
        comma_pause:      쉼표 사이 멈춤 (ms)
        chime_begin:      시작 차임 파일명 (없으면 빈 문자열)
        chime_end:        종료 차임 파일명 (없으면 빈 문자열)
        chime_mix:        차임과 TTS 혼합 여부
        connect_timeout:  연결 타임아웃 (초)
        read_timeout:     읽기 타임아웃 (초)
    """
    host: str = ""
    port: int = 80
    username: str = ""
    password: str = ""
    volume: int = 1
    tts_language: str = "kor"
    tts_gender: str = "female"
    tts_pitch: int = 100
    tts_speed: int = 100
    tts_volume: int = 1
    sentence_pause: int = 200
    comma_pause: int = 200
    chime_begin: str = ""
    chime_end: str = ""
    chime_mix: bool = True
    connect_timeout: int = 3
    read_timeout: int = 7

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.username and self.password)


# ===========================================================================
# TTS 안내 문구
# ===========================================================================

_TTS_MESSAGES: Dict[str, str] = {
    "head":              "안전모 미착용이 감지되었습니다. 즉시 착용 바랍니다.",
    "fall_detected":     "낙상이 감지되었습니다. 즉시 확인 바랍니다.",
    "danger_zone":       "위험 구역 침입이 감지되었습니다. 즉시 대피 바랍니다.",
    "intrusion":         "위험 구역 침입이 감지되었습니다. 즉시 대피 바랍니다.",
    "tilt_alert":        "기울기 이상이 감지되었습니다. 즉시 현장을 확인 바랍니다.",
    "temperature_alert": "온도 이상이 감지되었습니다. 즉시 현장을 확인 바랍니다.",
    "vibration_alert":   "진동 충격이 감지되었습니다. 즉시 현장을 확인 바랍니다.",
    "sensor_fault":      "센서 이상이 감지되었습니다. 장비 상태를 확인 바랍니다.",
    "critical":          "중요 위험 이벤트가 감지되었습니다.",
}

def build_tts_text(event_type: str, severity: str = "") -> str:
    """이벤트 타입·심각도에 따른 TTS 안내 문구를 반환한다."""
    msg = _TTS_MESSAGES.get(event_type.lower())
    if msg:
        return msg
    if "fall" in event_type.lower():
        return _TTS_MESSAGES["fall_detected"]
    if severity.lower() == "critical":
        return _TTS_MESSAGES["critical"]
    return "안전 이벤트가 감지되었습니다."


# ===========================================================================
# InterM API 파라미터 빌더 (인수인계 parameter.py 통합)
# ===========================================================================

def _param_header() -> Dict:
    return {"Content-Type": "application/json"}


def _param_create_tts(cfg: SpeakerConfig, title: str, text: str) -> Dict:
    """POST /TTS/Create 요청 바디."""
    body: Dict = {
        "Title":    title,
        "Text":     text,
        "Language": cfg.tts_language,
        "Gender":   cfg.tts_gender,
        "Option": {
            "Pitch":         cfg.tts_pitch,
            "Speed":         cfg.tts_speed,
            "Volume":        cfg.tts_volume,
            "SentencePause": cfg.sentence_pause,
            "CommaPause":    cfg.comma_pause,
        },
        "Storage": "internal",
    }
    if cfg.chime_begin or cfg.chime_end:
        body["Chime"] = {
            "Begin": cfg.chime_begin,
            "End":   cfg.chime_end,
            "Mix":   cfg.chime_mix,
        }
    return body


def _param_control_tts(tts_id: str) -> Dict:
    """POST /TTS/ToBGM, /TTS/Remove 요청 바디."""
    return {"ID": [tts_id]}


def _param_volume(volume: int) -> Dict:
    """POST /Audio/Output/PlayCtrl (Volume) 요청 바디."""
    return {
        "CHIndex":  1,
        "PlayType": "FilePlay",
        "ActionType": "Volume",
        "Volume":   volume,
    }


def _param_play(file_hash: str, loop_count: int = 1) -> Dict:
    """POST /Audio/Output/PlayCtrl (Play) 요청 바디."""
    return {
        "CHIndex":   1,
        "PlayType":  "FilePlay",
        "ActionType": "Play",
        "Play": [{"FileHash": file_hash, "FileLoopCount": loop_count}],
    }


def _param_stop() -> Dict:
    """POST /Audio/Output/PlayCtrl (Stop) 요청 바디."""
    return {
        "CHIndex":   1,
        "PlayType":  "FilePlay",
        "ActionType": "PlayStop",
    }


# 전원 제어 엔드포인트 (InterM API 11. Power)
_POWER_CTRL_PATH      = "/System/Power"
# 펜웨어 업데이트 엔드포인트 (InterM API 13. FirmwareUpdate)
_FIRMWARE_UPDATE_PATH = "/System/FirmwareUpdate"
# 수동 방송 제어 엔드포인트 (InterM API 26. Broadcast/Manual)
_BROADCAST_MANUAL_PATH = "/Controller/Broadcast/Manual"
# DSP 입력 설정 엔드포인트 (InterM API 9. DSP/Input)
_DSP_INPUT_PATH        = "/Audio/DSP/Input"
# DSP 출력 설정 엔드포인트 (InterM API 10. DSP/Output)
_DSP_OUTPUT_PATH       = "/Audio/DSP/Output"

# BeamSteering 허용 값 (Laser 모델 전용)
_BEAM_STEERING_STEPS: tuple[int, ...] = (
    -60, -45, -30, -25, -20, -15, -10, -5,
    0, 5, 10, 15, 20, 25, 30, 45, 60,
)


def _snap_beam_steering(value: int) -> int:
    """임의 정수를 BeamSteering 허용 목록에서 가장 가까운 값으로 스냅한다."""
    return min(_BEAM_STEERING_STEPS, key=lambda v: abs(v - value))


# ---------------------------------------------------------------------------
# DSP/Input 파라미터 빌더
# ---------------------------------------------------------------------------

def _param_dsp_input(mode: str, gain: Optional[float] = None) -> Dict:
    """POST /Audio/DSP/Input (IP-SPEAKER) 요청 바디.

    매개변수:
        mode: 접점 타입 ("Condenser" 또는 "Disabled")
        gain: MICGain (Mode=Condenser 일 때만 유효, -12 ~ 55.25 Float)
    """
    body: Dict = {"Mode": mode}
    if gain is not None and mode == "Condenser":
        body["Gain"] = round(float(gain), 3)
    return body


# ---------------------------------------------------------------------------
# DSP/Output 파라미터 빌더
# ---------------------------------------------------------------------------

def _param_dsp_output_speaker(
    is_mute: Optional[bool] = None,
    volume:  Optional[int]  = None,
) -> Dict:
    """POST /Audio/DSP/Output (IP-SPEAKER) 요청 바디.

    매개변수:
        is_mute: 음소거 여부 (True=Mute On, False=Mute Off)
        volume:  볼륨 값 (0~100)
    """
    body: Dict = {}
    if is_mute is not None:
        body["IsMute"] = bool(is_mute)
    if volume is not None:
        body["Volume"] = max(0, min(100, int(volume)))
    return body


def _param_dsp_output_laser(
    output_mode:   Optional[str]  = None,
    delay:         Optional[int]  = None,
    volume:        Optional[int]  = None,
    is_mute:       Optional[bool] = None,
    beam_steering: Optional[int]  = None,
) -> Dict:
    """POST /Audio/DSP/Output (Laser 모델) 요청 바디.

    매개변수:
        output_mode:   오디오 방송 타입 ("Analog" 또는 "Digital")
        delay:         출력 딜레이 (ms, 0~680)
        volume:        출력 볼륨 (dB, -80~20)
        is_mute:       음소거 여부
        beam_steering: 출력 각도, 허용 목록 외 값은 가장 가까운 값으로 스냅
                       (-60, -45, -30, -25, -20, -15, -10, -5,
                         0, 5, 10, 15, 20, 25, 30, 45, 60)
    """
    body: Dict = {}
    if output_mode   is not None:
        body["OutputMode"]    = output_mode
    if delay is not None:
        body["Delay"]         = max(0, min(680, int(delay)))
    if volume is not None:
        body["Volume"]        = max(-80, min(20, int(volume)))
    if is_mute is not None:
        body["IsMute"]        = bool(is_mute)
    if beam_steering is not None:
        body["BeamSteering"]  = _snap_beam_steering(int(beam_steering))
    return body


# ---------------------------------------------------------------------------
# Broadcast/Manual 파라미터 빌더
# ---------------------------------------------------------------------------

def _param_broadcast_start(source_id: int, zone_list: list[str]) -> Dict:
    """POST /Controller/Broadcast/Manual (Start) 요청 바디.

    매개변수:
        source_id: 방송 소스 ID (GET /Controller/Source/List 에서 확인)
        zone_list: 방송 대상 존 리스트
                   예) ["all"], ["z1", "z2"], ["g1"], ["z1-z4"], ["g2-g3"]
    """
    return {
        "Action": "Start",
        "Start":  {"SourceID": source_id, "ZoneList": zone_list},
    }


def _param_broadcast_stop(zone_list: list[str]) -> Dict:
    """POST /Controller/Broadcast/Manual (Stop) 요청 바디.

    매개변수:
        zone_list: 중지할 존 리스트
    """
    return {
        "Action": "Stop",
        "Stop":   {"ZoneList": zone_list},
    }


def _param_broadcast_all_stop() -> Dict:
    """POST /Controller/Broadcast/Manual (AllStop) 요청 바디."""
    return {"Action": "AllStop"}


def _param_broadcast_volume(zone_id: int, volume: int) -> Dict:
    """POST /Controller/Broadcast/Manual (Volume) 요청 바디.

    매개변수:
        zone_id: 볼륨 제어 대상 존 ID
        volume:  볼륨 값 (0~100)
    """
    return {
        "Action": "Volume",
        "Volume": {"ZoneID": zone_id, "Volume": max(0, min(100, volume))},
    }


def _param_power(method: str) -> Dict:
    """POST /System/Power 요청 바디.

    매개변수:
        method: 전원 설정 방법 (예: "On", "Off", "Reboot")
    """
    return {"Method": method}


# ===========================================================================
# 커스텀 예외
# ===========================================================================

class SpeakerNetworkError(Exception):
    """스피커 디바이스 네트워크 오류 (연결 불가·타임아웃)."""


# ===========================================================================
# InterM HTTP 클라이언트 (인수인계 communicate.py + api.py 통합)
# ===========================================================================

class _IntermClient:
    """InterM 디바이스 HTTP 클라이언트.

    인수인계 자료의 RequestCommunicator + Interm 클래스를 단일 클래스로 통합.
    Digest 인증 사용.
    """

    def __init__(self, cfg: SpeakerConfig):
        self._cfg = cfg
        self._base = f"http://{cfg.host}/interm-api"
        self._auth = HTTPDigestAuth(cfg.username, cfg.password)
        self._timeout = (cfg.connect_timeout, cfg.read_timeout)

    def _request(self, method: str, path: str, **kwargs) -> Dict:
        url = f"{self._base}{path}"
        try:
            resp = requests.request(
                method, url, headers=_param_header(),
                auth=self._auth, timeout=self._timeout, **kwargs,
            )
            resp.raise_for_status()
            return resp.json()
        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            logger.warning("[InterM] %s %s 네트워크 오류 (스피커 오프라인?): %s", method, path, exc)
            raise SpeakerNetworkError(str(exc)) from exc
        except Exception as exc:
            logger.error("[InterM] %s %s 오류: %s", method, path, exc)
            raise

    def _post(self, path: str, body: Dict) -> Dict:
        return self._request("POST", path, json=body)

    def _get(self, path: str) -> Dict:
        return self._request("GET", path)

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def create_tts(self, title: str, text: str) -> Dict:
        """TTS 파일을 생성한다. POST /TTS/Create"""
        body = _param_create_tts(self._cfg, title, text)
        return self._post("/TTS/Create", body)

    def get_tts_id(self, title: str) -> Optional[str]:
        """생성된 TTS의 ID를 조회한다. GET /TTS/Status"""
        resp = self._get("/TTS/Status")
        for item in resp.get("result", {}).get("FileList", []):
            if item.get("Title") == title:
                return item.get("ID")
        return None

    def convert_tts_to_bgm(self, tts_id: str) -> Dict:
        """TTS → BGM(WAV) 변환. POST /TTS/ToBGM"""
        return self._post("/TTS/ToBGM", _param_control_tts(tts_id))

    def remove_tts(self, tts_id: str) -> None:
        """임시 TTS 파일을 제거한다. POST /TTS/Remove"""
        try:
            self._post("/TTS/Remove", _param_control_tts(tts_id))
        except Exception:
            pass  # 정리 실패는 무시

    def _get_bgm_internal_list(self) -> list:
        """/Audio/File/Status 에서 BGM Internal 목록을 반환한다 (공유 파싱 헬퍼)."""
        resp = self._get("/Audio/File/Status")
        return (
            resp.get("result", {})
                .get("FileList", {})
                .get("BGM", {})
                .get("Internal", [])
        )

    def list_bgm_files(self) -> list:
        """내부 BGM 파일 목록을 반환한다. GET /Audio/File/Status"""
        try:
            return self._get_bgm_internal_list()
        except Exception:
            return []

    def remove_bgm_files(self, file_hashes: list) -> None:
        """BGM 파일을 일괄 삭제한다. POST /Audio/File/Remove"""
        if not file_hashes:
            return
        try:
            body = {
                "Type": "BGM",
                "FileList": [{"FileHash": int(h)} for h in file_hashes],
            }
            self._post("/Audio/File/Remove", body)
        except Exception:
            pass  # 정리 실패는 무시

    # ------------------------------------------------------------------
    # 오디오 파일
    # ------------------------------------------------------------------

    def get_file_hash(self, title: str) -> Optional[str]:
        """BGM 파일의 해시를 조회한다. GET /Audio/File/Status"""
        resp = self._get("/Audio/File/Status")
        result = resp.get("result", {})
        # 오류 응답 검사
        if "Error" in result:
            err = result["Error"]
            if err.get("code") != 1210:  # 1210 = 음량 조절 관련 무시 코드
                logger.warning("[InterM] get_file_hash 오류 응답: %s", err)
                return None
        for f in result.get("FileList", {}).get("BGM", {}).get("Internal", []):
            name = f.get("FileName", "")
            if name in (f"{title}.mp3", f"{title}.wav", f"TTS_{title}.wav"):
                return f.get("FileHash")
        return None

    def control_volume(self, volume: int) -> Dict:
        """볼륨을 조절한다. POST /Audio/Output/PlayCtrl"""
        return self._post("/Audio/Output/PlayCtrl", _param_volume(volume))

    def play_file(self, file_hash: str, loop_count: int = 1) -> Dict:
        """오디오 파일을 재생한다. POST /Audio/Output/PlayCtrl"""
        return self._post("/Audio/Output/PlayCtrl", _param_play(file_hash, loop_count))

    def stop_file(self) -> Dict:
        """재생을 중지한다. POST /Audio/Output/PlayCtrl"""
        return self._post("/Audio/Output/PlayCtrl", _param_stop())


# ===========================================================================
# 응답 검증
# ===========================================================================

def _is_error_response(resp: Dict) -> bool:
    """InterM API 오류 응답 여부를 반환한다."""
    result = resp.get("result", {})
    if isinstance(result, dict):
        err = result.get("Error", {})
        code = err.get("code") if isinstance(err, dict) else None
        return code is not None and code != 1210  # 1210 = 볼륨 무시 코드
    return False


# ===========================================================================
# SpeakerDevice (인수인계 control.py Speaker 클래스 통합)
# ===========================================================================

class SpeakerDevice:
    """InterM 스피커 디바이스 컨트롤러.

    인수인계 자료의 Speaker 클래스 로직을 그대로 사용하며,
    CCTV 프로젝트 이벤트 구조에 맞는 공개 인터페이스를 추가한다.
    """

    def __init__(self, config: SpeakerConfig):
        self.config = config
        self._client: Optional[_IntermClient] = None
        self._bgm_cleaned: bool = False

    def _get_client(self) -> Optional[_IntermClient]:
        """클라이언트 인스턴스를 반환한다 (설정이 없으면 None)."""
        if not self.config.is_configured:
            logger.warning("[Speaker] host/username/password 미설정 - 스피커 비활성화")
            return None
        if self._client is None:
            self._client = _IntermClient(self.config)
        return self._client

    @staticmethod
    def _check_ok(resp: Dict) -> bool:
        """InterM API Execute==OK 성공 응답 여부를 반환한다."""
        return (
            resp.get("Execute") == "OK"
            or resp.get("result", {}).get("Execute") == "OK"
        )

    def _get_setting(self, path: str, label: str) -> Optional[Dict]:
        """GET 요청 → result 딕셔너리 반환 공통 헬퍼."""
        client = self._get_client()
        if client is None:
            return None
        try:
            resp = client._get(path)
            if resp.get("code") == 200:
                return resp.get("result", {})
            logger.error("[Speaker] %s 오류 응답: %s", label, resp)
            return None
        except Exception as exc:
            logger.error("[Speaker] %s 오류: %s", label, exc, exc_info=True)
            return None

    def _post_setting(self, path: str, body: Dict, label: str) -> bool:
        """POST 요청 → Execute==OK 확인 공통 헬퍼."""
        client = self._get_client()
        if client is None:
            return False
        try:
            resp = client._post(path, body)
            if self._check_ok(resp):
                logger.info("[Speaker] %s 완료: %s", label, body)
                return True
            logger.error("[Speaker] %s 실패: %s", label, resp)
            return False
        except Exception as exc:
            logger.error("[Speaker] %s 오류: %s", label, exc, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # 내부 흐름 (인수인계 control.py make_file 로직)
    # ------------------------------------------------------------------

    def _make_file(self, client: _IntermClient, title: str, text: str) -> bool:
        """TTS 생성 → BGM 변환 흐름을 실행한다.

        ① POST /TTS/Create
        ② sleep(2)  ← 디바이스 처리 대기
        ③ GET  /TTS/Status   → tts_id
        ④ POST /TTS/ToBGM
        ⑤ Timer(1s): POST /TTS/Remove  ← 임시 TTS 정리
        """
        try:
            resp = client.create_tts(title, text)
            if _is_error_response(resp):
                logger.error("[Speaker] TTS 생성 실패: %s", resp)
                return False
            logger.debug("[Speaker] TTS 생성 완료: title=%r", title)

            time.sleep(2)  # 디바이스 처리 대기 (원본 코드 동일)

            tts_id = client.get_tts_id(title)
            if not tts_id:
                logger.error("[Speaker] TTS ID 조회 실패: title=%r", title)
                return False

            resp = client.convert_tts_to_bgm(tts_id)
            if _is_error_response(resp):
                logger.error("[Speaker] TTS→BGM 변환 실패: %s", resp)
                return False
            logger.debug("[Speaker] TTS→BGM 변환 완료: id=%s", tts_id)

            # 임시 TTS 파일 비동기 정리 (1초 후)
            threading.Timer(1.0, client.remove_tts, args=(tts_id,)).start()
            return True

        except SpeakerNetworkError as exc:
            logger.warning("[Speaker] 스피커 오프라인으로 _make_file 건너뜀: %s", exc)
            return False
        except Exception as exc:
            logger.error("[Speaker] _make_file 오류: %s", exc, exc_info=True)
            return False

    # ------------------------------------------------------------------
    def cleanup_old_bgm_files(self) -> int:
        """기동 시 축적된 구형 CCTV BGM 파일(타임스탬프 포함)을 정리한다."""
        client = self._get_client()
        if client is None:
            return 0
        files = client.list_bgm_files()
        # 구형 파일: TTS_cctv_<camera>_<event>_<timestamp>.wav 패턴
        old_pattern = re.compile(r'TTS_cctv_.*_\d{9,}\.wav$')
        to_delete = [
            f["FileHash"] for f in files
            if old_pattern.search(f.get("FileName", ""))
        ]
        if not to_delete:
            logger.info("[Speaker] 정리 대상 구형 BGM 파일 없음")
            return 0
        logger.info("[Speaker] 구형 BGM 파일 %d개 정리 시작", len(to_delete))
        batch_size = 50
        deleted = 0
        for i in range(0, len(to_delete), batch_size):
            batch = to_delete[i : i + batch_size]
            client.remove_bgm_files(batch)
            deleted += len(batch)
        logger.info("[Speaker] 구형 BGM 파일 %d개 정리 완료", deleted)
        return deleted

    # 공개 API
    # ------------------------------------------------------------------

    def play(self, event_type: str, severity: str = "", camera_id: str = "") -> bool:
        """CCTV 이벤트에 대한 TTS 방송을 실행한다.

        인수인계 자료의 Speaker.start_broadcast() 흐름:
          ① 파일 해시 조회
          ② (없으면) 파일 생성
          ③ 볼륨 설정
          ④ 파일 재생
        """
        client = self._get_client()
        if client is None:
            return False

        # 첫 호출 시 구형 BGM 파일 백그라운드 정리
        if not self._bgm_cleaned:
            self._bgm_cleaned = True
            threading.Thread(target=self.cleanup_old_bgm_files, daemon=True).start()

        text = build_tts_text(event_type, severity)
        # event_type별 고정 title → 파일 1회 생성 후 재사용 (타임스탬프 제거)
        title = f"cctv_{event_type}"
        logger.info("[Speaker] 방송 시작: title=%r, text=%r", title, text)

        try:
            # ① 기존 BGM 파일 해시 확인
            file_hash = client.get_file_hash(title)

            # ② 파일 없으면 생성
            if not file_hash:
                ok = self._make_file(client, title, text)
                if not ok:
                    return False
                file_hash = client.get_file_hash(title)

            if not file_hash:
                logger.error("[Speaker] BGM 파일 해시 조회 실패: title=%r", title)
                return False

            # ③ 볼륨 설정
            resp = client.control_volume(self.config.volume)
            if _is_error_response(resp):
                logger.warning("[Speaker] 볼륨 설정 실패 (무시): %s", resp)

            # ④ 재생
            resp = client.play_file(file_hash, loop_count=1)
            if _is_error_response(resp):
                logger.error("[Speaker] 재생 실패: %s", resp)
                return False

            logger.info("[Speaker] 방송 성공: title=%r", title)
            return True

        except SpeakerNetworkError:
            logger.warning(
                "[Speaker] 스피커 오프라인 (%s:%s) - 방송 건너뜀",
                self.config.host, self.config.port,
            )
            return False
        except Exception as exc:
            logger.error("[Speaker] play() 오류: %s", exc, exc_info=True)
            return False

    def stop(self) -> bool:
        """현재 방송을 중지한다."""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.stop_file()
            logger.info("[Speaker] 방송 중지")
            return True
        except Exception as exc:
            logger.error("[Speaker] stop() 오류: %s", exc)
            return False

    # ------------------------------------------------------------------
    # 전원 제어
    # ------------------------------------------------------------------

    def _power_ctrl(self, method: str) -> bool:
        """전원 제어 공통 내부 메서드."""
        client = self._get_client()
        if client is None:
            return False
        try:
            resp = client._post(_POWER_CTRL_PATH, _param_power(method))
            if resp.get("Execute") != "OK":
                logger.error("[Speaker] 전원 %s 실패: %s", method, resp)
                return False
            logger.info("[Speaker] 전원 %s 성공", method)
            return True
        except requests.exceptions.ConnectTimeout:
            logger.warning("[Speaker] 전원 제어 연결 타임아웃 (%s)", self.config.host)
            return False
        except Exception as exc:
            logger.error("[Speaker] power_ctrl(%r) 오류: %s", method, exc, exc_info=True)
            return False

    def power_on(self) -> bool:
        """스피커 전원을 켠다.

        반환값:
            성공 시 True, 실패 시 False.
        """
        return self._power_ctrl("On")

    def power_off(self) -> bool:
        """스피커 전원을 끈다.

        반환값:
            성공 시 True, 실패 시 False.
        """
        return self._power_ctrl("Off")

    def reboot(self) -> bool:
        """스피커 장치를 재부팅한다.

        반환값:
            성공 시 True, 실패 시 False.
        """
        return self._power_ctrl("Reboot")

    def firmware_update(
        self,
        file_path: str,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> bool:
        """스피커 펜웨어를 업데이트한다 (.imkp 파일만 가능).

        매개변수:
            file_path:   업로드할 .imkp 평성 경로
            on_progress: 진행상황 콜백 (Dict 수신, 없으면 로그만 출력)
                         Dict 필드: Status, Process, Step, Progress(Download 시), ErrNo(Fail 시)

        반환값:
            업데이트 완료(Status=="End") 시 True, 실패 시 False.
        """
        client = self._get_client()
        if client is None:
            return False

        if not file_path.lower().endswith(".imkp"):
            logger.error("[Speaker] 펜웨어 파일은 .imkp 구문만 가능합니다: %s", file_path)
            return False

        def _log_progress(part: Dict) -> None:
            status   = part.get("Status", "")
            process  = part.get("Process", "")
            step     = part.get("Step", "")
            progress = part.get("Progress")
            if progress is not None:
                logger.info("[Speaker] FW 업데이트 [%s/%s] %d%%", status, process, progress)
            else:
                logger.info("[Speaker] FW 업데이트 [%s] Process=%s Step=%s", status, process, step)
            if on_progress:
                on_progress(part)

        url  = f"http://{self.config.host}/interm-api{_FIRMWARE_UPDATE_PATH}"
        auth = HTTPDigestAuth(self.config.username, self.config.password)
        try:
            with open(file_path, "rb") as fp:
                filename = file_path.replace("\\", "/").split("/")[-1]
                resp = requests.post(
                    url,
                    files={"UploadedFile": (filename, fp, "application/octet-stream")},
                    auth=auth,
                    timeout=(self.config.connect_timeout, 300),
                    stream=True,
                )
                resp.raise_for_status()

            success = False
            buf = ""
            for chunk in resp.iter_content(chunk_size=None):
                buf += chunk.decode("utf-8", errors="replace")
                while "{" in buf and "}" in buf:
                    start = buf.index("{")
                    end   = buf.index("}", start) + 1
                    try:
                        part = json.loads(buf[start:end])
                        buf  = buf[end:]
                    except json.JSONDecodeError:
                        buf = buf[end:]
                        continue
                    _log_progress(part)
                    if part.get("Status") == "End":
                        success = True
                    elif part.get("Status") == "Fail":
                        logger.error("[Speaker] FW 업데이트 실패: ErrNo=%s", part.get("ErrNo"))
                        return False

            if success:
                logger.info("[Speaker] 펜웨어 업데이트 완료")
            else:
                logger.error("[Speaker] 펜웨어 업데이트 실패")
            return success
        except Exception as exc:
            logger.error("[Speaker] firmware_update() 오류: %s", exc, exc_info=True)
            return False

    def upload_file(self, file_path: str, file_type: str = "BGM", storage_type: str = "Internal") -> bool:
        """음원 파일(MP3/WAV)을 스피커에 업로드한다.

        매개변수:
            file_path:    로컬 파일 경로 (MP3 또는 WAV)
            file_type:    음원 타입 ("BGM" 또는 "CHIME", 기본: "BGM")
            storage_type: 저장소 타입 ("Internal" 또는 "External", 기본: "Internal")

        반환값:
            업로드 성공 시 True, 실패 시 False.
        """
        if not self.config.is_configured:
            logger.warning("[Speaker] 스피커 미설정 - 업로드 불가")
            return False
        url  = f"http://{self.config.host}/interm-api/Audio/File/Upload"
        auth = HTTPDigestAuth(self.config.username, self.config.password)
        try:
            with open(file_path, "rb") as fp:
                filename = file_path.replace("\\", "/").split("/")[-1]
                resp = requests.post(
                    url,
                    files={"File": (filename, fp)},
                    data={"Type": file_type, "StorageType": storage_type},
                    auth=auth,
                    timeout=(self.config.connect_timeout, self.config.read_timeout),
                )
                resp.raise_for_status()
                data = resp.json()
            if data.get("Execute") == "OK":
                logger.info("[Speaker] 파일 업로드 성공: %r (type=%s, storage=%s)", file_path, file_type, storage_type)
                return True
            logger.error("[Speaker] 파일 업로드 실패: %s", data)
            return False
        except Exception as exc:
            logger.error("[Speaker] upload_file() 오류: %s", exc, exc_info=True)
            return False

    def remove_file(self, file_list: list[str], file_type: str = "BGM") -> bool:
        """스피커에 저장된 음원 파일을 삭제한다.

        매개변수:
            file_list: 삭제할 파일 식별자 리스트
                       - file_type="BGM"  → 파일 해시 문자열 목록
                       - file_type="Chime" → 파일 명 문자열 목록
            file_type: 음원 타입 ("BGM" 또는 "Chime", 기본: "BGM")

        반환값:
            삭제 성공 시 True, 실패 시 False.
        """
        return self._post_setting(
            "/Audio/File/Remove",
            {"Type": file_type, "FileList": file_list},
            f"remove_file(type={file_type!r})",
        )

    def replace_file(
        self,
        old_file_list: list[str],
        new_file_path: str,
        file_type: str = "BGM",
        storage_type: str = "Internal",
        *,
        skip_remove_error: bool = False,
    ) -> bool:
        """스피커에 저장된 음원 파일을 새 파일로 교체한다.

        내부적으로 remove_file() → upload_file() 순서로 동작한다.

        매개변수:
            old_file_list:     삭제할 기존 파일 식별자 리스트
                               - file_type="BGM"   → 파일 해시 문자열 목록
                               - file_type="Chime" → 파일 명 문자열 목록
            new_file_path:     업로드할 새 파일의 로컬 경로 (MP3 또는 WAV)
            file_type:         음원 타입 ("BGM" 또는 "Chime", 기본: "BGM")
            storage_type:      저장소 타입 ("Internal" 또는 "External", 기본: "Internal")
            skip_remove_error: True 이면 삭제 실패 시에도 업로드를 계속 시도한다.
                               False(기본) 이면 삭제 실패 시 즉시 False 반환.

        반환값:
            삭제 + 업로드 모두 성공 시 True, 실패 시 False.
        """
        logger.info(
            "[Speaker] replace_file() 시작: old=%s → new=%r (type=%s)",
            old_file_list, new_file_path, file_type,
        )

        removed = self.remove_file(old_file_list, file_type)
        if not removed:
            if skip_remove_error:
                logger.warning(
                    "[Speaker] 기존 파일 삭제 실패 — skip_remove_error=True 이므로 업로드 계속 진행"
                )
            else:
                logger.error("[Speaker] 기존 파일 삭제 실패 — 교체 중단")
                return False

        uploaded = self.upload_file(new_file_path, file_type, storage_type)
        if uploaded:
            logger.info("[Speaker] replace_file() 완료: %r 업로드 성공", new_file_path)
        else:
            logger.error("[Speaker] replace_file() 실패: 업로드 오류")
        return uploaded

    # ------------------------------------------------------------------
    # 수동 방송 제어 (InterM API 26. Broadcast/Manual)
    # ------------------------------------------------------------------

    def _broadcast_manual(self, body: Dict, action: str) -> bool:
        """수동 방송 제어 공통 내부 메서드."""
        client = self._get_client()
        if client is None:
            return False
        try:
            resp = client._post(_BROADCAST_MANUAL_PATH, body)
            if resp.get("Execute") == "OK":
                logger.info("[Speaker] 수동 방송 %s 성공", action)
                return True
            logger.error("[Speaker] 수동 방송 %s 실패: %s", action, resp)
            return False
        except Exception as exc:
            logger.error("[Speaker] broadcast_%s() 오류: %s", action.lower(), exc, exc_info=True)
            return False

    def broadcast_start(self, source_id: int, zone_list: list[str]) -> bool:
        """수동 방송을 시작한다.

        매개변수:
            source_id: 방송 소스 ID (GET /Controller/Source/List 에서 확인)
            zone_list: 방송 대상 존 리스트
                       예) ["all"]            - 모든 존
                            ["z1", "z3"]       - 특정 존 지정
                            ["g1"]             - 그룹 1 전체
                            ["z1-z4"]          - 존 1~4 범위
                            ["g2-g3"]          - 그룹 2~3 범위
        반환값:
            성공 시 True, 실패 시 False.
        """
        return self._broadcast_manual(_param_broadcast_start(source_id, zone_list), "Start")

    def broadcast_stop(self, zone_list: list[str]) -> bool:
        """수동 방송을 중지한다.

        매개변수:
            zone_list: 중지할 존 리스트 (예: ["all"], ["z1", "z2"])

        반환값:
            성공 시 True, 실패 시 False.
        """
        return self._broadcast_manual(_param_broadcast_stop(zone_list), "Stop")

    def broadcast_all_stop(self) -> bool:
        """모든 존의 수동 방송을 일괄 중지한다.

        반환값:
            성공 시 True, 실패 시 False.
        """
        return self._broadcast_manual(_param_broadcast_all_stop(), "AllStop")

    def broadcast_volume(self, zone_id: int, volume: int) -> bool:
        """특정 존의 수동 방송 볼륨을 설정한다.

        매개변수:
            zone_id: 볼륨 제어 대상 존 ID
            volume:  볼륨 값 (0~100, 범위 밀어나면 자동 클램핑)

        반환값:
            성공 시 True, 실패 시 False.
        """
        return self._broadcast_manual(_param_broadcast_volume(zone_id, volume), "Volume")

    # ------------------------------------------------------------------
    # DSP 입력 설정 (InterM API 9. DSP/Input)
    # ------------------------------------------------------------------

    def get_dsp_input(self) -> Optional[Dict]:
        """스피커 DSP 입력 설정을 조회한다 (GET /Audio/DSP/Input).

        반환값:
            성공 시 result 딕셔너리 (예: {"Mode": "Condenser", "MICGain": 12.5}),
            실패 시 None.
        """
        return self._get_setting(_DSP_INPUT_PATH, "get_dsp_input()")

    def set_dsp_input(
        self,
        mode: str,
        gain: Optional[float] = None,
    ) -> bool:
        """스피커 DSP 입력 설정을 변경한다 (POST /Audio/DSP/Input, IP-SPEAKER).

        매개변수:
            mode: 접점 타입 ("Condenser" 또는 "Disabled")
            gain: MICGain (mode=\"Condenser\" 일 때만 적용, -12 ~ 55.25 Float)

        반환값:
            성공 시 True, 실패 시 False.
        """
        return self._post_setting(
            _DSP_INPUT_PATH,
            _param_dsp_input(mode, gain),
            f"set_dsp_input(mode={mode!r})",
        )

    # ------------------------------------------------------------------
    # DSP 출력 설정 (InterM API 10. DSP/Output)
    # ------------------------------------------------------------------

    def get_dsp_output(self) -> Optional[Dict]:
        """스피커 DSP 출력 설정을 조회한다 (GET /Audio/DSP/Output).

        반환값:
            성공 시 result 딕셔너리 (예: {"IsMute": false, "Volume": 50}),
            실패 시 None.
        """
        return self._get_setting(_DSP_OUTPUT_PATH, "get_dsp_output()")

    def set_dsp_output(
        self,
        *,
        is_mute: Optional[bool] = None,
        volume:  Optional[int]  = None,
    ) -> bool:
        """스피커 DSP 출력 설정을 변경한다 (POST /Audio/DSP/Output, IP-SPEAKER).

        매개변수:
            is_mute: 음소거 여부 (True=Mute On, False=Mute Off)
            volume:  볼륨 값 (0~100)

        반환값:
            성공 시 True, 실패 시 False.
        """
        body = _param_dsp_output_speaker(is_mute=is_mute, volume=volume)
        if not body:
            logger.warning("[Speaker] set_dsp_output(): 변경할 파라미터가 없습니다.")
            return False
        return self._post_setting(_DSP_OUTPUT_PATH, body, "set_dsp_output()")

    def set_dsp_output_laser(
        self,
        *,
        output_mode:   Optional[str]  = None,
        delay:         Optional[int]  = None,
        volume:        Optional[int]  = None,
        is_mute:       Optional[bool] = None,
        beam_steering: Optional[int]  = None,
    ) -> bool:
        """Laser 모델의 DSP 출력 설정을 변경한다 (POST /Audio/DSP/Output, Laser).

        모든 파라미터는 키워드 전용이며, 변경이 필요한 항목만 전달한다.

        매개변수:
            output_mode:   오디오 방송 타입 ("Analog" 또는 "Digital")
            delay:         출력 딜레이 (ms, 0~680, 자동 클램핑)
            volume:        출력 볼륨 (dB, -80~20, 자동 클램핑)
            is_mute:       음소거 여부
            beam_steering: 출력 각도 (-60~60 사이 허용된 스텝 값,
                           그 외 값은 자동으로 가장 가까운 스텝으로 스냅)

        반환값:
            성공 시 True, 실패 시 False.
        """
        body = _param_dsp_output_laser(
            output_mode=output_mode,
            delay=delay,
            volume=volume,
            is_mute=is_mute,
            beam_steering=beam_steering,
        )
        if not body:
            logger.warning("[Speaker] set_dsp_output_laser(): 변경할 파라미터가 없습니다.")
            return False
        return self._post_setting(_DSP_OUTPUT_PATH, body, "set_dsp_output_laser()")
