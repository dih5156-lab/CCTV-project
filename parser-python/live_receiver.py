"""
live_receiver.py
================
실시간 MQTT 데이터 수신 + TLV 파싱 모니터링 스크립트

.env 파일에서 설정을 읽어 NS_PARK / LAB 브로커에 연결한 뒤
수신되는 모든 센서 메시지를 파싱해서 콘솔에 출력합니다.

실행 방법:
    cd parser-python
    python live_receiver.py
    python live_receiver.py --env ../aiot-tlv-parser/.env   (다른 .env 경로)
    python live_receiver.py --broker ns_park                 (특정 브로커만)
"""

import sys
import os
import json
import base64
import struct
import argparse
import threading
import signal
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 의존성 체크 ──────────────────────────────────────
try:
    from dotenv import load_dotenv
except ImportError:
    print("[오류] python-dotenv 가 설치되지 않았습니다: pip install python-dotenv")
    sys.exit(1)

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("[오류] paho-mqtt 가 설치되지 않았습니다: pip install paho-mqtt")
    sys.exit(1)

from tlv.parser import Parser

# ─────────────────────────────────────────────────────
# 설정 로드
# ─────────────────────────────────────────────────────

def load_env(env_path: str) -> dict:
    """지정된 .env 파일 로드 후 필요한 설정 반환"""
    load_dotenv(env_path, override=True)
    return {
        "ns_park": {
            "host":     os.getenv("NS_PARK_MQTT_HOST", "localhost"),
            "port":     int(os.getenv("NS_PARK_MQTT_PORT", "1883")),
            "username": os.getenv("NS_PARK_MQTT_ID", ""),
            "password": os.getenv("NC_PW", ""),
        },
        "lab": {
            "host":     os.getenv("LAB_MQTT_HOST", "localhost"),
            "port":     int(os.getenv("LAB_MQTT_PORT", "1883")),
            "username": os.getenv("LAB_MQTT_ID", ""),
            "password": os.getenv("NC_PW", ""),
        },
    }

# ─────────────────────────────────────────────────────
# TLV 파싱 + 출력
# ─────────────────────────────────────────────────────

_parser = Parser()
_lock   = threading.Lock()

TABLE_NAMES = {
    "t3":     "디바이스 정보",
    "t34950": "하천 모니터링 (수위/유속/강수량)",
    "t34952": "침수 감지",
    "t34954": "온도/습도",
    "t34955": "경사계",
    "t34956": "화재 경보",
    "t34957": "복합 요약1 (온도+경사)",
    "t34958": "복합 요약2 (가속도+자이로+경사)",
}


def parse_and_print(broker_name: str, topic: str, raw_json: dict):
    """
    JSON 메시지에서 base64 payload 를 추출해 TLV 파싱 후 출력
    TTS 형식(frm_payload)과 커스텀 형식(payload) 모두 지원
    """
    # payload 필드 탐색: TTS 형식 우선, 없으면 기본 payload 키
    uplink = raw_json.get("uplink_message", {})
    b64_payload = (
        uplink.get("frm_payload")          # TTS /devices/.../up
        or raw_json.get("payload")          # 커스텀 da/{appEUI}/{devEUI}/up
        or raw_json.get("frm_payload")      # 최상위에 있는 경우
    )

    if not b64_payload:
        _print(broker_name, topic, "[스킵] payload 필드 없음")
        return

    # Base64 디코딩
    try:
        buf = base64.b64decode(b64_payload)
    except Exception as e:
        _print(broker_name, topic, f"[오류] base64 디코딩 실패: {e}")
        return

    # TLV 파싱 (start_index=8)
    try:
        result = _parser.decode_lwm2m_tlv(buf, 8)
    except ValueError as e:
        _print(broker_name, topic, f"[스킵] {e}")
        return
    except Exception as e:
        _print(broker_name, topic, f"[오류] TLV 파싱 실패: {e}")
        return

    if result is None:
        _print(broker_name, topic, "[스킵] 파서가 None 반환")
        return

    # 출력
    label = TABLE_NAMES.get(result.table_name, result.table_name)
    now   = datetime.now().strftime("%H:%M:%S")

    data  = {k: v for k, v in result.data.items() if k != "tableName"}

    with _lock:
        print()
        print(f"┌─ [{now}] {broker_name.upper()} │ {topic}")
        print(f"│  테이블 : {result.table_name}  ({label})")
        for key, val in data.items():
            if isinstance(val, float):
                print(f"│  {key:<35} = {val:.6f}")
            else:
                print(f"│  {key:<35} = {val}")
        print(f"└{'─' * 60}")


def _print(broker_name: str, topic: str, msg: str):
    with _lock:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {broker_name.upper()} │ {topic} │ {msg}")


# ─────────────────────────────────────────────────────
# MQTT 클라이언트
# ─────────────────────────────────────────────────────

def build_client(broker_name: str, cfg: dict) -> mqtt.Client:
    try:
        # paho-mqtt 2.x
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION1,
            client_id=f"live-receiver-{broker_name}",
            protocol=mqtt.MQTTv311,
        )
    except AttributeError:
        # paho-mqtt 1.x 하위 호환
        client = mqtt.Client(
            client_id=f"live-receiver-{broker_name}",
            protocol=mqtt.MQTTv311,
        )
    client.username_pw_set(cfg["username"], cfg["password"])

    def on_connect(c, userdata, flags, rc):
        codes = {
            0: "연결 성공",
            1: "잘못된 프로토콜",
            2: "클라이언트 ID 거부",
            3: "서버 사용 불가",
            4: "잘못된 인증 정보",
            5: "권한 없음",
        }
        msg = codes.get(rc, f"알 수 없는 코드: {rc}")
        if rc == 0:
            print(f"[{broker_name}] ✅  {cfg['host']}:{cfg['port']} → {msg}")
            # 모든 토픽 구독 (TTS: v3/# , da: #)
            c.subscribe("#", qos=0)
            print(f"[{broker_name}] 구독 시작: # (모든 토픽)")
        else:
            print(f"[{broker_name}] ❌  연결 실패 → {msg}")

    def on_disconnect(c, userdata, rc):
        if rc != 0:
            print(f"[{broker_name}] ⚠️  연결 끊김 (rc={rc}), 재연결 대기중...")

    def on_message(c, userdata, msg):
        try:
            raw_json = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return  # JSON 이 아닌 메시지는 무시
        parse_and_print(broker_name, msg.topic, raw_json)

    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message
    return client


# ─────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="실시간 MQTT TLV 수신기")
    parser.add_argument(
        "--env",
        default=os.path.join(os.path.dirname(__file__), ".env"),
        help=".env 파일 경로 (기본: parser-python/.env)",
    )
    parser.add_argument(
        "--broker",
        choices=["ns_park", "lab", "both"],
        default="both",
        help="연결할 브로커 (기본: both)",
    )
    args = parser.parse_args()

    env_path = os.path.abspath(args.env)
    if not os.path.exists(env_path):
        print(f"[오류] .env 파일을 찾을 수 없습니다: {env_path}")
        sys.exit(1)

    print(f"[설정] .env 로드: {env_path}")
    brokers = load_env(env_path)

    # 사용할 브로커 목록 결정
    target_brokers = (
        {k: v for k, v in brokers.items() if k == args.broker}
        if args.broker != "both"
        else brokers
    )

    print("\n연결 대상 브로커:")
    for name, cfg in target_brokers.items():
        print(f"  {name}: {cfg['host']}:{cfg['port']}  (user={cfg['username']})")
    print()

    # 클라이언트 생성 및 연결
    clients = []
    for name, cfg in target_brokers.items():
        client = build_client(name, cfg)
        try:
            client.connect(cfg["host"], cfg["port"], keepalive=60)
            client.loop_start()
            clients.append(client)
        except Exception as e:
            print(f"[{name}] ❌  연결 시도 실패: {e}")

    if not clients:
        print("[오류] 연결된 브로커가 없습니다.")
        sys.exit(1)

    # Ctrl+C 종료 처리
    stop_event = threading.Event()

    def handle_signal(sig, frame):
        print("\n\n[종료] Ctrl+C 감지, 연결을 닫습니다...")
        for c in clients:
            c.loop_stop()
            c.disconnect()
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)

    print("메시지 대기 중... (종료: Ctrl+C)\n")
    stop_event.wait()
    print("[완료]")


if __name__ == "__main__":
    main()
