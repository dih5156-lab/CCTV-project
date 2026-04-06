"""run_kuiper_rules.py - Kuiper 규칙 배포 도구"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _replace_tokens(payload: str, replacements: Dict[str, Any]) -> str:
    result = payload
    for key, value in replacements.items():
        result = result.replace(f"{{{{{key}}}}}", str(value))
    return result


def _find_unresolved_tokens(payload: str) -> list[str]:
    return sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", payload)))


def _request(method: str, url: str, **kwargs):
    response = requests.request(method, url, timeout=10, **kwargs)
    return response


def ensure_mqtt_source_config(kuiper_api: str, mqtt_broker: str, mqtt_port: int) -> None:
    """eKuiper 기본 MQTT 소스 설정을 올바른 브로커 주소로 패치합니다."""
    url = f"{kuiper_api}/metadata/sources/mqtt/confKeys/default"
    payload = {
        "server": f"tcp://{mqtt_broker}:{mqtt_port}",
        "qos": 1,
        "protocolVersion": "3.1.1",
        "clientid": "",
        "username": "",
        "password": "",
    }
    resp = _request("PUT", url, json=payload)
    if resp.status_code in (200, 201):
        logger.info(f"MQTT 소스 기본 설정 완료: tcp://{mqtt_broker}:{mqtt_port}")
    else:
        logger.warning(f"MQTT 소스 설정 응답: {resp.status_code} - {resp.text}")


def ensure_stream(kuiper_api: str, stream_name: str, stream_sql: str) -> None:
    stream_url = f"{kuiper_api}/streams/{stream_name}"
    create_stream_url = f"{kuiper_api}/streams"

    get_resp = _request("GET", stream_url)
    if get_resp.status_code == 200:
        # 항상 재생성하여 최신 소스 설정을 반영
        del_resp = _request("DELETE", stream_url)
        if del_resp.status_code not in (200, 204):
            logger.warning(f"스트림 삭제 실패 ({stream_name}): {del_resp.status_code} - {del_resp.text}")
        else:
            logger.info(f"기존 스트림 삭제: {stream_name}")

    create_resp = _request("POST", create_stream_url, json={"sql": stream_sql})
    if create_resp.status_code in (200, 201):
        logger.info(f"스트림 생성 완료: {stream_name}")
        return

    raise RuntimeError(f"스트림 생성 실패: {create_resp.status_code} - {create_resp.text}")


def upsert_rule(kuiper_api: str, rule: Dict[str, Any]) -> None:
    rule_id = rule["id"]
    rule_url = f"{kuiper_api}/rules/{rule_id}"
    create_rule_url = f"{kuiper_api}/rules"

    get_resp = _request("GET", rule_url)
    if get_resp.status_code == 200:
        delete_resp = _request("DELETE", rule_url)
        if delete_resp.status_code not in (200, 204):
            raise RuntimeError(
                f"기존 룰 삭제 실패 ({rule_id}): {delete_resp.status_code} - {delete_resp.text}"
            )
        logger.info(f"기존 룰 삭제: {rule_id}")

    create_payload = {
        "id": rule_id,
        "sql": rule["sql"],
        "actions": rule["actions"],
        "options": {
            "isEventTime": False,
        },
    }

    create_resp = _request("POST", create_rule_url, json=create_payload)
    if create_resp.status_code in (200, 201):
        logger.info(f"룰 배포 완료: {rule_id}")
        return

    raise RuntimeError(f"룰 배포 실패 ({rule_id}): {create_resp.status_code} - {create_resp.text}")


def _env_float(name: str, default: float) -> float:
    """환경변수를 float로 읽는다. 설정되지 않으면 default 반환."""
    val = os.environ.get(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    """환경변수를 int로 읽는다. 설정되지 않으면 default 반환."""
    val = os.environ.get(name)
    return int(val) if val is not None else default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CCTV Kuiper 규칙 묶음 배포",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kuiper-api", default=os.environ.get("KUIPER_API", "http://localhost:9081"), help="Kuiper REST API 기본 URL [env: KUIPER_API]")
    parser.add_argument(
        "--rules-file",
        default=os.environ.get("KUIPER_RULES_FILE", "kuiper/rules/cctv_intrusion_rules.json"),
        help="규칙 묶음 JSON 경로 [env: KUIPER_RULES_FILE]",
    )
    parser.add_argument("--mqtt-broker", default=os.environ.get("MQTT_BROKER", "localhost"), help="MQTT 브로커 호스트 [env: MQTT_BROKER]")
    parser.add_argument("--mqtt-port", type=int, default=_env_int("MQTT_PORT", 1883), help="MQTT 브로커 포트 [env: MQTT_PORT]")
    parser.add_argument("--intrusion-confidence", type=float, default=_env_float("INTRUSION_CONFIDENCE", 0.7), help="침입 이벤트 신뢰도 임계값 [env: INTRUSION_CONFIDENCE]")
    parser.add_argument("--critical-confidence", type=float, default=_env_float("CRITICAL_CONFIDENCE", 0.9), help="중요 이벤트 라우팅 임계값 [env: CRITICAL_CONFIDENCE]")
    parser.add_argument("--persist-hit-count", type=int, default=_env_int("PERSIST_HIT_COUNT", 5), help="5초 윈도우 내 최소 검출 횟수 [env: PERSIST_HIT_COUNT]")
    parser.add_argument("--tilt-threshold", type=float, default=_env_float("TILT_THRESHOLD", 10.0), help="기울기 임계값 (도) [env: TILT_THRESHOLD]")
    parser.add_argument("--temp-high-threshold", type=float, default=_env_float("TEMP_HIGH_THRESHOLD", 60.0), help="고온 임계값 (°C) [env: TEMP_HIGH_THRESHOLD]")
    parser.add_argument("--retry-count", type=int, default=_env_int("KUIPER_RETRY_COUNT", 3), help="Kuiper API 재시도 횟수 [env: KUIPER_RETRY_COUNT]")
    parser.add_argument("--retry-delay", type=int, default=_env_int("KUIPER_RETRY_DELAY", 2), help="Kuiper API 재시도 간격(초) [env: KUIPER_RETRY_DELAY]")
    args = parser.parse_args()

    if not (0.0 <= args.intrusion_confidence <= 1.0):
        parser.error("--intrusion-confidence는 0.0~1.0 이어야 합니다")
    if not (0.0 <= args.critical_confidence <= 1.0):
        parser.error("--critical-confidence는 0.0~1.0 이어야 합니다")
    if args.persist_hit_count <= 0:
        parser.error("--persist-hit-count는 양수여야 합니다")
    if args.retry_count <= 0:
        parser.error("--retry-count는 양수여야 합니다")
    if args.retry_delay < 0:
        parser.error("--retry-delay는 0 이상이어야 합니다")
    if args.mqtt_port <= 0:
        parser.error("--mqtt-port는 양수여야 합니다")

    kuiper_api = args.kuiper_api.rstrip("/")

    rules_path = Path(args.rules_file)
    if not rules_path.exists():
        raise FileNotFoundError(f"룰 파일이 없습니다: {rules_path}")

    raw = rules_path.read_text(encoding="utf-8")
    hydrated = _replace_tokens(
        raw,
        {
            "MQTT_BROKER": args.mqtt_broker,
            "MQTT_PORT": args.mqtt_port,
            "INTRUSION_CONFIDENCE": args.intrusion_confidence,
            "CRITICAL_CONFIDENCE": args.critical_confidence,
            "PERSIST_HIT_COUNT": args.persist_hit_count,
            "TILT_THRESHOLD": args.tilt_threshold,
            "TEMP_HIGH_THRESHOLD": args.temp_high_threshold,
        },
    )

    unresolved_tokens = _find_unresolved_tokens(hydrated)
    if unresolved_tokens:
        raise ValueError(f"치환되지 않은 토큰이 남아 있습니다: {', '.join(unresolved_tokens)}")

    pack = json.loads(hydrated)
    # stream(단수) 또는 streams(복수) 모두 지원
    if "streams" in pack:
        streams = pack["streams"]
    elif "stream" in pack:
        streams = [pack["stream"]]
    else:
        streams = []
    rules = pack["rules"]

    last_error = None
    for attempt in range(1, args.retry_count + 1):
        try:
            ensure_mqtt_source_config(kuiper_api, args.mqtt_broker, args.mqtt_port)
            for stream in streams:
                ensure_stream(kuiper_api, stream["name"], stream["sql"])

            for rule in rules:
                upsert_rule(kuiper_api, rule)

            logger.info("Kuiper 룰 배포 완료")
            return
        except Exception as error:
            last_error = error
            logger.warning(f"Kuiper 룰 배포 실패 (시도 {attempt}/{args.retry_count}): {error}")
            if attempt < args.retry_count:
                time.sleep(args.retry_delay)

    raise RuntimeError(f"Kuiper 룰 배포 최종 실패: {last_error}")


if __name__ == "__main__":
    main()
