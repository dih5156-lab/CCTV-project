"""내부 공통 MQTT 클라이언트 팩토리.

MqttEventPublisher 와 MqttTopicSubscriber 양쪽에서 사용한다.
이 모듈은 protocols 패키지 외부로 노출하지 않는다.
"""

from __future__ import annotations

import uuid
from typing import Optional

import paho.mqtt.client as mqtt

_PAHO_V2 = hasattr(mqtt, "CallbackAPIVersion")

# 재연결 백오프 공통 상수 (publisher/subscriber 공유)
RECONNECT_MIN_DELAY: float = 1.0    # 최초 재시도 대기 시간 (초)
RECONNECT_MULTIPLIER: float = 2.0   # 대기 시간 배율
# MAX_DELAY 는 각 클라이언트마다 별도 지정 (publisher: 60s, subscriber: 30s)


def create_mqtt_client(
    client_id_prefix: str,
    client_id: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> mqtt.Client:
    """재사용 가능한 paho MQTT 클라이언트를 생성한다.

    Paho v2 / v1 API 차이를 추상화하며, username/password 인증을 선택적으로 적용한다.

    Args:
        client_id_prefix: 자동 생성 시 접두사 (UUID 8자리 접미사 추가)
        client_id:        지정할 경우 해당 ID 사용, None 이면 접두사+UUID 사용
        username:         MQTT 브로커 인증 사용자명 (없으면 None)
        password:         MQTT 브로커 인증 비밀번호 (없으면 None)

    Returns:
        콜백이 설정되지 않은 mqtt.Client 인스턴스
    """
    cid = client_id or f"{client_id_prefix}-{uuid.uuid4().hex[:8]}"
    try:
        if _PAHO_V2:
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=cid,
                clean_session=True,
            )
        else:
            client = mqtt.Client(client_id=cid, clean_session=True)
    except Exception:
        # 최후 폴백: 키워드 없이 시도 (구버전 paho 대응)
        client = mqtt.Client(client_id=cid)

    if username:
        client.username_pw_set(username, password)

    return client
