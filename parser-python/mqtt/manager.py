"""
mqtt/manager.py
===============
Go 원본: aiot-tlv-parser/pkg/mqtt/manager.go

여러 MQTT 브로커 연결을 관리하는 매니저 모듈입니다.
Go의 goroutine + sync.RWMutex → Python의 threading.RLock 으로 변환되었습니다.

의존 라이브러리:
  paho-mqtt (pip install paho-mqtt)

연결 브로커:
  - ns_park : 실제 LoRa 네트워크 서버
  - lab     : 랩 테스트 브로커
  (proxy, lab_test, local 은 주석 처리된 상태 유지)
"""

import json
import logging
import os
import threading
import time
from typing import Optional

try:
    import paho.mqtt.client as mqtt_client
except ImportError:
    mqtt_client = None

from mqtt.interfaces import SensorDataProcessor
from mqtt.uplink import UplinkDecoder

logger = logging.getLogger(__name__)


class ClientInfo:
    """
    단일 MQTT 클라이언트 정보 컨테이너
    Go: type ClientInfo struct { Client mqtt.Client; Name string; Config MQTTConfig; Connected bool; mu sync.RWMutex }
    """

    def __init__(self, client, name: str, config):
        """
        Args:
            client : paho MQTT 클라이언트 인스턴스
            name   : 브로커 식별 이름 (예: "ns_park")
            config : MQTTConfig 인스턴스
        """
        self.client = client
        self.name = name
        self.config = config
        self.connected = False
        self._lock = threading.RLock()  # Go: mu sync.RWMutex

    def set_connected(self, value: bool) -> None:
        with self._lock:
            self.connected = value

    def is_connected(self) -> bool:
        with self._lock:
            return self.connected


class Manager(UplinkDecoder):
    """
    MQTT 멀티 브로커 매니저
    Go: type Manager struct { clients map[string]*ClientInfo; processor SensorDataProcessor; mu sync.RWMutex; ... }

    여러 MQTT 브로커에 동시 연결하고 메시지를 단일 처리기로 라우팅합니다.
    """

    def __init__(self, configs, *, durable=False, client_id_prefix=None):
        """
        Go: func NewManager(configs config.MQTTConfigs) *Manager
        """
        self._durable = durable
        self._subscribers: dict = {}
        self._client_id_prefix = (client_id_prefix or os.environ.get("AIOT_MQTT_CLIENT_ID_PREFIX", "")).strip()
        if durable and not self._client_id_prefix:
            raise ValueError("Durable TLV MQTT requires AIOT_MQTT_CLIENT_ID_PREFIX")
        self._clients: dict = {}
        self._processor: Optional[SensorDataProcessor] = None
        self._lock = threading.RLock()  # Go: mu sync.RWMutex

    def set_processor(self, processor: SensorDataProcessor) -> None:
        """
        센서 데이터 처리기 설정
        Go: func (m *Manager) SetProcessor(processor SensorDataProcessor)
        """
        self._processor = processor

    def connect_client(self, name: str, config) -> None:
        """
        MQTT 브로커에 연결
        Go: func (m *Manager) ConnectClient(name string, config config.MQTTConfig) error

        연결 옵션:
          - KeepAlive   : 60초
          - AutoReconnect: True
          - 최대 재연결 간격: 5분
        """
        if self._durable:
            from src.protocols.mqtt_subscriber import MqttTopicSubscriber
            if self._processor is None or not callable(getattr(self._processor, "process_mqtt_message", None)):
                raise ValueError("Durable TLV MQTT requires a durable sensor processor")
            if name in self._subscribers:
                raise ValueError(f"client {name} already exists")
            subscriber = MqttTopicSubscriber(
                broker=config.host, port=config.port,
                topics=("+/+/up", "v3/+/devices/+/up"),
                client_id=f"{self._client_id_prefix}-{name}",
                username=config.username, password=config.password,
                durable=True, message_handler=self._processor.process_mqtt_message,
            )
            self._subscribers[name] = subscriber
            subscriber.connect()
            return
        if mqtt_client is None:
            raise RuntimeError("paho-mqtt is not installed. Run: pip install paho-mqtt")

        with self._lock:
            if name in self._clients:
                raise ValueError(f"client {name} already exists")

        # paho MQTT 클라이언트 옵션 설정
        # Go 원본의 mqtt.NewClientOptions() 생성에 해당한다.
        client = mqtt_client.Client(
            client_id=f"aiot-sensor-{name}-{int(time.time())}",
        )

        if config.username and config.password:
            client.username_pw_set(config.username, config.password)

        client.keepalive = 60

        # 연결 끊김 핸들러
        # Go 원본의 연결 끊김 콜백 등록에 해당한다.
        def on_disconnect(c, userdata, rc):
            logger.warning(f"MQTT client {name} connection lost: rc={rc}")
            with self._lock:
                if name in self._clients:
                    self._clients[name].set_connected(False)

        # 연결 성공 핸들러
        # Go 원본의 연결 성공 후 토픽 구독 콜백 등록에 해당한다.
        def on_connect(c, userdata, flags, rc):
            if rc == 0:
                logger.info(f"MQTT client {name} connected successfully")
                with self._lock:
                    if name in self._clients:
                        self._clients[name].set_connected(True)
                self._subscribe_to_topics(c, name, config)
            else:
                logger.error(f"MQTT client {name} connection failed: rc={rc}")

        # 메시지 수신 핸들러
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = self._message_handler

        # 브로커 연결
        client.connect_async(config.host, config.port, keepalive=60)
        client.loop_start()

        client_info = ClientInfo(client=client, name=name, config=config)
        with self._lock:
            self._clients[name] = client_info

        logger.info(f"MQTT client {name} connecting to {config.host}:{config.port}")

    def _subscribe_to_topics(self, client, name: str, config) -> None:
        """
        MQTT 토픽 구독
        Go: func (m *Manager) subscribeToTopics(client mqtt.Client, name string, config config.MQTTConfig) error

        dcaLPWAN 표준 Uplink와 호환 포맷만 구독
        """
        topics = ["+/+/up", "v3/+/devices/+/up"]
        for topic in topics:
            client.subscribe(topic, qos=1)
            logger.info(f"Subscribed to topic: {topic}")

    def _message_handler(self, client, userdata, msg) -> None:
        """
        MQTT 메시지 수신 핸들러
        Go: func (m *Manager) messageHandler(client mqtt.Client, msg mqtt.Message)

        처리 조건: 토픽에 "up" 포함 여부 확인
        """
        topic = msg.topic
        logger.debug(f"Received message on topic: {topic}")

        # 업링크 메시지만 처리
        if not topic.split("/")[-1:] == ["up"]:
            return

        # JSON 파싱
        try:
            message_data = json.loads(msg.payload)
            if not isinstance(message_data, dict):
                raise ValueError("JSON payload must be an object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse JSON message: {e}")
            return

        app_id, dev_eui = self._extract_app_and_device(topic, message_data)
        payload = self._extract_payload(message_data)

        if not app_id or not dev_eui:
            logger.warning("Missing appID or devEUI in message: topic=%s", topic)
            return
        if not payload:
            logger.info("Missing payload in message: appID=%s devEUI=%s topic=%s", app_id, dev_eui, topic)
            return

        # rx_metadata 에서 채널/주파수/시각 추출
        channel = 0
        frequency = 0
        received_at = 0
        uplink = message_data.get("uplink_message", {})
        rx_metadata = message_data.get("rx_metadata") or uplink.get("rx_metadata", [])
        primary_rx_metadata = {}
        if isinstance(rx_metadata, list) and rx_metadata and isinstance(rx_metadata[0], dict):
            primary_rx_metadata = rx_metadata[0]
            channel = self._parse_integer(primary_rx_metadata.get("channel", 0))
            frequency = self._parse_integer(primary_rx_metadata.get("frequency", 0))
            received_at = self._parse_received_at(primary_rx_metadata.get("time", 0))

        # 센서 데이터 처리기 호출
        if self._processor:
            try:
                self._processor.process_sensor_data(
                    app_id=app_id,
                    dev_eui=dev_eui,
                    payload=payload,
                    channel=channel,
                    frequency=frequency,
                    received_at=received_at,
                    uplink_metadata={
                        "message_id": message_data.get("message_id", ""),
                        "f_port": message_data.get("f_port", 0),
                        "f_cnt_up": message_data.get("f_cnt_up", 0),
                        "is_confirmed": bool(message_data.get("is_confirmed", False)),
                        "is_ack": bool(message_data.get("is_ack", False)),
                        "radio": self._extract_radio_metadata(primary_rx_metadata),
                    },
                )
            except Exception as e:
                logger.error(f"Failed to process sensor data: {e}")

    def poll(self) -> None:
        """Drive retry/reconnect outside the Paho callback thread."""
        for subscriber in self._subscribers.values():
            subscriber.connect()

    def disconnect_client(self, name: str) -> None:
        """
        특정 MQTT 클라이언트 연결 해제
        Go: func (m *Manager) DisconnectClient(name string) error
        """
        if self._durable:
            self._subscribers.pop(name).disconnect()
            return
        with self._lock:
            client_info = self._clients.get(name)
            if not client_info:
                raise KeyError(f"client {name} not found")

            client_info.client.disconnect()
            client_info.client.loop_stop()
            del self._clients[name]
            logger.info(f"MQTT client {name} disconnected")

    def disconnect_all(self) -> None:
        """
        모든 MQTT 클라이언트 연결 해제
        Go: func (m *Manager) DisconnectAll()
        """
        if self._durable:
            for subscriber in self._subscribers.values():
                subscriber.disconnect()
            self._subscribers.clear()
            return
        with self._lock:
            for name, client_info in list(self._clients.items()):
                client_info.client.disconnect()
                client_info.client.loop_stop()
                logger.info(f"MQTT client {name} disconnected")
            self._clients.clear()

    def get_client_status(self, name: str) -> bool:
        """
        특정 클라이언트 연결 상태 반환
        Go: func (m *Manager) GetClientStatus(name string) (bool, error)
        """
        if self._durable:
            return self._subscribers[name].is_connected
        with self._lock:
            client_info = self._clients.get(name)
            if not client_info:
                raise KeyError(f"client {name} not found")
        return client_info.is_connected()

    def get_all_clients_status(self) -> dict:
        """
        모든 클라이언트 연결 상태 반환
        Go: func (m *Manager) GetAllClientsStatus() map[string]bool
        """
        if self._durable:
            return {name: sub.is_connected for name, sub in self._subscribers.items()}
        with self._lock:
            return {name: info.is_connected() for name, info in self._clients.items()}

    def init(self, configs, processor: SensorDataProcessor) -> None:
        """
        설정에 따라 MQTT 클라이언트들 초기화 및 연결
        Go: func (m *Manager) Init(configs config.MQTTConfigs, processor SensorDataProcessor) error

        활성화 브로커:
          - ns_park (Go에서 활성화됨)
          - lab     (Go에서 활성화됨)
          proxy, lab_test, local 은 주석 처리됨
        """
        logger.info("Initializing MQTT clients...")

        self.set_processor(processor)

        # 연결할 클라이언트 목록 (ns_park, lab 만 활성)
        clients_to_connect = {
            "ns_park": configs.ns_park,
            "lab":     configs.lab,
        }

        connected_targets = set()
        for name, config in clients_to_connect.items():
            if config.host:
                target_key = (
                    str(config.host).strip().lower(),
                    int(config.port),
                    str(config.username or ""),
                )
                if target_key in connected_targets:
                    logger.info(
                        "Skipping MQTT client %s because %s:%s is already connected",
                        name,
                        config.host,
                        config.port,
                    )
                    continue
                try:
                    self.connect_client(name, config)
                    connected_targets.add(target_key)
                except Exception as e:
                    if self._durable:
                        self.disconnect_all()
                        raise
                    logger.warning(f"Failed to connect MQTT client {name}: {e}")

        with self._lock:
            logger.info(f"MQTT initialization completed. Connected clients: {len(self._clients)}")
