# AIoT TLV Parser: Go → Python 변환 비교 문서

> 작성일: 2026-04-01  
> 원본: `aiot-tlv-parser/` (Go 코드)  
> 변환본: `parser-python/` (Python 코드)  
> 목적: Go 언어를 모르는 사람도 코드 구조와 동작을 이해할 수 있도록 상세 비교

---

## 목차

1. [프로젝트 구조 비교](#1-프로젝트-구조-비교)
2. [언어 개념 대조표](#2-언어-개념-대조표)
3. [파일별 상세 비교](#3-파일별-상세-비교)
   - [main.go → main.py](#31-maingo--mainpy)
   - [config/config.go → config/config.py](#32-configconfiggo--configconfigpy)
   - [config/validation.go → config/validation.py](#33-configvalidationgo--configvalidationpy)
   - [tlv/parser.go → tlv/parser.py](#34-tlvparsergo--tlvparserpy)
   - [tlv/transformer_v0.go → tlv/transformer_v0.py](#35-tlvtransformer_v0go--tlvtransformer_v0py)
   - [tlv/transformer_v1.go → tlv/transformer_v1.py](#36-tlvtransformer_v1go--tlvtransformer_v1py)
   - [database/models.go → database/models.py](#37-databasemodelsgo--databasemodelspy)
   - [database/connection.go → database/connection.py](#38-databaseconnectiongo--databaseconnectionpy)
   - [database/processor.go → database/processor.py](#39-databaseprocessorgo--databaseprocessorpy)
   - [database/queries.go → database/queries.py](#310-databasequeriesgo--databasequeriespy)
   - [mqtt/interfaces.go → mqtt/interfaces.py](#311-mqttinterfacesgo--mqttinterfacespy)
   - [mqtt/classifier.go → mqtt/classifier.py](#312-mqttclassifiergo--mqttclassifierpy)
   - [mqtt/manager.go → mqtt/manager.py](#313-mqttmanagergo--mqttmanagerpy)
   - [service/device_info_service.go → service/device_info_service.py](#314-servicedevice_info_servicego--servicedevice_info_servicepy)
   - [service/event_service.go → service/event_service.py](#315-serviceevent_servicego--serviceevent_servicepy)
   - [service/sensor_service.go → service/sensor_service.py](#316-servicesensor_servicego--servicesensor_servicepy)
   - [batch/manager.go → batch/manager.py](#317-batchmanagergo--batchmanagerpy)
   - [batch/devices_batch.go → batch/devices_batch.py](#318-batchdevices_batchgo--batchdevices_batchpy)
4. [주요 변수 정의 모음](#4-주요-변수-정의-모음)
5. [TLV 프로토콜 해설](#5-tlv-프로토콜-해설)
6. [전체 데이터 흐름](#6-전체-데이터-흐름)
7. [의존성 패키지 비교](#7-의존성-패키지-비교)

---

## 1. 프로젝트 구조 비교

```
Go 원본 (aiot-tlv-parser/)          Python 변환본 (parser-python/)
────────────────────────────────    ────────────────────────────────────
main.go                         →   main.py
go.mod                          →   requirements.txt
pkg/
  config/
    config.go                   →   config/config.py
    validation.go               →   config/validation.py
  database/
    models.go                   →   database/models.py
    connection.go               →   database/connection.py
    processor.go                →   database/processor.py
    queries.go                  →   database/queries.py
  mqtt/
    interfaces.go               →   mqtt/interfaces.py
    classifier.go               →   mqtt/classifier.py
    manager.go                  →   mqtt/manager.py
  service/
    device_info_service.go      →   service/device_info_service.py
    event_service.go            →   service/event_service.py
    sensor_service.go           →   service/sensor_service.py
  tlv/
    parser.go                   →   tlv/parser.py
    transformer_v0.go           →   tlv/transformer_v0.py
    transformer_v1.go           →   tlv/transformer_v1.py
  batch/
    manager.go                  →   batch/manager.py
    devices_batch.go            →   batch/devices_batch.py
```

---

## 2. 언어 개념 대조표

| Go 개념 | Python 대응 | 설명 |
|---------|-------------|------|
| `struct` | `@dataclass` | 데이터 구조 정의. Go는 타입이 엄격, Python은 유연 |
| `interface` | `ABC` (추상 베이스 클래스) | 구현 계약 정의 |
| `goroutine` | `threading.Thread` | 동시 실행 단위. Go는 경량 고루틴, Python은 OS 스레드 |
| `channel` | `queue.Queue` | 스레드 간 데이터 전달 |
| `sync.RWMutex` | `threading.RLock` | 읽기/쓰기 동시성 제어 |
| `context.Context` | `threading.Event` | 취소/종료 신호 전달 |
| `atomic.Pointer` | `threading.Lock + 변수 교체` | 원자적 포인터 교체 |
| `log.Printf(...)` | `logger.info(...)` | 로그 출력 |
| `defer` | `finally` / `with` / `atexit` | 클린업 코드 지연 실행 |
| `error` 반환값 | `raise Exception` | 오류 처리 |
| `type assertion` | `isinstance()` | 런타임 타입 확인 |
| `go.mod` | `requirements.txt` | 의존성 관리 파일 |
| `gin` (HTTP) | `Flask` (HTTP) | REST API 서버 프레임워크 |
| `bun` ORM | `psycopg2` | PostgreSQL 드라이버/ORM |
| `paho.mqtt.golang` | `paho-mqtt` | MQTT 클라이언트 라이브러리 |
| `encoding/binary` | `struct` 모듈 | 바이너리 데이터 인코딩/디코딩 |
| `math.Float32frombits` | `struct.unpack(">f", ...)` | IEEE 754 float32 변환 |
| `binary.BigEndian.Uint16` | `struct.unpack(">H", ...)` | Big-Endian uint16 변환 |
| `time.Duration` | `datetime.timedelta` | 시간 간격 표현 |
| `time.UnixMilli(ms)` | `datetime.fromtimestamp(ms/1000)` | Unix 밀리초 → 날짜시간 |

---

## 3. 파일별 상세 비교

### 3.1 main.go → main.py

**역할**: 전체 서비스 초기화 및 HTTP 서버 실행

| 항목 | Go (main.go) | Python (main.py) |
|------|-------------|-----------------|
| 진입점 | `func main()` | `def main()` + `if __name__ == "__main__"` |
| .env 로드 | `godotenv.Load()` | `load_dotenv()` (python-dotenv) |
| HTTP 서버 | `gin.Default()` + `srv.ListenAndServe()` | `Flask(__name__)` + `app.run()` |
| 종료 신호 | `signal.Notify(quit, SIGINT, SIGTERM)` | `signal.signal(SIGINT, ...)` |
| 고루틴 서버 | `go func() { srv.ListenAndServe() }()` | `threading.Thread(target=run_server)` |
| Graceful Shutdown | `srv.Shutdown(ctx)` (5초 타임아웃) | `threading.Event.wait()` 후 cleanup |
| CORS | `router.Use(func(c *gin.Context){...})` | `@app.after_request` 훅 |

**초기화 순서** (Go와 Python 동일):
```
1. .env 로드 → 2. Config 로드 → 3. DB 초기화 → 4. SensorService →
5. BatchManager → 6. 디바이스 목록 대기(최대63초) → 7. MQTT 초기화 →
8. HTTP 서버 시작 → 9. 종료 대기 → 10. 정리(cleanup)
```

---

### 3.2 config/config.go → config/config.py

**역할**: 환경변수 → 설정 객체 변환

#### 구조체(struct) 비교

| Go struct | Python dataclass | 설명 |
|-----------|-----------------|------|
| `ServerConfig` | `ServerConfig` | 서버 포트, 앱 버전 |
| `DatabaseConfig` | `DatabaseConfig` | DB 호스트, 포트, 인증 등 |
| `MQTTConfig` | `MQTTConfig` | 단일 MQTT 브로커 설정 |
| `MQTTConfigs` | `MQTTConfigs` | 5개 브로커 묶음 |
| `BatchConfig` | `BatchConfig` | 배치 작업 설정 |
| `Config` | `Config` | 모든 설정의 최상위 컨테이너 |

#### 헬퍼 함수 비교

| Go 함수 | Python 함수 | 환경변수 타입 |
|---------|------------|-------------|
| `getEnvString(key, default)` | `_get_env_string(key, default)` | 문자열 |
| `getEnvInt(key, default)` | `_get_env_int(key, default)` | 정수 |
| `getEnvBool(key, default)` | `_get_env_bool(key, default)` | 불리언 |
| `getEnvDuration(key, default)` | `_get_env_duration(key, default)` | 시간 간격 |
| `getEnvStringSlice(key, default)` | `_get_env_string_slice(key, default)` | 쉼표 구분 리스트 |

#### 주요 환경변수 → 설정 매핑

| 환경변수 | 설정 필드 | 기본값 |
|---------|---------|--------|
| `ROUTER` | `server.port` | `"3500"` |
| `APP_VERSION` | `server.app_version` | `"1.0.0"` |
| `DB_HOST` | `database.host` | `"localhost"` |
| `DB_PORT` | `database.port` | `5432` |
| `DB_USER` | `database.user` | `"postgres"` |
| `DB_PW` | `database.password` | `""` |
| `DB_NAME` | `database.database` | `"aiot_sensor"` |
| `DB_MAX_CONNECTIONS` | `database.max_connections` | `20` |
| `PROXY_MQTT_HOST` | `mqtt.proxy.host` | `"localhost"` |
| `NS_PARK_MQTT_HOST` | `mqtt.ns_park.host` | `"localhost"` |
| `NC_API_RUI` | `batch.device_api_url` | API URL |
| `NC_APPLICATION_IDS` | `batch.application_ids` | 쉼표 구분 |
| `NC_PW` | `batch.token` | `""` |
| `BATCH_SKIP_TLS_VERIFY` | `batch.skip_tls_verify` | `false` |

---

### 3.3 config/validation.go → config/validation.py

**역할**: 설정값 유효성 검사

| Go 메서드 | Python 함수 | 검사 내용 |
|-----------|------------|---------|
| `(c *Config) Validate()` | `validate_config(cfg)` | 전체 설정 유효성 검사 진입점 |
| `(s *ServerConfig) Validate()` | `validate_server_config(server)` | 포트 번호 유효성 |
| `(d *DatabaseConfig) Validate()` | `validate_database_config(database)` | DB 연결 설정 필수값 |
| `(m *MQTTConfigs) Validate()` | `validate_mqtt_configs(mqtt_configs)` | 5개 브로커 설정 |
| `(m *MQTTConfig) Validate(name)` | `validate_mqtt_config(config, name)` | 단일 브로커 호스트/포트 |
| `(b *BatchConfig) Validate()` | `validate_batch_config(batch)` | API URL 형식, 재시도 횟수 |

---

### 3.4 tlv/parser.go → tlv/parser.py

**역할**: LwM2M TLV 바이너리 프로토콜 파서

#### 클래스/타입 비교

| Go 타입 | Python 클래스 | 설명 |
|---------|--------------|------|
| `AllowedTable []int` | `ALLOWED_TABLE: list` | 허용 테이블 ID 상수 |
| `TLVItem struct` | `TLVItem class` | TLV 아이템 (ID + 값) |
| `ParsedData struct` | `ParsedData class` | 파싱 결과 (테이블명 + 데이터 딕셔너리) |
| `Parser struct` | `Parser class` | 메인 파서 클래스 |

#### 메서드 비교

| Go 메서드 | Python 메서드 | 설명 |
|-----------|--------------|------|
| `NewParser()` | `Parser.__init__()` | 파서 생성 |
| `(p *Parser) DecodeLwM2MTLV(buffer, startIndex)` | `decode_lwm2m_tlv(buffer, start_index)` | 메인 파싱 진입점 |
| `(p *Parser) decodeTLV(tableName, buffer, oldVersion)` | `_decode_tlv(table_name, buffer, old_version)` | 테이블별 디코딩 |
| `(p *Parser) parseTLVItems(tableName, buffer)` | `_parse_tlv_items(table_name, buffer)` | TLV 아이템 순차 파싱 |
| `(p *Parser) readValue(tableName, id, typeByte, buffer, index)` | `_read_value(table_name, id_, type_byte, buffer, index)` | 타입별 값 읽기 |
| `isTableAllowed(tableID)` | `_is_table_allowed(table_id)` | 허용 테이블 확인 |

#### TLV 타입 바이트 해석 비교

| 타입 바이트 | 바이트 수 | Go 처리 | Python 처리 |
|-----------|---------|--------|------------|
| `0xc1` | 1 | `buffer[index] != 0` → bool | `buffer[index] != 0` → bool |
| `0xc2` | 2 | `string(buffer[i:i+2])` | `.decode("latin-1")` |
| `0xc3` | 3 | `string(buffer[i:i+3])` | `.decode("latin-1")` |
| `0xc4` | 4 | `math.Float32frombits(bits)` | `struct.unpack(">f", ...)` |
| `0xc5` | 5 | `string(buffer[i:i+5])` | `.decode("latin-1")` |
| `0xe4` | 4 | `int64(BigEndian.Uint32(...))` | `struct.unpack(">I", ...)` → int |

#### 버전 판별 로직

```
Go:   oldVersion := len(buffer) > 0 && buffer[0] == '1'
Python: old_version = len(buffer) > 0 and buffer[0] == ord('1')

✅ 첫 바이트가 ASCII '1'(= 0x31 = 49) 이면 구버전(V1) 처리
```

---

### 3.5 tlv/transformer_v0.go → tlv/transformer_v0.py

**역할**: 신버전(V0) TLV 데이터 → 필드 딕셔너리 변환

| Go 메서드 | Python 메서드 | 처리 대상 |
|-----------|--------------|---------|
| `Transform(tableName, data, tlvItems)` | `transform(table_name, data, tlv_items)` | 테이블별 분기 |
| `parse3(data, tlvItems)` | `_parse3(data, tlv_items)` | 디바이스 장치 정보 |
| `parse34950(data, tlvItems)` | `_parse34950(data, tlv_items)` | 하천 수위/유속/강수량 |
| `parse34952(data, tlvItems)` | `_parse34952(data, tlv_items)` | 침수 감지 |
| `parse34954(data, tlvItems)` | `_parse34954(data, tlv_items)` | 온도/습도 |
| `parse34955(data, tlvItems)` | `_parse34955(data, tlv_items)` | 경사계 (각도) |
| `parse34956(data, tlvItems)` | `_parse34956(data, tlv_items)` | 화재 경보 |
| `parse34957(data, tlvItems)` | `_parse34957(data, tlv_items)` | 복합 요약1 |
| `parse34958(data, tlvItems)` | `_parse34958(data, tlv_items)` | 복합 요약2 |

#### 특수 로직: parse34957 (V0)

```go
// Go: angle_x, angle_y 모두 있으면 event_code = 1
if angleX, ok := data["angle_x"]; ok && angleX != nil {
    if angleY, ok := data["angle_y"]; ok && angleY != nil {
        data["event_code"] = 1
    }
}
```
```python
# Python
if data.get("angle_x") is not None and data.get("angle_y") is not None:
    data["event_code"] = 1
```

#### reporting_period 단위 변환 (ms → 초)

```go
// Go: ID=26241, uint32 값 / 1000.0
if val, ok := tlv.Value.(uint32); ok {
    data["reporting_period"] = float64(val) / 1000.0
}
```
```python
# Python
if tlv.id == 26241 and isinstance(tlv.value, int):
    data["reporting_period"] = tlv.value / 1000.0
```

---

### 3.6 tlv/transformer_v1.go → tlv/transformer_v1.py

**역할**: 구버전(V1) TLV 데이터 → 필드 딕셔너리 변환

V0과 V1의 주요 차이점:

| 특징 | V0 (신버전) | V1 (구버전) |
|------|------------|------------|
| 테이블 3 지원 | ✅ 지원 | ❌ 미지원 |
| created_at 필드 | ❌ 없음 | ✅ Unix초 × 1000 |
| 34950/34952 처리 | 실제 파싱 | data 그대로 반환 |
| TLV ID 배치 | ID=0부터 시작 | ID=1부터 시작 |

#### created_at 처리 (V1 공통 패턴)

```go
// Go: int64 타입 타임스탬프 × 1000 (초 → 밀리초)
if val, ok := tlv.Value.(int64); ok {
    data["created_at"] = val * 1000
}
```
```python
# Python
if isinstance(tlv.value, int):
    data["created_at"] = tlv.value * 1000
```

#### parse34958 V1 특수 로직: event_code 기본값

```go
// Go: ID=10 없으면 event_code = 0
hasEventCode := false
for _, tlv := range tlvItems {
    if tlv.ID == 10 { hasEventCode = true; break }
}
if !hasEventCode { data["event_code"] = 0 }
```
```python
# Python
has_event_code = any(tlv.id == 10 for tlv in tlv_items)
if not has_event_code:
    data["event_code"] = 0
```

---

### 3.7 database/models.go → database/models.py

**역할**: DB 테이블에 대응하는 데이터 모델 정의

#### 테이블 모델 비교

| Go struct | Python dataclass | DB 테이블 | 주요 필드 |
|-----------|-----------------|---------|---------|
| `DefaultSensorData` | `DefaultSensorData` | (공통 임베딩) | app_eui, dev_eui, device_id, payload, channel, frequency |
| `T3` | `T3` | `t3` | manufacturer, model_number, firmware_version, battery_level |
| `T34950` | `T34950` | `t34950` | water_level, flow_velocity, rain_fall, reporting_period |
| `T34952` | `T34952` | `t34952` | flood_level, reporting_period |
| `T34954` | `T34954` | `t34954` | temperature, humidity, reporting_period |
| `T34955` | `T34955` | `t34955` | angle_x, angle_y, reporting_angle_threshold, relative_angle_value_reset |
| `T34956` | `T34956` | `t34956` | fire_alarm, reporting_period |
| `T34957` | `T34957` | `t34957` | temperature, angle_x, angle_y, event_code |
| `T34958` | `T34958` | `t34958` | acc_x~z, gyro_x~z, angle_x, angle_y, event_code |
| `SensorData` | `SensorData` | `sensor_data` | object_id, payload_tlv(JSON), is_event |
| `Notification` | `Notification` | `notifications` | user_id, app_eui, dev_eui, device_id, object_id |

#### Go 임베딩 vs Python 컴포지션

```go
// Go: struct 임베딩 (DefaultSensorData 의 필드들이 T34950 에 그대로 포함)
type T34950 struct {
    DefaultSensorData          // 임베딩
    WaterLevel float64
    ...
}
```
```python
# Python: dataclass 컴포지션 (sensor_data 필드로 보유)
@dataclass
class T34950:
    sensor_data: DefaultSensorData = field(default_factory=DefaultSensorData)
    water_level: float = 0.0
    ...
```

---

### 3.8 database/connection.go → database/connection.py

**역할**: DB 커넥션 풀 초기화 및 쿼리 실행

| Go 함수/메서드 | Python 함수/메서드 | 설명 |
|--------------|-----------------|------|
| `Init(cfg)` | `init(cfg)` | DB 초기화, 커넥션 풀 생성 |
| `(db *DB) Close()` | `DB.close()` | 커넥션 풀 종료 |
| `(db *DB) ExecuteQuery(query, args...)` | `DB.execute_query(query, args)` | SELECT → dict 리스트 |
| `(db *DB) ExecuteQueryRow(query, args...)` | `DB.execute_query_row(query, args)` | 단일 행 SELECT |
| `(db *DB) ExecuteInsert(query, args...)` | `DB.execute_insert(query, args)` | INSERT |
| `(db *DB) ExecuteInTransaction(fn)` | `DB.execute_in_transaction(fn)` | 트랜잭션 내 실행 |
| `(db *DB) HealthCheck()` | `DB.health_check()` | `SELECT 1` 핑 |
| `(db *DB) GetStats()` | `DB.get_stats()` | 커넥션 풀 통계 |
| `db.NewInsert().Model(&batch).Exec(ctx)` | `bulk_insert(db, table, records)` | 배치 INSERT |

#### 커넥션 풀 설정 비교

```go
// Go (database/sql)
sqldb.SetMaxOpenConns(cfg.MaxConnections)
sqldb.SetMaxIdleConns(cfg.MaxConnections / 2)
sqldb.SetConnMaxLifetime(cfg.IdleTimeout)
```
```python
# Python (psycopg2.pool)
pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=max(1, cfg.max_connections // 2),
    maxconn=cfg.max_connections,
    dsn=dsn,
)
```

---

### 3.9 database/processor.go → database/processor.py

**역할**: 센서 데이터 메모리 큐 → 배치 DB INSERT

#### 주요 변수

| Go 변수 | Python 변수 | 타입 | 설명 |
|---------|-----------|------|------|
| `dp.db` | `self._db` | DB | DB 인스턴스 |
| `dp.threshold` | `self._threshold` | int | 처리 임계값 |
| `dp.interval` | `self._interval` | float(초) | 처리 주기 |
| `dp.t3 []T3` | `self._t3: List[T3]` | list | T3 큐 |
| `dp.t34950 []T34950` | `self._t34950: List[T34950]` | list | T34950 큐 |
| `dp.ctx` | `self._stop_event` | threading.Event | 종료 신호 |
| `dp.cancel` | `self._stop_event.set()` | - | 종료 트리거 |
| `batchSize = 1000` | `_BATCH_SIZE = 1000` | int | 1회 처리 최대 건수 |

#### 배치 처리 로직 비교

```go
// Go: 슬라이스 앞부분 추출 후 제거
batch := (*data)[:batchSize]
*data = (*data)[batchSize:]
_, err := dp.db.NewInsert().Model(&batch).Exec(dp.ctx)
```
```python
# Python: 리스트 슬라이싱
batch = getattr(self, attr)[:_BATCH_SIZE]
setattr(self, attr, current[len(batch):])
bulk_insert(self._db, table_name, batch)
```

---

### 3.10 database/queries.go → database/queries.py

**역할**: 센서 데이터 조회 SQL

| Go 메서드 | Python 메서드 | 설명 |
|-----------|--------------|------|
| `NewQueryService(db)` | `QueryService.__init__(db)` | 서비스 생성 |
| `GetUserIDByAppEUI()` | `get_user_id_by_app_eui()` | 앱EUI → 사용자ID 매핑 조회 |
| `scanRowsWithArray(rows)` | (내부 psycopg2 처리) | PostgreSQL 배열 타입 스캔 |

---

### 3.11 mqtt/interfaces.go → mqtt/interfaces.py

**역할**: MQTT 처리기 인터페이스 정의

| Go interface | Python ABC | 메서드 |
|-------------|-----------|-------|
| `SensorDataProcessor` | `SensorDataProcessor(ABC)` | `ProcessSensorData` / `process_sensor_data` |
| `EventDataProcessor` | `EventDataProcessor(ABC)` | `ProcessEventData`, `StartApplicationIdsUpdate` |

---

### 3.12 mqtt/classifier.go → mqtt/classifier.py

**역할**: MQTT 메시지 분류 및 라우팅

#### 토픽 파싱 로직

```
target 타입 토픽: v3/{appID}/devices/eui-{devEUI}/up
  → topicArr[1] = appID
  → topicArr[3] = "eui-116DE0A1425200D7" → "eui-" 제거 → 대문자화

da 타입 토픽: {appEUI}/{devEUI}/up
  → topicArr[0] = appEUI
  → topicArr[1] = devEUI → 대문자화
```

| Go 메서드 | Python 메서드 | 설명 |
|-----------|--------------|------|
| `NewClassifier(allowedDevices, processor)` | `Classifier.__init__(allowed_devices, sensor_processor)` | 생성 |
| `ClassifyMessage(topic, message)` | `classify_message(topic, message)` | 메인 처리 |
| `getParserType(topic)` | `_get_parser_type(topic)` | "target" or "da" 판별 |
| `parseTopic(topicArr, parserType)` | `_parse_topic(topic_arr, parser_type)` | AppID, DevEUI 추출 |
| `isDeviceAllowed(devEUI)` | `_is_device_allowed(dev_eui)` | 허용 디바이스 확인 |

---

### 3.13 mqtt/manager.go → mqtt/manager.py

**역할**: 다중 MQTT 브로커 연결 관리

| Go 구조체 필드 | Python 인스턴스 변수 | 타입 | 설명 |
|-------------|-----------------|------|------|
| `clients map[string]*ClientInfo` | `self._clients: dict` | dict | 클라이언트 맵 |
| `processor SensorDataProcessor` | `self._processor` | SensorDataProcessor | 데이터 처리기 |
| `mu sync.RWMutex` | `self._lock` | threading.RLock | 동시성 보호 |

| Go 메서드 | Python 메서드 | 설명 |
|-----------|--------------|------|
| `NewManager(configs)` | `Manager.__init__(configs)` | 매니저 생성 |
| `Init(configs, processor)` | `init(configs, processor)` | 클라이언트 초기화 |
| `ConnectClient(name, config)` | `connect_client(name, config)` | 단일 브로커 연결 |
| `DisconnectAll()` | `disconnect_all()` | 전체 연결 해제 |
| `GetClientStatus(name)` | `get_client_status(name)` | 연결 상태 확인 |
| `messageHandler(client, msg)` | `_message_handler(client, userdata, msg)` | 메시지 수신 핸들러 |

**활성화된 브로커** (Go와 Python 동일):
- `ns_park` (NS_PARK_MQTT_HOST)
- `lab` (LAB_MQTT_HOST)

---

### 3.14 service/device_info_service.go → service/device_info_service.py

**역할**: DevEUI → DeviceID 원자적 매핑 관리

#### 변수 정의

| Go 변수 | Python 변수 | 타입 | 설명 |
|---------|-----------|------|------|
| `d.devices atomic.Pointer[map[string]string]` | `self._devices: Dict[str, str]` | dict | DevEUI → DeviceID 매핑 |
| (atomic.Pointer 내부) | `self._lock: threading.RLock` | RLock | 동시성 보호 |

| Go 메서드 | Python 메서드 | 설명 |
|-----------|--------------|------|
| `NewDeviceInfoService()` | `DeviceInfoService.__init__()` | 서비스 생성, 빈 map 초기화 |
| `UpdateDevicesFromBatch(mappings)` | `update_devices_from_batch(device_mappings)` | 배치 데이터로 전체 갱신 |
| `GetDeviceID(devEUI)` | `get_device_id(dev_eui)` | DevEUI → DeviceID |
| `GetDeviceIDSafe(devEUI)` | `get_device_id_safe(dev_eui)` | (DeviceID, bool) 튜플 반환 |
| `ListDeviceEUIs()` | `list_device_euis()` | 전체 EUI 목록 |
| `ListAllowedDevices()` | `list_allowed_devices()` | 전체 DeviceID 목록 |
| `GetDeviceCount()` | `get_device_count()` | 등록 디바이스 수 |

---

### 3.15 service/event_service.go → service/event_service.py

**역할**: 알림 이벤트 비동기 배치 처리

#### 주요 상수

| Go 상수/초기값 | Python 상수 | 값 | 설명 |
|--------------|-----------|---|------|
| `make(chan Notification, 5000)` | `_QUEUE_SIZE = 5000` | 5000 | 알림 큐 최대 크기 |
| `batchSize: 200` | `_BATCH_SIZE = 200` | 200 | 1회 배치 처리 최대 건수 |
| `time.NewTicker(36 * time.Second)` | `_UPDATE_INTERVAL = 36` | 36초 | applicationIDs 갱신 주기 |
| `time.NewTicker(1 * time.Second)` | `_PROCESS_INTERVAL = 1` | 1초 | 큐 처리 주기 |

#### 변수 정의

| Go 변수 | Python 변수 | 설명 |
|---------|-----------|------|
| `e.db` | `self._db` | DB 인스턴스 |
| `e.queryService` | `self._query_service` | 쿼리 서비스 |
| `e.applicationIDs map[string][]string` | `self._application_ids: Dict[str, List[str]]` | 앱EUI → 사용자ID 목록 |
| `e.mu sync.RWMutex` | `self._lock: threading.RLock` | 동시성 보호 |
| `e.notificationQueue chan Notification` | `self._notification_queue: queue.Queue` | 알림 큐 |

---

### 3.16 service/sensor_service.go → service/sensor_service.py

**역할**: MQTT 수신 데이터의 전체 처리 파이프라인

#### 헬퍼 함수 (TLV 딕셔너리에서 타입별 값 추출)

| Go 함수 | Python 함수 | 반환 타입 |
|---------|-----------|---------|
| `getFloat64FromTLV(data, key)` | `_get_float(data, key)` | float |
| `getStringFromTLV(data, key)` | `_get_str(data, key)` | str |
| `getIntFromTLV(data, key)` | `_get_int(data, key)` | int |
| `getBoolFromTLV(data, key)` | `_get_bool(data, key)` | bool |
| `mustMarshalJSON(v)` | `_must_marshal_json(v)` | bytes |
| `getCreatedAt(data)` | `_get_created_at(data)` | datetime |
| `isEvent(data)` | `_is_event(data)` | bool |

#### 이벤트 처리 흐름

```
event_code가 True/1 이면 →
  EventService.get_user_ids_by_app_eui(appID) 로 사용자 목록 조회 →
  각 사용자에게 Notification 생성 →
  EventService.add_notification_to_queue() 에 추가 →
  1초 배치 스케줄러가 DB INSERT
```

---

### 3.17 batch/manager.go → batch/manager.py

**역할**: 배치 작업 라이프사이클 관리

| Go 변수 | Python 변수 | 설명 |
|---------|-----------|------|
| `bm.jobs []BatchJob` | `self._jobs: List[DeviceScheduler]` | 배치 작업 목록 |
| `bm.ctx context.Context` | `self._stop_event: threading.Event` | 종료 신호 |
| `bm.cancel context.CancelFunc` | `self._stop_event.set()` | 종료 트리거 |
| `bm.wg sync.WaitGroup` | `self._threads: List[threading.Thread]` | 스레드 추적 |
| `bm.cfg config.BatchConfig` | `self._cfg` | 배치 설정 |

---

### 3.18 batch/devices_batch.go → batch/devices_batch.py

**역할**: 외부 API에서 디바이스 목록 배치 조회

#### API 응답 구조체

| Go struct | Python (dict 처리) | 필드 |
|-----------|------------------|------|
| `APIApplicationIDs` | `ids["application_id"]` | 애플리케이션 ID |
| `APIDeviceIDs` | `ids["device_id"]`, `ids["dev_eui"]` | 디바이스 IDs |
| `APIDevice` | `device["ids"]`, `device["name"]` | 디바이스 정보 |
| `APIDeviceResponse` | `response["end_devices"]` | 디바이스 목록 |

#### 함수 비교

| Go 함수 | Python 함수 | 설명 |
|---------|-----------|------|
| `GetDevicesBatch(url, token, skip)` | `get_devices_batch(url, token, skip)` | 단일 HTTP GET |
| `GetDevicesBatchWithRetry(url, token, maxRetries, skip)` | `get_devices_batch_with_retry(url, token, max_retries, skip)` | 재시도 로직 포함 |
| `ProcessDevicesBatch(mapping, callback)` | `process_devices_batch(mapping, callback)` | 콜백 호출 |
| `convertDeviceInfoToMapping(devices)` | `_convert_device_info_to_mapping(devices)` | 내부 변환 함수 |

#### 병렬 API 호출 비교

```go
// Go: channel 기반 병렬 처리
results := make(chan map[string]string, len(applicationIds))
for _, appId := range ds.config.ApplicationIds {
    go func(appId string) { results <- fetchDevicesForApp(appId) }(appId)
}
```
```python
# Python: ThreadPoolExecutor 기반 병렬 처리
with ThreadPoolExecutor(max_workers=len(application_ids)) as executor:
    futures = {executor.submit(fetch_for_app, app_id): app_id
               for app_id in self._config.application_ids}
    for future in as_completed(futures):
        result = future.result()
```

---

## 4. 주요 변수 정의 모음

### 허용 테이블 ID와 의미

| 테이블 ID | 변수/상수 | 의미 |
|---------|---------|------|
| `3` | `ALLOWED_TABLE[0]` | 디바이스 장치 정보 (LwM2M Object 3) |
| `34950` | `ALLOWED_TABLE[1]` | 하천 모니터링 (수위, 유속, 강수량) |
| `34952` | `ALLOWED_TABLE[2]` | 침수 감지 |
| `34954` | `ALLOWED_TABLE[3]` | 온도/습도 |
| `34955` | `ALLOWED_TABLE[4]` | 경사계 (각도 센서) |
| `34956` | `ALLOWED_TABLE[5]` | 화재 경보 |
| `34957` | `ALLOWED_TABLE[6]` | 복합 요약 1 (온도+경사) |
| `34958` | `ALLOWED_TABLE[7]` | 복합 요약 2 (가속도+자이로+경사) |

### TLV 타입 바이트 정의

| 十六进制 | 십진수 | 의미 | 값 크기 |
|--------|-------|------|--------|
| `0xc1` | 193 | 1바이트 불리언/정수 | 1 byte |
| `0xc2` | 194 | 2바이트 문자열 | 2 bytes |
| `0xc3` | 195 | 3바이트 문자열 | 3 bytes |
| `0xc4` | 196 | 4바이트 float32 또는 uint32 | 4 bytes |
| `0xc5` | 197 | 5바이트 문자열 | 5 bytes |
| `0xe4` | 228 | 4바이트 타임스탬프 + 16비트 ID | 4 bytes |

### 데이터 처리 타이밍 상수

| 상수/변수 | 값 | 용도 |
|---------|---|------|
| `_BATCH_SIZE` (processor.py) | `1000` | DB 1회 최대 INSERT 건수 |
| `DataProcessor interval` (main) | `1초` | 배치 처리 주기 |
| `_QUEUE_SIZE` (event_service) | `5000` | 알림 큐 최대 크기 |
| `_BATCH_SIZE` (event_service) | `200` | 알림 1회 최대 INSERT 건수 |
| `_UPDATE_INTERVAL` (event_service) | `36초` | applicationIDs 갱신 주기 |
| `_PROCESS_INTERVAL` (event_service) | `1초` | 알림 큐 처리 주기 |
| 디바이스 목록 대기 | `최대 63초` (21회 × 3초) | 초기 디바이스 로딩 대기 |
| 배치 종료 타임아웃 | `30초` | BatchManager.stop_all() |

---

## 5. TLV 프로토콜 해설

TLV는 **Type-Length-Value** 의 약자로, IoT 디바이스가 바이너리 형식으로 데이터를 전송하는 프로토콜입니다.

### 전체 페이로드 구조

```
[byte 0]    : 버전 정보 ('1' = 구버전 V1, 기타 = 신버전 V0)
[byte 1-4]  : (헤더 영역, 미사용)
[byte 5-6]  : 테이블 ID (Big-Endian uint16, 예: 34950)
[byte 7]    : (헤더 영역)
[byte 8+]   : TLV 데이터 영역 (start_index = 8)
```

### TLV 아이템 구조

```
각 TLV 아이템:
[1 byte]  Type Byte  → 값의 크기와 ID 크기를 결정
[1-2 byte] ID        → 리소스 식별자 (0xe4이면 2바이트 ID)
[N byte]  Value      → 실제 데이터
```

### 파싱 예시

```
수신 바이너리: 30 00 00 00 00 88 86 ...
                ^               ^^
                |               byte[5]=0x88, byte[6]=0x86
                버전('0' → V0)  → uint16 = 34950 (0x8886)
                
→ 테이블 34950 허용 ✅
→ 신버전(V0) 처리
→ byte[8:]을 TLV 파싱 시작
```

---

## 6. 전체 데이터 흐름

```
LoRa 디바이스
    │ TLV 바이너리 페이로드 (Base64 인코딩)
    ▼
LoRa 네트워크 서버 (TTS 등)
    │ MQTT 메시지 발행 (JSON with payload field)
    ▼
MQTT Broker (NsPark / Lab)
    │ topic: {appEUI}/{devEUI}/up 또는 v3/{appID}/devices/eui-{devEUI}/up
    ▼
Manager._message_handler()
    │ "up" 포함 여부 확인
    ▼
SensorService.process_sensor_data()
    │
    ├─ Base64 디코딩 → bytes
    ├─ DeviceInfoService.get_device_id(devEUI) → deviceID
    │
    ├─ Parser.decode_lwm2m_tlv(buffer, 8)
    │     ├─ 버전 판별 (byte[0] == '1')
    │     ├─ 테이블 ID 추출 (byte[5:7])
    │     ├─ parseTLVItems() → [TLVItem, ...]
    │     └─ TransformerV0/V1.transform() → ParsedData
    │
    ├─ 테이블별 dataclass 생성 (T3, T34950, ... T34958)
    ├─ DataProcessor.add_data() → 메모리 큐
    │
    ├─ SensorData 생성 → DataProcessor.add_data()
    │
    └─ isEvent? → EventService.add_notification_to_queue()

DataProcessor (백그라운드 스레드, 1초 주기)
    └─ bulk_insert(db, "t34950", batch) → PostgreSQL

EventService (백그라운드 스레드, 1초 주기)
    └─ INSERT notifications → PostgreSQL

BatchManager (백그라운드 스레드, 1시간 주기)
    └─ HTTP GET /applications/{appId}/devices
        → DeviceInfoService.update_devices_from_batch()
```

---

## 7. 의존성 패키지 비교

| Go 패키지 | Python 패키지 | 버전 | 용도 |
|----------|-------------|------|------|
| `gin-gonic/gin` | `flask` | 3.0+ | HTTP REST 서버 |
| `joho/godotenv` | `python-dotenv` | 1.0+ | .env 파일 로드 |
| `lib/pq` + `uptrace/bun` | `psycopg2-binary` | 2.9+ | PostgreSQL 드라이버 |
| `eclipse/paho.mqtt.golang` | `paho-mqtt` | 1.6+ | MQTT 클라이언트 |
| `google/uuid` | `uuid` (표준 라이브러리) | - | UUID 생성 |
| `encoding/binary` | `struct` (표준 라이브러리) | - | 바이너리 인코딩 |
| `encoding/base64` | `base64` (표준 라이브러리) | - | Base64 인코딩/디코딩 |
| `encoding/json` | `json` (표준 라이브러리) | - | JSON 직렬화 |
| `math` | `struct.unpack(">f"...)` | - | float32 변환 |
| `net/http` | `flask.jsonify` | - | HTTP 응답 |
| `sync` | `threading` | - | 동시성 기본 도구 |
| `context` | `threading.Event` | - | 취소/종료 신호 |

---

*이 문서는 `aiot-tlv-parser` Go 코드를 Python으로 변환하면서 생성된 비교 참고문서입니다.*  
*Go 코드 원본은 `aiot-tlv-parser/` 디렉토리, Python 변환본은 `parser-python/` 디렉토리에 있습니다.*
