# EdgeX 디바이스 서비스 점검 결과 (2026-08-28)

## 결론

현재 장치는 통신 방향에 따라 두 경로로 운영한다.

- LoRa 센서: 장치가 값을 올리는 uplink 방식이므로 MQTT와 EdgeX Core Data에서 조회한다.
- Dabit 전광판: 서버가 장치에 명령을 내리는 output 방식이므로 EdgeX Core Command를 사용한다.

센서를 `device-rest`로 즉시 조회하는 Core Command는 실제 하드웨어 동작과 맞지 않아 숨겼다. 센서 프로파일은 이벤트 스키마와 단위 이름만 제공한다.

## 운영 데이터 흐름

```text
LoRa 센서
  -> AIoT Parser (TLV decode)
  -> MQTT aiot/sensors/{dev_eui}/{table}
  -> EdgeX Core Data event
  -> Alert/Public API 및 Grafana

AI/Rules/Action Layer
  -> EdgeX Core Command
  -> cctv-device-dabit
  -> Dabit TCP 전광판
```

## 확인된 구성

| 구분 | 운영 이름 | 상태 | 용도 |
| --- | --- | --- | --- |
| 범용 EdgeX Device Service | `device-rest` | 실행 중 | 카메라 등록 및 uplink 센서 메타데이터 연결 |
| 센서 Parser | `aiot-parser` | 실행 중 | LoRa/TLV decode, MQTT·Core Data·Alert 전달 |
| 전광판 Device Service | `cctv-device-dabit` | 정상 | EdgeX 명령을 Dabit TCP 명령으로 변환 |
| 전광판 장치 | `cctv-signboard-01` | 등록됨/UP | `display`, `clear`, `power` 명령 제공 |
| 실센서 장치 | `aiot-SNIOT-F-RVM-001` | 등록됨/UP | `aiot-t34950-river` 프로파일 사용 |

## 이번 점검에서 수정한 문제

### 1. 센서 Metadata 누락

Core Data에는 `aiot-SNIOT-F-RVM-001` 이벤트가 5,406건 있었지만 Metadata에는 장치가 없었다. 등록 도구에 `--device DEVICE_ID:TABLE` 입력을 추가하고 실센서를 등록했다.

```bash
rtk .venv/bin/python edgex/register_aiot_devices.py \
  --metadata-url http://127.0.0.1:59881 \
  --device SNIOT-F-RVM-001:t34950
```

### 2. 센서 이벤트와 프로파일 리소스명 불일치

Parser 원본 필드와 EdgeX 프로파일 필드가 달랐다.

| Parser 원본 | EdgeX 프로파일 |
| --- | --- |
| `water_level_m` | `water_level` |
| `flow_velocity_mps` | `flow_velocity` |
| `rain_fall_mm` | `rain_fall` |
| `temperature_c` | `temperature` |
| `angle_x_deg` | `angle_x` |
| `acc_x_g` | `acc_x` |
| `gyro_z_dps` | `gyro_z` |

MQTT와 Alert API에는 기존 원본 필드를 유지하고, Core Data event를 만들 때만 프로파일 이름으로 변환한다. 프로파일에 없는 IMU 보조 필드는 Core Data reading에서 제외한다.

### 3. 동작하지 않는 센서 GET 명령 노출

LoRa uplink 센서를 Core Command로 조회하면 `No End device parameters defined` 500 오류가 발생했다. 센서 리소스와 묶음 명령을 `isHidden: true`로 바꿔 Core Command에서 제거했다. 센서 최근 값은 다음 경로를 사용한다.

```text
GET http://127.0.0.1:59880/api/v3/event/device/name/{deviceName}
GET http://<Jetson>:9000/api/v1/sensor-readings
```

### 4. 전광판 장치 ID 불일치

EdgeX Metadata는 `cctv-signboard-01`, Device Service 기본값은 `signboard-01`이었다. 기본값을 `cctv-signboard-01`로 통일했다. 명시적으로 `SIGNBOARD_DEVICE_ID`를 지정하면 해당 값을 우선한다.

## 자동 계약 검사

다음 명령은 Metadata 장치, 프로파일, 최근 Core Data event를 비교한다. 데이터를 변경하지 않는다.

```bash
rtk .venv/bin/python scripts/health/check_edgex_device_contracts.py --event-limit 500
```

검사 항목:

- 이벤트 장치가 Metadata에 존재하는지
- 장치와 event의 프로파일이 같은지
- reading 이름이 프로파일에 정의됐는지
- uplink 센서가 실패하는 polling 명령을 노출하는지

종료 코드는 정상 `0`, 계약 불일치 `1`, API 접근 실패 `2`다.

AIoT 운영 점검을 실행하면 같은 계약 검사가 자동으로 포함된다.

```bash
AIOT_PILOT_CHECK=1 rtk ./scripts/ops/run_operation_check.sh
```

최근 이벤트 조회량은 기본 500건이며 필요할 때만 조절한다.

```bash
AIOT_PILOT_CHECK=1 EDGEX_CONTRACT_EVENT_LIMIT=1000 \
  rtk ./scripts/ops/run_operation_check.sh
```

계약 검사는 읽기 전용이다. 문제가 발견돼도 장치나 프로파일을 자동으로
변경하지 않고 운영 점검 보고서에 `EdgeX Device Contracts` 실패로 기록한다.

## 배포 상태와 다음 반영

EdgeX Metadata의 센서 프로파일 갱신과 실센서 등록은 적용됐다. Parser 필드 정규화와 전광판 기본 ID 변경은 소스 및 테스트까지 완료했지만, 진행 중인 외형 학습의 CPU/I/O 간섭을 피하기 위해 컨테이너 재빌드는 아직 하지 않았다.

학습 종료 후 다음 두 서비스만 선택적으로 재빌드한다.

```bash
rtk docker compose --env-file .env.jetson -f docker-compose.jetson.yml \
  up -d --no-deps --build aiot-parser cctv-device-dabit
```

반영 후 계약 검사와 상태를 확인한다.

```bash
rtk .venv/bin/python scripts/health/check_edgex_device_contracts.py --event-limit 500
rtk docker ps --filter name=aiot-parser --filter name=cctv-device-dabit
```

기존 Core Data에 저장된 과거 reading 이름은 자동 변경하지 않는다. 재배포 이후 생성되는 새 event부터 정규화된다.
