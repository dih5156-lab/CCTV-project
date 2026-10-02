# Jetson 배포 준비도 검증 기록

## 시험 정보

| 항목 | 내용 |
|---|---|
| 시험 일시 | 2026-09-07 09:55 KST 전후 |
| 대상 환경 | Jetson ARM64 |
| 검사 명령 | `.venv/bin/python scripts/health/check_deployment_readiness.py` (`auto` 대상) |
| 결과 | PASS |

## 통과 항목

- 민감정보 기본값 및 런타임 secret 연결
- Dockerfile 복사 경로와 Compose 런타임 가정
- 모델 manifest·artifact·디렉터리 구조
- Grafana 대시보드 JSON
- 카메라-출력 장치 매핑
- Python·parser 컴파일
- 기본·Jetson Compose 설정
- Action Layer·Alert API·EdgeX Adapter·Kuiper 실행 진입점
- Docker build 및 runtime cleanup 스크립트 문법

## EdgeX Device 계약 후속 검증

초기 실행 중인 parser 이미지가 오래된 상태여서 하천 센서 이벤트의
리소스명이 프로파일 정의와 달랐다. `aiot-parser` 이미지를 최신
`parser-python/mqtt/edgex_forwarder.py` 기준으로 재빌드·재기동한 뒤,
장치 제어가 없는 테스트 센서 이벤트 1건을 Core Data에 생성해 검증했다.

```text
devices=31
profiles=17
events=500
issues=0
result=PASS
```

현재 parser 컨테이너 내부의 하천 리소스 매핑은 다음과 같다.

```text
water_level_m     → water_level
flow_velocity_mps → flow_velocity
rain_fall_mm      → rain_fall
reporting_period_s → reporting_period
```

기존에 저장된 잘못된 이벤트가 아니라 최신 코드로 생성된 이벤트를 기준으로
계약 검사가 통과했음을 확인했다. 이 시험은 센서 데이터 계약 검증용이며,
실제 하천 센서 uplink 수신을 대체하지 않는다.

## 검사 대상 주의사항

배포 준비도 검사기는 기본적으로 `auto` 대상에서 실행된다. ARM64 호스트에서는
자동으로 Jetson 대상을 선택하므로 기본 `docker-compose.yml`의 일부 EdgeX
이미지가 amd64 전용일 가능성 때문에 잘못 실패하지 않는다. 이는 실제 배포
구성인 `docker-compose.jetson.yml`을 기준으로 검사하기 위한 동작이다.

Jetson 현장에서는 다음 명령을 사용한다.

```bash
.venv/bin/python scripts/health/check_deployment_readiness.py
```

검사 대상을 명시해야 하는 경우에는 `--deployment-target jetson`을 사용한다.
일반 Compose의 ARM64 호환성을 별도로 엄격하게 확인하려면
`--deployment-target generic`을 사용한다.

## 운영 상태 확인

검증 당시 Action Layer는 Direct 모드, MQTT 연결 상태, health `up`이었다.
전광판 Device Service와 EdgeX 결과 수집기도 healthy 상태였다.

## 데이터 흐름 smoke 재검증

최초 실행에서는 Action Layer의 `INTERNAL_SERVICE_TOKEN`이 실행 셸에 전달되지
않아 `/events` 검사만 `401 Unauthorized`로 실패했다. 서비스 장애나 이벤트
처리 오류가 아니며, 운영 Compose에 설정된 인증 토큰을 smoke 프로세스에도
주입해야 하는 실행 절차 문제였다.

`.env.jetson`을 셸에 안전하게 주입한 뒤 같은 검사를 재실행한 결과는 다음과
같다. 토큰 값 자체는 로그에 출력하지 않았다.

```bash
set -a
source .env.jetson
set +a
.venv/bin/python scripts/smoke/smoke_test_data_flow.py
```

```text
alert api accepts alert                         PASS (202)
alert api accepts sensor reading                PASS (202)
action layer accepts event                      PASS (200)
action layer metrics expose handled events     PASS (200)
public api metrics endpoint                     PASS (200)
result                                         PASS
```

운영 절차는 [OPERATIONS_RUNBOOK.md](../guides/OPERATIONS_RUNBOOK.md)의 배포 전
체크리스트에도 반영했다.

## 2026-09-07 운영 재검증

11:29 KST에 실행 중인 Jetson 스택을 대상으로 호스트·컨테이너·현장
전광판 연결 상태를 다시 확인했다.

검사 명령:

```bash
.venv/bin/python scripts/health/check_jetson_edgex_stack.py \
  --host 127.0.0.1 --timeout 3 \
  --signboard-host 192.168.88.91 --signboard-port 5000 \
  --check-device-contracts --json
```

검증 결과:

```text
MQTT Broker             PASS  127.0.0.1:1883 연결 성공
Redis                   PASS  127.0.0.1:6379 연결 성공
Signboard Device        PASS  192.168.88.91:5000 연결 성공
EdgeX Core Metadata     PASS  HTTP 200
EdgeX Core Data         PASS  HTTP 200
AIoT Parser             PASS  컨테이너 running/healthy fallback
Alert API               PASS  HTTP 200
Action Layer            PASS  health=up, mqtt=connected, mode=auto
Public API              PASS  HTTP 200
EdgeX Device Contracts  PASS  devices=31, profiles=17, events=500, issues=0
failures                0
```

AIoT Parser DB와 Parser HTTP 포트는 호스트에 직접 노출되지 않지만,
컨테이너 상태가 `running/healthy`여서 Jetson Compose의 의도된 포트 구성으로
판정했다. 이 결과는 현장 전광판 TCP 연결과 EdgeX 계약까지 확인한 재검증이며,
스피커·사이렌의 실제 물리 출력 검증을 포함하지 않는다.

## 2026-09-07 낙상 추적 경로 코드 검증

실제 영상·장치가 아닌 인프로세스 통합 테스트로 대표 낙상 이벤트의
식별자 연결을 검증했다.

```text
MQTT fall_detected 이벤트
  → ActionBridge 이벤트 정규화
  → speaker 명령 생성
  → command_id = fall-e2e-1:speaker
  → 장치 결과 acknowledged
  → SQLite 명령 이력 저장
```

검증 항목:

- 입력 `event_id=fall-e2e-1` 보존
- 입력 `request_id=fall-request-1` 보존
- 논리 명령 `command_id=fall-e2e-1:speaker` 생성
- 모의 스피커 응답 `acknowledged` 기록

실행 결과는 `tests/test_integration_action_pipeline.py`의 낙상 시나리오와
관련 Action/Transport 테스트에서 `82 passed`로 확인했다. 이는 영상 모델의
실제 낙상 판정이나 물리 장치 출력 증빙이 아니므로, 현장 UAT 완료로 간주하지 않는다.

## 미완료 항목

- 스피커·사이렌 실제 물리 출력 UAT
- EdgeX 결과 timeout·재시도 정책의 운영 확정
- 시각적 장치 출력 사진 증빙
