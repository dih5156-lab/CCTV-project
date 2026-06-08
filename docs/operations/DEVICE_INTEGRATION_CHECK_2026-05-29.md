# 디바이스 연동 점검 기록 - 2026-05-29

## 결론

API와 내부 이벤트 흐름은 정상이다.
운영 점검 wrapper의 `FAIL`은 Docker 권한 문제에서 발생했다.
실제 알람 장비 연결은 아직 완료로 볼 수 없다.

- Public API / Alert API / Action Layer: PASS
- AI Engine 내부 관리 API: PASS
- 데이터 흐름 smoke test: PASS
- 주요 Docker 컨테이너: running/healthy
- 스피커 TCP 연결: FAIL
- 전광판 TCP 연결: FAIL
- 경광등 설정: FAIL
- 운영 점검 wrapper 전체 결과: FAIL

## 실행 기준

- 점검일: 2026-05-29
- 시작 기준 시각: 10:04 KST
- 작업 브랜치: main

## 운영 점검 결과

```bash
./scripts/ops/run_operation_check.sh
```

결과:

- runtime secret consistency: FAIL
- deployment smoke: PASS
- data flow smoke: PASS

실패 원인:

```text
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

판단:

- 핵심 API나 이벤트 흐름 실패가 아니다.
- 현재 사용자 권한으로 Docker socket 접근이 막혀 컨테이너 런타임 비밀값 확인 단계가 실패했다.

## 데이터 흐름 smoke test

```bash
.venv/bin/python scripts/smoke/smoke_test_data_flow.py
```

결과: PASS

통과 항목:

- Alert API alert 수신
- Alert API sensor reading 수신
- Action Layer event 수신
- Action Layer metrics 노출
- Public API metrics endpoint

## 서비스 health 확인

확인한 항목:

- Public API health: PASS
- Public API readiness: PASS
- Action Layer health: PASS
- Alert API health: PASS
- Zone API health: PASS
- Camera Model API health: PASS
- Face API health: PASS

Public API readiness 기준으로 `action-layer`, `alert-api` dependency는 모두 `up` 상태였다.

## Docker 컨테이너 상태

`sudo docker ps` 기준 주요 서비스는 실행 중이다.

- cctv-ai-engine: Up, healthy
- cctv-edgex-adapter: Up, healthy
- cctv-action-layer: Up, healthy
- cctv-sensor-rule-bridge: Up, healthy
- aiot-parser: Up, healthy
- cctv-alert-api: Up, healthy
- cctv-public-api: Up, healthy
- edgex-mqtt-broker: Up
- cctv-public-demo-ui: Up
- cctv-media-server: Up
- EdgeX core services: Up

## 알람 장비 연결 확인

```bash
.venv/bin/python scripts/health/check_alarm_devices.py
```

결과: FAIL

- speaker
  - host: 192.168.88.92
  - port: 80
  - reachable: false
  - detail: timed out
- signboard
  - host: 192.168.88.91
  - port: 5000
  - reachable: false
  - detail: No route to host
- siren
  - configured: false
  - missing env: SIREN_HOST, SIREN_USER, SIREN_PASSWORD

## 필드 네트워크 확인

```bash
.venv/bin/python scripts/health/check_field_network.py
```

결과: FAIL

- speaker route: OK
- signboard route: OK
- siren: not configured

판단:

- 스피커/전광판 대역으로 라우팅 자체는 잡힌다.
- 실제 TCP 연결은 실패하므로 장비 전원, 케이블, IP, 포트, 방화벽 확인이 우선이다.

## Docker 권한 조치

운영 점검 wrapper의 Docker 권한 실패를 해결하기 위해 현재 사용자 `sawwave`를 `docker` 그룹에 추가했다.

```bash
sudo -S usermod -aG docker sawwave
```

결과:

- 명령 실행: 성공
- 현재 세션의 `docker ps`: 아직 FAIL

현재 세션에서 바로 실패한 이유:

```text
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

판단:

- Linux 그룹 변경은 기존 로그인 세션에 바로 반영되지 않는다.
- 로그아웃/로그인 또는 재부팅 후 새 터미널에서 다시 확인해야 한다.

재확인 명령:

```bash
docker ps
./scripts/ops/run_operation_check.sh
```

## 다음 조치

1. 로그아웃/로그인 또는 재부팅 후 `docker ps` 확인
2. `./scripts/ops/run_operation_check.sh` 재실행
3. 스피커 `192.168.88.92:80` 전원/IP/포트 확인
4. 전광판 `192.168.88.91:5000` 전원/IP/포트 확인
5. 경광등 사용 여부에 따라 `SIREN_HOST`, `SIREN_USER`, `SIREN_PASSWORD` 설정
