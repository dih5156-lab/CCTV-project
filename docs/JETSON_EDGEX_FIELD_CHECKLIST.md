# Jetson EdgeX 현장 점검 체크리스트

이 문서는 Jetson Orin 배포 후, EdgeX 연동과 출력 디바이스 전송 경로를 현장 테스트 전에 점검하기 위한 운영 체크리스트입니다.

## 현재 구조 평가

현재 저장소는 현장 테스트를 진행할 수 있는 수준의 배포 자산을 이미 갖추고 있습니다.

- Jetson 전용 배포 파일: [docker-compose.jetson.yml](/C:/Users/dih51/Documents/GitHub/CCTV-project/docker-compose.jetson.yml)
- Jetson 전용 컨테이너 이미지: [Dockerfile.jetson](/C:/Users/dih51/Documents/GitHub/CCTV-project/Dockerfile.jetson)
- TLV 파서: [Dockerfile.parser](/C:/Users/dih51/Documents/GitHub/CCTV-project/Dockerfile.parser)
- EdgeX 어댑터: [runners/run_edgex_adapter.py](/C:/Users/dih51/Documents/GitHub/CCTV-project/runners/run_edgex_adapter.py)
- 센서 규칙 브리지: [runners/run_sensor_rule_bridge.py](/C:/Users/dih51/Documents/GitHub/CCTV-project/runners/run_sensor_rule_bridge.py)
- Action Layer: [runners/run_action_bridge.py](/C:/Users/dih51/Documents/GitHub/CCTV-project/runners/run_action_bridge.py)
- Alert API: [runners/run_alert_api.py](/C:/Users/dih51/Documents/GitHub/CCTV-project/runners/run_alert_api.py)
- TLV Parser DB 환경값: [.env.jetson](/C:/Users/dih51/Documents/GitHub/CCTV-project/.env.jetson)

즉, 방향은 맞습니다.  
다만 현장 안정성을 위해서는 배포 성공 여부보다 아래 3가지를 확인해야 합니다.

1. `AI -> MQTT -> EdgeX Adapter -> EdgeX` 경로가 실제로 이어지는지
2. `TLV -> aiot-parser -> aiot/sensors/# -> sensor-rule-bridge -> aiot/rules/sensor/#` 경로가 실제로 이어지는지
3. `Rule/Alert -> Action Layer -> 스피커/전광판/사이렌` 경로가 실제 장비까지 닿는지
4. 장애 시 `outbox` 와 재시도가 기대대로 동작하는지

## 지금 꼭 확인해야 하는 항목

### 1. JetPack / L4T 버전 정합성

[Dockerfile.jetson](/C:/Users/dih51/Documents/GitHub/CCTV-project/Dockerfile.jetson)은 `r36.4.0` 기준인데, [.env.jetson](/C:/Users/dih51/Documents/GitHub/CCTV-project/.env.jetson)에는 `L4T_TAG=r36.2.0-pth2.1-py3`가 남아 있습니다.

- 실제 Jetson이 JetPack 6.2면 `r36.4.x` 기준으로 맞추는 편이 안전합니다.
- 현재 compose는 `L4T_TAG`를 직접 쓰지 않더라도, 운영 문서와 실제 장비 버전이 어긋나 있으면 현장 대응이 꼬이기 쉽습니다.

### 2. 민감 정보 외부 분리

[.env.jetson](/C:/Users/dih51/Documents/GitHub/CCTV-project/.env.jetson)에 스피커 비밀번호가 평문으로 들어 있습니다.

- 현장 배포 전에는 `.env.jetson`을 실운영용 비밀 파일로 분리하는 것을 권장합니다.
- 저장소에는 예시 파일만 남기고, 실제 장비 비밀번호는 배포 서버의 별도 env 파일이나 비밀 저장소로 옮기는 편이 안전합니다.

### 3. AI 엔진 / 어댑터 헬스체크 강도

현재 `cctv-ai-engine`, `cctv-edgex-adapter`의 compose 헬스체크는 `kill -0 1` 수준입니다.

- 프로세스가 살아 있어도 MQTT 연결이나 EdgeX 연결이 끊긴 상태일 수 있습니다.
- 현장에서는 반드시 별도 점검 스크립트로 `MQTT`, `Core Metadata`, `Core Data`, `Action Layer`를 함께 확인해야 합니다.

## 추가한 점검 스크립트

현장 점검용 스크립트를 추가했습니다.

- [scripts/check_jetson_edgex_stack.py](/C:/Users/dih51/Documents/GitHub/CCTV-project/scripts/check_jetson_edgex_stack.py)

기본 점검 항목:

- MQTT 브로커 `1883`
- Redis `6379`
- AIoT Parser DB `5432`
- EdgeX Core Metadata `/api/v3/ping`
- EdgeX Core Data `/api/v3/ping`
- AIoT Parser `/health`
- Alert API `/health`
- Action Layer `/health`
- Public API `/api/v1/health`
- 선택적으로 Public API `/api/v1/appearances/status`
- 선택적으로 스피커 / 전광판 / 사이렌 TCP 연결

예시:

```bash
python scripts/check_jetson_edgex_stack.py --host 127.0.0.1
```

외형 검색 상태까지 포함:

```bash
python scripts/check_jetson_edgex_stack.py \
  --host 127.0.0.1 \
  --check-appearance-status \
  --public-api-key <PUBLIC_API_KEY>
```

`/api/v1/appearances/status`의 응답 해석 기준은
[APPEARANCES_STATUS_API.md](APPEARANCES_STATUS_API.md)
문서를 기준으로 확인하는 것을 권장합니다.

출력 장비까지 포함:

```bash
python scripts/check_jetson_edgex_stack.py \
  --host 127.0.0.1 \
  --speaker-host 192.168.88.92 --speaker-port 80 \
  --signboard-host 192.168.88.91 --signboard-port 5000
```

JSON 출력:

```bash
python scripts/check_jetson_edgex_stack.py --json
```

## 권장 현장 점검 순서

1. Jetson 호스트에서 GPU 런타임과 Docker를 확인합니다.
2. `docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --build`로 스택을 올립니다.
3. `docker compose -f docker-compose.jetson.yml ps`로 컨테이너 상태를 확인합니다.
4. `python scripts/check_jetson_edgex_stack.py ...`로 인프라 상태를 확인합니다.
5. 외형 검색을 운영할 경우 `--check-appearance-status` 옵션으로 `appearances/status`도 함께 확인합니다.
6. `cctv-ai-engine` 로그에서 카메라 입력과 MQTT 발행 여부를 확인합니다.
7. `cctv-edgex-adapter` 로그에서 카메라 등록, Core Metadata/Core Data 연결 여부를 확인합니다.
8. `cctv-action-layer` 로그에서 MQTT 구독, 장비 제어 성공 여부를 확인합니다.
9. 실제 이벤트 1건을 발생시켜 스피커/전광판/사이렌 반응을 확인합니다.
10. MQTT 또는 EdgeX를 잠시 끊었다가 복구해 outbox 재전송이 동작하는지 확인합니다.

## 현장 테스트 전 최종 권장사항

반드시 필요한 항목:

- Jetson JetPack 버전과 컨테이너 기준 버전 재확인
- `.env.jetson`의 민감 정보 외부 분리
- 카메라 RTSP 주소 고정 확인
- 스피커 / 전광판 / 사이렌 IP 고정 확인
- 현장 네트워크에서 `1883`, `8000`, `8080` 접근 정책 확인
- 장애 복구 테스트 1회 수행

있으면 더 좋은 항목:

- `cctv-ai-engine`, `cctv-edgex-adapter`에 더 강한 readiness 체크 추가
- 현장용 `docker compose logs` 수집 스크립트 추가
- 테스트 이벤트 자동 발생 스크립트 추가

## 한 줄 결론

현재 저장소는 Jetson + EdgeX + 출력 디바이스 현장 테스트를 진행할 수 있는 구조입니다.  
다만 배포 전에 `버전 정합성`, `비밀값 분리`, `실제 네트워크/장비 연결 점검 자동화` 세 가지는 꼭 챙기는 것이 좋습니다.
