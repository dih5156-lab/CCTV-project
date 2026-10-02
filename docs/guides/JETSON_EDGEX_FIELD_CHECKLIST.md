# Jetson EdgeX 현장 점검 체크리스트

이 문서는 Jetson Orin 배포 후, EdgeX 연동과 출력 디바이스 전송 경로를 현장 테스트 전에 점검하기 위한 운영 체크리스트입니다.

## 현재 구조 평가

현재 저장소는 현장 테스트를 진행할 수 있는 수준의 배포 자산을 이미 갖추고 있습니다.

- Jetson 전용 배포 파일: [docker-compose.jetson.yml](../../docker-compose.jetson.yml)
- Jetson 전용 컨테이너 이미지: [Dockerfile.jetson](../../Dockerfile.jetson)
- TLV 파서: [Dockerfile](../../Dockerfile)의 `parser` build target
- EdgeX 어댑터: [runners/run_edgex_adapter.py](../../runners/run_edgex_adapter.py)
- 센서 규칙 브리지: [runners/run_sensor_rule_bridge.py](../../runners/run_sensor_rule_bridge.py)
- Action Layer: [runners/run_action_bridge.py](../../runners/run_action_bridge.py)
- Alert API: [runners/run_alert_api.py](../../runners/run_alert_api.py)
- TLV Parser DB 환경값: `.env.jetson`

즉, 방향은 맞습니다.  
다만 현장 안정성을 위해서는 배포 성공 여부보다 아래 3가지를 확인해야 합니다.

1. `AI -> MQTT -> EdgeX Adapter -> EdgeX` 경로가 실제로 이어지는지
2. `TLV -> aiot-parser -> aiot/sensors/# -> sensor-rule-bridge -> aiot/rules/sensor/#` 경로가 실제로 이어지는지
3. `Rule/Alert -> Action Layer -> 스피커/전광판/사이렌` 경로가 실제 장비까지 닿는지
4. 장애 시 `outbox` 와 재시도가 기대대로 동작하는지

## 지금 꼭 확인해야 하는 항목

### 1. JetPack / L4T 버전 정합성

[Dockerfile.jetson](../../Dockerfile.jetson)의 베이스 이미지는 `l4t-jetpack:r36.4.0`이며,
2026-10-02 확인한 운영 호스트는 JetPack 6.2.3/L4T 36.5.2·CUDA 12.6·TensorRT 10.3·cuDNN 9.3·DeepStream 7.1입니다.
호스트와 이미지 태그를 같은 버전이라고 해석하지 않습니다. 현재 검증한 조합을 기준으로 유지하고,
새 장비에서는 아래 설치·실행 검사를 다시 통과해야 합니다.
[NVIDIA JetPack 6.2.3 지원 정보](https://developer.nvidia.com/embedded/jetpack-sdk-623).

### 1-1. 호스트와 컨테이너 의존성 준비

2026-10-02 사용자 결정: 별도 Jetson 설치 재현은 프로젝트 기능 완성 후·정식 배포 전에 수행합니다.
현재 장비 점검을 새 장비 설치 합격으로 표시하지 않습니다. 같은 날 RF 활성화/모델 종류는
현재 운영값이 재생성 후에도 유지되도록 `.env.jetson`에 명시하고 Compose 해석으로 확인했습니다.
외형 모델 경로와 기존 배포 검사 기준의 불일치는 [I46](../planning/STATUS.md)에 남아 있습니다.

현재 Compose는 호스트의 `/opt/nvidia/deepstream`과 NVIDIA 라이브러리를 연결합니다.
Docker 이미지 배포만으로 호스트 준비가 끝나지 않습니다. 대상 Jetson에는 먼저 지원하는
L4T/드라이버·Docker/NVIDIA Container Toolkit·DeepStream 7.1 SDK를 준비해야 합니다.
SDK 배포 파일은 NVIDIA의 Jetson용 패키지를 사용하고 임의로 버전을 올리지 않습니다.
[공식 설치 절차](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Installation.html#jetson-setup).

호스트 SDK 설치 후에는 다음 도구로 필수 Ubuntu 런타임 패키지를 준비합니다.
기본 실행은 모의 설치이며 `--apply`에서만 실제 설치합니다. SDK/JetPack 교체 및 패키지 제거는 하지 않습니다.

```bash
rtk proxy bash scripts/ops/install_jetson_deepstream_dependencies.sh --dry-run
rtk proxy sudo bash scripts/ops/install_jetson_deepstream_dependencies.sh --apply
rtk proxy python3 scripts/health/check_deepstream_env.py --native-only --timeout 15 --json
rtk proxy docker exec cctv-ai-engine python /app/scripts/health/check_deepstream_env.py --native-only --timeout 15 --json
```

검사 항목은 실제 `deepstream-app --version-all` 실행/SDK 7.1 버전과
`nvstreammux`, `nvinfer`, `nvtracker`, `nvv4l2decoder`, `nvvideoconvert` 로드입니다.
하나라도 실패하면 종료 코드 1로 배포 준비 실패를 표시합니다.
컨테이너의 `libgstrtspserver-1.0-0` 등은 Dockerfile 설치 목록에 포함돼 있습니다.
모델/Python 바인딩 검사에는 `--native-only`를 빼고 실제 사용하는 `--config`를 지정합니다.
기본 config 목록을 현재 활성 모델 목록으로 간주하지 않습니다.

2026-10-02에는 호스트의 `libgstrtspserver-1.0.so.0` 누락을 재현했고,
사용자가 도구로 설치한 뒤 실제 호스트·컨테이너 점검을 수행했습니다.
실제 GPU/디스플레이 접근이 제한된 샌드박스의 timeout은 호스트 설치 실패와 구분합니다.

### 1-2. X11 준비와 자동 시작

현재 EGL 경로는 로그인 세션의 X11 접근을 필요로 합니다.
`deploy/systemd/cctv-jetson-stack.service`의 계정·작업 경로·`DISPLAY`·`XAUTHORITY`를 대상 장비에 맞춥니다.
`prepare_jetson_display.sh`는 인증 성공까지 재시도하고 로컬 root만 허용합니다.
서비스는 X11 준비 → 호스트 SDK 검사 → Compose 시작 → 컨테이너 SDK 검사 순서입니다.
SDK 검사 통과만으로 실제 영상 처리나 재부팅 복구까지 합격한 것은 아닙니다.

```bash
rtk proxy systemd-analyze verify deploy/systemd/cctv-jetson-stack.service
rtk proxy sudo install -m 0644 deploy/systemd/cctv-jetson-stack.service /etc/systemd/system/cctv-jetson-stack.service
rtk proxy sudo systemctl daemon-reload
```

설치 후 서비스 활성화 여부를 확인합니다. 위 명령은 실행 중인 스택을 재시작하지 않습니다.
이 서비스의 `start/restart`는 AI·Public API·UI를 재생성하므로 운영 복구 시험 시간에 수행합니다.
새 로그인 세션/재부팅 후 X11 인증, 자동 시작, 실제 프레임 증가를 확인해야 합니다.

### 2. 민감 정보 외부 분리

`.env.jetson`에 스피커 비밀번호가 평문으로 들어갈 수 있습니다.

- 현장 배포 전에는 `.env.jetson`을 실운영용 비밀 파일로 분리하는 것을 권장합니다.
- 저장소에는 예시 파일만 남기고, 실제 장비 비밀번호는 배포 서버의 별도 env 파일이나 비밀 저장소로 옮기는 편이 안전합니다.

### 3. AI 엔진 / 어댑터 헬스체크 강도

현재 `cctv-ai-engine`은 Zone API `8765/health`를 확인하는 HTTP healthcheck를 사용합니다. `cctv-edgex-adapter`는 아직 `kill -0 1` 기반 프로세스 생존 확인만 사용합니다.

- DeepStream의 AI health는 실행 상태와 실제 처리 프레임의 나이를 확인합니다.
  첫 프레임이 없거나 30초 넘게 멈추거나 통계 조회에 실패하면 HTTP 503입니다.
  전체 파이프라인 진행 검사이며 카메라별 영상 품질·모델 정확도·MQTT 전달 합격을 뜻하지 않습니다.
- 어댑터는 프로세스가 살아 있어도 MQTT 또는 EdgeX 연결이 끊긴 상태일 수 있습니다.
- 현장에서는 반드시 별도 점검 스크립트로 `MQTT`, `Core Metadata`, `Core Data`, `Action Layer`를 함께 확인해야 합니다.

## 추가한 점검 스크립트

현장 점검용 스크립트를 추가했습니다.

- [scripts/health/check_jetson_edgex_stack.py](../../scripts/health/check_jetson_edgex_stack.py)

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
.venv/bin/python scripts/health/check_jetson_edgex_stack.py --host 127.0.0.1
```

외형 검색 상태까지 포함:

```bash
.venv/bin/python scripts/health/check_jetson_edgex_stack.py \
  --host 127.0.0.1 \
  --check-appearance-status \
  --public-api-key <PUBLIC_API_KEY>
```

`/api/v1/appearances/status`의 응답 해석 기준은
[../features/APPEARANCES_STATUS_API.md](../features/APPEARANCES_STATUS_API.md)
문서를 기준으로 확인하는 것을 권장합니다.

출력 장비까지 포함:

```bash
.venv/bin/python scripts/health/check_jetson_edgex_stack.py \
  --host 127.0.0.1 \
  --speaker-host 192.168.88.92 --speaker-port 80 \
  --signboard-host 192.168.88.91 --signboard-port 5000
```

JSON 출력:

```bash
.venv/bin/python scripts/health/check_jetson_edgex_stack.py --json
```

DeepStream Python 모듈까지 확인하려면 `--deepstream`을 추가합니다. 이 검사는 명령을 실행한 호스트 Python 환경의 `gi`, `pyds` import를 확인하므로, 컨테이너 내부 런타임을 검증하려면 컨테이너 안에서 실행하거나 `check_deepstream_env.py`를 함께 사용합니다.

## 권장 현장 점검 순서

1. Jetson 호스트에서 GPU 런타임과 Docker를 확인합니다.
2. `docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --build`로 스택을 올립니다.
3. `docker compose --env-file .env.jetson -f docker-compose.jetson.yml ps`로 컨테이너 상태를 확인합니다.
4. AI 엔진만 재시작할 때는 `docker compose --env-file .env.jetson -f docker-compose.jetson.yml restart cctv-ai-engine`을 사용합니다.
5. `.venv/bin/python scripts/health/check_jetson_edgex_stack.py ...`로 인프라 상태를 확인합니다.
6. 외형 검색을 운영할 경우 `--check-appearance-status` 옵션으로 `appearances/status`도 함께 확인합니다.
7. `cctv-ai-engine` 로그에서 카메라 입력과 MQTT 발행 여부를 확인합니다.
8. `cctv-edgex-adapter` 로그에서 카메라 등록, Core Metadata/Core Data 연결 여부를 확인합니다.
9. `cctv-action-layer` 로그에서 MQTT 구독, 장비 제어 성공 여부를 확인합니다.
10. 실제 이벤트 1건을 발생시켜 스피커/전광판/사이렌 반응을 확인합니다.
11. MQTT 또는 EdgeX를 잠시 끊었다가 복구해 outbox 재전송이 동작하는지 확인합니다.

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
