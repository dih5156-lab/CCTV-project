# 스마트폴 Jetson·미니 PC 서비스 분리안

10-23 현재 우선 범위는 **시험 미니PC 1대(Windows 10·128GB·192.168.0.20)의 탐지 결과·센서값·이벤트 이미지 저장 POC**다. Windows를 유지하고 상시 녹화는 제외한다. OpenSSH/SFTP 전달은 검토/인증 대기이며 운영 전환·자동 이미지 전송/복구 검증 완료가 아니다. 아래 MQTT/EdgeX 등 서비스 분리안은 별도 후속 설계로 유지한다. 시험 장비 1대의 확인을 전체 폴 장비 사양 확정으로 확대하지 않는다. [확정 범위·접속 상태](../planning/POC_SCOPE.md#minipc-storage-20261023).

10-22 **A 호스트 전체 장애 대응**의 독립 영상 경로는 각 폴의 스위치/AP를 무선 백홀로 연결하는 안이다. 연차보고서의 AEC 간 원본 데이터 전달도 재확인했으므로, A가 살아 있는 연산 분산에는 운영 설정을 유지하는 SSH 전달 경로부터 시험한다. [범위 구분](../planning/EDGE_ORCHESTRATION_POC.md#report-reassessment-20261022). A/B를 같은 물리 스위치에 연결하는 것은 로컬 실험 대안이다. CCTV→A 폴 네트워크→무선 백홀→B 경로에서 A Jetson을 중계기로 쓰지 않는다. AP 모델/브리지·라우팅 지원과 실제 배선은 미확인이며 이번 운영 변경은 없다. [확인된 구성·주소 선택·단계별 검증](../planning/EDGE_ORCHESTRATION_POC.md#wireless-plan-20261022).

10-22 연계: 고정 역할 분리에 더해 사용자 요청의 **Jetson2대·CCTV1대 분석 담당 전환**은 [엣지 오케스트레이션 구현 계획](../planning/EDGE_ORCHESTRATION_POC.md)을 따른다. 기존 개발 PC/미니PC의 단일 제어기·전용 AI 프로세스 인계·결과 검증을 제안하며, 아래 서비스 분리나4폴 전체를 구현 완료한 것으로 표시하지 않는다.

2026-10-19 KST · 상태: 설계 제안, 분리 Compose/제품 코드 미구현·운영 미반영.

사용자 확인은 **4폴×CCTV4대=16카메라, 4폴×SAWWAVE AP4대=16AP**, 여분 Orin AGX2대·NX2대다. 폴마다 Jetson·미니 PC와 SFC8000GHP/NEXT-POE1411GDT를 배치하는 안을 기준으로 정리했다. 전체 폴 미니 PC의 OS/CPU/RAM과 두 스위치의 실제 확보 수량·전원, AP 세부 모델/용도는 미확정이다(시험용 1대의 후속 확인은 상단 참조). [POC 원본](../planning/POC_SCOPE.md#smartpole-four-sites-20261019).

## 1. 권장 경계: 데이터 소유권을 유지하는 첫 분리

**Jetson은 AI·영상·영상 조회 묶음, 미니 PC는 MQTT·EdgeX 서버·센서·장치 제어 묶음**으로 나눈다. 처음 제시했던 역할 개념을 현재 공유 파일 구조에 맞춰 구체화한 안이다. 모든 비GPU 서비스를 미니 PC로 옮기는 것보다 제품 코드 변경이 작다.

소스 [Compose](../../docker-compose.jetson.yml)와 실제 실행 목록을 함께 확인했다. 실행 중인 `cctv-device-speaker`, `cctv-device-siren`, `cctv-edgex-command-result`, Prometheus/Grafana의 일부 정의가 현재 Jetson Compose에는 없다. 기존 컨테이너가 실행 중이라는 사실을 새 장비에서 재현 가능하다는 뜻으로 해석하지 않는다. 실제 AI는 별도 현장 override를 사용한다. 비밀값이 포함된 현장 override를 문서/저장소에 복제하지 말고 이미지·모델·허용 설정을 선별해 배포 명세로 옮긴다.

### Jetson에 유지

| 서비스/실행 이름 | 이유와 함께 둘 데이터 |
|---|---|
| `cctv-ai-engine` | DeepStream/TensorRT·카메라/구역/모델/얼굴/스트림 API. 모델·엔진·GPU venv·얼굴 DB·외형 DB·crop·이벤트 outbox |
| `cctv-media-server` | AI RTSP 출력과 WebRTC 중계. 내부 송출 주소를 유지 |
| `cctv-public-api` | 외형 기록/조건/crop, 카메라 설정, 이벤트/센서 JSONL, 장치 결과 DB를 로컬에서 읽는 API |
| `cctv-alert-api` | Public API가 읽는 `alert_api_events.jsonl`, `sensor_readings.jsonl` 기록. 미니 PC의 발신자들은 Jetson으로 HTTP 전송 |
| `cctv-edgex-adapter` | AI 이벤트→미니 PC EdgeX. 선택형 `AIOT_COMMANDS_ENABLED` 기능이 외형 DB/crop을 직접 조회하므로 함께 유지 |
| `cctv-edgex-command-result` | MQTT 결과를 Public API가 읽는 `edgex_command_results.db`에 기록. 서비스 정의 재현 필요 |
| `public-demo-ui` (컨테이너 `cctv-public-demo-ui`) | Nginx가 같은 장비의 Public API·AI API·MediaMTX를 중계. 기존 프록시 이름과 웹 경로 유지 |

어댑터의 AIoT 명령 기능은 기본 Compose에서 `false`다. 이번 분리를 이유로 켜거나 끄지 않는다. 기능을 유지할 수 있는 배치를 선택한 것이다.

### 미니 PC에 배치

| 서비스 묶음 | 실제 Compose 서비스/실행 이름 |
|---|---|
| 메시지·저장 | `edgex-mqtt-broker`, `consul`, `database`(Redis), `aiot-parser-db`(PostgreSQL) |
| EdgeX 코어 | `core-common-config-bootstrapper`(초기화), `core-metadata`, `core-data`, `core-command`, `device-rest` |
| 규칙 처리 | `rules-engine`(eKuiper), `app-rules-engine`, `cctv-kuiper-rule-loader`(규칙 등록 작업) |
| 센서 | `aiot-parser`, `cctv-sensor-rule-bridge` |
| 현장 제어 | `cctv-action-layer`, `cctv-device-signboard`, `cctv-device-speaker`, `cctv-device-siren` |
| 운영 화면/지표 | `ui`(EdgeX 상태 화면), `prometheus`, `grafana` |
| 기존 지원 서비스 | `support-notifications`, `support-scheduler`는 첫 분리에서 유지하고 실제 사용 여부를 별도 확인 |

사이렌 서비스 배치와 사이렌 실물 보유/시험은 다르다. 실물 미보유 상태에서 대상 활성화를 추가하지 않는다. `device-virtual`은 가상 장치 시험용으로 구분하고 현장 상시 필수 서비스로 자동 포함하지 않는다. 개발용 진행판·외형 검수 웹·학습용 Compose는 현장 기능에 필요한지 확인 후 제외한다.

미니 PC는 Linux/Docker를 가정한 제안이다. 비GPU 서비스는 기존 [Dockerfile](../../Dockerfile)의 `action`/`parser` target을 재사용하고 미니 PC 아키텍처로 빌드·검증한다. Jetson용 ARM64 이미지/SDK를 그대로 복사하지 않는다. AP 관리용 회사 APC는 이 프로젝트의 서비스로 확인되지 않았으며, 필요한 경우 별도 시스템으로 연계한다.

## 2. 공유 파일 때문에 함께 두는 이유

| 데이터 | 현재 접근 지점 | 첫 분리에서 소유할 장비 |
|---|---|---|
| `appearances.db`·외형 조건·crop | AI의 AppearancePipeline, Public API `appearances.py`/`search.py`, 어댑터 AIoT 조회 | Jetson |
| 이벤트·센서 JSONL | Alert API 기록, Public API `events.py`/`sensor_readings.py` 직접 조회 | Jetson |
| `edgex_command_results.db` | 결과 collector 기록, Public API `command_results.py` 직접 조회 | Jetson |
| `event_reviews.db` | Public API 평가 기록 | Jetson |
| 얼굴 DB·사진·모델·TensorRT 캐시 | AI 엔진/내부 API | Jetson |
| AI 이벤트 outbox·어댑터 outbox/명령 이력 | AI/어댑터 발행·복구 | Jetson |
| Action DB·입력 보존 DB·HTTP outbox | Action Layer | 미니 PC |
| 장치별 `device_requests/*.sqlite3` | 각 장치 서비스의 중복/실행 불명 처리 | 미니 PC |
| 센서 원본/투영·parser outbox·PostgreSQL, EdgeX Redis | 파서·센서/EdgeX 서비스 | 미니 PC |

근거: [AIoT 조회 구성](../../runners/run_edgex_adapter.py), [외형 조건](../../src/api/v1/appearances.py), [검색/crop](../../src/api/v1/search.py), [장치 결과 조회](../../src/api/v1/command_results.py), [Alert 기록](../../runners/run_alert_api.py).

같은 SQLite 파일을 NFS/SMB로 양쪽에 공유하지 않는다. 장비별 영속 경로를 유지하며 복원할 때도 데이터의 소유 장비를 지킨다. Action 이력과 장치 실행 이력을 함께 보존하고, 이전 Jetson의 제어 서비스와 새 미니 PC 제어 서비스를 동시에 활성화하지 않는다.

## 3. 어디를 수정할 것인가

| 우선순위 | 파일/범위 | 수정 내용 |
|---|---|---|
| 1 | [docker-compose.jetson.yml](../../docker-compose.jetson.yml), [docker-compose.yml](../../docker-compose.yml) | 기존 정의를 근거로 Jetson/미니 PC 두 실행 구성을 작성. 로컬 서비스만 `depends_on`에 남기고, 원격 서비스는 접속·재시도/복구 대상으로 처리. 실행 중인데 정의가 없는 장치/결과/지표 서비스를 복원 |
| 1 | [.env.jetson.example](../../.env.jetson.example), 새 미니 PC용 환경 예시 | 장비별 주소·데이터 경로·식별자·바인딩 분리. 비밀값은 현장별 별도 주입. 아래 주소는 신규 제안 값이지 이미 지원되는 env라고 가정하지 않음 |
| 1 | Compose `extra_hosts`, `command`, `environment`, `ports` | Jetson 컨테이너에서 미니 PC의 MQTT·Metadata·Core Data·Redis·Action을 찾게 하고, 미니 PC 발신자에서 Jetson Alert API를 찾게 함. 아래 연결표 참조 |
| 1 | [src/edgex/adapter_service.py](../../src/edgex/adapter_service.py), [src/edgex/device_service.py](../../src/edgex/device_service.py) | 어댑터 생성부의 `redisHost=edgex-redis`, `redisPort=6379`와 검증 응답기까지 확인. **첫 분리는 `extra_hosts`로 기존 이름을 보존하면 Python 수정 없이 가능**. 환경변수 방식 선택 시 생성부에서 host/port를 전달하는 최소 수정과 시험 필요 |
| 1 | EdgeX 서비스 등록 정보 | `baseUrl=http://cctv-device-service:59986` 기본값과 실제 `device-rest` 등록을 대조. 서비스가 소비하는 위치에서 접근 가능한 주소인지 확인하고, 변경 필요 시 명시적으로 이행. 검증 응답/등록 실패를 단순 MQTT 문제로 처리하지 않음 |
| 1 | [config/output_devices.json](../../config/output_devices.json), [config/event_type_map.json](../../config/event_type_map.json), [config/sensor_devices.json](../../config/sensor_devices.json), `cameras.json`, `zones_config.json` | 폴별 입력/출력 대상·카메라/구역 매핑. `AIOT_JETSON_ID`와 `EDGEX_JETSON_ID`, 장치 구독 대상 일치 확인. 폴 사이 명령 유출 방지 |
| 1 | [config/mediamtx.yml](../../config/mediamtx.yml) | 고정된 `webrtcAdditionalHosts`를 실제 접근 가능한 폴별 주소로 정리하고 카메라별 path와 AI 출력 연결 확인. 브라우저 WebRTC UDP 도달성 검증 |
| 2 | [scripts/ops/compose_stack.py](../../scripts/ops/compose_stack.py), [systemd 서비스](../../deploy/systemd/cctv-jetson-stack.service), [현장 override 연결](../../deploy/systemd/cctv-jetson-stack.service.d/10-live-profile.conf) | 두 역할의 실행 파일/서비스 목록을 선택하도록 보완. 현재 Jetson unit의 X11/EGL 준비를 미니 PC로 복사하지 않음. 기존 현장 override가 다른 서비스를 다시 띄우지 않도록 점검 |
| 2 | [monitoring/prometheus.jetson.yml](../../monitoring/prometheus.jetson.yml) | 미니 PC의 Prometheus가 Jetson 지표 주소를 수집하도록 변경. AI8770·어댑터9105·Public API9000·Alert8000의 필요한 바인딩과 수집 허용 범위 확인 |
| 2 | [scripts/ops/backup_runtime_databases.sh](../../scripts/ops/backup_runtime_databases.sh), [백업 unit](../../deploy/systemd/cctv-runtime-backup.service) | 양쪽 로컬 데이터와 어댑터 전용 볼륨을 각각 백업. PostgreSQL/Redis/MQTT 영속 데이터도 별도 복원 대상. 기존 SQLite 스크립트만으로 전체 백업 완료로 판정하지 않음 |
| 2 | [Compose 검사](../../scripts/health/check_compose_runtime_assumptions.py), [Jetson 검사](../../scripts/health/check_jetson_edgex_stack.py) | 한 호스트에 모든 서비스가 있다는 검사 가정을 두 역할에 맞게 조정. 보안/SDK/데이터 보존 검사를 삭제해 통과시키지 않음 |

Public API의 [기존 URL 설정](../../src/api/dependencies/_settings.py)과 어댑터의 [기존 CLI](../../runners/run_edgex_adapter.py)는 재사용한다. 환경변수만 적고 Compose `command`/`environment`의 고정값이 덮어쓰는 상태를 남기지 않는다. 예를 들어 AI `--mqtt-broker edgex-mqtt-broker`는 실제 호스트 이름 매핑 또는 command 치환까지 바꿔야 한다.

`public-demo-ui`를 Jetson에 두므로 [Nginx template](../../config/nginx/public-demo.conf.template)의 AI/Public API/MediaMTX 내부 이름은 첫 분리에서 유지할 수 있다. 이후 UI를 미니 PC로 옮기면 이 template의 각 upstream, 내부 토큰 전달, WebRTC 경로도 함께 바꿔야 한다.

### 주소 연결표

`JETSON_HOST_IP`, `MINIPC_HOST_IP`는 설계상 표기다. 실제 env 이름과 IP는 구현 시 정하고, 비밀값/실장비 IP를 코드에 고정하지 않는다. 내부 이름 유지안에서는 **외부 서비스 이름만** 해당 소비 컨테이너의 `extra_hosts`에 매핑한다.

| 출발 → 도착 | 현재 참조 / 첫 분리 시 처리 |
|---|---|
| Jetson AI·어댑터·결과 collector → 미니 PC | `edgex-mqtt-broker:1883` → 미니 PC 관리망 IP. 브로커는 기본127.0.0.1 바인딩을 관리망에 한정해 변경; 인증/토픽 ACL 유지 |
| Jetson 어댑터 → 미니 PC EdgeX | `edgex-core-metadata:59881`, `edgex-core-data:59880`, `edgex-redis:6379`. 현재 Redis 메시지버스/장치 검증도 사용하므로 MQTT 주소만 수정하면 불완전 |
| Jetson Public API → 미니 PC Action | `cctv-action-layer:8080`. 기존 `ACTION_LAYER_URL` 또는 이름 매핑 사용 |
| 미니 PC Action → Jetson Alert | `EXTERNAL_API_URL`의 `/api/alerts`까지 유지 |
| 미니 PC parser → Jetson Alert | `ALERT_API_URL`의 `/api/sensor-readings`까지 유지 |
| 미니 PC app-rules-engine → Jetson Alert | `WRITABLE_PIPELINE_FUNCTIONS_HTTPEXPORT_PARAMETERS_URL`의 `/api/alerts`까지 유지 |
| 관제 브라우저 → Jetson | 웹7000, WebRTC 미디어UDP8189. Public API9000은 직접 접근이 필요한 관제/수집기만 허용. 나머지 AI 내부 API는 같은 Jetson Nginx에서 중계 |
| 미니 PC Prometheus → Jetson | 지표8770/9105/9000/8000. `127.0.0.1` 기본 바인딩과 Docker 서비스 DNS에 의존하지 않게 조정 |

Redis6379·EdgeX59880/59881·Action8080·Alert8000은 지정한 장비 사이 관리망에서만 허용하고 AP 이용자망/인터넷에 노출하지 않는다. 이 안은 현재 Redis 결합을 유지하는 첫 분리이며, 네트워크 정책상 Redis의 호스트 간 통신이 금지된다면 어댑터를 미니 PC로 옮기고 AIoT 조회를 원격화하는 별도 설계가 필요하다.

## 4. 확인된 스위치로 가능한 배선안

- SFC8000GHP: 제조사 기준 기가비트 RJ45 PoE8개 + SFP4개, PoE 총240W 표기. 전원장치와 실물 리비전 확인 필요. [솔텍 공식 사양](https://soltech.co.kr/prod/sfc8000ghp/).
- NEXT-POE1411GDT: 국립전파연구원 인증상 기가비트 PoE4개 + TP1개 + SFP1개인 비관리형 스위치. TP 업링크를 써도 AP용4포트를 남길 수 있다. [인증 정보](https://www.rra.go.kr/ko/license/A_b_popup_keyno.do?key_no=R-R-NXT-NEXT-POE1411GD). NEXT의 총 PoE 예산·전원 사양은 이번에 확인하지 못했다.

**NEXT의 전력 공급이 AP4대의 최대 요구 전력을 충족하는 경우** 아래 배치를 권한다. 실제 전면 포트 라벨과 대응시켜야 하며 표 번호는 배선 계획이다.

| 장비 | 포트 배정안 |
|---|---|
| SFC8000GHP RJ45 1~4 | CCTV1~4 |
| SFC8000GHP RJ45 5~6 | Jetson / 미니 PC. 각각 자체 전원 사용 |
| SFC8000GHP RJ45 7 | NEXT의 별도 TP 업링크에 연결 |
| SFC8000GHP RJ45 8 | 유선 상위망 또는 센서 게이트웨이용 후보. 둘 다 필요하면 추가 포트 계획 |
| SFC8000GHP SFP | 상위 광망/추가 연결 후보. 사용 시 호환 모듈·매체·속도 확인 |
| NEXT PoE 1~4 | SAWWAVE AP1~4 |
| NEXT SFP | 여유. RJ45 연결과 동시에 같은 두 스위치를 임의로 이중 연결하지 않음 |

AP4대가 영상 입력 경로에 끼지 않도록 CCTV·Jetson·미니 PC를 솔텍에 직접 연결한다. 공공 Wi-Fi를 제공한다면 솔텍 VLAN과 상위 라우터/방화벽에서 이용자망을 분리한다. NEXT는 비관리형이므로 관리형 VLAN 설정을 가정하지 않으며, AP의 SSID별 VLAN trunk가 필요하면 해당 모델의 태그 전달을 실물 검증한다. 솔텍은 L2 스위치이므로 VLAN 간 접근 통제와 인터넷 게이트웨이는 별도 라우팅 장비 역할이다.

이 배치의 NEXT 업링크는 AP4대가 공유하는1Gbps 경로다. AP별 무선 최대 속도의 합을 보장하지 않는다. AP가 공공 서비스/백홀/센서 게이트웨이 중 무엇을 담당하는지는 세부 모델과 현장 설계로 정한다. 전광판·스피커·게이트웨이의 유선 포트가 추가된다면 전체 장비 포트표를 다시 대조한다.

## 5. 이행 순서와 검증 범위

1. 기존 운영 이미지/override·서비스·볼륨·비밀값 위치를 목록화하고, 두 역할의 후보 Compose를 작성한다. 지금의 일반 `docker-compose.yml`을 미니 PC에서 그대로 올리면 AI와 중복 서비스까지 시작할 수 있으므로 사용하지 않는다.
2. 격리한 후보에서 Compose 해석·서비스 목록·상호 주소·마운트/권한·기존 이름 매핑을 검사한다. MQTT 인증/ACL, Redis 장치 검증, EdgeX 등록/저장까지 확인한다.
3. 새 미니 PC의 출력 대상은 모의 장치로 시작한다. 이벤트 ID·폴/카메라/장치 매핑·중복·만료·실행 불명·결과 DB·Public API를 대조한다. 실제 전환 시 이전 제어 서비스 중지와 이력 이행/복귀를 한 계획으로 수행한다.
4. Jetson 외형 조건 등록→AI 반영→기록 검색→crop, 얼굴/구역/카메라 설정, 라이브/WebRTC, 센서 JSONL 조회를 확인한다. 두 장비 분리 후에도 기존 기능이 빠지지 않아야 한다.
5. 미니 PC 단절/재시작 동안 Jetson 추론·로컬 보존, Jetson 단절 동안 미니 PC 센서/장치 경로, 외부망 단절과 복구를 각각 시험한다. 미니 PC 장애 중 실시간 제어 지속을 보장하지 않는다. Jetson 장애 중 조회 화면/Alert 복제 로그는 중단되며 센서 원본과 Action 이력은 미니 PC에 남도록 검증한다.
6. AGX/NX별1→2→4채널 후4폴 통합 시험을 진행한다. AP16대 동시 접속 부하와 공공망 격리, 카메라별 추론율/지연·온도/메모리·함체 전원/열을 별도 측정한다. AP 수가 늘었다고 AI 채널 목표를 자동 변경하지 않는다.

관련 기존 시험: `tests/test_compose_stack.py`, `tests/test_check_compose_runtime_assumptions.py`, `tests/test_check_jetson_edgex_stack.py`, `tests/test_adapter_service.py`, `tests/test_edgex_adapter_aiot_commands.py`, `tests/test_public_api.py`, `tests/test_appearance_log.py`, `tests/test_command_results_api.py`, `tests/test_public_demo_nginx_config.py`, `tests/test_runtime_database_backup.py`. 구현 시 변경 범위에 맞게 실행하고, 기존 단일호스트 모의 시험만으로 두 장비 실증을 대체하지 않는다.

관제 공통 저장 경로는 별도 주의가 필요하다. `AIOT_INGEST_DB` 선택형 경로와 기존 MQTT outbox는 동일 기능의 배포 완료 상태가 아니며 동시에 켤 수 없는 조합이 있다. [MonitoringDeliveryWorker](../../src/monitoring_worker.py)는 원본 DB 소유 장비에 둔다. [원격 POC 러너](../../runners/run_monitoring_remote_poc.py)는 전용 POC DB/binding을 요구하므로 바로 운영 전송 서비스로 간주하지 않는다. [수신 POC 구성](../../deploy/monitoring-poc/compose.yaml)의 중앙 배포와 두 원본 장비의 고유 origin/binding·ACK 대조는 별도 확인한다.

## 6. 이번에 확인한 것과 남은 것

- 그래프와 실제 소스로 서비스 호출/공유 파일 경계를 확인했고, Compose 서비스 목록과 읽기 전용 실행 목록을 대조했다. 실행 중31컨테이너 목록을 확인했으며 전체 서비스 기능 합격을 뜻하지 않는다.
- SFC8000GHP 제조사 포트 사양과 NEXT 인증 포트 구성을 확인했다. AP 전력/모델·NEXT 전원/PoE 예산·실제 설치 수량은 미확인이다.
- 이번 산출물은 분리 설계 문서다. 제품 코드/Compose/env/systemd 변경, 이미지 빌드, 컨테이너 재시작, GPU 부하, 두 장비/무선/실물 출력 시험은 수행하지 않았다. P6.4 및 기존 완료9/37을 유지한다.
