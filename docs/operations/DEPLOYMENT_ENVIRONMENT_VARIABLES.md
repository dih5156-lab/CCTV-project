# CCTV 운용환경 최종 변수 기준

이 문서는 운영 배포에서 실제로 입력해야 하는 환경 변수 파일과 우선순위를 한 곳에 정리한 기준 문서입니다.

## 1. 변수 파일 기준

| 배포 형태 | 기준 파일 | 실행 방식 | 비고 |
| --- | --- | --- | --- |
| 서버/PC 기본 스택 | `.env` | `docker compose -f docker-compose.yml up -d --build` | Public API, Alert API, Action Layer, EdgeX, AIoT parser 공통 |
| Jetson 통합 스택 | `.env.jetson` | `docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --build` | compose 치환과 일부 서비스 `env_file` 입력을 동시에 맞춤 |
| 모니터링 추가 | `.env` 또는 `.env.jetson` | 기존 compose 명령에 monitoring compose 추가 | `GRAFANA_ADMIN_PASSWORD`만 추가되면 됨 |
| 런타임 정리 systemd | `/etc/default/cctv-runtime-cleanup` | `sudo ./scripts/ops/install_runtime_cleanup_timer.sh` | 없으면 기본값 사용 |

기준 원칙:

- 서버/PC 배포는 `.env`를 단일 입력으로 사용합니다.
- Jetson 배포는 `.env.jetson`을 단일 입력으로 사용하고, `--env-file .env.jetson`을 항상 명시합니다.
- `.env.jetson`을 `.env`로 복사해 운용하지 않습니다. 같은 장비에 두 파일이 공존할 수 있으므로 명시 실행이 안전합니다.
- 민감값은 `.env.example`, `.env.jetson.example`에 채우지 않고 빈 값으로 유지합니다.

## 2. 공통 필수 변수

아래 값은 운영 기준으로 비워두면 안 됩니다.

| 변수 | 적용 배포 | 설명 |
| --- | --- | --- |
| `MQTT_USER` | `.env`, `.env.jetson` | Mosquitto 인증 사용자 |
| `MQTT_PASSWORD` | `.env`, `.env.jetson` | Mosquitto 인증 비밀번호 |
| `AIOT_DB_PASSWORD` | `.env`, `.env.jetson` | `aiot-parser-db`와 `aiot-parser`가 함께 쓰는 PostgreSQL 비밀번호 |
| `PUBLIC_API_KEY` | `.env`, `.env.jetson` | Public API 외부 인증 키 |
| `INTERNAL_SERVICE_TOKEN` | `.env`, `.env.jetson` | Public API, Alert API, Action Layer 내부 호출 토큰 |
| `STREAM_API_TOKEN` | `.env`, `.env.jetson` | 선택 사항. Stream API 전용 토큰, 비우면 `INTERNAL_SERVICE_TOKEN`을 사용 |
| `GRAFANA_ADMIN_PASSWORD` | `.env`, `.env.jetson` | Grafana 관리자 비밀번호 |

권장값:

- `APP_ENV=production`: 운영 보안 강제 기준으로 사용합니다.
- `REQUIRE_PUBLIC_API_KEY=1`: `PUBLIC_API_KEY`가 비어 있으면 Public API 요청을 거부합니다.
- `REQUIRE_CORS_ORIGINS=1`: `CORS_ORIGINS` 누락 또는 `*` 설정을 거부합니다.
- `REQUIRE_STREAM_API_TOKEN=1`: Stream API의 `/cameras`, `/stream`, `/snapshot` 접근에 토큰을 요구합니다.
- `CORS_ORIGINS`: 운영 도메인만 쉼표로 제한합니다.
- `RATE_LIMIT_ENABLED=true`: 특별한 이유가 없으면 유지합니다.

## 3. 장비 연동 변수

현장 장비를 실제로 붙일 때만 설정합니다. 호스트 값이 비어 있으면 해당 장비 제어는 비활성화됩니다.

| 변수 | 설명 |
| --- | --- |
| `SPEAKER_HOST`, `SPEAKER_PORT`, `SPEAKER_USER`, `SPEAKER_PASSWORD`, `SPEAKER_VOLUME` | 스피커 제어 |
| `SIGNBOARD_HOST`, `SIGNBOARD_PORT` | 전광판 제어 |
| `SIREN_HOST`, `SIREN_PORT`, `SIREN_USER`, `SIREN_PASSWORD`, `SIREN_AUTO_STOP` | 경광등 제어 |
| `ACTION_ALARM_COOLDOWN` | 동일 이벤트 재발행 억제 간격 |

## 4. AIoT parser / EdgeX 관련 변수

| 변수 | 설명 |
| --- | --- |
| `AIOT_DB_NAME` | AIoT parser PostgreSQL DB 이름 |
| `AIOT_DB_USER` | AIoT parser PostgreSQL 사용자 |
| `AIOT_ALLOW_UNKNOWN_DEVICES` | 미등록 센서 허용 여부 |
| `KUIPER_RETRY_COUNT`, `KUIPER_RETRY_DELAY` | Kuiper 룰 로더 재시도 설정 |
| `INTRUSION_CONFIDENCE`, `CRITICAL_CONFIDENCE`, `PERSIST_HIT_COUNT`, `TILT_THRESHOLD`, `TEMP_HIGH_THRESHOLD` | 룰 임계값 |

## 5. Jetson 전용 핵심 변수

Jetson 통합 스택에서 자주 조정하는 값만 추렸습니다. 나머지 고급 튜닝 값은 compose 기본값을 우선 사용합니다.

| 변수 | 설명 |
| --- | --- |
| `L4T_TAG` | Jetson 베이스 이미지 태그 |
| `USE_GSTREAMER`, `USE_DEEPSTREAM`, `AI_DEVICE` | 추론 런타임 선택 |
| `FRAME_SKIP`, `FPS`, `RTSP_BUFFER_SIZE` | 입력 처리 성능 조정 |
| `DS_YOLO_TASK`, `DS_YOLO_CLASS_IDS`, `DS_YOLO_LABELS` | DeepStream 추론 라우팅 |
| `DS_HELMET_ENABLED`, `DS_FACE_ENABLED`, `DS_APPEARANCE_ENABLED` | 기능 on/off |
| `DS_PREVIEW_ENABLED`, `STREAM_API_ENABLED`, `STREAM_PORT`, `STREAM_FPS`, `STREAM_JPEG_QUALITY`, `STREAM_API_TOKEN`, `REQUIRE_STREAM_API_TOKEN` | 미리보기/스트림 API |
| `ZONE_API_PORT`, `CAMERA_MODEL_API_PORT`, `FACE_API_PORT` | 내부 관리 API 포트 |
| `APPEARANCE_ENABLED`, `APPEARANCE_BACKEND`, `APPEARANCE_SAVE_CROPS` | 외형 검색 기본 동작 |
| `FACE_RECOGNITION_BACKEND`, `FACE_SNAPSHOT_ENABLED` | 얼굴 인식/스냅샷 |

고급 DeepStream/스트리밍 튜닝 변수(`DS_H264_*`, `DS_PREVIEW_*`, `STREAM_WIDTH`, `STREAM_HEIGHT` 등)는 기본값이 이미 compose에 정의되어 있으므로, 운영 이슈가 있을 때만 `.env.jetson`에 추가합니다.

## 6. systemd 런타임 정리 변수

`deploy/systemd/cctv-runtime-cleanup.service.in`은 선택적으로 `/etc/default/cctv-runtime-cleanup`을 읽습니다.

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `RUNTIME_DATA_DIR` | `<project>/data` | 런타임 데이터 루트 |
| `RUNTIME_DIR` | `<data>/runtime` | SQLite DB, crop, snapshot 산출물 루트 |
| `RUNTIME_LOG_DIR` | `<data>/logs` | JSONL 로그 루트 |
| `APPEARANCE_CROP_DIR` | `<data>/runtime/appearance_crops` | crop 저장 경로 |
| `ALERT_LOG_PATH` | `<data>/logs/alert_api_events.jsonl` | 알림 로그 경로 |
| `SENSOR_LOG_PATH` | `<data>/logs/sensor_readings.jsonl` | 센서 로그 경로 |
| `APPEARANCES_DB` | `<data>/runtime/appearances.db` | 외형 검색 DB 경로 |
| `CROP_RETENTION_DAYS` | `7` | crop 보존 일수 |
| `LOG_MAX_MB` | `200` | JSONL 로그 회전 기준 |
| `PYTHON_BIN` | `<project>/.venv/bin/python`, 없으면 `python3` | crop DB 참조 정리에 사용할 Python 실행 파일 |

## 7. 검증 명령

서버/PC 기본 스택:

```bash
.venv/bin/python scripts/health/check_sensitive_defaults.py
.venv/bin/python scripts/health/check_compose_runtime_assumptions.py --json
.venv/bin/python scripts/health/check_runtime_secret_consistency.py --env-file .env --json
./scripts/ops/run_operation_check.sh
```

Jetson 통합 스택:

```bash
.venv/bin/python scripts/health/check_sensitive_defaults.py
.venv/bin/python scripts/health/check_compose_runtime_assumptions.py --json
.venv/bin/python scripts/health/check_runtime_secret_consistency.py --env-file .env.jetson --json
RUNTIME_ENV_FILE=.env.jetson ./scripts/ops/run_operation_check.sh
```

## 8. 최종 정리

- `.env.example`와 `.env.jetson.example`는 시작 템플릿입니다.
- 실제 운영값은 `.env` 또는 `.env.jetson`에만 둡니다.
- Jetson은 `--env-file .env.jetson`을 표준으로 고정합니다.
- 운영 점검과 문서 검토 시 이 문서를 최종 기준으로 사용합니다.
