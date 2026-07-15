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

## 2. 운영 필수 및 조건부 변수

개발 기본값에서는 일부 값이 비어 있어도 기동할 수 있지만, 운영에서는 사용하는 서비스와 보안 옵션에 맞춰 설정해야 합니다.

| 변수 | 필요 조건 | 설명 |
| --- | --- | --- |
| `MQTT_USER`, `MQTT_PASSWORD` | 인증이 활성화된 Mosquitto 사용 시 필수 | MQTT 인증 정보 |
| `MQTT_BIND_HOST` | Jetson MQTT 브로커를 외부 장치에 공개할 때 설정 | 기본 `127.0.0.1`, 외부 장치 연동 시 `0.0.0.0` |
| `AIOT_DB_PASSWORD` | `aiot-parser-db` 사용 시 필수 | parser와 PostgreSQL이 같은 값을 사용 |
| `PUBLIC_API_KEY` | `REQUIRE_PUBLIC_API_KEY=1` 또는 외부 노출 시 필수 | Public API 외부 인증 키 |
| `PUBLIC_API_BIND_HOST` | Public API를 외부 네트워크에 공개할 때 설정 | 기본 `127.0.0.1`, 외부 공개 시 `0.0.0.0` |
| `PUBLIC_DEMO_BIND_HOST` | Demo UI를 외부 네트워크에 공개할 때 설정 | 기본 `127.0.0.1`, 외부 공개 시 `0.0.0.0` |
| `MEDIA_BIND_HOST` | Jetson MediaMTX RTSP/WebRTC 포트를 외부 공개할 때 설정 | 기본 `127.0.0.1`, 외부 공개 시 `0.0.0.0` |
| `MEDIA_API_BIND_HOST` | MediaMTX API 포트를 외부 공개할 때 설정 | 기본 `127.0.0.1`, 가능하면 내부망/localhost 유지 |
| `INTERNAL_SERVICE_TOKEN` | 내부 서비스 인증을 강제할 때 필수 | Public API, Alert API, Action Layer 내부 호출 토큰 |
| `STREAM_API_TOKEN` | `REQUIRE_STREAM_API_TOKEN=1`일 때 필요 | 비우면 `INTERNAL_SERVICE_TOKEN`으로 fallback |
| `GRAFANA_ADMIN_PASSWORD` | monitoring profile을 운영할 때 필수 | Grafana 관리자 비밀번호 |

권장값:

- `APP_ENV=production`: 운영 보안 강제 기준으로 사용합니다.
- `MQTT_BIND_HOST=127.0.0.1`: 기본은 로컬호스트에만 공개합니다. 외부 장치가 MQTT로 직접 붙어야 할 때만 `0.0.0.0`로 열고, Mosquitto 인증과 현장 네트워크 제한을 함께 적용합니다.
- `REQUIRE_PUBLIC_API_KEY=1`: `PUBLIC_API_KEY`가 비어 있으면 Public API 요청을 거부합니다.
- `REQUIRE_CORS_ORIGINS=1`: `CORS_ORIGINS` 누락 또는 `*` 설정을 거부합니다.
- `REQUIRE_STREAM_API_TOKEN=1`: Stream API의 `/cameras`, `/stream`, `/snapshot` 접근에 토큰을 요구합니다.
- `PUBLIC_API_BIND_HOST=127.0.0.1`: 기본은 로컬호스트에만 공개합니다. 현장 대시보드에서 직접 접근해야 할 때만 `0.0.0.0`로 열고, 반드시 `PUBLIC_API_KEY`와 CORS 제한을 함께 설정합니다.
- `PUBLIC_DEMO_BIND_HOST=127.0.0.1`, `MEDIA_BIND_HOST=127.0.0.1`: 데모 UI와 영상 포트도 기본은 로컬에 묶습니다. 외부 브라우저에서 직접 WebRTC를 볼 때만 `0.0.0.0`로 열고, 현장 방화벽/VLAN으로 접근 대상을 제한합니다.
- `MEDIA_API_BIND_HOST=127.0.0.1`: MediaMTX API는 운영에서 외부 공개하지 않는 것을 권장합니다.
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
| `DS_YOLO_POSTPROCESS_MODE` | `vectorized`가 운영 기본값이며, 장애 시 `legacy`로 즉시 롤백 |
| `DS_HELMET_ENABLED`, `DS_FACE_ENABLED`, `DS_APPEARANCE_ENABLED` | 기능 on/off |
| `DS_PREVIEW_ENABLED`, `STREAM_API_ENABLED`, `STREAM_PORT`, `STREAM_FPS`, `STREAM_JPEG_QUALITY`, `STREAM_API_TOKEN`, `REQUIRE_STREAM_API_TOKEN` | 미리보기/스트림 API |
| `ZONE_API_PORT`, `CAMERA_MODEL_API_PORT`, `FACE_API_PORT` | 내부 관리 API 포트 |
| `APPEARANCE_ENABLED`, `APPEARANCE_BACKEND`, `APPEARANCE_SAVE_CROPS` | 외형 검색 기본 동작 |
| `FACE_RECOGNITION_BACKEND`, `FACE_SNAPSHOT_ENABLED` | 얼굴 인식/스냅샷 |

### 스트림과 H.264 출력 튜닝

| 변수 | 일반 Compose 기본값 | Jetson Compose 기본값 | 설명 |
|---|---:|---:|---|
| `STREAM_FPS` | `15` | `15` | MJPEG 목표 FPS |
| `STREAM_WIDTH`, `STREAM_HEIGHT` | `960`, `540` | `960`, `540` | MJPEG 송출 크기. 둘 중 하나가 `0`이면 원본 크기 유지 |
| `STREAM_JPEG_QUALITY` | `65` | `.env.jetson.example`은 `75` | JPEG 품질. 코드에서 `30~95`로 보정 |
| `DS_RTSP_LOCATION_TEMPLATE` | `rtsp://cctv-media-server:8554/{camera_id}` | 동일 | 카메라 ID별 분석 영상 RTSP 게시 URL 템플릿 |
| `DS_H264_ENCODER` | 구성별 값 | `nvv4l2h264enc` | H.264 인코더 |
| `DS_H264_WIDTH`, `DS_H264_HEIGHT` | `1280`, `720` | `960`, `540` | H.264 출력 해상도 |
| `DS_H264_BITRATE` | `4000000` | `3000000` | H.264 bitrate |
| `DS_H264_POC_FIX_ENABLED` | `.env.example`은 `false` | Compose fallback `true` | H.264 POC 보정 활성화 |
| `DS_H264_POC_TYPE` | `2` | `2` | POC 보정값 |

`DS_H264_POC_FIX_ENABLED`는 Python buffer handoff를 사용하므로 CPU 부하와 복사가 늘어날 수 있습니다. 영상 호환 문제가 없으면 현장 부하 측정 후 비활성화를 검토합니다.

카메라별 WebRTC 라이브 주소는 `http://<Jetson-IP>:8889/<camera_id>/`입니다. 기존 `DS_RTSP_LOCATION`은 활성 카메라가 하나이고 `DS_RTSP_LOCATION_TEMPLATE`이 없을 때만 호환됩니다. 다중 카메라에서는 `{camera_id}` 템플릿을 사용해야 합니다.

### falldata 보조 검증

| 변수 | 기본값 | 설명 |
|---|---:|---|
| `FALLDATA_AUX_ENABLED` | `false` | 보조 검증 활성화 |
| `FALLDATA_AUX_MODE` | `shadow` | `shadow` 또는 `confirm` |
| `FALLDATA_AUX_MAX_EXTRACT_FRAMES` | `120` | MediaPipe feature 추출 최대 프레임 |
| `FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE` | `true` | 검증기 장애 시 원본 낙상 알람 유지 |
| `FALLDATA_AUX_CONFIRM_BORDERLINE` | `false` | DeepStream borderline 후보 확인 실험 옵션 |
| `FALLDATA_AUX_COMPARE_MODEL_PATH` | 미설정 | 후보 모델 비교 결과 기록 |
| `FALLDATA_AUX_COMPARE_VETO_ENABLED` | `false` | compare 결과 기반 차단 실험 옵션 |

현재 코드 점검 기준으로 OpenCV 경로는 위 설정을 직접 읽습니다. DeepStream의 borderline/compare 정책은 처리 메서드는 있지만 초기화 연결이 확인되지 않았으므로, 환경변수만 설정하고 활성화되었다고 판단하지 않습니다. 자세한 상태는 [falldata 통합 문서](../features/FALLDATA_INTEGRATION.md)를 참고합니다.

### Jetson external volume 사전 준비

`docker-compose.jetson.yml`에서 `db-data`, `aiot-db-data`, `aiot-parser-data`, `kuiper-*`, `trt-cache`는 external volume입니다. 새 장비에서는 첫 실행 전에 한 번 생성합니다.

```bash
docker volume create edgex-jetson_db-data
docker volume create edgex-jetson_aiot-db-data
docker volume create edgex-jetson_aiot-parser-data
docker volume create edgex-jetson_kuiper-data
docker volume create edgex-jetson_kuiper-etc
docker volume create edgex-jetson_kuiper-log
docker volume create edgex-jetson_kuiper-plugins
docker volume create edgex-jetson_trt-cache
```

볼륨이 없으면 Compose가 `external volume ... not found`로 시작 전에 실패합니다.

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
| `FALL_SHADOW_REVIEW_LOG_PATH` | `<data>/fall_dataset/annotations/review.jsonl` | 낙상 shadow 검토 JSONL 로그 경로 |
| `FALL_SHADOW_CLIP_DIR` | `<data>/fall_dataset/clips/pending` | 라벨 대기 낙상 클립 저장 경로 |
| `APPEARANCES_DB` | `<data>/runtime/appearances.db` | 외형 검색 DB 경로 |
| `CROP_RETENTION_DAYS` | `7` | crop 보존 일수 |
| `FALL_REVIEW_RETENTION_DAYS` | `3` | 낙상 검토 클립 보존 일수 |
| `LOG_MAX_MB` | `200` | JSONL 로그 회전 기준 |
| `FALL_SHADOW_LOG_MAX_MB` | `50` | 낙상 shadow 로그 회전 기준 |
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

`check_compose_runtime_assumptions.py`는 현재 기본 `docker-compose.yml`의 아키텍처도 함께 검사합니다. ARM64 Jetson에서는 기본 Compose의 amd64 전용 가능성을 알리기 위해 `default compose architecture` 항목과 전체 `passed`가 실패할 수 있습니다. Jetson 배포에서는 이 항목을 `docker-compose.jetson.yml을 사용하라`는 경고로 해석하고, 나머지 wiring 검사와 Jetson Compose `config` 검사를 함께 확인합니다.

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml config --quiet
```

## 8. 최종 정리

- `.env.example`와 `.env.jetson.example`는 시작 템플릿입니다.
- 실제 운영값은 `.env` 또는 `.env.jetson`에만 둡니다.
- Jetson은 `--env-file .env.jetson`을 표준으로 고정합니다.
- 운영 점검과 문서 검토 시 이 문서를 최종 기준으로 사용합니다.
