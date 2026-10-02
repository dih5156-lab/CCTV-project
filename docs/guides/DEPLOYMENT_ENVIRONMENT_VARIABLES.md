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
| `DS_CONTEXT_MIN_INTERVAL_SEC` | 기본 `0.5`: 카메라·추적 인물별 얼굴/의상/색상 후처리 최소 간격(초). 새 인물/재연결 후 첫 관측은 즉시 대상, `0`은 제한 해제. 사람·헬멧·낙상 탐지와 생성된 이벤트 저장 주기는 변경하지 않음. 샘플 수 기반 속성 평활화는 실제 시간상 더 오래 걸릴 수 있음 |
| `DS_PREVIEW_ENABLED`, `STREAM_API_ENABLED`, `STREAM_PORT`, `STREAM_FPS`, `STREAM_JPEG_QUALITY`, `STREAM_API_TOKEN`, `REQUIRE_STREAM_API_TOKEN` | 미리보기/스트림 API |
| `DS_RTSP_DROP_ON_LATENCY` | 미설정 시 SDK 기본값 유지(7.1: true). RTSP/RTSPS에만 적용. false는 격리 재생의 압축 패킷 폐기 대조용이며 처리량 부족 시 영상 지연/메모리가 누적될 수 있다. 운영 변경 전 입력 FPS·처리량·지연을 함께 검증한다. |
| `DS_NVVIDEOCONVERT_COPY_HW` | 빈 값: SDK 기본값 유지. `1`: GPU 복사, `2`: VIC 복사. Jetson surface-array 메모리의 CUDA 복사 오류에 대한 NVIDIA 우회 후보. OSD 앞·H.264·preview의 nvvideoconvert에 적용하며 모델 추론 장치는 바꾸지 않는다. 그 외 값은 시작 실패로 처리한다. |
| `ZONE_API_PORT`, `CAMERA_MODEL_API_PORT`, `FACE_API_PORT` | 내부 관리 API 포트 |
| `APPEARANCE_ENABLED`, `APPEARANCE_BACKEND`, `APPEARANCE_SAVE_CROPS` | 외형 검색 기본 동작 |
| `FACE_RECOGNITION_BACKEND`, `FACE_SNAPSHOT_ENABLED` | 얼굴 인식/스냅샷 |

### 스트림과 H.264 출력 튜닝

2026-09-29 v0.62: `DS_NVVIDEOCONVERT_COPY_HW=2`는 격리 RC3에서 시험하는 설정이다.
Compose 기본값은 빈 값이며 운영 `.env.jetson`은 변경하지 않았다. VIC는 Jetson용이고,
surface-array(메모리 유형0/4) 환경을 전제로 검증한다. CUDA 메모리 유형이나 dGPU에 그대로 적용하지 않는다.
[NVIDIA 오류 대응](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_troubleshooting.html)과
[시험 결과·제약](../planning/TRACKER_VALIDATION.md#v062--cuda-메모리-오류-진단과-vic-복사-후보)을 함께 확인한다.

| 변수 | 일반 Compose 기본값 | Jetson Compose 기본값 | 설명 |
|---|---:|---:|---|
| `STREAM_FPS` | `15` | `15` | MJPEG 목표 FPS |
| `STREAM_WIDTH`, `STREAM_HEIGHT` | `960`, `540` | `960`, `540` | MJPEG 송출 크기. 둘 중 하나가 `0`이면 원본 크기 유지 |
| `STREAM_JPEG_QUALITY` | `65` | `.env.jetson.example`은 `75` | JPEG 품질. 코드에서 `30~95`로 보정 |
| `DS_H264_ENCODER` | 구성별 값 | `nvv4l2h264enc` | H.264 인코더 |
| `DS_H264_WIDTH`, `DS_H264_HEIGHT` | `1280`, `720` | `960`, `540` | H.264 출력 해상도 |
| `DS_H264_FPS` | 코드 fallback `30` | Compose fallback `20` | H.264 출력 목표 프레임률(최소1). 출력 videorate 반복/드랍 가능, AI 추론 FPS와 구분 |
| `DS_H264_BITRATE` | `4000000` | `3000000` | H.264 bitrate |
| `DS_H264_POC_FIX_ENABLED` | `.env.example`은 `false` | Compose fallback `true` | H.264 POC 보정 활성화 |
| `DS_H264_POC_TYPE` | `2` | `2` | POC 보정값 |

**2026-09-29 v0.56 판단 보완:** 수정 전/출력 없는 경로에도 초기 정지가 있어 출력 변경 원인 단정은 정정한다. 확정 수정안과 통합 판정이 없어 운영 보류는 유지한다.

**2026-09-29 v0.55 당시 적용 보류:** 아래 출력 수정 후보가 15FPS 입력에서 정지하여 I33을 재개했다. 출력 제외/수정 전 대조는 진행됐으나 내부 원인은 추가 검증이 필요하다.

2026-09-29 v0.54 당시 I33 로컬 수정: H.264 출력에 videorate를 사용해 mux의0/1 FPS가 인코더에 전달되는 것을 방지한다. 초기 프레임 전 생성은 건너뛰며 반복 가능한 간격은100ms로 제한한다. 운영 적용·장시간/다중 성능 검증은 별도다.

`DS_H264_POC_FIX_ENABLED`는 Python buffer handoff를 사용하므로 CPU 부하와 복사가 늘어날 수 있습니다. 영상 호환 문제가 없으면 현장 부하 측정 후 비활성화를 검토합니다.

### DeepStream pose 낙상 규칙

| 변수 | 기본값 | 설명 |
|---|---:|---|
| `POSE_CROP_MODE` | `0` | crop-pose POC 활성화. 기본값 `0`; `_run_pose_crop()`가 없으면 full-frame으로 fallback |
| `DS_FALL_SCORE_THRESHOLD` | `3.0` | pose 낙상 점수 판정 기준 |
| `DS_FALL_ANGLE_HORIZONTAL` / `DS_FALL_ANGLE_INVERTED` | `55` / `125` | 몸통 수평·역수평 각도 기준(도) |
| `DS_FALL_BBOX_ASPECT_RATIO` | `1.35` | 가로형 bbox 강한 후보 기준 |
| `DS_FALL_SPAN_BBOX_ASPECT_RATIO` | `1.20` | 키포인트 수직 분산 규칙 적용 기준 |
| `DS_FALL_KEYPOINT_SPAN_RATIO` | `0.55` | 가로형 pose의 수직 분산 상한 |
| `DS_FALL_ENABLE_FOLDED_POSE` | `false` | 접힌 바닥 자세를 운영 낙상 점수에 포함 |
| `DS_FALL_FOLDED_POSE_MAX_SPAN_RATIO` | `0.30` | 접힌 자세 후보의 하체 수직 분산 상한 |
| `DS_FALL_SUPPRESS_SITTING_LIKE_POSE` | `false` | 앉은 자세와 유사한 가로형 후보 감점 |

`DS_FALL_ENABLE_FOLDED_POSE`는 기본적으로 비활성화합니다. 활성화하기 전에 near-miss 로그를 정상/낙상으로 라벨링하고, 현장 precision과 temporal recall을 확인해야 합니다. `DS_FALL_FOLDED_POSE_MAX_SPAN_RATIO`는 활성화 여부와 별개로 후보 품질을 제한합니다.

### falldata 보조 검증

| 변수 | 기본값 | 설명 |
|---|---:|---|
| `FALLDATA_AUX_ENABLED` | `false` | 보조 검증 활성화 |
| `FALLDATA_AUX_MODE` | `shadow` | `shadow` 또는 `confirm` |
| `FALLDATA_AUX_MAX_EXTRACT_FRAMES` | `120` | MediaPipe feature 추출 최대 프레임 |
| `FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE` | `true` | 검증기 장애 시 원본 낙상 알람 유지 |
| `FALLDATA_AUX_CONFIRM_BORDERLINE` | `false` | DeepStream borderline 후보 확인 실험 옵션 |
| `FALLDATA_AUX_COMPARE_MODEL_PATH` | 미설정 | 후보 모델 비교 결과 기록 |
| `FALLDATA_AUX_TEMPORAL_POSE_IMGSZ` | `640` | 보조 TCN Pose 입력 크기. 2026-09-10 현장 비교 기준이며 자식 프로세스에 `--imgsz`로 명시 전달; 운영 적용 전 처리 시간 재검증 필요 |
| `FALLDATA_AUX_TEMPORAL_WORKER_ENABLED` | `false` | 선택 시 TCN/Pose 모델을 전용 자식 프로세스에서 재사용. 최초 호출/모델 교체/오류 후에는 다시 로딩. 기존 timeout을 대기·시작·응답에 적용하며 실패 시 자식 종료, 다음 호출에서 새로 시작. 60초 유휴 또는 부모 파이프 EOF 시 종료. GPU 메모리를 유지하므로 전체 AI 동시 부하·장시간 검증 후 적용 |
| `FALLDATA_AUX_TEMPORAL_SLIDING_WINDOW_SIZE` | `0` | `0`이면 기존 전체 영상 추론, 양수면 겹치는 temporal window 사용 |
| `FALLDATA_AUX_TEMPORAL_SLIDING_WINDOW_STRIDE` | `5` | sliding window 이동 간격 |
| `FALLDATA_AUX_TEMPORAL_MIN_CONFIRMED_WINDOWS` | `1` | 연속 window 확인 개수; shadow 튜닝에서만 변경 |
| `FALLDATA_AUX_TEMPORAL_NEAR_MISS_SHADOW_ENABLED` | `false` | 원본 detector near-miss를 temporal 모델에 shadow-only로 전달 |
| `FALLDATA_AUX_COMPARE_VETO_ENABLED` | `false` | compare 결과 기반 차단 실험 옵션 |

`FALLDATA_AUX_INLINE_POSE_RF=true` 후보에서는 모델 번들의
`model_kind=deepstream_pose_inline_rf`, `feature_source=deepstream_pose_inline`이 필요하다.
메타데이터 누락/다른 출처는 `incompatible_feature_contract`로 예측하지 않는다.
confirm 모드의 기존 fail-open 설정은 유지한다. 확률 필터 없는 명시적 특징 수집은 가능하지만
불일치 모델의 확률로 수집을 선별하지 않는다. 모델 파일에 출처 이름만 바꿔 넣어 우회하지 않는다.
현재 설정된 영상 추출 RF의 전환 영향은 [출처 검사 기록](../planning/MODEL_INFERENCE_METRICS.md#rf-source-v069)에 있다.

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
| `ACTION_HTTP_OUTBOX_DB` | `<data>/runtime/action_http_outbox.db` | HTTP 전송 outbox DB 경로 |
| `MQTT_EVENT_OUTBOX_DB` | `<data>/runtime/mqtt_event_outbox.db` | MQTT 전송 outbox DB 경로 |
| `CROP_RETENTION_DAYS` | `7` | crop 보존 일수 |
| `FALL_REVIEW_RETENTION_DAYS` | `3` | 낙상 검토 클립 보존 일수 |
| `OUTBOX_RETENTION_DAYS` | `7` | 전송 완료 outbox 보존 일수 |
| `OUTBOX_CLEANUP_BATCH_SIZE` | `25000` | DB별 1회 최대 삭제 행 수 |
| `LOG_MAX_MB` | `200` | JSONL 로그 회전 기준 |
| `FALL_SHADOW_LOG_MAX_MB` | `50` | 낙상 shadow 로그 회전 기준 |
| `PYTHON_BIN` | `<project>/.venv/bin/python`, 없으면 `python3` | 런타임 정리에 사용할 Python 실행 파일 |

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
