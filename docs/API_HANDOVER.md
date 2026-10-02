# CCTV Platform API 인수인계 문서

이 문서는 CCTV 프로젝트의 공개 API, AI Engine 보조 API, Action Layer 내부 API,
장치 제어 경로를 운영자와 신규 개발자가 재현할 수 있도록 정리한 문서다.

## 1. 서비스 구성과 기본 주소

Jetson 배포는 `docker-compose.jetson.yml`을 기준으로 한다.

| 서비스 | 기본 주소 | 역할 | 외부 공개 여부 |
|---|---|---|---|
| `cctv-public-api` | `http://localhost:9000` | 서버·대시보드용 공개 REST API | 프록시 정책에 따름 |
| `cctv-action-layer` | `http://cctv-action-layer:8080` | 이벤트 수신, 승인, 장치 명령 | 내부 전용 |
| `cctv-ai-engine` | `8765`~`8770` | zone, camera model, face, stream, metrics API | 내부 전용 |
| `cctv-alert-api` | `http://cctv-alert-api:8000` | 알람 저장·전달 | 내부 전용 |
| `edgex-core-data` | `http://edgex-core-data:59880` | EdgeX Reading 저장·조회 | 내부 전용 |

공개 API의 경로는 `/api/v1`을 사용한다. 실행 중인 공개 API의 정확한 계약은
`GET /openapi.json`으로 확인할 수 있으며, 브라우저 문서는 `GET /docs`다.

## 2. 인증과 공통 응답

### 공개 API 인증

인증이 필요한 공개 API에는 다음 헤더를 사용한다.

```http
X-API-Key: <PUBLIC_API_KEY>
```

`PUBLIC_API_ALLOW_QUERY_KEY=1`일 때만 `?api_key=...`를 사용할 수 있다. 운영에서는
URL·프록시·접속 로그에 키가 남을 수 있으므로 query 방식은 사용하지 않는다.

`APP_ENV=production` 또는 `REQUIRE_PUBLIC_API_KEY=1`이면 `PUBLIC_API_KEY`가 반드시
설정되어야 한다.

주요 인증 응답:

| 상태 | 의미 |
|---|---|
| `401` | 키가 전달되지 않음 |
| `403` | 키가 잘못됨 |
| `503` | 운영 필수 키가 서버에 설정되지 않음 |
| `422` | 요청 본문 또는 query 검증 실패 |
| `500` | 처리되지 않은 서버 오류 |

### 공통 응답 래퍼

대부분의 공개 API는 다음 형식으로 응답한다.

```json
{
  "success": true,
  "data": {},
  "error": null,
  "timestamp": "2026-09-09T09:00:00+09:00"
}
```

실패 시 `success=false`, `data=null`, `error`에 사람이 읽을 수 있는 메시지가
들어간다. 페이지 조회 API는 `data` 안에 `items`, `limit`, `offset`, `total`을
포함하는 `PaginatedResponse`를 사용한다.

## 3. 공개 REST API

### 상태·운영

| Method | 경로 | 용도 |
|---|---|---|
| GET | `/api/v1/health` | API·DB·외부 의존성 상태 확인 |
| GET | `/api/v1/readiness` | 배포 전 준비 상태 확인 |
| GET | `/api/v1/metrics` | Prometheus 메트릭 조회 |

상태 확인 예시:

```bash
curl -H "X-API-Key: $PUBLIC_API_KEY" \
  http://localhost:9000/api/v1/readiness
```

`health`는 서비스가 살아 있는지를, `readiness`는 실제 요청을 처리할 준비가
되었는지를 판단하는 용도다. 배포 자동화에서는 `readiness`를 우선 사용한다.

### 이벤트·알람

| Method | 경로 | 용도 |
|---|---|---|
| GET | `/api/v1/events` | AI/센서 이벤트 조회 |
| POST | `/api/v1/alerts` | 알람 수신 및 저장·전달 |
| POST | `/api/v1/event-reviews` | 이벤트 검수 결과 저장 |
| GET | `/api/v1/event-reviews/summary` | 검수 통계 조회 |

이벤트 조회 주요 query:

```text
camera_id, event_type, time_from, time_to, fall_direction,
limit, offset
```

예시:

```bash
curl -G -H "X-API-Key: $PUBLIC_API_KEY" \
  --data-urlencode "event_type=fall_detected" \
  --data-urlencode "limit=50" \
  http://localhost:9000/api/v1/events
```

알람을 직접 넣을 때는 `POST /api/v1/alerts`의 요청 schema를 반드시
`/openapi.json`에서 확인한다. 알람은 중복 방지용 event id와 발생 시각을 포함해야
하며, 저장 성공이 곧 장치 동작 성공을 의미하지는 않는다. 장치 결과는
`/api/v1/command-results`에서 별도로 확인한다.

### 카메라·사이트

| Method | 경로 | 용도 |
|---|---|---|
| GET | `/api/v1/cameras` | 등록 카메라 목록 조회 |
| GET | `/api/v1/cameras/{camera_id}` | 카메라 상세 조회 |
| GET | `/api/v1/sites` | 사이트 목록 조회 |
| POST | `/api/v1/sites` | 사이트 추가 |
| DELETE | `/api/v1/sites/{site_id}` | 사이트 삭제 |

카메라 응답에서는 비밀번호 등 접속 자격 증명이 제거된다. 카메라 설정을
변경할 때는 `cameras.json`과 실제 실행 컨테이너의 mount 경로가 일치하는지 확인한다.

### 장치 제어·수동 승인

| Method | 경로 | 용도 |
|---|---|---|
| GET | `/api/v1/control/mode` | 현재 전역/사이트 모드 조회 |
| POST | `/api/v1/control/mode` | `auto`, `manual`, `shadow` 등 모드 설정 |
| GET | `/api/v1/control/pending` | 수동 승인 대기 이벤트 조회 |
| GET | `/api/v1/control/devices` | 허용 출력 장치 목록 조회 |
| GET | `/api/v1/control/commands` | 최근 장치 명령 조회 |
| POST | `/api/v1/control/approve/{event_id}` | 이벤트 승인 |
| POST | `/api/v1/control/reject/{event_id}` | 이벤트 거부 |

모드 변경 예시:

```bash
curl -X POST \
  -H "X-API-Key: $PUBLIC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"mode":"manual"}' \
  http://localhost:9000/api/v1/control/mode
```

운영 중에는 장치 목록에 없는 장치 id를 명령 대상으로 지정하지 않는다. 승인·거부
요청의 성공은 Action Layer가 명령을 접수했다는 의미이며, 실제 장치 결과는 명령
결과 API와 장치 로그를 함께 확인한다.

### 센서 Reading

| Method | 경로 | 용도 |
|---|---|---|
| GET | `/api/v1/sensor-readings` | 센서 Reading 조회 |
| POST | `/api/v1/sensor-readings` | 센서 Reading 수신 |

주요 query는 `device_id`, `table`, `limit`, `offset`, `time_from`, `time_to`다.
Reading 수신은 EdgeX·AIoT parser 경로와 연결되므로 payload 필드명과 timestamp 형식을
임의로 바꾸지 않는다.

### 외형 검색·조건

| Method | 경로 | 용도 |
|---|---|---|
| GET | `/api/v1/search` | 사람·외형 로그 검색 |
| GET | `/api/v1/search/crops/{filename}` | 허용된 crop 파일 조회 |
| GET | `/api/v1/appearances/conditions` | 외형 조건 목록 |
| GET | `/api/v1/appearances/status` | 외형 분석 런타임 상태 |
| POST | `/api/v1/appearances/conditions` | 외형 조건 추가 |
| DELETE | `/api/v1/appearances/conditions/{condition_id}` | 외형 조건 삭제 |

crop 파일명은 서버가 허용한 디렉터리 내부로 제한된다. `../` 형태의 경로를
직접 전달하지 않으며, 파일이 없으면 `404`를 반환한다.

### 장치 명령 결과

| Method | 경로 | 용도 |
|---|---|---|
| GET | `/api/v1/command-results` | 최근 명령 결과 목록 |
| GET | `/api/v1/command-results/{request_id}` | 특정 요청 결과 조회 |

명령 상태는 `queued`, `success`, `failed`, `timeout`, `simulated` 등이 될 수 있다.
`request_id`는 재시도·중복 제거·결과 연결에 사용되므로 로그에서 보존해야 한다.

## 4. Action Layer 내부 API

Action Layer는 기본적으로 `http://cctv-action-layer:8080`에서 수신한다.
`INTERNAL_SERVICE_TOKEN`이 설정되면 `/health`, `/ping`, `/metrics`, `/`를 제외한
요청에 다음 헤더가 필요하다.

```http
X-Internal-Token: <INTERNAL_SERVICE_TOKEN>
```

| Method | 경로 | 용도 |
|---|---|---|
| GET | `/health`, `/ping` | Action Layer·MQTT 상태 |
| GET | `/sites` | 사이트 목록 |
| POST | `/sites` | 사이트 추가 |
| DELETE | `/sites/{site_id}` | 사이트 삭제 |
| GET | `/pending` | 승인 대기 목록 |
| GET | `/mode` | 현재 모드 |
| POST | `/mode` | 모드·장치 설정 변경 |
| GET | `/devices` | 출력 장치 목록 |
| GET | `/events` | 최근 이벤트 |
| GET | `/commands` | 최근 명령 |
| POST | `/events` | 이벤트 수신 및 큐 적재 |
| POST | `/approve/{event_id}` | 이벤트 승인 |
| POST | `/reject/{event_id}` | 이벤트 거부 |
| GET | `/metrics` | Prometheus 메트릭 |

`POST /events`는 JSON 본문을 받고, 선택적으로 `topic` 필드를 사용한다.
정상 응답의 `queued=true`는 비동기 큐에 들어갔다는 뜻이다. 큐가 가득 차면
`503`이 반환되며, 이때 호출자는 같은 이벤트를 무분별하게 즉시 재전송하지 말고
request id와 Action Layer 로그를 확인한다.

## 5. 장치 제어 모드

현재 Jetson 기본값은 다음과 같다.

```text
EDGEX_SHADOW_ENABLED=false
EDGEX_COMMAND_MODE=direct
```

따라서 실제 운영 장치 제어는 Action Layer가 speaker/siren/signboard/dabit
전송 계층을 직접 호출한다. 커스텀 Device Service 컨테이너는 compose에 정의되어
있지만, 기본 direct 모드에서는 주 경로가 아니다.

EdgeX 전환 시에는 다음 순서를 지킨다.

1. 한 장치만 `shadow` 모드로 전환한다.
2. `edgex/commands/cctv/...` publish와 `edgex/results/cctv/...` 회신을 확인한다.
3. `/api/v1/command-results/{request_id}`에서 결과를 대조한다.
4. direct 결과와 EdgeX 결과가 일치한 뒤 `edgex` 모드로 전환한다.
5. 실패하면 `direct`로 되돌리고 outbox·명령 결과를 보존한다.

`core-command` 컨테이너가 배포되어 있다는 사실만으로 애플리케이션이 해당 REST
API를 사용한다는 뜻은 아니다. 현재 소스의 장치 전송 분기는 Action Layer의
명령 모드 선택에 의해 결정된다.

## 6. crop-pose POC 계약

crop-pose는 현재 운영 기본 경로가 아니며 `POSE_CROP_MODE=0`이 기본값이다.
POC를 켜면 `ObjectDetectionPipeline.detect_primary_people()`가 analyzer의
`_run_pose_crop()`를 선택한다. 해당 메서드가 없으면 기존 full-frame pose로
자동 fallback한다.

구현에 사용하는 공통 함수는 다음과 같다.

- `expand_bbox_for_pose_crop()`: person bbox를 여백 포함 crop 영역으로 변환
- `extract_pose_crops()`: 프레임에서 crop 이미지와 원본 좌표를 생성
- `restore_crop_keypoints()`: crop keypoint에 원점 offset을 더해 원본 좌표 복원
- `attach_pose_results_to_tracks()`: 복원된 keypoint를 person `track_id`와 결합
- `FallDetector.detect_keypoints()`: YOLO wrapper 없이 `Nx3` keypoint 배열 판정

운영 전에는 동일 영상에서 full-frame과 crop 모드를 각각 실행해 pose 생성률,
낙상 recall, FPS, p95 지연시간, GPU·시스템 메모리를 비교한다. 검증 전에는
`POSE_CROP_MODE=1`을 운영 compose에 적용하지 않는다.

상세 비교 순서와 결과 JSON 형식은
`docs/testing/POSE_CROP_COMPARISON_CHECKLIST.md`를 따른다. 결과를 저장한 뒤에는
`python scripts/ops/validate_pose_crop_compare_report.py <report.json>`으로
인수인계용 최소 계약을 검사한다.

## 7. 장애 대응 순서

1. `GET /api/v1/health`와 `GET http://cctv-action-layer:8080/health`를 각각 확인한다.
2. `GET /api/v1/readiness`에서 실패한 의존성을 확인한다.
3. 공개 API 응답의 `request_id` 또는 event id를 기준으로 Action Layer
   `/commands`와 `/api/v1/command-results/{request_id}`를 조회한다.
4. MQTT 연결, 장치 서비스 health, direct/EdgeX 모드를 확인한다.
5. 장치가 실제로 동작하지 않으면 같은 명령을 반복하기 전에 장치별 로그와
   `action_events.db`, `edgex_command_results.db`를 보존한다.
6. AI 낙상 문제는 pose 진단 JSONL의 `person_count`, `keypoint_count`,
   `keypoint_confidence_avg`, `stage`, `elapsed_ms`를 먼저 확인한다.

## 8. 배포·검증 명령

Jetson 배포 전:

```bash
python scripts/health/check_deployment_readiness.py --deployment-target jetson
python scripts/health/check_sensitive_defaults.py
python scripts/ops/check_edgex_device_service_contracts.py
```

공개 API 계약 확인:

```bash
curl http://localhost:9000/openapi.json
curl http://localhost:9000/docs
```

코드 변경 후에는 관련 pytest와 compose 검사를 실행한다. 학습 중인
`cctv-ai-engine`을 재시작해야 하는 변경은 feature extraction·RF·temporal 학습이
완료된 뒤 적용한다.

## 9. 소스 오브 트루스

- 공개 API 라우터: `src/api/app.py`, `src/api/v1/`
- 인증: `src/api/dependencies/auth.py`
- 공통 응답: `src/api/schemas/common.py`
- Action Layer API: `src/protocols/rest.py`
- 명령 모드·전송: `src/services/action_bridge.py`, `src/services/device_command_transport.py`
- Jetson 서비스 구성: `docker-compose.jetson.yml`
- 실행 계약: `/openapi.json`

라우트나 payload를 변경할 때는 코드와 이 문서를 같은 변경에서 수정하고,
호환성이 깨지면 별도 마이그레이션 메모를 남긴다.

## 2026-09-28 추가: 정확한 추적 ID 문자열

`GET /api/v1/search`의 항목에 `track_id_text: string | null`을 추가했다. 기존 `track_id` 정수 응답도 유지하지만 JavaScript는 큰 정수를 정확하게 표현하지 못하므로 ID 비교·저장에는 문자열 필드를 사용한다. 정확한 값을 얻으려고 이미 파싱된 정수에 `String(track_id)`를 적용하면 안 된다. [DB 호환·검증·적용 범위](planning/APPEARANCE_TRACK_ID.md). 로컬 수정 상태이며 운영 API 반영은 별도다.
