# CCTV 프로젝트 완성도 점검 - 2026년 7월

## 결론

현재 프로젝트는 단순 POC를 넘어, Jetson/DeepStream 현장 배포를 전제로 한 통합형 엣지 CCTV 플랫폼 단계입니다.

완성도는 개발/검증 기준으로는 높지만, 운영 투입 전에는 아래 3가지를 우선 잠가야 합니다.

1. 낙상 보조 모델의 shadow 데이터 기반 승격 기준
2. DeepStream 장시간 안정성 및 카메라 장애 복구 기준
3. 운영 보안값과 외부 장비 장애 처리 기준

## 현재 검증 상태

2026-07-01 기준 로컬 검증 결과:

- 정적 검사: `ruff check src scripts tests` 통과
- 패치 형식 검사: `git diff --check` 통과
- 전체 테스트: `1178 passed, 5 skipped`
- 테스트 수집: `1183 items`
- falldata aux 환경 점검: 통과
- falldata sample manifest readiness: 실패
  - 원인: non-fall scene group이 1개뿐이라 모델 승격 기준인 class별 최소 2개 group을 만족하지 못함
- falldata model promotion check: 실패
  - `falldata_sample_rf_max120_standard_metrics.json` 기준 non-fall group 부족 및 false positive 8건
- fall shadow review 요약:
  - 전체 review record: 850건
  - clip 보유 record: 792건
  - label 상태: 850건 모두 `unlabeled`
  - 우선 라벨링 후보는 `scripts/ops/summarize_fall_shadow_review.py`의 `labeling_candidate_examples`에서 확인 가능
- compose runtime assumptions: Jetson/arm64 호스트에서 기본 `docker-compose.yml` EdgeX 이미지 사용 위험으로 실패
  - 해석: Jetson 배포는 `docker-compose.jetson.yml` 사용 또는 API/action 서비스만 선택 실행 필요

실행한 명령:

```bash
.venv/bin/python -m ruff check src scripts tests --fix
.venv/bin/python -m ruff check src scripts tests
git diff --check
.venv/bin/python -m pytest -q
.venv/bin/python scripts/health/check_compose_runtime_assumptions.py
.venv/bin/python scripts/health/check_falldata_aux.py
.venv/bin/python scripts/health/check_fall_manifest_readiness.py
.venv/bin/python scripts/health/check_falldata_model_report.py --metrics-json models/experiments/falldata_sample_rf_max120_standard_metrics.json
.venv/bin/python scripts/ops/summarize_fall_shadow_review.py
```

## 현재 변경 묶음

### 1. falldata 낙상 보조 검증

관련 파일:

- `src/core/ai/_falldata_aux.py`
- `src/core/_fall_shadow_review.py`
- `scripts/datasets/train_falldata_video_rf.py`
- `scripts/datasets/compare_falldata_video_models.py`
- `scripts/health/check_falldata_aux.py`
- `scripts/health/check_fall_manifest_readiness.py`
- `scripts/health/check_falldata_model_report.py`
- `scripts/ops/summarize_fall_shadow_review.py`
- `docs/features/FALLDATA_INTEGRATION.md`
- `docs/guides/MLOPS_MODEL_EVALUATION.md`

의미:

- pose 기반 낙상 후보를 falldata RF 모델로 shadow/confirm 검증할 수 있습니다.
- 검증 실패 시 fail-open/fail-closed 정책을 테스트로 확인합니다.
- 학습/비교/승격 점검 스크립트가 추가되어 모델 교체 판단 근거를 남길 수 있습니다.

운영 전 확인:

- 현장 카메라별 shadow 로그를 최소 1~2일 수집합니다.
- `false_positive`, `false_negative`, `near_miss`를 분리해서 확인합니다.
- confirm 모드 전환은 카메라 각도/조명별 성능 기준을 만족한 뒤 적용합니다.
- 현재 sample manifest는 non-fall group이 부족하므로 confirm 승격용 근거로 사용하지 않습니다.
- 현재 shadow review clip은 충분하지만 라벨이 없으므로, 먼저 clip을 `fall` / `non_fall`로 검수해야 합니다.

### 2. DeepStream 구조 분리 및 안정화

관련 파일:

- `src/core/deepstream_processor.py`
- `src/core/_deepstream_element_config.py`
- `src/core/_deepstream_env.py`
- `src/core/_deepstream_event_queue.py`
- `src/core/_deepstream_labels.py`
- `src/core/_deepstream_model_flags.py`
- `src/core/_deepstream_source_health.py`
- `src/core/_deepstream_topology.py`
- `tests/test_deepstream_processor.py`

의미:

- `deepstream_processor.py`의 책임을 설정, 라벨, 이벤트 큐, source health, topology helper로 분리했습니다.
- 카메라별 모델 플래그, source retry, DeepStream element 설정 검증 범위가 넓어졌습니다.

운영 전 확인:

- Jetson에서 4대 이상 카메라 기준 1시간, 8시간, 24시간 단계로 stability watch를 수행합니다.
- `frames_dropped`, source retry 횟수, 마지막 프레임 시간, GPU 메모리, 발열을 함께 봅니다.
- 특정 카메라 반복 장애 시 source 단위 재연결과 전체 pipeline 재시작 조건을 분리합니다.

### 3. Action Bridge 책임 분리

관련 파일:

- `src/services/action_bridge.py`
- `src/services/_action_bridge_support.py`
- `src/services/_action_bridge_alarm.py`
- `src/services/_action_bridge_executor.py`
- `src/services/_action_bridge_models.py`
- `src/services/_action_bridge_payloads.py`
- `src/services/_action_bridge_repo.py`
- `src/services/_action_bridge_rest_queue.py`
- `src/services/_action_bridge_site_registry.py`
- `src/services/_action_bridge_topics.py`
- `src/services/_device_reachability.py`

의미:

- 알람 판단, 실행, 저장소, REST queue, 사이트 설정, topic 정의를 분리했습니다.
- 기존 `_action_bridge_support.py`는 호환 import로 남겨 기존 테스트/외부 import 영향을 줄였습니다.

운영 전 확인:

- 스피커/경광등/전광판이 일부 장애일 때 나머지 장비와 HTTP forward가 계속 동작하는지 확인합니다.
- REST action queue full, 외부 API timeout, 장비 TCP unreachable이 metrics와 로그에 남는지 확인합니다.

### 4. 외형 분석/스트림 API 개선

관련 파일:

- `src/core/ai/_appearance_analyzer.py`
- `src/core/ai/_appearance_pipeline.py`
- `src/services/appearance_log.py`
- `src/services/stream_api.py`
- `scripts/ops/audit_appearance_colors.py`

의미:

- 외형 색상/속성 로그와 검색 정확도 확인 도구가 보강되었습니다.
- Stream API 접근 제한과 JPEG/프레임 처리 테스트가 강화되었습니다.

운영 전 확인:

- 현장 조명에서 HSV 색상 오탐 패턴을 audit 리포트로 확인합니다.
- crop 저장량과 정리 정책을 운영 주기에 맞춥니다.

### 5. 환경/Compose 점검

관련 파일:

- `.env.example`
- `.env.jetson.example`
- `docker-compose.yml`
- `docker-compose.jetson.yml`
- `scripts/health/check_compose_runtime_assumptions.py`

의미:

- falldata, appearance, runtime path, MQTT auth, 필수 secret 등 운영 가정 검사가 강화되었습니다.

운영 전 확인:

- 운영 환경에서는 `PUBLIC_API_KEY`, `INTERNAL_SERVICE_TOKEN`, `MQTT_USER`, `MQTT_PASSWORD`, `GRAFANA_ADMIN_PASSWORD`를 실제 값으로 설정합니다.
- 개발 편의용 wildcard CORS나 빈 token이 운영 Compose에 섞이지 않도록 readiness 단계에서 차단합니다.

## 오류 처리 보강 우선순위

### P0 - 운영 중단 방지

- MQTT publish 실패 시 SQLite outbox 또는 bounded retry 정책을 명확히 합니다.
- RTSP 카메라가 반복 실패할 때 카메라 단위 degraded 상태와 pipeline 재시작 조건을 분리합니다.
- disk full 또는 SQLite lock 발생 시 이벤트 저장 실패 counter를 metrics에 노출합니다.

### P1 - 오탐/미탐 관리

- 낙상 confirm 모드 전환 전 shadow review 기준을 문서화합니다.
- near-miss와 실제 fall 후보를 별도 집계합니다.
- 카메라별 threshold override를 운영 API 또는 설정으로 관리합니다.

### P2 - 외부 장비 장애 대응

- 스피커/경광등/전광판별 timeout, 인증 실패, unreachable 상태를 분리합니다.
- 한 장비 실패가 전체 Action Bridge 처리를 막지 않도록 장비별 실패 격리를 유지합니다.
- cooldown 정책과 demo override 정책을 운영 로그에 명확히 남깁니다.

### P3 - 운영 보안

- Public API, Stream API, 내부 제어 API의 token 필수 조건을 운영 모드에서 강제합니다.
- Grafana/MQTT/DB 기본 비밀번호 사용을 readiness에서 실패 처리합니다.
- 외부 노출 포트는 Public API 등 필요한 포트로 제한합니다.

## 다음 실행 순서

1. 변경 묶음을 기능별 커밋 단위로 분리합니다.
2. fall shadow review clip 중 `labeling_candidate_examples`부터 라벨링합니다.
3. 라벨링된 non-fall group을 manifest에 반영한 뒤 RF 모델을 재학습/재평가합니다.
4. Jetson에서 DeepStream stability watch를 다시 수행합니다.
5. 운영 Compose에서 secret/readiness 검사를 통과시킵니다.

권장 확인 명령:

```bash
.venv/bin/python -m ruff check src scripts tests
.venv/bin/python -m pytest -q
.venv/bin/python scripts/health/check_compose_runtime_assumptions.py
.venv/bin/python scripts/health/check_falldata_aux.py
.venv/bin/python scripts/health/check_fall_manifest_readiness.py
.venv/bin/python scripts/health/check_falldata_model_report.py --metrics-json models/experiments/falldata_sample_rf_max120_standard_metrics.json
.venv/bin/python scripts/ops/summarize_fall_shadow_review.py
./scripts/ops/run_deepstream_stability_watch.sh
```
