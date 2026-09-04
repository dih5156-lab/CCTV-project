# 수정·검증 가이드

## 1. 수정 전 공통 원칙

먼저 변경 지점을 하나의 흐름으로 적는다.

```text
입력 계약 → 정규화 → 판정/변환 → MQTT·HTTP·EdgeX 출력 → 저장·메트릭 → 테스트
```

환경 변수·토픽·DB 스키마·EdgeX profile·Compose를 바꾸는 경우에는 관련 문서와 rollback 방법도 함께 수정한다. 비밀번호·토큰은 코드나 문서에 실제 값을 넣지 않는다.

## 2. 대표 변경별 위치

| 변경 요구 | 우선 확인할 위치 | 함께 수정할 것 |
|---|---|---|
| 낙상 민감도 조정 | `src/core/ai/_fall_detector.py` | `.env*.example`, pose 문서, 분석 테스트 |
| DeepStream 후처리 | `src/core/_yolo_postprocess.py`, `deepstream_processor.py` | 배포 문서, 성능 로그, replay 테스트 |
| 새 AI 이벤트 | AI analyzer·event publisher | MQTT 계약, Alert/Action 구독, API 문서 |
| 새 센서 | `parser-python/tlv`, `sensor_rule_bridge.py` | fixture, rule, EdgeX profile 여부 |
| 새 출력 장치 | `src/devices`, `src/services/action_bridge.py` | device API 문서, cooldown·실패 처리 |
| API 변경 | `src/api` 또는 `src/protocols/rest.py` | 인증, schema, API 테스트 |
| Compose 변경 | `docker-compose*.yml`, `.env*.example` | healthcheck, volume, 운영 런북 |

## 3. 최소 검증 순서

```bash
git diff --check
python -m pytest -q tests parser-python/tests
ruff check src tests parser-python
docker compose --env-file .env.jetson -f docker-compose.jetson.yml config --quiet
```

장비·Docker·GPU가 없는 환경에서는 unit test와 Compose config까지만 실행하고, 현장 검증 미실행 사실을 기록한다. Jetson 변경은 operation check와 짧은 DeepStream stability watch를 추가한다.

## 4. 변경 후 인수인계 기록

다음 네 가지를 PR 또는 작업 기록에 남긴다.

1. 변경 파일과 이유
2. 입력·출력 계약 변화 여부
3. 실행한 검증 명령과 결과
4. 운영 중 rollback 방법과 영향 범위

## 5. 장애 대응 순서

1. health/readiness로 어느 계층부터 끊겼는지 찾는다.
2. MQTT topic을 원본→정규화→규칙 결과 순서로 구독한다.
3. 해당 서비스 로그와 outbox pending을 확인한다.
4. 장치별 API를 직접 호출해 네트워크·인증·프로토콜을 분리 확인한다.
5. 수정 후 같은 입력 fixture 또는 실제 영상으로 재현·검증한다.

