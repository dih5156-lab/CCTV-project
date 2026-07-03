# DeepStream 성능 측정 결과 및 안정성 테스트

## 결론

- 기존 30분 DeepStream 안정성 관찰 결과는 양호합니다.
- 2026-05-26 16:35 KST 기준 현재 재검증에서도 DeepStream 관련 단위 테스트와 데이터 플로우 스모크 루프는 통과했습니다.
- 단, 이번 재검증에서 `scripts/health/check_jetson_edgex_stack.py`의 일부 항목은 실패했습니다. 실패 원인은 DeepStream 자체 장애라기보다 호스트 기준 포트/인증/런타임 위치 차이로 보입니다.

## 기존 DeepStream 성능/안정성 결과

출처:

- `docs/reviews/PROJECT_REVIEW_2026-06.md`
- `/tmp/deepstream_stability_20260514_110527.log`

30분 안정성 관찰 요약:

| 항목 | 결과 |
| --- | --- |
| 샘플 수 | 30/30 정상 |
| 컨테이너 상태 | `healthy` 유지 |
| RestartCount | 0 |
| dropped | 0 |
| Stream API | `ok` 유지 |
| frames | `1,916,816 -> 1,956,999` (+40,183) |
| events | `298,157 -> 336,218` (+38,061) |
| cctv-ai-engine 메모리 | `694.5~694.8MiB` |
| Jetson RAM | `16,169~17,010MB / 62,828MB` |
| GPU 온도 | `47.9~49.2C` |
| TJ 온도 | `51.2~53.8C` |

해석:

- 30분 동안 컨테이너 재시작 없이 프레임과 이벤트가 계속 증가했습니다.
- `dropped=0`이 유지되어 해당 관찰 구간에서는 프레임 드롭 이슈가 확인되지 않았습니다.
- 메모리 사용량이 약 695MiB 근처에서 안정적으로 유지되어 짧은 관찰 기준 메모리 증가 징후는 보이지 않았습니다.

## 2026-05-26 현재 재검증

실행 시각:

- 2026-05-26 16:20~16:35 KST

### DeepStream 단위 테스트

명령:

```bash
.venv/bin/python -m pytest tests/test_deepstream_event_factory.py tests/test_deepstream_face_context.py tests/test_deepstream_processor.py
```

결과:

```text
32 passed, 5 skipped in 21.69s
```

해석:

- DeepStream 이벤트 변환, 얼굴 context 후처리, Processor 인터페이스 테스트는 통과했습니다.
- `5 skipped`는 현재 테스트 정책상 DeepStream 런타임 의존 테스트가 환경에 따라 건너뛰어진 항목입니다.

### 현재 컨테이너 상태

명령:

```bash
sudo docker inspect cctv-ai-engine --format 'Status={{.State.Status}} Health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} RestartCount={{.RestartCount}} StartedAt={{.State.StartedAt}} Runtime={{.HostConfig.Runtime}}'
```

결과:

```text
Status=running Health=healthy RestartCount=0 StartedAt=2026-05-26T06:27:56.474580701Z Runtime=nvidia
```

현재 리소스:

```text
CPU=114.20%
MEM=628.9MiB / 61.36GiB
MEM_PERC=1.00%
PIDS=70
```

컨테이너 내부 DeepStream Python 바인딩:

```text
gi+pyds import ok
```

해석:

- `cctv-ai-engine`은 `nvidia` runtime으로 실행 중이고 health도 정상입니다.
- 컨테이너 내부에서는 `gi`, `pyds` import가 정상입니다.

### 데이터 플로우 스모크 테스트

1회 실행:

```bash
.venv/bin/python scripts/smoke/smoke_test_data_flow.py
```

결과:

```text
passed=true
alert api accepts alert: PASS
alert api accepts sensor reading: PASS
action layer accepts event: PASS
action layer metrics expose handled events: PASS
public api metrics endpoint: PASS
```

3분 반복 실행:

```bash
./scripts/smoke/run_smoke_loop.sh 3 30
```

결과:

```text
총 실행: 6회
PASS: 6
FAIL: 0
실패율: 0%
```

해석:

- Alert API, Action Layer, Public API metrics로 이어지는 기본 데이터 플로우는 3분 동안 안정적으로 통과했습니다.
- 이 테스트는 DeepStream 영상 처리 부하 자체를 장시간 검증하는 테스트는 아니고, 주변 이벤트 처리 경로의 짧은 안정성 확인입니다.

## 재검증 중 확인된 주의 사항

### 1. `check_jetson_edgex_stack.py` 일부 실패

명령:

```bash
.venv/bin/python scripts/health/check_jetson_edgex_stack.py --host localhost --public-api-port 9000 --deepstream --check-appearance-status --json
```

실패 항목:

| 항목 | 결과 | 해석 |
| --- | --- | --- |
| AIoT Parser DB | `localhost:5432 Connection refused` | DB 컨테이너는 실행 중이지만 호스트 포트로 publish되어 있지 않음 |
| AIoT Parser | `localhost:3500 Connection refused` | `aiot-parser`는 컨테이너 내부 `4000/tcp`로 실행 중이며 호스트 `3500` 노출 없음 |
| Public API Appearance Status | `401 Unauthorized` | API key 없이 호출되어 인증 실패 |
| GStreamer Python | `No module named gi` | 호스트 Python 기준 검사 실패 |
| DeepStream pyds | `No module named pyds` | 호스트 Python 기준 검사 실패 |

중요:

- 현재 배포 형태에서는 DeepStream 런타임이 `cctv-ai-engine` 컨테이너 내부에 있으므로, `gi/pyds` 최종 판단은 컨테이너 내부 import 결과를 기준으로 보는 것이 맞습니다.
- Parser 관련 실패는 컨테이너 상태와 포트 노출 정책을 분리해서 봐야 합니다. `docker ps` 기준 `aiot-parser`, `aiot-parser-db`는 모두 `healthy`였습니다.

### 2. Stream API 포트 차이

문서 일부에서는 `8769`를 Stream API health로 사용하지만, 현재 `cctv-ai-engine`은 호스트에 `8765-8767`만 노출되어 있습니다.

확인 결과:

| 포트 | 서비스 | 결과 |
| --- | --- | --- |
| 8765 | `cctv-zone-api` | `ok` |
| 8766 | `cctv-camera-model-api` | `ok` |
| 8767 | `cctv-face-api` | `ok` |
| 8769 | Stream API 문서 기준 | 연결 거부 |

## 현재 판단

- 확실히 검증됨: DeepStream 단위 테스트, AI 엔진 컨테이너 health, 컨테이너 내부 `gi/pyds`, 기본 데이터 플로우 3분 안정성.
- 기존 근거로 검증됨: 2026-05-14 30분 DeepStream 처리 안정성, `dropped=0`, 메모리 안정.
- 추가 확인 필요: 현재 실행 상태에서 장시간 DeepStream stats 증가량, `8769` Stream API 포트 문서와 compose 설정 일치 여부, `check_jetson_edgex_stack.py`의 호스트/컨테이너 검사 기준 정리.


## 표준 안정성 통과 기준

운영/데모 배포 전 DeepStream 안정성은 아래 기준으로 판단합니다.

| 기준 | 통과 조건 | 확인 방법 |
| --- | --- | --- |
| 실행 지속성 | 관찰 시간 동안 `cctv-ai-engine`이 `running`/`healthy` 유지 | `docker inspect` |
| 재시작 | `RestartCount=0` 유지 | `docker inspect` |
| API 상태 | Public API, Zone API, Model API, Face API health 모두 통과 | stability watch 로그 |
| 데이터 흐름 | `scripts/smoke/smoke_test_data_flow.py` 반복 통과 | stability watch 로그 |
| 리소스 | 메모리 사용량이 지속 증가하지 않음 | `docker stats`, `tegrastats` |
| 프레임 처리 | `dropped=0` 또는 원인 설명 가능한 수준 유지 | DeepStream 로그 |
| 장애 흔적 | 반복 `ERROR`/timeout 없음 | 컨테이너 로그 |

판정 기준:

- `PASS`: 전체 샘플 `FAIL=0`, `RestartCount=0`, 컨테이너 health 정상 유지
- `NEEDS REVIEW`: 일시 실패가 1회 이상 있으나 원인이 확인되고 재검증 통과
- `FAIL`: 반복 실패, 컨테이너 재시작, health 비정상, 메모리 지속 증가 중 하나라도 확인

## 표준 실행 방법

DeepStream 관찰 전 기본 운영 점검 wrapper를 먼저 실행합니다.

```bash
./scripts/ops/run_operation_check.sh
```

기본 12시간 관찰:

```bash
sudo -v
./scripts/ops/run_deepstream_stability_watch.sh
```

짧은 재확인:

```bash
sudo -v
./scripts/ops/run_deepstream_stability_watch.sh 30 30
```

결과 파일은 기본적으로 아래 위치에 생성됩니다.

```text
reports/deepstream-stability/deepstream_stability_<timestamp>.log
reports/deepstream-stability/deepstream_stability_<timestamp>.summary
```

요약 파일에서 우선 확인할 값:

```text
result=pass
samples=<샘플 수>
fail=0
failure_rate=0.0%
```

주의:

- `fail=1`이어도 실패율이 반올림되어 작게 보일 수 있으므로, 최종 판정은 항상 `fail` 값을 기준으로 봅니다.
- `data flow smoke` timeout이 다시 발생하면 Action Layer 로그와 실제 장비 제어 timeout을 같이 확인합니다.
- `8769` Stream API는 현재 표준 health 경로에서 제외하고, `8765/8766/8767`을 기준으로 확인합니다.

## 다음 실행 권장

장시간 안정성 재측정은 표준 스크립트로 실행합니다.

```bash
sudo -v
./scripts/ops/run_deepstream_stability_watch.sh 720 60
```

30분 단위 빠른 재확인이 필요하면 아래처럼 줄여서 실행합니다.

```bash
sudo -v
./scripts/ops/run_deepstream_stability_watch.sh 30 30
```

수동으로 원인을 더 볼 때만 아래 항목을 별도 터미널에서 확인합니다.

```bash
sudo docker inspect cctv-ai-engine --format 'Status={{.State.Status}} Health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} RestartCount={{.RestartCount}} Runtime={{.HostConfig.Runtime}}'
sudo docker stats cctv-ai-engine --no-stream
sudo docker logs --tail 120 cctv-ai-engine
```


## 12시간 안정성 관찰 후속 조치

2026-05-27 04:49 KST 종료된 12시간 관찰 결과:

```text
총 샘플: 665
PASS: 664
FAIL: 1
실패율: 0%
```

실패 1건의 원인:

- 실패 샘플: `sample=17`, `2026-05-26T17:07:05+09:00`
- 실패 항목: `data flow smoke`
- 세부 실패: `action layer accepts event`, `POST http://localhost:8080/events`, `detail=timed out`
- 같은 샘플에서 DeepStream 컨테이너는 `running`, `healthy`, `RestartCount=0`이었고 Public API, Zone API, Camera Model API, Face API는 모두 정상
- Action Layer 로그상 실제 카메라 이벤트와 smoke 이벤트가 겹친 시점에 전광판 `display()` timeout이 발생함

조치:

- `POST /events` REST 경로는 이벤트를 백그라운드 큐에 넣고 즉시 `200 {"status": "ok", "queued": true}`를 반환하도록 변경
- 실제 장비 제어와 외부 전송은 `RestActionWorker`에서 비동기로 처리
- MQTT 수신 경로와 수동 승인 경로는 기존 동기 동작 유지

검증:

```bash
.venv/bin/python -m pytest tests/test_action_bridge.py tests/test_integration_action_pipeline.py tests/test_smoke_test_data_flow.py
.venv/bin/python -m py_compile src/services/action_bridge.py src/protocols/rest.py
.venv/bin/python scripts/smoke/smoke_test_data_flow.py
curl -fsS -X POST http://localhost:8080/events -H 'Content-Type: application/json' -d '{"camera_id":"smoke-cam-async","type":"helmet","severity":"low","confidence":0.99}'
```

검증 결과:

- 관련 테스트: `54 passed`
- 문법 검사: 통과
- 실서비스 smoke test: 통과
- 직접 REST 호출 응답: `{"status": "ok", "queued": true}`
