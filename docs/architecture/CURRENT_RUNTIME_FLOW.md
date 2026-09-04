# 현재 운영 경로

## 문서 목적

이 문서는 현재 코드가 실제로 어떤 경로로 이벤트를 받고 장치를 제어하는지 기록한다. 목표 아키텍처와 혼동하지 않도록 현재 동작과 아직 활성화되지 않은 경로를 분리해서 설명한다.

## 현재 확인된 핵심 진입점

```text
카메라·센서 이벤트
        ↓
MQTT / Alert API
        ↓
ActionBridge._on_message
        ↓
ActionBridge._dispatch_command
        ↓
_ActionExecutor.execute
```

현재 코드 그래프 기준으로 `ActionBridge._dispatch_command`는 메시지 수신 이후 명령 실행을 시작하고, `_ActionExecutor.execute`는 다음 책임을 함께 가지고 있다.

- 이벤트 위험도와 경보 가능 여부 확인
- 사이트·카메라 기준 출력 장치 조회
- 직접 장치 호출
- EdgeX Shadow Command 발행
- 명령 상태·결과 저장
- 상태 이벤트 발행

## 현재 출력 경로

```text
_ActionExecutor.execute
    ├─ 직접 경로
    │   ├─ SpeakerDevice.play
    │   ├─ SirenClient.trigger
    │   └─ SignboardDevice.display / clear / power_on / power_off
    │
    ├─ EdgeX Shadow 경로
    │   └─ ActionBridge._publish_edgex_command
    │       └─ _publish_shadow_command
    │           └─ direct_status / edgex_publish_status 비교 기록
    │
    └─ 감사·상태 경로
        ├─ EventRepo.record_command / save
        ├─ CommandResultStore.get
        └─ 상태 MQTT 발행
```

## 현재 상태 판단

| 영역 | 현재 상태 | 판단 |
|---|---|---|
| AI 이벤트 생성 | 동작 중 | DeepStream/센서 이벤트를 공통 이벤트로 전달 |
| Action Layer | 동작 중 | 이벤트 수신, 위험도, 장치 선택, 명령 실행 담당 |
| 직접 장치 제어 | 운영 기본 경로 | 실제 장치별 클라이언트를 직접 호출 |
| EdgeX Command | 코드·테스트 준비 | Shadow 또는 호환 경계가 있으나 운영 기본 경로는 아님 |
| 다중 디바이스 레지스트리 | 구현됨 | `device_id` 기준 장치 선택 가능 |
| 명령 결과 수집 | 구현됨 | 결과 저장·조회 및 Action Layer 이력 반영 가능 |
| Shadow 비교 기록 | 구현됨 | 같은 `command_id`에 직접 결과와 EdgeX 발행 결과를 함께 저장 |
| 실제 장치 UAT | 미완료 | 물리 장치 연결 후 검증 필요 |

## 현재 환경 기준선

공유 가능한 환경 템플릿과 Jetson Compose 기준으로 현재 기본값은 다음과 같다.

| 설정 | 현재 기본값 | 의미 |
|---|---|---|
| `EDGEX_SHADOW_ENABLED` | `false` | EdgeX Shadow 명령 발행 비활성화 |
| `EDGEX_COMMAND_MODE` | 비어 있음 | 비어 있으면 기존 Shadow 설정과 호환하며 `direct` 또는 `shadow`로 결정 |
| `EDGEX_ALLOWED_DEVICES` | 비어 있음 | EdgeX/Shadow 모드에서 허용할 장치 목록; 예: `signboard` |
| `EDGEX_DEVICE_REGISTRY_PATH` | 비어 있음 | 다중 출력 장치 레지스트리 비활성화 |
| `EDGEX_COMMAND_TOPIC_PREFIX` | `edgex/commands/cctv` | 공통 명령 topic 접두사 |
| `EDGEX_RESULT_TOPIC_PREFIX` | `edgex/results/cctv` | 명령 결과 topic 접두사 |
| `EDGEX_COMMAND_RESULT_DB` | `/app/data/runtime/edgex_command_results.db` | 명령 결과 감사 저장소 |
| `SPEAKER_HOST` | 설정 가능 | 서비스 경계는 있으나 실제 스피커 연결·동작 검증 불가 |
| `SPEAKER_DRY_RUN` | `false` | 호스트가 설정되면 실제 호출을 시도하는 기본값 |
| `SIREN_SERVICE_PORT` | `59992` | 사이렌 Device Service HTTP 경계만 존재하며 물리 사이렌은 없음 |

따라서 현재 구조는 EdgeX 서비스 자체가 Compose에 포함되어 있어도, 공유 환경 템플릿 기준으로는 직접 장치 제어가 기본이고 EdgeX 명령은 opt-in 상태다. 이 값을 바꾸는 작업은 실제 장치 UAT와 shadow 비교 이후에 수행해야 한다.

스피커와 사이렌을 사용할 수 없는 현장에서는 전체 모드를 바꾸기 전에 다음처럼 전광판만 허용한다.

```env
EDGEX_COMMAND_MODE=edgex
EDGEX_ALLOWED_DEVICES=signboard
```

이 설정은 사이트별 장치 선택 결과에서 스피커·사이렌을 제거한다. 비워두면 기존 사이트 설정과 호환되며 모든 선택 장치를 허용한다.

### 2026-09-04 실행 환경 확인

- CCTV AI Engine, Action Layer, EdgeX Core Command, 전광판 Device Service가 실행 중이다.
- Action Layer 실행 환경에 `EDGEX_COMMAND_MODE`가 지정되지 않아 현재 이벤트 장치 제어는 direct 호환 경로로 동작한다.
- `cctv-signboard-01`은 EdgeX Core Command의 `power`, `clear`, `display` 실제 호출을 통과했다.
- 사이렌은 현재 물리 디바이스가 없어 실제 UAT 대상이 아니며, 스피커는 서비스 경계는 있으나 실제 연결·동작 검증이 불가능하다.
- 전광판의 문구·색상·배경색·크기·속도·밝기 전달을 실제 Dabit 장치 응답으로 확인했다.
- 따라서 전체 Action Layer를 `edgex` 모드로 전환하지 않고, 전광판 단독 라우팅만 운영 검증 대상으로 삼아야 한다.

### Shadow 비교 결과 형식

Shadow 모드에서는 직접 장치 제어를 계속 수행하고, 같은 `command_id`로 EdgeX Command를 비교 발행한다. 비교 결과는 `action_commands.payload_json.shadow_comparison`에 저장된다.

```json
{
  "mode": "shadow",
  "direct_status": "acknowledged",
  "edgex_publish_status": "acknowledged",
  "comparison": "match"
}
```

`edgex_publish_status=acknowledged`는 EdgeX Command MQTT 발행 성공을 의미한다. 실제 스피커·사이렌·전광판이 물리적으로 동작했다는 뜻은 아니며, 물리 동작 확인은 Device Service 결과와 현장 UAT로 별도 검증해야 한다.

## 구조상 확인된 문제

현재 `_ActionExecutor.execute`가 경보 정책, 장치 선택, 직접 호출, EdgeX 발행, 저장을 동시에 다룬다. 이 구조는 기능은 동작하지만 다음 문제가 있다.

1. 직접 제어와 EdgeX 제어 중 어떤 경로가 실제 운영 기준인지 코드만 보고 판단하기 어렵다.
2. 장치별 장애가 Action Layer의 이벤트 처리와 섞일 수 있다.
3. EdgeX 전환 시 실행 정책과 전송 방식이 함께 바뀌어 롤백 범위가 커진다.
4. 다중 디바이스 라우팅과 장치별 재시도 정책을 독립적으로 검증하기 어렵다.

## 당장 유지해야 하는 것

- AI 추론과 낙상 판정은 Jetson 로컬 경로에서 유지한다.
- 이벤트 계약과 `event_id`, `request_id`, `camera_id`, `device_id` 추적 필드는 유지한다.
- 실제 장치 UAT 전까지 직접 경로를 제거하지 않는다.
- EdgeX 장애가 AI 이벤트 생성과 저장을 중단시키지 않도록 한다.
