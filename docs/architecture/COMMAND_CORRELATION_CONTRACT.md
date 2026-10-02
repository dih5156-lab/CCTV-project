# 명령 식별자 상관관계 계약

## 목적

다중 CCTV와 다중 출력 디바이스 환경에서 이벤트, 논리 명령, 물리 장치 요청,
결과를 서로 혼동하지 않도록 식별자 관계를 고정한다.

## 식별자 관계

```text
event_id
  └─ command_id = event_id:device
       └─ request_id = command_id:device_id
```

| 식별자 | 생성 위치 | 의미 | 예시 |
|---|---|---|---|
| `event_id` | 이벤트 정규화 | 원본 AI 이벤트 1건 | `event-1` |
| `command_id` | Action Layer | 이벤트에 대한 논리 장치 명령 | `event-1:signboard` |
| `request_id` | EdgeX Transport | 특정 물리 장치로 보낸 요청 | `event-1:signboard:signboard-1` |

## 처리 규칙

1. 하나의 `event_id`는 여러 `command_id`를 가질 수 있다.
   - 스피커: `event-1:speaker`
   - 전광판: `event-1:signboard`
2. 하나의 `command_id`는 여러 `request_id`를 가질 수 있다.
   - 동일한 논리 장치가 여러 물리 장치에 연결될 수 있기 때문이다.
3. 공통 EdgeX 결과 감사 저장소는 `request_id`를 기준으로 저장한다.
4. Action Layer의 논리 명령 상태는 물리 장치 접미사를 제거한 `command_id`를 기준으로 갱신한다.
5. `request_id`가 없는 결과는 어떤 물리 명령의 응답인지 확인할 수 없으므로 무시한다.
6. `request_id`가 `command_id:device_id` 형식이면 `device_id`와 일치하는 마지막 접미사만 제거한다.
7. EdgeX 레지스트리에서 대상 물리 장치를 찾지 못하면 명령을 발행하지 않고 `failed`로 기록한다.
   - 물리 장치 ID가 없는 유형 공통 토픽으로 발행하면 의도하지 않은 장치가 동작할 수 있기 때문이다.

## 상태 흐름

```text
이벤트 수신
  ↓
event_id 생성·확정
  ↓
command_id 생성 및 queued/sent 기록
  ↓
request_id 생성 및 EdgeX 발행
  ↓
EdgeX 결과 수신
  ↓
request_id 결과 기록
  ↓
devices/result 상태 발행
```

Action Layer 명령 이력은 별도로 다음처럼 갱신된다.

```text
request_id = event-1:signboard:signboard-1
device_id  = signboard-1
        ↓
command_id = event-1:signboard
        ↓
action_commands 상태 갱신
```

## Action 상태 이벤트 추적 필드

Action Layer가 발행하는 `cctv/status/action/events/pending`,
`events/filtered`, `events/executed` 상태에도 다음 필드를 포함한다.

| 필드 | 포함 조건 | 용도 |
|---|---|---|
| `event_id` | 항상 포함 | 원본 이벤트와 상태 이벤트 연결 |
| `request_id` | 입력 이벤트에 값이 있을 때 포함 | 상위 요청·외부 호출 연결 |
| `device_results[].command_id` | 장치 명령이 실행된 경우 포함 | 논리 장치 명령 결과 연결 |
| `device_results[].device_ids` | EdgeX 물리 대상이 있는 경우 포함 | 실제 대상 장치 목록 연결 |
| `device_results[].request_ids` | EdgeX 발행을 시도한 경우 포함 | 물리 장치별 요청 결과 연결 |

따라서 운영 로그는 `event_id → command_id → request_id` 순서로 조회하고,
Action Layer 실행 응답에서는 `device_results[].device_ids`와
`device_results[].request_ids`로 물리 장치별 결과를 확인한다.
영상 추론에서 생성된 이벤트인지 합성 UAT 이벤트인지 `source`·`metadata`로 구분한다.

## 실행 결과 예시

EdgeX 모드에서 하나의 논리 명령이 두 물리 장치로 발행된 경우:

```json
{
  "device": "signboard",
  "command_id": "event-1:signboard",
  "event_id": "event-1",
  "status": "acknowledged",
  "device_ids": ["signboard-1", "signboard-2"],
  "request_ids": [
    "event-1:signboard:signboard-1",
    "event-1:signboard:signboard-2"
  ]
}
```

Direct 모드는 기존 장치 클라이언트를 직접 호출하므로 물리 EdgeX 대상이
없다. 따라서 `device_ids`와 `request_ids`는 빈 배열이며, 논리 명령의
`event_id`·`command_id`·상태로 실행 이력을 추적한다.

## 운영 시 주의사항

- `request_id`를 새로 생성해 결과에 사용하면 안 된다. 발행 시 사용한 값을 그대로 전달해야 한다.
- 장치 재시도 시 같은 요청을 재전송할지, 새 요청 ID를 만들지 정책을 별도로 정해야 한다.
- 카메라 ID가 레지스트리의 `camera_ids`와 일치하지 않으면 대상 없음으로 처리되므로 운영 ID를 통일해야 한다.
- 현재 구현은 장치별 EdgeX 발행 결과를 `request_id`로 기록하지만, 비동기 재시도·타임아웃 정책은 다음 운영 검증 단계에서 확정해야 한다.
