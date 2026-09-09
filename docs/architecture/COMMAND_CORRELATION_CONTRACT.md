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

## 운영 시 주의사항

- `request_id`를 새로 생성해 결과에 사용하면 안 된다. 발행 시 사용한 값을 그대로 전달해야 한다.
- 장치 재시도 시 같은 요청을 재전송할지, 새 요청 ID를 만들지 정책을 별도로 정해야 한다.
- 현재 구현은 장치별 EdgeX 발행 결과를 `request_id`로 기록하지만, 비동기 재시도·타임아웃 정책은 다음 운영 검증 단계에서 확정해야 한다.
