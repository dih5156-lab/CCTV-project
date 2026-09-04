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
| 실제 장치 UAT | 미완료 | 물리 장치 연결 후 검증 필요 |

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

