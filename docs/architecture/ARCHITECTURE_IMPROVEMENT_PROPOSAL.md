# Edge AIoT 아키텍처 보완안

기준일: 2026-09-04

## 결론

현재 구조는 `입력 → AI·센서 분석 → MQTT → 규칙 → Action Layer → EdgeX → Device Service → 장치` 방향으로 가는 것이 맞다.
다만 실시간 현장 제어와 정부과제 실증을 안정적으로 운영하려면 Action Layer를 단일 책임 모듈로 유지하지 말고 정책·매핑·전송·결과 수집으로 분리해야 한다.

전임자의 `Device Manager` 개념은 유지하되, 제조사 직접 통신은 Device Service 안에만 두고 Action Layer에서는 표준 명령만 생성한다.

## 권장 구조

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Edge AIoT Data Plane                                 │
│                                                                             │
│  [CCTV] ─┐                                                                  │
│  [AIoT] ─┼→ [입력 어댑터/정규화] → [AI·센서 분석] → [공통 이벤트]             │
│          │                                      │                            │
│          └──────────────────────────────────────┴→ [로컬 MQTT]               │
│                                                         │                    │
│                                               [eKuiper / Rule Bridge]         │
│                                                         │                    │
│                                               [Action Policy]                 │
│                                      ┌──────────┼──────────┐                  │
│                                      │          │          │                  │
│                               [중복·쿨다운] [장치매핑] [안전승인]              │
│                                      └──────────┼──────────┘                  │
│                                         [공통 Command]                        │
│                                                 │                             │
│                                         [EdgeX Core Command]                  │
│                                                 │                             │
│                       ┌─────────────────┬──────┴──────┬────────────────┐    │
│                       │                 │             │                │    │
│                [Signboard DS]   [Speaker DS]   [Siren DS]      [CCTV DS]   │
│                       │                 │             │                │    │
│                 TCP/Dabit          HTTP/InterM     GPIO/HTTP        RTSP    │
│                       │                 │             │                │    │
│                    전광판            스피커          사이렌              │    │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌────────────────────────────┐
                         │ 병렬 감사·관측 경로          │
                         │ 결과 수집·SQLite·API·메트릭 │
                         └────────────────────────────┘
```

## 전임자 구조에서 계승할 부분

| 전임자 개념 | 보완 후 위치 | 계승 이유 |
|---|---|---|
| Event Receiver | 입력 어댑터·공통 이벤트 | 영상·센서 입력 경계가 필요함 |
| Platform Bridge | Public API·외부 연동 | 상위 플랫폼과의 양방향 연동 필요 |
| Device Manager | 장치 레지스트리·Device Resolver | 카메라·사이트·장치 매핑 필요 |
| Device Control Module | EdgeX Device Service | 제조사 프로토콜을 장치별로 격리 |
| Threading 병렬 제어 | MQTT·명령 큐·장치별 실행 | 한 장치 장애가 다른 장치를 막지 않게 함 |

## 현재 구조에서 보완할 부분

### 1. Action Layer 책임 분리

현재 `_ActionExecutor` 주변에 다음 책임이 함께 있다.

- 이벤트 위험도와 승인 정책
- 대상 장치 결정
- Direct 호출
- EdgeX 발행
- 결과 저장
- 상태 이벤트 발행

권장 분리:

```text
EventNormalizer
  → PolicyEngine
  → DeviceResolver
  → CommandBuilder
  → EdgeXCommandDispatcher
  → CommandResultCollector
```

이렇게 하면 EdgeX로 전환할 때 정책 코드는 유지하고 전송기만 교체할 수 있다.

### 2. 데이터 평면과 관리 평면 분리

실시간 제어는 외부 서버나 관제 화면에 의존하면 안 된다.

```text
데이터 평면: CCTV·센서 → AI → MQTT → 규칙 → Action → EdgeX → 장치
관리 평면: 설정·장치등록·모델배포·로그조회·승인·백업
```

관리 평면이 장애가 나도 데이터 평면은 마지막으로 승인된 설정으로 현장 제어를 계속해야 한다.

### 3. 이벤트와 명령의 의미 분리

AI 이벤트는 “위험 상황이 감지됨”을 뜻하고, Command는 “장치에 이 동작을 실행하라”를 뜻한다.
두 메시지를 하나의 payload로 섞지 않는다.

필수 상관관계 필드:

```text
event_id, message_id, request_id, command_id,
camera_id, site_id, zone_id, device_id, schema_version
```

### 4. 장치 제어 안전장치

실시간 경보는 오탐보다 놓침이 위험할 수 있지만, 반복 명령과 잘못된 장치 제어도 위험하다.

필수 정책:

- 이벤트별 cooldown
- 동일 `event_id` 중복 실행 차단
- 장치별 허용 command 목록
- 장치별 최대 실행 시간
- 장치별 retry 횟수와 backoff
- 미등록 `device_id` 차단
- 장치 장애 시 다른 장치 계속 실행
- 수동 중지 명령의 우선순위 보장
- 전원·밝기·음량 등 위험한 설정값 범위 검증

### 5. 규칙 엔진의 최종 판정 주체 확정

eKuiper와 Python `SensorRuleBridge`가 같은 조건을 최종 알람으로 발행하면 장치가 중복 동작할 수 있다.

권장 책임:

```text
Python: TLV 파싱·단위 변환·복잡한 물리 계산
eKuiper: threshold·window·집계·라우팅
Action Layer: 승인·cooldown·장치 명령 생성
```

같은 조건에서 최종 장치 명령은 Action Layer에서만 생성한다.

## 정부과제 관점의 보완 효과

| 평가 관점 | 보완 전 위험 | 보완 후 설명 |
|---|---|---|
| 확장성 | 장치 추가 시 플랫폼 코드 수정 | Device Service와 레지스트리 추가 |
| 실시간성 | 외부 서버·DB 의존 가능 | 로컬 데이터 평면 우선 |
| 상호운용성 | 제조사 API가 플랫폼에 결합 | EdgeX 명령과 Device Service로 분리 |
| 신뢰성 | 한 장치 오류가 전체 흐름에 영향 | 장치별 장애 격리 |
| 추적성 | 명령 결과 추적이 불명확 | event/request/command/result 상관관계 |
| 재현성 | 수동 시연 중심 | 테스트·로그·모델·환경 버전 고정 |
| 실증성 | 구현과 실제 장치 동작 혼동 | 구현·dry-run·실제 UAT 분리 |

## 단계적 적용 순서

1. 현재 계약과 결과 필드를 고정한다.
2. `DeviceResolver`와 `CommandBuilder`를 먼저 분리한다.
3. EdgeX 전광판 단독 경로를 운영 검증한다.
4. `CommandResultCollector`를 별도 책임으로 분리한다.
5. 스피커·사이렌은 장치 확보 전까지 dry-run으로 유지한다.
6. Action Layer의 기본 전송 모드를 EdgeX로 변경한다.
7. Direct 경로는 한 릴리스 동안 롤백용으로 유지한다.
8. 실제 2대 이상 장치에서 장애 격리와 다중 라우팅을 검증한다.
9. 안정화 후 Direct 호출 코드를 제거한다.

## 이번 단계에서 하지 않는 것

- AI 추론을 EdgeX 내부로 이동하지 않는다.
- 외부 클라우드 경로를 실시간 제어의 필수 경로로 만들지 않는다.
- 실제 장치가 없는 스피커·사이렌을 운영 완료로 표시하지 않는다.
- 전체 Action Layer를 검증 없이 한 번에 EdgeX 모드로 전환하지 않는다.
