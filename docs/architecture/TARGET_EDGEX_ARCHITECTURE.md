# 목표 EdgeX 중심 아키텍처

## 목표

Edge AIoT 현장에서 AI 이벤트를 빠르게 생성하고, 모든 출력 장치 제어는 EdgeX Device Service를 통해 일관되게 처리한다. EdgeX는 장치 제어·장치 상태·프로토콜 차이를 흡수하고, AI 엔진은 실시간 추론에 집중한다.

## 목표 흐름

```text
카메라·센서 입력
        ↓
Jetson AI 엔진
  ├─ 사람·헬멧·낙상·외형 분석
  └─ 이벤트 정규화
        ↓
공통 MQTT 이벤트
        ↓
Action Layer
  ├─ 위험도·중복·승인 정책
  ├─ camera_id → site/zone → device_id 매핑
  └─ 공통 Command 생성
        ↓
EdgeX Core Command
        ↓
장치별 Device Service
  ├─ Speaker Device Service
  ├─ Siren Device Service
  └─ DABIT Signboard Device Service
        ↓
현장 장치 제어
        ↓
명령 결과·장치 상태 수집
        ↓
Action 이력·Public API·모니터링
```

## 책임 분리

| 계층 | 책임 | 하지 않는 일 |
|---|---|---|
| AI 엔진 | 영상/센서 분석, 이벤트 생성, 실시간 판정 | 장치별 프로토콜 직접 처리 |
| MQTT 이벤트 계층 | 이벤트 전달, 재연결, outbox | 장치 제어 정책 결정 |
| Action Layer | 위험도, 중복 억제, 승인, 대상 장치 결정 | 장치 프로토콜 세부 구현 |
| EdgeX Core Command | 표준 명령 라우팅 | AI 판정 및 경보 우선순위 결정 |
| Device Service | 장치 프로토콜 변환, 장치별 timeout/retry | 다른 장치의 정책 처리 |
| 결과 수집 계층 | 요청·수락·완료·실패 기록 | 명령을 다시 임의로 실행 |
| Public API/모니터링 | 조회, 감사, 운영 상태 표시 | 실시간 제어 경로 대체 |

## 전환 모드

### 1. direct

기존 직접 장치 제어만 실행한다. 실제 운영 중단을 막기 위한 기본 롤백 모드다.

### 2. shadow

직접 제어를 실행하면서 같은 명령을 EdgeX로도 기록·전송한다. EdgeX 결과는 비교용으로만 사용하고 실제 경보 성공 여부를 대체하지 않는다.

### 3. edgex

EdgeX Command와 Device Service를 실제 제어 경로로 사용한다. 장치 결과가 완료 또는 허용된 실패 상태로 추적되어야 한다.

## 전환 완료 조건

- 모든 명령에 `event_id`, `request_id`, `device_id`가 존재한다.
- 장치별 timeout과 retry 정책이 Device Service에 정의되어 있다.
- 한 장치 장애가 다른 장치의 명령 실행을 막지 않는다.
- EdgeX 장애 시 안전한 fallback 또는 명확한 실패 상태가 기록된다.
- direct/shadow/edgex 모드를 환경변수로 되돌릴 수 있다.
- 실제 장치 UAT에서 명령 지연시간과 결과 상태를 확인한다.

## 실시간성 기준

EdgeX를 사용하더라도 AI 추론을 EdgeX 안으로 옮기지 않는다. 카메라 프레임 처리와 낙상 판단은 Jetson에서 수행하고, EdgeX는 판단 이후 장치 제어 경계로 사용한다. 따라서 측정해야 할 시간은 다음처럼 나눈다.

```text
프레임 입력 → AI 판정
AI 판정 → Action 승인
Action 승인 → EdgeX Command 수락
Command 수락 → 장치 동작 완료
```

