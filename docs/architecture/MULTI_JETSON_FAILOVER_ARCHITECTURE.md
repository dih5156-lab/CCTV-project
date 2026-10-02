# 멀티 Jetson 운영 및 중앙 스케줄러 아키텍처

작성일: 2026-09-09  
상태: 설계 문서(현재 업무 완료 후 착수)

## 1. 목적

스마트폴 A, B, C에 Jetson 보드가 각각 설치된 환경에서 한 보드가 고장 나도 관제 서버가 장애를 감지하고, 해당 스마트폴의 **이벤트 처리 업무**를 정상 보드로 임시 전환하는 구조를 정의한다.

이 문서는 지금 즉시 중앙 스케줄러를 구현하기 위한 작업 지시서가 아니다. 현재 진행 중인 모델 학습과 낙상 인식 검증을 마친 뒤, 운영 안정성을 확보하면서 단계적으로 도입하기 위한 기준 문서다.

## 2. 현재 상태와 작업 순서

### 현재 운영 구조

- Jetson이 카메라 입력, YOLO/pose 추론, 낙상 판정, 로컬 이벤트 저장을 수행한다.
- 장치 제어는 현재 direct 모드가 기본이며, Action Layer가 장비 프로토콜을 직접 호출한다.
- EdgeX Reading 투영과 외부 export 경로는 준비되어 있다.
- 중앙 스케줄러, Jetson 등록부, heartbeat 기반 failover, 작업 큐는 아직 구현되어 있지 않다.
- 센서 원시 시계열은 PostgreSQL, 외형/액션/outbox 계열은 SQLite를 사용한다.

### 현재 먼저 끝낼 업무

1. pose 보강 학습과 feature extraction/RF/temporal 학습 완료 확인
2. 실제 낙상 영상 replay 및 pose 진단 로그 확인
3. API 인수인계 문서와 배포 readiness 점검
4. 장치 제어 경로와 인증 설정 점검
5. 위 결과를 기준으로 단일 Jetson의 정확도·지연·메모리 기준선 확정

중앙 스케줄러 설계는 위 기준선이 확보된 후 착수한다. 그래야 보드의 실제 처리 용량과 장애 판정 시간을 근거로 정할 수 있다.

## 3. 목표 구조

중앙 Windows PC 또는 관제 서버에는 GPU가 필요하지 않다. 중앙 서버는 Jetson이 보낸 상태를 관리하고 작업을 배정하는 CPU 중심 control plane이다. 실제 영상 추론은 계속 각 Jetson에서 수행한다.

구성 요소:

- **Jetson Edge Worker**: 카메라 ingest, YOLO/pose/fall 추론, 로컬 SQLite outbox, 장치 제어
- **Heartbeat/Registry**: 보드 ID, 스마트폴 ID, 카메라 목록, 모델 버전, 마지막 heartbeat, 현재 부하 기록
- **Task Router/Scheduler**: 장애 보드의 업무를 후보 보드에 배정하고 lease를 관리
- **Queue/Result Store**: 이벤트와 재처리 작업을 저장하며 중복 처리를 방지
- **관제 API/UI**: 보드 상태, 장애 전환, 미처리 이벤트, 수동 복귀를 표시

기본 통신은 MQTT 또는 내부 HTTPS API를 사용한다. 원본 RTSP 영상 전체를 중앙으로 보내는 방식은 네트워크와 CPU/GPU 비용이 커서 1단계 범위에서 제외한다.

## 4. B 보드 장애 시나리오

1. B가 일정 시간 heartbeat를 보내지 않는다.
2. Registry가 B를 `SUSPECT`로 표시하고 짧은 grace period를 둔다.
3. 연속 heartbeat 누락과 health check 실패가 확인되면 B를 `OFFLINE`으로 전환한다.
4. Scheduler가 B의 미완료 **이벤트/센서 처리 lease**를 회수한다.
5. A와 C 중 정책상 허용된 보드만 후보로 남긴다.
6. 후보의 queue depth, GPU 사용률, 메모리, 현재 카메라 수를 이용해 capacity score를 계산한다.
7. score가 가장 높은 보드에 작업을 배정한다. 동률이면 고정 우선순위 또는 라운드로빈을 사용한다.
8. 작업에는 `task_id`, `source_device_id`, `event_time`, `model_version`, `lease_expiry`를 포함한다.
9. 결과 저장 시 `task_id`를 unique key로 사용해 A와 C가 동시에 처리해도 중복 이벤트를 만들지 않는다.
10. B가 복구되면 자동 복귀하지 않고, health 안정화와 수동 승인 후 원래 배치로 되돌린다.

### 처리 범위

1단계 failover는 원본 영상 스트림이 아니라 다음 데이터를 대상으로 한다.

- 낙상/출입/센서 이벤트
- 이벤트 시각 전후의 짧은 snapshot 또는 clip 요청
- 미전송 outbox와 재처리 작업
- 장치 제어 요청의 상태 추적

원본 RTSP를 A/C가 대신 받아야 하는 경우에는 카메라 접근 권한, 네트워크 경로, 스트림 중복 수신, 지연을 별도로 검증한 뒤 2단계로 확장한다.

## 5. 상태와 배정 정책

### 보드 상태

`BOOTING → READY → BUSY → SUSPECT → OFFLINE → RECOVERING → READY`

- `READY`: 작업 수신 가능
- `BUSY`: 처리 중이지만 capacity threshold 이내
- `SUSPECT`: heartbeat 지연 또는 health check 실패가 시작됨
- `OFFLINE`: 장애 확정, 새 작업 배정 금지
- `RECOVERING`: 복귀 검증 중, 자동 배정 금지

### capacity score 예시

초기에는 설명 가능한 고정식으로 시작한다.

```text
score = 0.35 * (1 - gpu_utilization)
      + 0.25 * (1 - memory_utilization)
      + 0.25 * (1 - queue_utilization)
      + 0.15 * (1 - camera_load)
```

실제 가중치는 단일 Jetson 기준선 측정 후 조정한다. score가 낮은 보드로 작업을 보내지 않도록 최소 여유 threshold와 최대 동시 작업 수를 둔다.

## 6. 단계별 도입 계획

### Phase 0 — 현재 업무 마무리

- 학습 산출물과 pose replay 결과 확인
- pose retry/diagnostics 로그 기준 정리
- API 및 배포 문서 인수인계 완료
- 단일 보드의 FPS, 지연, GPU/메모리 사용량 측정

### Phase 1 — 관찰만 하는 control plane

- 고유한 `device_id`와 `camera_id` 규칙 확정
- heartbeat 수집과 Registry 구축
- 관제 화면에 READY/BUSY/SUSPECT/OFFLINE 표시
- 실제 작업 배정은 하지 않고 로그로만 검증

### Phase 2 — 이벤트 failover

- MQTT/API 기반 task lease와 queue 도입
- B 장애 시 이벤트와 outbox 재처리만 A/C로 전환
- idempotency key와 재시도 횟수 적용
- 장애 주입 테스트로 중복/유실 여부 검증

### Phase 3 — 용량 기반 배정과 운영 자동화

- GPU·메모리·queue·카메라 부하 기반 score 적용
- 복구 승인, drain, 재배치 runbook 제공
- 인증서/토큰, 감사 로그, 알림 연동

### Phase 4 — 선택적 영상 failover

- 카메라 이중 접근과 RTSP 재연결 검증
- snapshot/clip 우선 전환 후 필요한 카메라만 live stream 전환
- 네트워크 대역폭과 end-to-end 지연 기준을 통과한 경우에만 운영 적용

## 7. 안전장치와 운영 기준

- heartbeat timeout, grace period, lease timeout은 설정값으로 관리한다.
- 중앙 서버가 내려가도 Jetson은 로컬 추론과 SQLite outbox를 계속 수행한다.
- 중앙 서버는 장치 제어를 직접 수행하지 않고, 인증된 Action Layer 경계를 유지한다.
- 모든 배정/회수/복귀는 `task_id`, `device_id`, `reason`, `model_version`과 함께 감사 로그를 남긴다.
- 모델 버전이 다른 보드에는 호환되지 않는 작업을 배정하지 않는다.
- 장애 전환 성공률, 이벤트 유실률, 중복률, failover 시간, 복구 시간을 운영 지표로 수집한다.

## 8. 착수 전 결정 목록

- 중앙 서버 위치: Windows PC(개발/POC) 또는 상시 운영 관제 서버
- 메시지 경로: 기존 MQTT 우선 여부와 HTTPS fallback
- 작업 단위: 이벤트/outbox부터 시작할지, snapshot/clip까지 포함할지
- heartbeat 주기와 장애 확정 시간
- 보드별 카메라 소유권과 failover 허용 범위
- 자동 복귀 허용 여부와 운영자 승인 절차

## 9. 완료 기준

중앙 스케줄러 구현을 시작해도 되는 시점은 다음을 모두 충족한 때다.

- 단일 Jetson에서 pose 낙상 검출의 재현 가능한 평가 결과가 있다.
- 학습 산출물 경로와 모델 버전 기록 방식이 고정되어 있다.
- API/인증/배포 설정을 인수인계 문서에서 확인할 수 있다.
- heartbeat와 failover 장애 주입 테스트의 기대 결과가 합의되어 있다.
- 이벤트 중복 방지와 로컬 outbox 보존 정책이 정해져 있다.

관련 문서: [API_HANDOVER.md](../API_HANDOVER.md), [tasks/todo.md](../../tasks/todo.md)
