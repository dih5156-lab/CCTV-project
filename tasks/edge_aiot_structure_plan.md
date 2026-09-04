# Edge AIoT 구조화 정리 계획

## 개요

현재 Action Layer에 함께 존재하는 경보 정책, 장치 선택, 직접 장치 호출, EdgeX Shadow 발행, 결과 저장 책임을 단계적으로 분리한다. 목표는 Jetson AI 추론의 실시간성을 유지하면서 EdgeX를 모든 현장 장치 제어의 표준 경계로 사용하는 것이다.

## 기준선

- 현재 핵심 경로는 `ActionBridge._on_message → _dispatch_command → _ActionExecutor.execute`다.
- `_ActionExecutor.execute`에서 직접 장치 호출과 EdgeX Shadow 발행이 함께 가능하다.
- 다중 장치 레지스트리, 장치별 Device Service, 명령 결과 저장 구조는 이미 추가되어 있다.
- 실제 운영 기본 경로는 아직 direct이며, 실제 장치 UAT와 Shadow 비교가 남아 있다.
- AI 추론은 Jetson 로컬 경로로 유지한다.

## 구현 원칙

- 계약을 먼저 고정하고 코드를 분리한다.
- 한 번에 한 장치 종류 또는 한 개의 수직 흐름만 전환한다.
- 미완성 경로는 기본 비활성으로 둔다.
- 각 단계마다 테스트·시뮬레이션·롤백 방법을 남긴다.
- 기존 직접 경로는 실제 장치 검증 전까지 삭제하지 않는다.

## 단계별 작업

### Phase 1: 현재 경로 기준선 고정

- [ ] 현재 direct/shadow/EdgeX 환경변수와 Docker 서비스 상태 기록
- [ ] 이벤트 수신부터 장치 결과 저장까지 대표 흐름 1개 추적
- [ ] 장치별 명령 topic·payload·결과 상태 기준 확정
- [ ] 기준선 테스트와 Compose 검증 결과 기록

### Phase 2: 실행 정책과 전송 방식 분리

- [ ] `ActionExecutor`에서 경보 정책과 전송 방식을 분리
- [ ] direct/shadow/edgex 실행 모드를 하나의 정책으로 통합
- [ ] 공통 명령 객체에 `event_id`, `request_id`, `device_id` 보장
- [ ] 정책 테스트와 전송 테스트를 별도 구성

### Phase 3: 장치별 EdgeX 수직 전환

- [ ] 스피커 direct/shadow 비교
- [ ] 스피커 EdgeX 실제 제어 및 결과 수집
- [ ] 사이렌 direct/shadow 비교
- [ ] 사이렌 EdgeX 실제 제어 및 결과 수집
- [ ] 전광판 direct/shadow 비교
- [ ] 전광판 EdgeX 실제 제어 및 결과 수집

### Phase 4: 다중 장치·장애 격리

- [ ] 같은 장치 종류 2대 이상 fan-out 검증
- [ ] 한 장치 timeout이 다른 장치에 영향을 주지 않는지 검증
- [ ] 미등록 `device_id` 차단 검증
- [ ] retry·timeout·outbox 동작 검증

### Phase 5: 운영 전환

- [ ] 사이트 단위로 EdgeX 모드 활성화
- [ ] 운영 지연시간·실패율·재시도율 모니터링
- [ ] direct 롤백 절차 실제 검증
- [ ] 검증 완료 후에만 direct 코드 제거 여부 결정

## 체크포인트

### Checkpoint 1: 기준선

- [ ] 현재 실제 운영 경로를 문서만 보고 재현할 수 있다.
- [ ] 대표 낙상 이벤트의 `event_id`부터 장치 결과까지 추적된다.

### Checkpoint 2: 장치 전환

- [ ] 장치 하나가 direct/shadow/edgex 세 모드에서 동일한 명령 계약을 사용한다.
- [ ] 실제 장치가 연결되지 않아도 시뮬레이션 테스트가 통과한다.

### Checkpoint 3: 운영 승인

- [ ] 다중 장치와 장애 격리 테스트가 통과한다.
- [ ] EdgeX 모드에서 현장 제어 지연시간이 허용 범위다.
- [ ] direct 롤백이 설정 변경만으로 가능하다.

## 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| direct와 EdgeX 중복 제어 | 이중 경보 | shadow 결과는 비교용으로만 처리하고 장치 실행 주체를 하나로 제한 |
| EdgeX 장애 | 제어 실패 | timeout·retry·outbox와 명확한 실패 상태 기록 |
| 장치 ID 매핑 오류 | 다른 장치 제어 | 등록 장치만 허용하고 카메라·사이트·구역 매핑 테스트 추가 |
| 구조 분리 중 회귀 | 기존 경보 중단 | 수직 슬라이스마다 관련 테스트와 direct fallback 유지 |

