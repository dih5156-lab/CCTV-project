# Edge AIoT 플랫폼 성숙도 평가

## 1. 평가 결론

현재 플랫폼은 **현장 실증·통합 검증 단계(Production Candidate 이전 단계)**로
평가한다. 단순 데모나 단일 모델 POC는 넘어섰지만, 모든 출력 장치의 실제 운영
증거와 장기 안정성 증거가 부족하므로 상용 운영 완료로 분류하지 않는다.

평가 기준은 다음 5단계로 정의한다.

| 단계 | 의미 |
|---:|---|
| 1 | 개념 검증·단순 데모 |
| 2 | 기능별 POC |
| 3 | 서비스 통합 POC |
| 4 | 현장 실증·운영 후보 |
| 5 | 상용 운영·장기 검증 완료 |

현재 종합 평가는 **3.5~4단계**다. 플랫폼 소프트웨어와 자동 검증은 4단계에
가깝고, 물리 장치와 AI 성능의 현장 검증은 3단계 수준이다.

## 2. 영역별 평가

| 평가 영역 | 수준 | 확인된 근거 | 남은 조건 |
|---|---:|---|---|
| 이벤트·서비스 구조 | 4 | Core, Services, API, EdgeX, Devices 분리 | 장기 운영 중 결합도와 장애 전파 측정 |
| 실시간 AI 파이프라인 | 3~4 | DeepStream·MQTT·Action Layer 흐름 검증 | 현장 FPS·지연시간·프레임 드롭 기준 확정 |
| EdgeX 제어 경계 | 3 | 전광판 Core Command canary와 결과 상관관계 확인 | 전광판 운영 전환 및 다른 장치 실장치 UAT |
| 출력 장치 운영 | 3 | 전광판 online, 명령 출력 확인 | 스피커 연결 및 사이렌 장치 확보 |
| 장애 격리·복구 | 3~4 | 장치별 resolver, dry-run, Direct rollback, idempotency | 다중 장치 실제 장애·타임아웃 UAT |
| 배포 자동화 | 4 | Jetson Compose, readiness, health, smoke 검사 | 재기동·전원 장애 후 자동 복구 증거 |
| 테스트 체계 | 4 | 전체 pytest와 API·장치·EdgeX 단위 테스트 | 실제 장치와 장기 부하 테스트 확대 |
| 모델 운영 | 3 | manifest, shadow, 평가·재학습 절차 | 고정 세트와 현장 라벨 기반 정량 성능표 |
| 보안·감사 | 3 | 내부 토큰, request/event/device ID 추적 | 권한 분리, 비밀값 교체, 감사 로그 운영 검증, 내부 포트 host_ip 제한 |

## 3. 현재 확인된 운영 증거

- Jetson ARM64 배포 준비도 검사 전체 통과
- 오프라인 readiness 검사 통과
- Alert API, Action Layer, Public API health 및 readiness 통과
- 데이터 흐름 smoke 검사 통과
- 전광판 EdgeX 명령의 실제 수신·출력·결과 기록 확인
- `event_id`, `request_id`, `device_id` 상관관계 구조 확인
- 동일 `request_id`의 중복 물리 명령 방지 검증
- 물리 출력 없이 검증하는 `ACTION_DEVICE_DRY_RUN` 경로 확인
- 기본 Direct 복귀와 EdgeX canary용 Compose override 분리

## 4. 아직 수준을 올리지 못하는 이유

다음 항목은 코드가 없어서가 아니라 실제 증거가 부족해서 미완료로 판단한다.

1. 스피커는 설정은 있으나 현재 네트워크에 연결되지 않았다.
2. 사이렌은 현재 물리 장치가 없어 실제 명령 결과를 확인할 수 없다.
3. 전광판은 EdgeX canary는 통과했지만 기본 운영 경로의 영구 전환은 보류 중이다.
4. 두 개 이상의 실제 장치가 동시에 동작하는 장애 격리 시험이 남아 있다.
5. 낙상 모델의 현장 정밀도·재현율·오탐률·지연시간·GPU 사용량을 지속 수집해야 한다.
6. 24시간 이상 장기 운전, 전원 재기동, 네트워크 단절 후 복구 증거가 부족하다.
7. Jetson compose에서 Alert API, Action Layer, 일부 AI 디버그 포트의 `host_ip`
   제한 여부를 서비스별로 확정하고 방화벽 정책과 함께 검증해야 한다.

## 5. 다음 승급 조건

### 4단계 확정 조건

- 전광판 EdgeX 운영 라우팅 전환 후 rollback 기록 확보
- 다중 장치 synthetic 장애 격리 및 실제 전광판 장애 복구 시험 완료
- 고정 평가 세트와 현장 라벨 세트의 모델 성능표 작성
- 24시간 이상 Jetson 안정성 시험과 재기동 복구 결과 기록

### 5단계 승급 조건

- 스피커·사이렌을 포함한 모든 운영 대상 장치의 실장치 UAT 완료
- 장치·EdgeX·MQTT·AI Engine 장애의 자동 복구 또는 명확한 운영 대응 절차 확보
- 인증·비밀값 교체·감사 로그·접근권한 검증 완료
- 부하 상황의 FPS, end-to-end 지연시간, 메모리, GPU 사용량 기준 통과
- 정부과제 요구사항별 구현·시험·현장 증빙 연결 완료

## 6. 정부과제 보고서용 표현

현재 단계는 다음과 같이 표현하는 것이 사실과 가장 잘 맞는다.

> 본 시스템은 Jetson 기반 Edge AIoT 실시간 영상·센서 이벤트 처리와 출력 장치
> 제어 구조를 통합하고, 전광판을 대상으로 EdgeX 제어 경로를 실증한 현장
> 실증 후보 단계이다. 스피커·사이렌 및 장기 안정성 검증은 장치 확보와 추가
> 현장 시험 후 운영 단계로 승급한다.

관련 운영 증거는 다음 문서에서 관리한다.

- [EdgeX 운영 모드 전환 정책](EDGEX_OPERATING_MODE_POLICY.md)
- [Jetson 배포 준비도 검증 기록](../handover/DEPLOYMENT_READINESS_20260907.md)
- [EdgeX 실장치 UAT 기록](../handover/EDGEX_LIVE_UAT_20260904.md)
- [정부과제 요구사항 추적표](../handover/REQUIREMENT_TRACEABILITY.md)
