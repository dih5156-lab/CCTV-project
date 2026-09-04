# EdgeX 중심 전환 현황

기준일: 2026-09-04

## 결론

현재 프로젝트는 **EdgeX 전환을 위한 코드·인프라 준비가 약 70% 완료**된 상태다.

다만 운영 환경의 실제 이벤트 제어 경로는 아직 기존 direct 경로다. 현재 `.env.jetson`에서
`EDGEX_SHADOW_ENABLED`와 `EDGEX_DEVICE_REGISTRY_PATH`가 활성화되지 않았기 때문이다.

따라서 다음 두 수치는 구분해야 한다.

| 구분 | 현재 상태 |
|---|---|
| EdgeX 전환 준비도 | 약 70% |
| 실제 운영 이벤트의 EdgeX 제어 비율 | 0%에 가까움 |
| EdgeX Device Service 실행 | 완료 |
| EdgeX Metadata 장치 등록 | 완료 |
| 다중 장치 라우팅 코드 | 완료 |
| Shadow 비교 운영 | 미활성화 |
| 실제 장치 현장 UAT | 미완료 |

## 완료된 범위

1. 스피커·경광등·전광판용 EdgeX Device Service 구현
2. Core Command HTTP 호환 경계 구현
3. MQTT Command 및 결과 토픽 구현
4. 결과 SQLite 저장 및 조회 API 구현
5. EdgeX Metadata에 출력 장치 등록
6. `device_id` 기반 다중 장치 레지스트리 구현
7. 카메라별 장치 매핑 및 Shadow fan-out 구현
8. 러너의 장치별 클라이언트 풀 생성 구현
9. 등록되지 않은 장치 요청 차단
10. 다중 장치 테스트 및 기존 전체 테스트 통과

검증 결과:

- 전체 테스트: `1659 passed, 0 failed, 72 skipped`
- 세 러너의 2대 장치 풀 생성 테스트: 통과
- Device Service 컨테이너 3종: `healthy`
- 실제 물리 장치 명령: 실행하지 않음

## 아직 direct 경로인 이유

현재 Action Layer는 다음 순서로 동작한다.

```text
AI 이벤트
  -> Action Layer
  -> 기존 스피커·전광판·경광등 direct 호출
  -> EdgeX Shadow Command는 비활성
```

EdgeX 중심 전환 후 목표 경로는 다음과 같다.

```text
AI 이벤트
  -> Action Layer
  -> EdgeX Command
  -> EdgeX Core Command
  -> Device Service
  -> 물리 장치
```

현재는 물리 장치가 연결되지 않은 상태에서 자동으로 전환하면 현장 제어 실패나 중복 동작이 발생할 수 있으므로 운영 플래그를 켜지 않았다.

## 다음 전환 순서

1. 물리 장치 없이 Shadow 활성화 후 Command 발행·결과 수집 확인
2. 장치별 결과와 기존 direct 결과 비교
3. 한 장치만 EdgeX 경로로 전환하는 부분 전환
4. 실패 시 direct 경로로 되돌리는 롤백 확인
5. 장치 2대 이상으로 장애 격리 UAT
6. 모든 장치의 EdgeX 경로 전환

전환 시작 전 다음 설정을 운영 환경에 추가한다.

```env
EDGEX_SHADOW_ENABLED=true
EDGEX_DEVICE_REGISTRY_PATH=/app/config/output_devices.json
```

단, 실제 장치에 명령이 전달될 수 있으므로 위 설정은 장치 연결 상태와
`SPEAKER_DRY_RUN`, `SIREN_DRY_RUN`, `SIGNBOARD_DRY_RUN` 값을 확인한 뒤 적용한다.
