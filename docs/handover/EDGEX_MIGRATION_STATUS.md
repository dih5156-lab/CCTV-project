# EdgeX 중심 전환 현황

기준일: 2026-09-07 — 아래 본문은 당시 전환 기록이다.

> **2026-09-18 최신 상태:** 실행 중인 Action Layer는 `EDGEX_COMMAND_MODE=edgex`,
> 허용 장치는 `speaker,signboard,siren`이다. 따라서 아래의 “현재 Direct / 0%”는
> 현재 상태로 사용하지 않는다. 전광판은 09-17 로그·결과 DB UAT가 있고, 스피커는
> 보유·실물 재검증 필요, 사이렌은 미보유다. 최신 근거와 남은 작업은
> [기존 성과·이슈](../planning/STATUS.md)를 따른다. 현재 실제 EdgeX 처리 비율은
> 측정하지 않았으며 설정만으로 100%라고 표시하지 않는다.

## 결론

장치별 timeout·재시도 기준은 [ADR-003](../decisions/ADR-003-edge-device-timeout-retry-policy.md)에
고정했다. 현재 기준은 멱등성이 확인된 전광판만 내부 재시도를 허용하고,
스피커·사이렌은 중복 동작 방지를 위해 자동 재시도하지 않는 것이다.

현재 프로젝트는 EdgeX 전환에 필요한 코드·인프라와 전광판 단일 장치
canary 검증까지 완료된 상태다. 다만 운영 기본 모드는 Direct로 유지한다.

다만 운영 환경의 실제 이벤트 제어 경로는 아직 기존 direct 경로다. 현재 `.env.jetson`에서
`EDGEX_SHADOW_ENABLED`와 `EDGEX_DEVICE_REGISTRY_PATH`가 활성화되지 않았기 때문이다.

따라서 다음 두 수치는 구분해야 한다.

| 구분 | 현재 상태 |
|---|---|
| EdgeX 전환 준비도 | 구현·자동 검증·전광판 canary 완료 |
| 실제 운영 이벤트의 EdgeX 제어 비율 | 0% (운영 기본값은 Direct) |
| EdgeX Device Service 실행 | 완료 |
| EdgeX Metadata 장치 등록 | 완료 |
| 다중 장치 라우팅 코드 | 완료 |
| Shadow 비교 운영 | 미활성화 |
| 전광판 실제 장치 canary UAT | 통과 |
| 스피커·사이렌 실제 장치 UAT | 미검증 |
| EdgeX timeout·재시도 운영 정책 | ADR-003으로 장치별 기준 확정, 실제 스피커·사이렌 UAT 후 재평가 |

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
10. 다중 장치 테스트 및 전체 테스트 통과
11. `sample_eval → cctv-signboard-01` EdgeX canary 실장치 검증
12. AIoT parser EdgeX 리소스 매핑 및 Device Contract 검증
13. 세 Device Service의 `request_id` 중복 실행 방지 캐시 구현
14. 멱등성 코드가 포함된 세 Device Service 이미지 빌드 및 컨테이너 반영
15. 세 Device Service 동일 request_id 재전송 런타임 UAT

검증 결과:

- 전체 테스트: `1699 passed, 0 failed, 72 skipped`
- 세 러너의 2대 장치 풀 생성 테스트: 통과
- Device Service 컨테이너 3종: `healthy`
- 전광판 실제 물리 장치 명령: `acknowledged`
- EdgeX Device Contract: `issues=0`

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

현재 운영 기본값은 Direct다. 전광판은 EdgeX canary를 통과했지만 스피커와
사이렌은 실제 장치 UAT가 끝나지 않았고, 결과 timeout·재시도 정책도
확정되지 않았으므로 전체 운영 플래그를 켜지 않았다.

## 다음 전환 순서

1. 스피커·사이렌 연결 후 장치별 물리 UAT
2. 결과 timeout·재시도 정책 확정 및 장애 UAT
3. 전광판 canary 결과를 운영 증빙으로 승인
4. 장치 2대 이상으로 장애 격리 UAT
5. 사이트 또는 장치 단위 부분 전환
6. 모든 장치의 EdgeX 경로 전환

전환 시작 전 다음 설정을 운영 환경에 추가한다.

```env
EDGEX_COMMAND_MODE=shadow
EDGEX_DEVICE_REGISTRY_PATH=/app/config/output_devices.json
```

단, 실제 장치에 명령이 전달될 수 있으므로 위 설정은 장치 연결 상태와
`SPEAKER_DRY_RUN`, `SIREN_DRY_RUN`, `SIGNBOARD_DRY_RUN` 값을 확인한 뒤 적용한다.
