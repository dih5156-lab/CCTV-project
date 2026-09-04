# 최종 인수인계 진행 현황

## 1. 현재까지 완료된 문서

| 영역 | 문서 | 상태 |
|---|---|---|
| 프로젝트 구조 | `docs/architecture/` | 문서화 완료 |
| EdgeX·센서 | `EDGEX_INTEGRATION.md`, `SENSOR_INTEGRATION.md` | 문서화 완료 |
| Docker 운영 | `DOCKER_AND_EXECUTION.md` | 문서화 완료 |
| 모델 | `docs/models/MODEL_HANDOVER.md` | 모델별 정리 완료 |
| 장치 API | `docs/devices/` | API·이벤트 Payload 정리 완료 |
| 백업·복구 | `BACKUP_RESTORE_RUNBOOK.md` | 절차 완료, 현장 리허설 필요 |
| 현장 검증 | `FIELD_UAT_CHECKLIST.md` | 시나리오 완료, 결과 입력 필요 |
| 장애 대응 | `INCIDENT_ESCALATION.md` | 절차 완료, 연락망 입력 필요 |
| 보안·개인정보 | `SECURITY_PRIVACY_OPERATIONS.md` | 기본 원칙 완료, 회사 정책 반영 필요 |

전체 문서 시작점은 [문서 인덱스](../README.md)다.

## 2. 확인된 검증 증거

| 검증 | 결과 |
|---|---|
| Markdown 변경 공백 검사 | 통과 (`git diff --check`) |
| Jetson Compose 렌더링 | 통과 (`docker compose ... config --quiet`) |
| 과거 낙상 영상 replay | `fall_detected` 이벤트 확인 |

### 낙상 replay 기록

| 항목 | 값 |
|---|---|
| 입력 | `data/fall_demo/20260902_142824/overlay.mp4` |
| 결과 | `data/fall_eval/sample_deepstream_results.jsonl` |
| 방식 | DeepStream replay + runtime event 확인 |
| 결과 | TP 1, FN 0, 최대 score 6.0, 최대 probability 약 0.784 |
| 주의 | 단일 과거 영상 검증이며 현장 전체 정확도 인증 아님 |

이 결과는 해당 영상에서 현재 파이프라인이 이벤트를 생성했다는 증거다. 모든 카메라·조명·거리 조건의 모델 정확도로 확대 해석하지 않는다.

## 3. 인수인계 당일 순서

### 시작 전

- [ ] 전체 인수인계서와 이 문서를 인수자에게 공유
- [ ] Git commit, Compose, 모델 manifest의 기준 버전 기록
- [ ] 장비 인벤토리의 `<현장 확인>` 항목 수집
- [ ] 운영 secret은 문서가 아닌 안전한 방식으로 전달

### 함께 실행

- [ ] Compose `config --quiet`
- [ ] 서비스 `ps`, health, readiness
- [ ] 카메라 영상 수신
- [ ] AI 이벤트 생성
- [ ] 센서 정상·위험 payload 처리
- [ ] EdgeX Reading 조회
- [ ] 스피커·전광판·사이렌 테스트
- [ ] Action Layer 자동/수동 모드 확인
- [ ] 이벤트·명령 이력 조회

### 종료 전

- [ ] 장애 상황 1개 재현 및 복구
- [ ] 백업 위치와 복구 담당자 확인
- [ ] 미해결 이슈·임시 설정·알려진 한계 기록
- [ ] 인수자 질문과 답변을 회의록 또는 이슈에 기록
- [ ] 인수자·인계자·책임자의 승인 기록

## 4. 미완료 항목

| 항목 | 담당자 | 목표일 | 상태 | 증거 위치 |
|---|---|---|---|---|
| 실제 장비 IP·담당자 입력 | `<입력>` | `<입력>` | 미착수 | 인벤토리 |
| DB/volume 복구 리허설 | `<입력>` | `<입력>` | 미착수 | 백업 런북 |
| 현장 UAT 전체 실행 | `<입력>` | `<입력>` | 미착수 | UAT 체크리스트 |
| 장애 연락망·SLA 입력 | `<입력>` | `<입력>` | 미착수 | 장애 대응 |
| 개인정보 보존 기간 확정 | `<입력>` | `<입력>` | 미착수 | 보안 문서 |
| Grafana 알림 기준 확정 | `<입력>` | `<입력>` | 미착수 | 모니터링 가이드 |

## 5. 승인

| 역할 | 이름 | 일시 | 서명/승인 |
|---|---|---|---|
| 인계자 | `<입력>` | `<입력>` | `<입력>` |
| 인수자 | `<입력>` | `<입력>` | `<입력>` |
| 운영 책임자 | `<입력>` | `<입력>` | `<입력>` |

