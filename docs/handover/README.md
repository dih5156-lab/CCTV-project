# 운영 인수인계 문서

프로젝트를 실제 담당자에게 넘길 때 사용하는 운영 문서 모음이다. `<현장 입력 필요>` 표시는 회사·현장 정보를 전달받은 뒤 채운다.

최종 진행 상태는 [최종 인수인계 진행 현황](FINAL_HANDOVER_STATUS.md)에서 확인한다.

전임자 자료의 원본은 삭제하지 않고 [전임자 자료 분류 기준](LEGACY_MATERIAL_CLASSIFICATION.md)에 따라 현재 기준·정부과제 증빙·참고·레거시로 구분한다. 정부과제 요구사항과 구현·시험·실증의 연결은 [정부과제 요구사항 추적표](REQUIREMENT_TRACEABILITY.md)를 사용한다.

| 문서 | 용도 |
|---|---|
| [장비 인벤토리](EQUIPMENT_INVENTORY_TEMPLATE.md) | 카메라·센서·출력 장치·담당자 관리 |
| [백업·복구 런북](BACKUP_RESTORE_RUNBOOK.md) | DB·볼륨 백업과 장애 복구 |
| [현장 UAT 체크리스트](FIELD_UAT_CHECKLIST.md) | 배포 전후 PASS/FAIL 검증 |
| [장애 대응·에스컬레이션](INCIDENT_ESCALATION.md) | 장애 분류와 보고 절차 |
| [데이터 라벨링·재학습](DATA_LABELING_AND_RETRAINING.md) | 데이터 수집부터 모델 승격까지 |
| [릴리스·모델 버전 관리](RELEASE_AND_MODEL_VERSIONING.md) | 코드·모델·환경 버전 기록 |
| [보안·개인정보 운영](SECURITY_PRIVACY_OPERATIONS.md) | 비밀값·영상·얼굴 데이터 관리 |
| [모니터링 지표 가이드](OBSERVABILITY_GUIDE.md) | Prometheus/Grafana 정상 범위 확인 |
| [정부과제 요구사항 추적표](REQUIREMENT_TRACEABILITY.md) | 요구사항·구현·시험·실증 연결 |
| [정부과제 증빙 등록 양식](EVIDENCE_REGISTER_TEMPLATE.md) | 로그·영상·스크린샷·모델 결과 등록 |
| [정부과제 증빙 초기 등록 목록](EVIDENCE_REGISTER.md) | 현재 저장소에서 확인된 증빙 목록 |
| [Edge AIoT 아키텍처 보완안](../architecture/ARCHITECTURE_IMPROVEMENT_PROPOSAL.md) | 전임자 구조와 현재 구조의 보완 방향 |
| [명령 식별자 상관관계 계약](../architecture/COMMAND_CORRELATION_CONTRACT.md) | event·command·request ID 추적 규칙 |
| [전광판 EdgeX 실장치 UAT](EDGEX_LIVE_UAT_20260904.md) | 실제 MQTT 명령·물리 출력·결과 DB 검증 |
