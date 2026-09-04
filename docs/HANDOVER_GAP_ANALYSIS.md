# 인수인계 문서 추가 작성 점검표

프로젝트 구조, 파이프라인, Docker, EdgeX, 센서, 장치 API, 이벤트 Payload, 모델별 인수인계에 이어 운영 인수인계 문서도 작성했다. 아래 문서는 실제 현장값을 채우고 리허설해야 완성된다.

## 우선순위 높음: 현장값·리허설 필요

### 1. 환경·장비 인벤토리

템플릿 작성 완료: [장비 인벤토리](handover/EQUIPMENT_INVENTORY_TEMPLATE.md). 실제 카메라 ID, 센서 DevEUI, 장치 IP, 설치 위치, 담당자를 채워야 한다.

### 2. 백업·복구 절차

절차 작성 완료: [백업·복구 런북](handover/BACKUP_RESTORE_RUNBOOK.md). 실제 백업 저장소·보존 기간·복구 리허설 결과를 채워야 한다.

### 3. 현장 수용 테스트(UAT) 문서

체크리스트 작성 완료: [현장 UAT 체크리스트](handover/FIELD_UAT_CHECKLIST.md). 실제 현장에서 PASS/FAIL 증거를 채워야 한다.

### 4. 장애 연락망·에스컬레이션

AI 모델, Jetson/Docker, 네트워크, EdgeX, 제조사 장치 담당자를 장애 유형별로 지정해야 한다. 장애 발생 시 로그·영상·시간·camera_id·event_id를 어떤 형식으로 전달할지도 정한다.

## 우선순위 중간: 운영 정책 확정 필요

### 5. 데이터 라벨링·재학습 운영 가이드

가이드 작성 완료: [데이터 라벨링·재학습](handover/DATA_LABELING_AND_RETRAINING.md). 실제 데이터 보관 위치와 승인 절차를 확정해야 한다.

### 6. 릴리스·모델 버전 관리 규칙

템플릿 작성 완료: [릴리스·모델 버전 관리](handover/RELEASE_AND_MODEL_VERSIONING.md). 팀의 릴리스 번호와 승인자를 지정해야 한다.

### 7. 보안·개인정보 처리 문서

가이드 작성 완료: [보안·개인정보 운영](handover/SECURITY_PRIVACY_OPERATIONS.md). 법무·보안 정책의 보존 기간과 책임자를 반영해야 한다.

### 8. 관측성 대시보드 설명서

가이드 작성 완료: [모니터링 지표 가이드](handover/OBSERVABILITY_GUIDE.md). 현장별 정상 범위와 알림 threshold를 정해야 한다.

## 현재 문서로 이미 커버되는 항목

- 프로젝트 구조·파이프라인: `docs/architecture/`
- Docker 실행·수정: `docs/architecture/DOCKER_AND_EXECUTION.md`
- EdgeX·센서: `docs/architecture/EDGEX_INTEGRATION.md`, `SENSOR_INTEGRATION.md`
- 장치 API·이벤트 Payload: `docs/devices/`
- 모델 History·평가·승격: `docs/models/MODEL_HANDOVER.md`
- 운영 점검·복구 기본 절차: `docs/guides/OPERATIONS_RUNBOOK.md`

## 권장 다음 순서

1. 실제 설치 장비 기준 인벤토리와 담당자 표 작성
2. DB/volume 백업·복구 리허설
3. 현장 UAT 시나리오와 PASS/FAIL 기준 확정
4. 개인정보·보안·장애 연락망을 회사 정책에 맞게 보완

운영 인수인계 문서 모음은 [handover/README.md](handover/README.md)에서 확인한다.
