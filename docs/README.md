# 문서 분류

이 디렉터리는 프로젝트 운영, API, AI 모델, EdgeX 연동, 아키텍처 문서를 보관한다.
새 문서를 추가할 때는 아래 분류를 기준으로 둔다.

## 운영 / 현장 점검

- [operations/OPERATION_CHECKLIST.md](operations/OPERATION_CHECKLIST.md) - 운영/데모 전 표준 점검 순서
- [operations/OPERATIONS_RUNBOOK.md](operations/OPERATIONS_RUNBOOK.md) - 운영 중 확인과 복구 절차
- [operations/JETSON_EDGEX_FIELD_CHECKLIST.md](operations/JETSON_EDGEX_FIELD_CHECKLIST.md) - Jetson/EdgeX 현장 점검 체크리스트
- [operations/DEEPSTREAM_PERFORMANCE_STABILITY_2026-05-26.md](operations/DEEPSTREAM_PERFORMANCE_STABILITY_2026-05-26.md) - DeepStream 성능/안정성 관찰 기록
- [operations/DEVICE_INTEGRATION_CHECK_2026-05-29.md](operations/DEVICE_INTEGRATION_CHECK_2026-05-29.md) - 디바이스 연동 점검 기록

## API / 이벤트 계약

- [api/PUBLIC_API_GUIDE.md](api/PUBLIC_API_GUIDE.md) - Public API 사용 가이드
- [api/PUBLIC_API_EXAMPLES.md](api/PUBLIC_API_EXAMPLES.md) - Public API 호출 예시
- [api/APPEARANCES_STATUS_API.md](api/APPEARANCES_STATUS_API.md) - 외형 검색 상태 API 계약
- [api/EVENT_SCHEMA_STANDARD.md](api/EVENT_SCHEMA_STANDARD.md) - 표준 이벤트 스키마
- [api/EXTERNAL_INGEST.md](api/EXTERNAL_INGEST.md) - 외부 이벤트 수신 구조

## AI 모델 / 비전 파이프라인

- [operations/MLOPS_MODEL_EVALUATION.md](operations/MLOPS_MODEL_EVALUATION.md) - 모델 평가 워크플로우
- [operations/FACE_RECOGNITION_SETUP.md](operations/FACE_RECOGNITION_SETUP.md) - 얼굴 인식 환경 구성
- [operations/PPHUMAN_ATTRIBUTE_INTEGRATION.md](operations/PPHUMAN_ATTRIBUTE_INTEGRATION.md) - PP-Human 외형 속성 분석 연동

## 아키텍처 / 룰 엔진 / 데이터 연동

- [architecture/DEVICE_SERVICE_ARCHITECTURE.md](architecture/DEVICE_SERVICE_ARCHITECTURE.md) - EdgeX 디바이스 서비스 구조
- [architecture/ASC_RULE_ENGINE.md](architecture/ASC_RULE_ENGINE.md) - ASC 룰 엔진 구조
- [architecture/KUIPER_RULE_ENGINE.md](architecture/KUIPER_RULE_ENGINE.md) - eKuiper 룰 엔진 연동
- [architecture/EDGEX_SQLITE_DATA_ARCHITECTURE.md](architecture/EDGEX_SQLITE_DATA_ARCHITECTURE.md) - EdgeX/SQLite 데이터 구조
- [architecture/ACTION_LAYER_SPEAKER_BRIDGE.md](architecture/ACTION_LAYER_SPEAKER_BRIDGE.md) - Action Layer와 스피커 브리지

## 프로젝트 구조 / 아키텍처

- [architecture/PROJECT_OVERVIEW.md](architecture/PROJECT_OVERVIEW.md) - 프로젝트 개요
- [architecture/PROJECT_STRUCTURE.md](architecture/PROJECT_STRUCTURE.md) - 디렉터리와 모듈 구조
- [architecture/CODE_DIRECTORY_RELOCATION_PLAN.md](architecture/CODE_DIRECTORY_RELOCATION_PLAN.md) - 코드 디렉터리 이동 초안
- [architecture/COMPATIBILITY_SHIMS.md](architecture/COMPATIBILITY_SHIMS.md) - 호환성 shim 정리

## 도구 / 에이전트 자료

- [tooling/CODEX_MULTI_AGENT_SETUP.md](tooling/CODEX_MULTI_AGENT_SETUP.md) - Codex 멀티 에이전트 설정
- 프로젝트 필수 도구: `.codex/skills/`, `.github/agents/`
- 참고 자료: `tooling/agents/` (원본 agent 문서)

## 리뷰 / 보고서 / 작업 가이드

- [reports/CODE_REVIEW_REPORT.md](reports/CODE_REVIEW_REPORT.md) - 코드 리뷰 보고서
- [reports/PROJECT_REVIEW_2026-04.md](reports/PROJECT_REVIEW_2026-04.md) - 2026년 4월 프로젝트 리뷰
- [reports/PROJECT_REVIEW_2026-06.md](reports/PROJECT_REVIEW_2026-06.md) - 2026년 6월 프로젝트 리뷰
- [reports/TECHNICAL_WORK_REPORT_GUIDE.md](reports/TECHNICAL_WORK_REPORT_GUIDE.md) - 기술 업무 보고서 작성 가이드

## 루트 문서

- [../README.md](../README.md) - 프로젝트 진입 문서
- [../COMMANDS.md](../COMMANDS.md) - 자주 쓰는 명령어 모음

## 정리 원칙

- 운영 중 계속 생성되는 로그와 결과물은 `reports/`에 둔다.
- 팀에 공유하거나 git에 남길 문서는 `docs/`에 둔다.
- 날짜가 붙은 점검 기록은 운영 분류에 두고, 장기적으로 자주 참조되는 내용은 체크리스트나 runbook에 합친다.
- 파일 이동은 README와 문서 간 링크를 함께 수정할 수 있을 때 진행한다.
