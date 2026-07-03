# 모듈화 정리

코드 디렉터리, 서비스별 책임, 데이터 흐름, 구조 변경 계획을 모았습니다.

## 전체 구조

- [프로젝트 개요](PROJECT_OVERVIEW.md): 시스템 구성과 설계 의도
- [프로젝트 구조](PROJECT_STRUCTURE.md): 디렉터리와 주요 모듈의 역할
- [디렉터리 이동 계획](CODE_DIRECTORY_RELOCATION_PLAN.md): 단계별 구조 개선안
- [호환성 shim](COMPATIBILITY_SHIMS.md): 이전 경로와 인터페이스 호환 정책

## 서비스와 데이터 흐름

- [Device Service 아키텍처](DEVICE_SERVICE_ARCHITECTURE.md)
- [Action Layer와 스피커 브리지](ACTION_LAYER_SPEAKER_BRIDGE.md)
- [EdgeX·SQLite 데이터 아키텍처](EDGEX_SQLITE_DATA_ARCHITECTURE.md)
- [ASC 룰 엔진](ASC_RULE_ENGINE.md)
- [eKuiper 룰 엔진](KUIPER_RULE_ENGINE.md)

실제 실행 순서는 [실행 및 배포 문서](../guides/README.md)를 참고하세요.
