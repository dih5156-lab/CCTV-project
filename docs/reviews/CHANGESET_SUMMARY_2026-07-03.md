# 변경 및 검증 요약 - 2026-07-03

## 결론

이번 변경은 기존 기능을 크게 바꾸는 리팩터링보다, 운영 경로의 책임 분리와 검증 가능성 강화에 초점을 맞춥니다. 주요 범위는 DeepStream 소스 상태 관리, Action Bridge 모듈 분리, 낙상 보조 모델 운영 도구, 외형 분석·스트림 안정화, 문서 재분류입니다.

낙상 보조 모델의 `confirm` 전환과 Jetson 장시간 안정성은 현장 데이터와 실기 검증이 필요한 항목입니다. 코드 테스트 통과만으로 운영 검증이 완료된 것으로 보지 않습니다.

## 변경 범위

| 영역 | 핵심 변경 | 운영 영향 |
|---|---|---|
| DeepStream | 환경, 모델 플래그, 이벤트 큐, 소스 상태, topology helper 분리 | 기존 진입점은 유지하며 카메라 장애 진단 단위를 세분화 |
| Action Bridge | 알람, 실행기, 저장소, REST queue, site registry, topic 책임 분리 | 외부 import 호환 경로를 유지하고 장비별 장애 격리 범위를 명확화 |
| 낙상 보조 검증 | manifest 생성, RF 학습·비교, readiness·promotion 점검, shadow clip 라벨링 도구 추가 | 기본 낙상 판단을 대체하지 않으며 `shadow` 검증 후 `confirm` 전환 필요 |
| 외형·스트림 | 색상 audit, appearance 저장·검색, Stream API 인증·프레임 처리 보강 | 현장 조명과 네트워크 조건에서 별도 확인 필요 |
| 배포 | Compose 전제 조건 및 런타임 비밀값 점검 강화 | `.env`와 `.env.jetson`의 역할을 분리하고 Jetson 전용 Compose 사용 |
| 문서 | 기능·모듈·실행/배포·리뷰·도구 문서로 재분류 | 문서 진입점을 `docs/README.md`로 통일 |

## 기능 상태 구분

### 현재 기본 경로

- OpenCV/YOLO 기반 PC 개발 실행
- Jetson DeepStream/TensorRT 실행
- MQTT 이벤트 발행과 Public API/Action Layer 전달
- 외형 분석 로그와 Stream API
- 카메라 소스 상태 및 재연결 관리

### 환경에 따라 선택하는 경로

- PP-Human/Paddle 또는 PA100K SGIE 외형 분석
- 얼굴 인식 백엔드
- EdgeX/eKuiper/ASC 연동
- falldata RF 보조 검증

### 운영 전 추가 근거가 필요한 항목

- falldata `confirm` 모드: 카메라별 shadow 라벨링과 false positive/negative 기준 필요
- 24시간 DeepStream 안정성: Jetson 실기에서 프레임 드롭, 재연결, GPU 메모리·온도 확인 필요
- 실제 스피커·경광등·전광판 장애 격리: 현장 네트워크와 장비 인증값으로 확인 필요

## 배포 시 주의사항

- 운영 비밀값은 `.env` 또는 `.env.jetson`에만 두고 커밋하지 않습니다.
- Jetson에서는 `docker-compose.jetson.yml`을 사용합니다. 기본 Compose의 일부 EdgeX 이미지는 ARM64와 맞지 않을 수 있습니다.
- Compose, 환경변수 예제, 런타임 경로가 함께 변경되었으므로 배포 전 `config --quiet`과 운영 점검 스크립트를 실행합니다.
- 데이터베이스와 모델이 저장된 외부 volume은 Compose 종료만으로 삭제되지 않습니다.

## 검증 결과

2026-07-03 게시 직전 로컬 검증 결과:

- Python 정적 검사: `ruff check src scripts tests` 통과
- 전체 pytest: Git pre-push 기준 `1190 passed, 5 skipped`, 실패 0
- 기본 Compose 구문: `.env.example` 기준 `config --quiet` 통과
- Jetson Compose 구문: `.env.jetson.example` 기준 `config --quiet` 통과
- shell script 문법 검사: `scripts/**/*.sh` 전체 통과
- Markdown 내부 링크 검사: 끊어진 로컬 링크 0건
- Git 패치 형식 검사: `git diff --check` 통과

skip 항목은 Jetson/DeepStream 등 현재 실행 환경에서 사용할 수 없는 조건을 포함합니다. 따라서 `1190 passed`는 로컬 자동 검증 결과이며 Jetson 실기 검증을 대신하지 않습니다.

Jetson GPU, 실제 RTSP 카메라, 외부 장비가 필요한 검증은 로컬 자동 테스트와 구분해 기록합니다.

## 관련 문서

- [빠른 시작](../guides/QUICK_START.md)
- [프로젝트 구조](../modules/PROJECT_STRUCTURE.md)
- [프로젝트 구조의 DeepStream 모듈 설명](../modules/PROJECT_STRUCTURE.md)
- [Action Layer 구조](../modules/ACTION_LAYER_SPEAKER_BRIDGE.md)
- [falldata 연동](../features/FALLDATA_INTEGRATION.md)
- [운영 Runbook](../guides/OPERATIONS_RUNBOOK.md)
