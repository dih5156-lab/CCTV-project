# 프로젝트 구조·운영 문서

이 문서는 CCTV AIoT 안전 시스템을 처음 맡는 사람이 **전체 구조를 이해하고, 로컬 또는 Jetson에서 실행하며, 안전하게 수정**할 수 있도록 만든 안내서다.

## 먼저 읽을 순서

1. [프로젝트 구조와 처리 파이프라인](PROJECT_STRUCTURE_AND_PIPELINES.md)
2. [전체 아키텍처](ARCHITECTURE.md)
3. [Docker 구성과 실행 방법](DOCKER_AND_EXECUTION.md)
4. [EdgeX 연동](EDGEX_INTEGRATION.md)
5. [센서 연동 형식과 방법](SENSOR_INTEGRATION.md)
6. [수정·검증 가이드](MODIFICATION_GUIDE.md)
7. [적용 모델별 상세 인수인계서](../models/MODEL_HANDOVER.md)
8. [시스템 아키텍처 기준 인수인계 설명서](ARCHITECTURE_HANDOVER.md)
9. [EdgeX 중심 Edge AIoT 전환 계획](EDGEX_FIRST_EVENT_ROUTING_PLAN.md)

## Figma 다이어그램

- [CCTV AIoT 안전 시스템 파이프라인 v2 — 입력→출력](https://www.figma.com/board/3mUQ3pVDnAKbWqlyOtTJUd)
- [CCTV AIoT 시스템 아키텍처 v4 — 디바이스 입력→출력](https://www.figma.com/board/uickZg65te7UqNZuoD2yVd)

두 그림은 Figma/FigJam에서 직접 열어 확대·주석·수정할 수 있다. 문서의 설명과 실제 코드가 달라지면 코드와 실행 로그를 우선 확인하고 다이어그램을 갱신한다.

## 관련 기존 문서

- [전체 인수인계서](../HANDOVER_CCTV_AIOT_SAFETY_SYSTEM.md)
- [빠른 시작](../guides/QUICK_START.md)
- [운영 런북](../guides/OPERATIONS_RUNBOOK.md)
- [MQTT·EdgeX 데이터 계약](../integrations/MQTT_EDGEX_DATA_CONTRACT.md)
- [디바이스 API 문서](../devices/README.md)

## 문서의 신뢰 수준

- **코드 확인**: 현재 저장소의 코드·Compose·설정에서 확인한 내용
- **운영 확인**: 실제 실행 또는 리플레이 테스트로 확인한 내용
- **현장 확인 필요**: 장비 IP, 실제 payload, 네트워크 정책처럼 배포 환경에서 확인해야 하는 내용
