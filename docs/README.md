# CCTV 프로젝트 문서

프로젝트 문서를 목적별로 분리했습니다. 처음 보는 경우에는 **실행 방법 → 기능 → 모듈 구조** 순서로 읽는 것을 권장합니다.

## 빠른 이동

| 분류 | 이런 경우에 확인 | 시작 문서 |
|---|---|---|
| [기능 정리](features/README.md) | 제공 기능, API, 이벤트 계약, AI 기능을 확인할 때 | [Public API 가이드](features/PUBLIC_API_GUIDE.md) |
| [모듈화 정리](modules/README.md) | 디렉터리, 모듈 책임, 서비스 간 데이터 흐름을 파악할 때 | [프로젝트 구조](modules/PROJECT_STRUCTURE.md) |
| [실행 및 배포](guides/README.md) | 로컬·Docker·Jetson 실행, 설정, 운영 점검이 필요할 때 | [빠른 시작](guides/QUICK_START.md) |
| [리뷰 정리](reviews/README.md) | 코드 리뷰, 완성도 점검, 시점별 개선 내역을 볼 때 | [최신 변경 및 검증 요약](reviews/CHANGESET_SUMMARY_2026-07-03.md) |
| [개발 도구](tooling/README.md) | Codex 등 개발 보조 도구 설정을 확인할 때 | [Codex 멀티 에이전트 설정](tooling/CODEX_MULTI_AGENT_SETUP.md) |
| [구조·인수인계](architecture/README.md) | 전체 구조, EdgeX, Docker, 센서, 모델 인수인계를 확인할 때 | [프로젝트 구조·운영 문서](architecture/README.md) |
| [모델 인수인계](models/README.md) | 적용 모델, 평가 수치, 재학습·승격 절차를 확인할 때 | [모델별 상세 인수인계서](models/MODEL_HANDOVER.md) |
| [운영 인수인계](handover/README.md) | 장비, 백업, UAT, 장애, 보안, 모니터링을 확인할 때 | [운영 인수인계 문서](handover/README.md) |
| [EdgeX·eKuiper 인수인계](integrations/EDGEX_KUIPER_HANDOVER.md) | EdgeX 구조, 장치 등록, eKuiper 룰과 장애 대응을 확인할 때 | [EdgeX·eKuiper 상세 인수인계서](integrations/EDGEX_KUIPER_HANDOVER.md) |

## 루트 문서

- [프로젝트 README](../README.md): 프로젝트 소개와 빠른 시작
- [명령어 모음](../COMMANDS.md): 자주 사용하는 실행·점검 명령

## 문서 관리 원칙

- 기능의 동작과 외부 계약은 `features/`에 둡니다.
- 코드 책임, 의존 관계, 구조 변경 계획은 `modules/`에 둡니다.
- 설치, 설정, 실행, 배포, 운영 점검 절차는 `guides/`에 둡니다.
- 날짜가 있는 점검 결과, 코드 리뷰, 완료 보고서는 `reviews/`에 둡니다.
- 하나의 문서가 여러 분류에 걸치면 문서의 **주요 목적**을 기준으로 한 곳에 두고 다른 문서에서 링크합니다.
