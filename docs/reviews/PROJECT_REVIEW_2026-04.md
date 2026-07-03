# CCTV 프로젝트 리뷰 및 개선 제안

## 잘한 부분

- AI 엔진, Action Layer, EdgeX/Kuiper, 공개 API를 분리해 둬서 실서비스 확장 방향이 좋습니다.
- `src/bootstrap`, `src/core`, `src/services`, `src/api` 계층이 나뉘어 있어 책임 분리가 점점 선명해지고 있습니다.
- 테스트 파일이 기능별로 넓게 깔려 있어 회귀 방지 기반이 이미 있습니다.
- Jetson/Windows를 모두 고려한 구조와 Docker 자산이 있어 현장 배포를 염두에 둔 설계가 보입니다.

## 부족한 부분

- 런타임 산출물(`data/crops`, `snapshots`, `logs`)이 저장소를 쉽게 오염시켜 협업성이 떨어집니다.
- 운영 관점의 헬스체크, readiness, 배포 절차 문서가 서비스별로 완전히 정리되지는 않았습니다.
- 공개 API와 내부 서비스 API의 응답 규약, 인증, 에러 포맷이 아직 완전히 표준화되지는 않았습니다.
- 일부 기능은 잘 추가되고 있지만 "실사용 운영 기준"의 관찰 가능성(health, metrics, alerting)은 더 보강이 필요합니다.

## AI Engineer 관점

- 잘한 점:
  AI 추론 경로와 이벤트 필터링 경로가 분리되어 있어서 모델 교체와 후처리 튜닝이 비교적 쉽습니다.
- 보완점:
  모델별 precision/recall, false positive, latency 기준이 코드 밖에서 추적되지 않습니다.
- 추가 추천:
  카메라별 임계값, zone별 이벤트 정책, fall/head 재전송 정책을 설정 파일로 더 세분화하는 편이 좋습니다.
- 추가 추천:
  얼굴 인식은 운영 환경에서 privacy/audit 로그와 feature flag를 분리해 두는 것이 안전합니다.

## DevOps Automator 관점

- 권장 배포 방식:
  Jetson은 `ai-engine` 전용 노드로 두고, Windows 또는 Linux 서버는 `public-api`, `action-layer`, `edgex`, `mosquitto`, `redis`를 맡는 2계층 분리가 가장 운영하기 쉽습니다.
- 이유:
  GPU 의존 서비스와 API/메시징 서비스를 분리하면 장애 영향 범위와 배포 속도를 함께 관리할 수 있습니다.
- 필수 보강:
  각 서비스에 `/health` 또는 동등한 health endpoint를 두고, compose 또는 ingress 레벨 healthcheck를 붙이는 것이 좋습니다.
- 필수 보강:
  `.env` 샘플, 이미지 태그 정책, 롤백 기준, 로그 수집 위치를 문서화해야 운영자가 덜 흔들립니다.
- 권장 자동화:
  현재 CI는 좋은 출발점이며, 다음 단계로는 이미지 취약점 차단 기준과 배포 승인 단계를 분리하는 것이 좋습니다.

## Senior Developer 관점

- 표준화 추천:
  API 응답 모델, 에러 모델, 인증 의존성, 페이징/필터 파라미터를 공통 스키마로 통일하는 편이 좋습니다.
- 표준화 추천:
  `runner` 스크립트마다 로깅 초기화, 환경변수 로드, graceful shutdown 패턴을 공통화하면 유지보수가 쉬워집니다.
- 실활용 기준 추가 기능:
  알람 확인/해제 ACK, 이벤트 검색 필터, 카메라 상태 진단, 최근 스냅샷 조회, 구역별 통계 대시보드.
- 실활용 기준 추가 기능:
  장비 오프라인 감지, RTSP 품질 저하 경고, 모델 버전 조회, 카메라별 탐지 on/off 제어.

## 이번 정리 사항

- `stream_api`에 `/health` 엔드포인트를 추가해 운영 상태 점검 경로를 마련했습니다.
- `STREAM_FPS`, `STREAM_JPEG_QUALITY` 환경변수 파싱을 안전하게 바꿔 잘못된 설정으로 서버가 import 단계에서 죽지 않도록 했습니다.
- `.gitignore`에 대용량 런타임 산출물 경로를 추가해 저장소 오염을 줄였습니다.
- 공개 API 응답/에러 포맷을 공통 래퍼로 더 통일했습니다.
- 공개 API 테스트가 로컬 tempdir 이슈 없이 돌도록 `tests/conftest.py`의 임시 디렉터리 fixture를 안정화했습니다.
- Action Layer 프록시 에러 메시지를 공통 helper로 정리했습니다.
- 하위 호환 shim 유지 이유와 축소 계획을 `docs/COMPATIBILITY_SHIMS.md`에 문서화했습니다.

## 다음 우선순위

1. 공개 API와 내부 API의 응답/인증/에러 포맷을 공통화하기
2. 서비스별 healthcheck와 compose healthcheck를 연결하기
3. 카메라/이벤트/알람 운영 메트릭을 수집하기
4. 장기 보관이 필요 없는 로컬 산출물의 보존 정책을 스크립트로 자동화하기
