# Compatibility Shims

현재 코드베이스에는 즉시 삭제하면 회귀 가능성이 큰 하위 호환 연결부가 일부 남아 있습니다.

## 유지 중인 호환 연결부

- `src/core/processor.py` 의 `_AdaptiveGovernor` 재내보내기
  기존 import 사용처와 내부 참조 안정성을 위해 유지 중입니다.

## 지금 바로 삭제하지 않은 이유

- 일부 로컬 스크립트나 외부 운영 코드가 `_AdaptiveGovernor` 재내보내기 경로를 사용할 수 있습니다.
- 문서와 코드 리뷰 기록 일부가 과거 구조를 기준으로 작성되어 있습니다.

## 축소 계획

1. 외부 운영 스크립트에서 `_AdaptiveGovernor` 직접 import 여부 점검
2. 필요 시 명시적 deprecation notice 추가
3. 한 릴리즈 동안 관찰
4. 이후 shim 제거

## 권장 원칙

- shim 제거는 기능 리팩토링과 분리해서 진행
- 내부 import와 테스트 경로를 먼저 정리한 뒤 제거
- 사용자 스크립트 영향을 줄이기 위해 제거 시점을 문서에 명시
