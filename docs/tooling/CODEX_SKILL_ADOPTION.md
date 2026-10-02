# Codex Skill 도입 기준

이 문서는 OpenAI의 기존 `openai/skills` 저장소를 참고해, CCTV AIoT 프로젝트에서 반복 업무를 skill로 관리하는 기준을 정의한다.

## 결론

`openai/skills`의 예제를 그대로 설치하지 않는다. 해당 저장소는 deprecated 상태이므로 구조와 작성 방식만 참고하고, 필요한 절차는 프로젝트 로컬 skill 또는 현재 Plugins 형식으로 재작성한다.

참고 자료:

- [OpenAI skills 저장소](https://github.com/openai/skills)
- [OpenAI Plugins 저장소](https://github.com/openai/plugins)
- [공식 Build skills 문서](https://developers.openai.com/codex/plugins)

## 적용 대상

Skill은 Jetson의 상시 실행 서비스나 추론 파이프라인에 포함하지 않는다. 다음과 같은 개발·운영 보조 업무에만 사용한다.

| 우선순위 | 후보 | 목적 | 기본 안전 모드 |
| --- | --- | --- | --- |
| 1 | `cctv-model-evaluation` | fall/not-fall 평가, 모델 비교, 리포트 생성 | 읽기 전용, 학습 프로세스 접근 금지 |
| 2 | `cctv-jetson-ops` | compose·healthcheck·포트·리소스·로그 점검 | 읽기 전용, restart 금지 |
| 3 | `cctv-api-handover` | API 목록·인증·예시·장애 대응 문서 갱신 | 문서 디렉터리만 쓰기 |
| 4 | `cctv-code-review` | 호출 경로·영향 범위·테스트 공백 점검 | 코드 수정 없이 보고서 생성 |

## 작성 규칙

1. `SKILL.md`에는 목적, 적용 조건, 입력·출력, 실행 명령, 실패 시 대응을 명시한다.
2. 모든 실행 스크립트는 `--dry-run` 또는 읽기 전용 기본 동작을 제공한다.
3. 학습 컨테이너, 모델 파일, compose 상태를 변경하는 명령은 기본 흐름에서 제외한다.
4. API 키·토큰·비밀번호를 로그나 문서에 출력하지 않는다.
5. 결과물에는 실행 시각, 대상 커밋 또는 모델 버전, 명령, 성공·실패 상태를 기록한다.
6. 새 skill을 추가하기 전에 기존 `AGENTS.md`와 프로젝트 로컬 skill의 규칙을 먼저 적용하고 중복 지침을 제거한다.
7. skill 변경은 일반 코드 변경과 같은 브랜치에서 검토하며, 실제 적용 전 CPU 기반 샘플 테스트를 통과해야 한다.

## 도입 순서

1. 현재 학습이 끝난 뒤 모델 평가 skill을 먼저 만든다.
2. 기존 `scripts/ops`, `scripts/health`, `tests`를 호출하는 얇은 래퍼로 시작한다. 새로운 평가 로직을 skill 안에 복제하지 않는다.
3. 샘플 영상 한 개와 고정된 기대 결과로 실행 로그를 검증한다.
4. 문서·운영 점검 skill을 추가하고, CI 또는 인수인계 체크리스트에서 호출한다.
5. MCP나 외부 플러그인은 로컬 절차가 안정화된 뒤 필요한 경우에만 연결한다.

## 현재 상태

- 학습 및 실시간 추론 서비스에는 변경하지 않았다.
- skill 도입과 독립적인 pose crop·운영 문서 관련 테스트 13개가 통과했다.
- `skills/cctv-model-evaluation/SKILL.md`에 안전한 평가 절차의 초안이 추가되어 있다.
- 실제 평가 명령과 모델 입력·출력 형식은 학습 완료 후 확정하고, 이 skill에 연결한다.

## Ponytail 플러그인 설치 — 2026-10-02

사용자 요청에 따라 Codex CLI 0.160.0에서 [Ponytail 공식 저장소](https://github.com/DietrichGebert/ponytail)의 Codex용 플러그인을 설치했다. 2026-09-22 도입한 AGENTS.md 지침은 유지한다.

- 설치·활성화: `ponytail@ponytail` **4.10.0**, 공식 저장소 커밋 `e3ba2aa6f1e6f0bc4d69eb09c9f0d0a93af56156`.
- 설치 경로: `/home/sawwave/.codex/plugins/cache/ponytail/ponytail/4.10.0`.
- 설정 백업: `/home/sawwave/.codex/config.toml.pre-ponytail-20261002-150129`. 기존 설정은 보존하고 Ponytail 플러그인·마켓플레이스 항목만 추가했다.
- 포함 기능: `ponytail`, `ponytail-review`, `ponytail-audit`, `ponytail-debt`, `ponytail-gain`, `ponytail-help`.
- 검증: 플러그인 목록에서 `installed=true`, `enabled=true` 확인. 별도 임시 상태 디렉터리에서 시작·리뷰 모드 전환·full 복귀·하위 에이전트 훅의 출력과 상태 검증 **4건 통과**. 현재 Codex 대화에서의 자동 훅 실행·신뢰 승인은 미검증이다.
- 영향 범위: 사용자 Codex 개발 도구 설정 및 프로젝트 안내 문서. CCTV 운영 서비스나 배포 라이브러리 변경은 없다.

실행한 설치 명령과 상태 확인 명령:

```bash
rtk proxy codex plugin marketplace add DietrichGebert/ponytail --json
rtk proxy codex plugin add ponytail@ponytail --json
rtk proxy codex plugin list --marketplace ponytail --json
```

Codex를 재시작하고 새 대화에서 `@ponytail-review`로 리뷰 기능을 호출한다. 공식 안내에 따라 자동 실행 훅은 Codex의 `/hooks`에서 검토·신뢰 승인한 뒤 새 스레드에서 사용한다. 플러그인 설치 성공과 현재 대화의 자동 활성화는 구분한다.
