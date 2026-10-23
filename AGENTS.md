## Ponytail 기반 최소 구현·토큰 절약

- 사용자 요청(2026-09-22)에 따라 [Ponytail 공식 규칙](https://github.com/DietrichGebert/ponytail/blob/main/AGENTS.md)을 이 프로젝트에 맞춰 적용한다. 당시에는 AGENTS.md 지침만 도입했으며, 2026-10-02 사용자 요청으로 Codex용 `ponytail@ponytail` 4.10.0을 설치·활성화했다. 기존 프로젝트 제약은 계속 적용한다.
- 구조 파악은 `docs/architecture/archify/index.html`과 해당 README를 길잡이로 삼는다. 그림은 스냅샷이므로 변경 대상의 실제 흐름은 그래프 도구와 필요한 코드 조각으로 확인한다.
- 구현 전에 필요성 → 기존 코드 재사용 → 표준 라이브러리 → 플랫폼 기본 기능 → 설치된 의존성 순으로 검토하고, 충족하는 가장 단순한 방법을 선택한다.
- 작업과 직접 관련된 호출 흐름을 확인한 뒤 공통 원인을 최소 수정한다. 요청하지 않은 추상화·새 의존성·대규모 리팩토링을 추가하지 않는다. 줄 수보다 이해하기 쉬운 올바른 코드를 우선한다.
- 사용자 요청(2026-09-29): 코드를 수정할 때 변경한 함수나 블록 가까이에 한글 주석으로 수정 이유와 중요한 제약을 남긴다. 동작 설명이 필요하면 함께 적되, 코드의 단순 번역이나 변경 이력 나열은 피하고 동작이 바뀌면 주석도 갱신한다.
- 사용자 요청(2026-10-06): 업무로 변경하는 범위에서 불필요한 주석·구현과 어긋난 구형 주석·데드코드를 함께 정리한다. 주석과 설명용 docstring은 새로 쓰거나 갱신할 때 한글로 작성한다. 삭제 전 직접 호출·콜백·문자열 참조·시험을 확인하며, 그래프의 호출 수가 0이라는 이유만으로 삭제하지 않는다.
- 문서는 현재 작업판과 관련 절만 읽고, 이미 확인한 내용은 변경이 없으면 다시 읽지 않는다. 전체 파일·긴 로그 출력 대신 필요한 구간과 실패 요약을 사용한다. 명령은 기존 RTK 규칙을 따른다.
- 변경에 맞는 최소한의 유효한 검증을 수행한다. 통과 후에는 추가 변경·실패·미해결 우려가 있을 때만 재검증한다. 입력 검증·데이터 유실 방지·보안·실장비 보정은 절약 대상이 아니다.
- WBS와 이슈의 원본 및 기존 진행판을 갱신하되, 같은 내용을 반복하는 새 보고서·검증 스크립트는 필요할 때만 만든다. 답변은 결과·검증·남은 일 중심으로 짧게 쓴다.
- 플러그인 사용은 Codex 재시작 후 새 대화에서 확인하며, 자동 실행 훅은 `/hooks`에서 검토·신뢰 승인한다. 설치·검증 근거와 `@ponytail-review` 사용법은 `docs/tooling/CODEX_SKILL_ADOPTION.md`에 기록한다. 토큰 절감률은 측정 전 단정하지 않는다.

## 현재 개발 계획과 진행 기록

- 모듈 안정화·현장 POC 작업을 시작할 때 `docs/planning/README.md`, `docs/planning/PROGRESS.md`, `docs/planning/WBS.md`를 순서대로 확인한다.
- 한 번에 WBS 작업 하나를 작은 범위로 진행하고, 종료 시 상태·검증 근거·운영 반영 여부·다음 작업을 갱신한다.
- 기존 성과와 이슈는 `docs/planning/STATUS.md`, 확인된 장비·가용시간과 첫 시험 범위는 `docs/planning/POC_SCOPE.md`를 확인한다. 완료·실패·차단·범위 변경이 생기면 추가 요청을 기다리지 않고 같은 작업에서 WBS·진행 기록·이슈를 갱신한다.
- 2026-09-18 사용자 확인: 하루 순개발 6시간 이상, CCTV 1~2대·온습도/기울기 센서·전광판·스피커 보유, 사이렌 미보유. 주당 일수·외부 관제·현장 일정은 미확정이다. 보유를 연결/실물 합격으로 해석하거나 실물 4카메라 목표를 자동 축소하지 않는다.
- `tasks/plan.md`와 `tasks/todo.md`는 기존 이력이다. 과거 체크박스를 현재 완료 상태로 자동 사용하지 않는다.
- 목표 종료는 2026-12-31이다. 일정의 가정·미확정 조건을 실제 관측과 구분하고, 목표 미달을 완료로 표시하지 않는다.
- 사용자 지시로 범위·순서·일정이 바뀌면 해당 문서를 함께 갱신한다. 계획 문서는 운영 중지·배포·실물 출력의 포괄 승인을 뜻하지 않는다.

## Codex 병렬 작업과 충돌 방지

- 이 프로젝트의 모든 Codex 채팅에 적용한다. 기본 구성은 통합 담당 1개·개발 작업 2개부터 시작하고, 분석·리뷰는 읽기 전용으로 진행한다. 작업이 늘어나면 담당 파일과 공유 자원의 중복을 먼저 확인한다. 현재 통합 담당과 배정표는 `docs/tooling/CODEX_SKILL_ADOPTION.md`의 병렬 작업 절에서 관리한다.
- 같은 checkout의 파일·Git index를 수정하는 채팅은 하나만 둔다. 나머지는 읽기 전용 분석을 하거나 별도 worktree에서 작업한다. 채팅이나 브랜치 이름만 다르게 만드는 것은 작업 폴더 분리가 아니다.
- 앱의 기본 경로 표시가 원본 checkout으로 남아 있어도 배정표의 전용 worktree를 실제 작업 경로로 사용한다. 개발 채팅은 쓰기 명령의 `workdir`·Git의 `-C`·패치의 절대 경로를 명시하며, 원본은 승인된 읽기 전용 참조에만 사용한다.
- 수정 전에 `AGENTS.md`와 배정표를 읽고 실제 작업 경로·기준 커밋·미커밋 변경을 확인한다. 앱의 관련 채팅 조회가 가능하면 실행 상태도 확인한다. 담당 작업·수정할 파일과 시험·사용 장비를 짧게 알리고 시작한다. 현재 작업판을 기준으로 분리하며 기본 브랜치나 오래된 worktree를 최신 상태로 간주하지 않는다.
- 새 작업은 요청에서 범위를 정해 독립 worktree에서 진행하고 통합 담당에 배정 정보를 인계한다. 같은 작업의 후속 질문은 기존 채팅·worktree를 재사용한다. 같은 파일·공통 인터페이스를 바꾸는 작업은 순서를 조정하고, 담당이 불명확한 부분은 읽기 전용 분석을 이어가며 확인한다. 하위 에이전트도 작업 폴더가 자동 분리된다고 가정하지 않는다.
- 미커밋 변경이 있는 폴더를 분리할 때는 기존 작성자가 수정을 멈춘 시점에 변경을 보존하고, 새 worktree의 필요한 소스·시험·설정과 대조한다. 새 worktree 생성만으로 변경이 옮겨졌다고 가정하지 않는다. 다른 채팅의 변경을 일괄 stage·commit·stash하거나 reset·clean하지 않는다.
- 병렬 구현은 worktree마다 WBS 작업 하나와 담당 범위를 정한다. 통합 담당만 배정표·공통 WBS·PROGRESS·STATUS·진행판을 순서대로 갱신한다. 작업자는 변경 파일·검증 근거·미해결 사항을 인계해 기존 기록 갱신 의무를 이어간다. 단독 작업은 기존 문서 갱신 절차를 따른다.
- worktree는 GPU·카메라·Docker·MQTT·DB·네트워크 포트를 격리하지 않는다. 같은 장비의 부하 시험은 동시에 실행하지 않고, 장비 사용 담당을 확인한다. 운영 Compose는 고정 컨테이너 이름 등이 있으므로 프로젝트 이름만 바꿔 독립 시험 환경이라고 판단하지 않는다.
- 모델·영상·비밀 설정·가상환경은 필요한 항목만 확인한다. 운영 `.env`·DB·출력 경로를 그대로 복제하거나 공유 쓰기로 연결하지 않는다. 브랜치는 `codex/` 접두사를 사용하고, 검증된 변경을 하나씩 통합한다.
- 종료 시 작업 경로·기준/결과 커밋(없으면 미커밋 상태)·변경 파일·검증 결과·운영 반영 여부·실행 중 프로세스·남은 일을 전달한다. 개발 완료와 통합 완료를 구분한다. 통합 담당은 관련 검증을 마친 뒤 기록을 갱신하고, 미커밋 변경과 미통합 커밋이 남지 않았는지 확인한 worktree만 정리한다.
- 루트 `AGENTS.md`는 버전관리 대상으로 유지한다. 새 worktree에서 지침의 존재와 내용을 확인하고, 이 규칙 반영 전 커밋에서 만들었다면 최신 지침을 별도로 전달한다. 기존 지침이 있으면 덮어쓰기 전에 차이를 확인한다.
- 이 지침은 자동 잠금이 아니다. 이미 실행 중인 채팅은 지침을 재확인하고 담당 범위·작업 경로를 합의한 뒤 계속해야 한다. 전환 절차와 인계 문구는 `docs/tooling/CODEX_SKILL_ADOPTION.md`의 병렬 작업 절을 따른다.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
