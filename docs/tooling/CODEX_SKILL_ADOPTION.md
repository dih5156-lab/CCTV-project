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


## 2026-10-13 code-review-graph 파서 복구·웹 탐색

- 2.3.7 MCP 전체 생성에서838파일/1노드가 나와 정상이 아님을 확인했다. `/usr/bin/python3 -I`는 사용자 경로의 `tree_sitter_language_pack`을 찾지 못하지만 일반 Python은 로드했다. 격리 파서 검사와 사용자 패키지 설치 위치의 불일치가 원인이다.
- `data/runtime/module_graph_20261013/tooling-env`를 system-site-packages/without-pip로 만들고 `.pth`로 기존 사용자 도구 경로를 연결했다. 새 패키지 설치·전역 설정 변경 없이 실제 파서 검사를 유지했다. 전체 빌드 뒤 미추적 제품/시험 파일도 기존 API로 추가했다.
- 결과: 그래프7,564노드/68,808관계/454정적 흐름·700파일. igraph가 없어 그룹은 파일 기반 대체 방식이며 PowerShell 파서는 사용 불가였다. 모든 언어/동적 콜백/프로세스 간 통신을 완전하게 추적한다고 해석하지 않는다.
- MCP `query_graph_tool`에서 새 ActionMqttIntake의 관계와 Action 입력 분기를 확인했다. 전역 MCP build 환경은 그대로이므로 재생성은 [작업용 명령](../architecture/archify/README.md)을 사용한다. 향후 전역 설정 정비는 별도다.
- [모듈 웹](http://192.168.2.141:18081/module_flow.html)은 Archify 요약2개와 code-review-graph 자체 탐색 화면을 연결한다. 외부 CDN 요청을 없애기 위해 같은 D3 7.9.0의 고정 SRI와 라이선스를 확인해 문서 자산으로 제공했다. CCTV 실행 이미지의 의존성은 바꾸지 않았다.
- 문서 IP 재정합·허용 경로 추가 후8파일 HTTP200/해시 일치, 구조도 각각9/9·JS/JSON 검사 통과. Chrome/Chromium 미설치로 브라우저 조작·시각 검토는 미실행. [검증 근거](../planning/evidence/module_graph_20261013.json)

## 2026-10-13 Jetson 검증 도구 정리

추천8개 도구의 설치/재사용/보류와 공식 근거·실행법은 [Jetson 개발 도구 적용표](JETSON_DEVELOPMENT_TOOLS.md)에서 관리한다. WBS11행의 장기 운영 기준에 연결했고 Nsight CLI/pytest-benchmark를 설치·실행했다. 설치와 실제 GPU/장기 운전 검증은 구분한다.

## 2026-10-23 Codex 병렬 작업 규칙

사용자가 공유한 [병렬 작업 상담](https://chatgpt.com/share/6adaa6d1-ac78-83ec-9f2f-1400bcc244fb)을 확인하고, 후속 요청에 따라 작업별 worktree와 통합 담당을 두는 방식을 이 프로젝트의 기본 Codex 규칙으로 정했다. 규칙 원본은 루트 [AGENTS.md](../../AGENTS.md)의 병렬 작업 절이며, 이 문서는 배정·전환·인계를 관리한다. 파일 분리와 실장비 자원 분리는 별도로 확인한다.

### 초기 조사 상태

- 조사 시점의 활성 채팅 세 개가 모두 `/media/sawwave/Learning11/CCTV-project`를 사용했다. 채팅 제목은 아래 표와 같다.
- 기준 HEAD는 `fefa4ab04ee2`(2026-10-02), 브랜치는 `260907-260911-9월2주차`였다. `git status --porcelain=v1` 기준 추적 변경 224항목·미추적 238항목이 있었다. 미추적 디렉터리는 한 항목으로 집계되므로 파일 총수는 아니다. 이 수치는 이번 지침 추가 전의 관측이다.
- Git의 미해결 병합 항목은 0건이었다. 과거 동시 저장에 의한 변경 유실 여부까지 증명한 것은 아니다.
- 기존 worktree 네 개는 7~8월 커밋에서 시작했고, `appearance-multitask-v1`에는 미커밋 34항목이 있었다. 별도로 경로가 사라진 등록 한 개도 있었다. 재사용·정리하지 않았다.
- 설치된 `codex-cli 0.163.0`의 `--help`에서 `--worktree` 지원을 확인했다.
- 초기 조사에서는 루트 `AGENTS.md`가 Git에서 제외됐다. 후속 규칙화 작업에서 `.gitignore`에 `!/AGENTS.md` 예외를 추가했다. 이제 버전관리 대상이지만 아직 커밋하지 않았으므로 이전 커밋에서 만드는 worktree에는 별도 전달·읽기 확인이 필요하다.
- `docker-compose.jetson.yml`에는 `name: edgex-jetson`과 고정 `container_name`이 있다. 다른 폴더에서 같은 Compose를 실행해도 독립된 환경이 되지 않는다.

### 기본 역할과 기존 채팅의 전환 상태

10-26 소스 통합: A/B 자동복귀3파일과 ROI 도구4파일은 before/after 해시 대조 후 원본 커밋 `59624ed`에 포함했다. camera-offload 최신 후보14파일과 싸움 개발은 미통합 보존한다. 작업판/원시자료/index 백업과 검증 제한은 [통합 검토 기록](../planning/evidence/integration_review_20261026.json)에 있다. 아래10-23 역할별 세부 수치는 해당 시점 이력이다.

통합 창구는 **Codex 병렬 작업 충돌 검토** 채팅이다. 2026-10-23 재확인에서 개발 채팅이 이미 전용 worktree 네 개를 만들고 실제 명령에 사용한 것을 확인했으며, 중복 worktree를 만들지 않고 이를 유지했다. 분리 당시 다섯 checkout의 기준 HEAD는 `fefa4ab04ee2ca26bbc0ee433381e4b7648265e7`이었다. 10-26 원본은 문서 `6972047`와 검토 소스204파일 `59624ed`를 커밋했으며, 개발 worktree의 기존 미커밋/기준 커밋은 그대로 보존한다. 앱의 채팅 기본 `cwd` 표시는 원본으로 남아 있지만, 실제 수정·시험·Git 명령은 아래 경로를 명시한다.

| 채팅 / 실제 작업 경로·브랜치 | 담당 범위 | 관측 상태·자원 |
| --- | --- | --- |
| **Codex 병렬 작업 충돌 검토** — `/media/sawwave/Learning11/CCTV-project` / `260907-260911-9월2주차` | 규칙·배정표·변경 통합·공통 진행 문서 | 분리·보존 확인. 엔진 배치 사전점검 원본 통합77 PASS·Ruff PASS, 기존 변경/index 보존. 이번 통합의 GPU/카메라/Docker 접근0. |
| **다중 카메라 ROI·트래픽 수정 계획** — `/home/sawwave/.codex/worktrees/two-camera-baseline/CCTV-project` / `codex/two-camera-baseline` | P6.2 기준 측정·ROI/트래픽 분석, `TWO_CAMERA_BASELINE_HANDOFF_20261023.md`와 해당 증거 JSON; 전용 runner·모의 단위시험·개별 인계의 중단 수치 기록 보완 | 한 쌍·3조건 제한23 PASS/Ruff PASS. 실제 사전점검29.49FPS/14.75Hz PASS 후 첫 CPU 변환에서 보호 중단, 중단 수치 미기록. GPU/수집기/시험 컨테이너 미시작·6결과 NOT_RUN.12:32:28 자체 자원0·운영 보존 확인,12:35:02 조기 반환·추가 부하 보류. guard 원시값 저장 보완31 PASS/Ruff PASS 담당 인계 완료·통합 측 소스2개/고정3개 해시 대조, 제품 소스 통합/장비 재시험 미실행. 기존 frozen verification/pilot 증거 보존·새 소스는 장비 미실행. GPU/변환/운영 수집기/장비 진단/재인덱싱 요청0. |
| **다중 카메라 엔진 배치 문제 확인** — `/home/sawwave/.codex/worktrees/multicamera-engine-batch/CCTV-project` / `codex/multicamera-engine-batch` | P6.2/I56 사전점검·배치 후보 준비, health/convert 관련 스크립트·시험·사용법 | 기존 max4 raw batch1/2/4 PASS 유지.13:29·14:25 모두 운영 사전점검 보호 중단·각GPU0/9. attempt2 4.188319FPS/1.994438Hz,14:25:38.504 정리·14:27:02.015 반환. 고정177/영상·라벨8/격리9 준비 검증 보존. 정확도/영상4입력 미평가·새 배정 없음. |
| **오케스트레이션으로 프로젝트 진행하기** — `/home/sawwave/.codex/worktrees/edge-auto-failback/CCTV-project` / `codex/edge-auto-failback` | A↔B 자동 전환, `src/services/edge_auto_handoff.py`·해당 시험·전용 정책 설정; 사용자 개발도구 중복 인덱싱 원인/최소 수정 전담; 신규 운영FPS/RTSP 실패의 기존 증거·로그·소스 읽기 대조 | A→B 단일카메라 확인·연속 복귀/60초 미완료. 시험/연결 정리 완료·새 부하 없음. 중복 인덱싱 조치 적용·모의5항목/검색 PASS 담당 보고, 실제 설정/해시 및45.20초30FPS/15Hz 원시 대조. 이후worker0에서4.77FPS 재저하 원인 미확정. 자동복귀3파일 동결·소유권 유지, camera-offload의 선택3 hunk만 조율 허용. |
| **ROS2 DDS 연동 적합성 확인·카메라1대 이전** — `/home/sawwave/.codex/worktrees/camera-offload/CCTV-project` / `codex/camera-offload` | P6.2-O4 후속, edge_service/edge_node_agent/edge_orchestrator/edge_handoff 및 관련 시험·전용 인계. A/B 완성본 기준 edge_auto_handoff의 observed_camera_service import·_fresh 선택·_candidate services 조회3 hunk와 결합 회귀시험만 허용. 정책/자동복귀 조건 보존, 범위 확대 전 조율; 추가7파일(소유권·미리보기·runner·초기/동적1카메라guard) 한정 개발 배정 | 기준fefa4ab·최신AGENTS 일치, 현재 소스12개 고정 사본과 A/B 완성본3개 SHA 대조. 독립 카메라 프로세스별 감독자+노드1Agent service map 범위, shared4 hot분할/새복귀정책 제외. 소스 개발과 하단 CPU 전용 모형시험만 배정·GPU/카메라/수동 인덱싱 보류, 결합 동작 미검증. 기준/한정 범위: `data/runtime/coordination_1433_as3hte41/camera-offload-scope.json`. 추가7파일 한정배정 유지. 제품12+시험2 구현후 최종검증대기·운영미반영. READY patch637추가/55삭제(604/633추가는과거시점), 15:34 READY 고정614/제품14파일 해시일치로 갱신대기 해소. CPU시험24 PASS/1 FAIL/174 NOT_RUN·15:43:17 자체정리/15:47:38 반환기록 확인. HTTP크기경계63915>65536 단언실패·자동재시도0·제품미통합. 수정 READY050731d2…81a1fd1/614파일 해시대조 PASS(HTTP모형노드3→4, 제품변경0), 새패치542e63d2…ffa271. 이전df6fa96a/612파일 인계는 이력. 수정본 CPU재시험은 사전점검17.345FPS/8.870Hz 미달로pytest0·모형자식0·16:22:31 반환통보. 수정본NOT_RUN·새배정없음. |
| **미니PC 데이터 저장 관리 구성** — 채팅 `01a16768-1af3-7073-89ab-66375066d5ca`, `/home/sawwave/.codex/worktrees/minipc-storage/CCTV-project`, `codex/minipc-storage`, 기준 `6a362f0` | 사용자 실제 탐지·센서·이벤트 이미지 자동저장 요청. 미니PC 저장시각(KST) 기준 날짜/시간 분류, 원 발생시각 보존. 기존 내보내기/전송 재사용·완성본 검증 후 확정·재전송 중복방지·원본 보존. 정확한 파일목록은 흐름 분석 후 조율 | 단발합성3파일3348bytes 저장/회수SHA PASS·SSH/소켓0·반환. 새worktree clean/AGENTS 동일 확인, 원본은 읽기전용. 현재 소스준비만·CPU시험/SSH/실데이터 전송/상주/운영적용 미배정. .20은 미니PC 유지·B 비활성후보 활성화 금지. [범위](../planning/POC_SCOPE.md#minipc-storage-20261023). |

이 표는 작업 배정과 확인 시점의 상태이며 실시간 잠금이나 완료 보증이 아니다. 사용자 후속 요청에 따라 바뀌는 진행 상황은 관련 채팅의 최신 상태로 확인한다. ROI는 모델·가상환경·고정 설정·입력 영상을 원본에서 읽기 전용으로 참조하므로 재시험 전 해시·경로를 다시 확인한다.


### A GPU 사용 순서 요청 — 2026-10-23

확인 시각: `2026-10-26T12:07:55.317407+09:00`. 상태: **`MINIPC_INSTALL_RETURNED_PREFLIGHT_BLOCKED_CODEGRAPH_MOCK_PENDING` — 미니PC 설치 전 보호중단·반환, 새 실행 배정 없음.**

| 항목 | 현재 배정·조건 |
| --- | --- |
| 현재 시험 예약 | **새 배정 없음·ROI 저장 설정비교 BLOCK 보완 대기**.16:28:47 수정본 CPU 구간의 새30초/19모의/Ruff/최종5초 PASS,16:29:33 종료·자체PGID잔여0·16:30:39 반환. 별도16:31:48 저장사본 비교는 허용변경/카메라/roundtrip PASS·기준본 대 운영 사본 7항목 BLOCK(새 장비조회0). 담당은 비교기/시험2파일의 근거 있는 최소 보완·부정회귀 준비만 진행하며 새시험은 별도 배정한다. 실제 운영변경/GPU/변환/인덱싱 미배정. |
| 10-26 정적 준비 재개 | 당시 약63GiB 확보 근거로 전용 worktree의 작은 소스/인계 쓰기를 재개했다(현재 가용량 새 측정 아님). A/B는 동결 단일카메라3파일·기존 검증 스냅샷 기준의 재시험 실행기/보호/정리를 자기 시험폴더에서 정적으로 준비한다. 원본의 camera-offload 서비스 선택3hunk는 혼합하지 않고 현재 통합본 합격으로 표시하지 않는다. camera-offload는 자기 실행기/시험2파일 정적 보완 담당이다. 원본/공통문서는 통합 전담, 새 모의·장비·인덱싱 시험은 미배정이다. |
| CodeGraph checkout 기준 | **검색 호출 보류 해제**. 원본 명시 default, 다른 worktree는 정확한 `projectPath`·동일 Git common·해당 checkout READY/DB만 허용. 상위/원본 자동 fallback0, 미준비는 NOT_READY이며 실제 파일을 대조한다. scoped 제외·손상DB 재시도 차단·원본896파일 DB·단일 reload·검색을 검증했다. 새 index/rebuild/공유설정은 별도 배정. vendor 업데이트 시 패치 해시/경계/오류처리 재검증 필요. 다른MCP 전체무영향·장기FPS안정은 미확인. |
| ROI attempt2 결과 | 새30초 사전점검 PASS 뒤 낙상1개 변환 완료, 정상 영상 변환 중24.292493FPS/12.244997Hz로 보호 중단. GPU/수집0·6결과NOT_RUN·정확도/성능미평가. 실패 원시 카운터/시각·메모리·온도 저장, 부분영상 보존. 고정40파일과 과거12시자료 보존. |
| A/B 반환·보류 확정 | 최종 시험2026-10-23T12:21:43.124773+09:00 보호 종료, 인덱싱12:21:44.436862 원복, GPU12:23:32.839039 반환, 전체12:32:41.774800 정리. DDS A/B pending0·1170건 ID/hash/ACK 일치.13:08:01.356919 A 시험 프로세스/컨테이너0·SSH 소켓 부재 재확인. B는 종료 당시 증거이며 새 SSH 확인 없음. 최소 보류 구간12:55:17.387269~13:10:17.387269 이후에도 통합 담당 배정 전 재개하지 않는다. |
| 엔진 실행 승인 이력 | 종전 max4 생성·batch1/2/4 검증은 완료했다. 이번에는 사용자가 남은 정확도 비교와4카메라 동시 성능 검증을 새로 승인했다. 운영 engine/ONNX/config/컨테이너 변경 금지를 유지한다. 사용자 승인 재요청 없이 계획과 장비 구간을 조율한다. |
| 엔진 사용 완료 구간 | 예약11:38:24~11:53:24 KST 내11:51:15 반환. 별도 max4 생성52.14초·batch1/2/4 PASS·후보 내부 출력차0, 운영 미반영. 후보 파일은 전용 worktree에 보존하며 정확도/공유 앱/실물 성능 잔여로 worktree를 정리하지 않는다. |
| 다음 순서·반환 조건 | 미니PC새회차09:33:08 인증PASS,원격읽기응답로컬UTF8해석오류로판정미확정,09:34:02 자체SSH/소켓0·전송/저장0반환. rawbytes/exit분리보완READY는새시간대기. A/B관측실제반환/회복근거뒤미니PC소량시험·엔진16모의·ROI모의등을각각조율한다. 엔진120초진단은A/B단독관측으로대체,ROI3파일구현/정적준비만허용·camera-offload실장비는기존자료계획만허용한다. |
| 엔진 후속 실행 계획 | 고정 운영 이미지·전용 worktree·network none/별도DB·fakesink에서 기존 main.py→공유 mux/nvinfer→tracker/낙상→이벤트큐를 관측한다. p52_candidate_lock_20261016 trial000/001/002/004의 낙상2·정상2, 기존1080p30·10초 영상4개를 SHA 대조 후 재인코딩 없이 사용한다. 운영nano batch1 네클립→후보max4 batch1 네클립→후보 공유batch4 한회, 총9회. 준비/사전점검30초·각30~45초(약5~7분)·정리2분 예상이며 이전13:29·14:25 회차는 사전점검 보호 중단으로9회 모두 NOT_RUN. attempt3은 새30초 사전점검 PASS 후 첫 기존엔진 컨테이너 착수 중 보호중단하여 반환했다. 첫 회차 INVALID·나머지8회 NOT_RUN이며 재개에는 운영 회복 확인과 새 명시 구간이 필요하다. |
| 엔진 평가 범위 | heuristic person/fall의 정답 비교·영상4입력 개발회귀 계획은 유지한다. CPU12 PASS·Ruff PASS·격리 Compose9개와 통합 측 고정 해시177개/영상·라벨8개/격리9개 대조는 준비 검증이다. attempt3은 사전점검 PASS 후 첫 회차 INVALID·나머지8회 NOT_RUN(카메라별1 INVALID/11 NOT_RUN), 유효결과0이다. 실제GPU추론은 로그 부재로 미확인, 정확도 NOT_EVALUATED·공유4입력 성능 NOT_RUN이다. |
| 최신 시험 중 운영 관측 | ROI 반환 후 A/B단독09:11:27~09:11:57 30.298139초 평균28.153544FPS/14.027264Hz.09:11:32~37 창18.469373FPS/9.333984Hz로 기준미달,나머지5창29.09~31.21FPS·age최대0.055205초·오류/재시작/드롭증가0. 안정회복 미확인, 앞선30초PASS는 ROI착수전 이력이다. [공통 원시 근거](../planning/evidence/two_camera_baseline_20261023.json). |
| ROI 사용 완료 구간 | 첫12:28~12:43 예약은12:35:02 반환(변환0/GPU0). attempt2 15:55~16:10 예약은15:57:11.323231 실행 종료·15:58:16.992244 자체잔여0 확인·15:59:24.352801 반환(유효변환1/GPU0). 두 실행 모두6결과NOT_RUN이며 첫 실패의 미기록 수치는 소급 보완하지 않는다. |
| 후속 시험 운영 보호 조건 | 배정 구간의 운영 사전점검 PASS 후 실행한다. 운영27 FPS/13.5Hz 미만·마지막 프레임 age1초 기준·오류/재시작 증가·가용 메모리8GiB 미만·80도 이상이면 자체 시험만 중단·정리한다. CPU 변환도 같은 보호 범위이며, 중단 원인·해당 구간 원시 카운터/시간·FPS/Hz/메모리/온도를 보존한다. 운영/타 채팅 프로세스·인덱싱은 변경하지 않는다. max4 후보는 엔진 전용 격리 시험에만 사용하며 운영 엔진은 교체하지 않는다. |
| 보존 조건 | 기존 운영/다른 시험 프로세스를 임의 중지하지 않는다. 운영 엔진·원본 checkout은 보존하고 후보 출력은 엔진 worktree에만 둔다. 운영 부하와 추가 시험의 사용 가능 여부를 구분한다. |
| 현재 확인 범위 | 엔진13:29 사전점검 실패·GPU0/9회 NOT_RUN 및 기존 max4 raw batch1/2/4 PASS 유지. A/B 운영 진단의 FPS/CPU와30초 회복을 원시 카운터에서 재계산했다. 호스트13:03:28 부팅·운영AI13:04:48 시작으로12시 실행과 구분하며 재부팅 원인은 미확인. 메모리 부족/과열 근거 없음·네트워크 오류 증가0이나 네트워크 원인을 완전히 배제하지 않는다. [공통 근거](../planning/evidence/engine_batch_gpu_20261023.json). |
| 개발도구 수정 담당 | A/B 조치 완료: _config.db auto_watch/auto_index=false, Codex 해당 MCP index_repository 비노출/CBM_WORKERS=1, 단일flock·CPU1개/nice19/I/O idle helper.14:13:56 config/mcpServer/reload params={}는 loaded threads의 MCP 설정 refresh 예약이며 특정 서버만 대상으로 한 호출이 아니다. 다른 MCP별 재연결·지연 영향 미확인; 명시적 interrupt/kill/stop/앱·운영 재시작 요청0. 수동 갱신은 통합 조율 후 cctv-code-index <프로젝트절대경로>만 사용한다. |
| 진단 역할·전달 | A/B 제한 로그/저장지표/kernel·수동route/neighbor 조회 종료·추가예약0 보고 수신. 엔진/통합 중복계측0. 사용자PoE스위치에A·CCTV1대·미니PC연결 확인, 세부전원미확인·미니PC유선IP192.168.0.20 사용자확인(B비활성후보중복가능). camera-offload 추가7파일은기준/후보해시대조 후한정개발범위배정, 최종CPU시험은보호준비READY/새시간구간대기. 새RTSP/운영변경/재시작/인덱싱0 유지.  camera-offload 자체 기록: 운영RTSP/설정변경 미발견,14:20/14:23 CPU모형시험 있음·인과미확정. 15:34 READY/614해시대조 완료, CPU전용시간은 현재 예약행 참조. ROI에도 저장지표 회복/새배정없음 안내를 등록했다(수신확인 전). |



- 오케스트레이션에 종료·정리 및15분 인계 요청을 대기열로 전달했다: `01a1668f-e21d-7193-be0e-059fb49a42f4`.
- 엔진 배치 채팅에 대기 상태와 시작 조건을 회신했다: `01a16690-47c8-7483-a9af-13a603ceecb4`.
- 두 접수 ID는 메시지 대기열 등록 근거이며 담당자의 수신·동의나 장비 반환 완료를 뜻하지 않는다. 종료 확인을 받기 전에는 GPU 사용 가능 시각을 추정하지 않는다. 사용이15분을 넘길 경우 자동 연장하지 않고 현재 작업/프로세스 상태를 통합 담당에 알려 순서를 다시 조율한다.

11:26 KST 엔진 채팅 인계 당시: 전용 경로 `/home/sawwave/.codex/worktrees/multicamera-engine-batch/CCTV-project`, 후보 `data/runtime/engine_batch_20261023/batch4_candidate.json`, 원본 해시/운영 상태 근거 `data/runtime/engine_batch_20261023/gpu_validation_baseline.json`의 존재를 확인했다. baseline 기록 시각은 `2026-10-23T02:02:47.014606+00:00`이며 최신 실시간 장비 상태로 간주하지 않는다. 운영 engine/ONNX/config SHA256 불변·빌드/추론0·소스/인계 패치 변경 없음은 요청자 보고다. ROI 채팅에도 실행/예정 시험과15분 인계 가능 여부를 대기열로 질의했다. 접수 ID는 `01a16695-2d3c-79c3-939c-934699c47e0d`다. 종료 확인 전 기존 P6.2/I56 판정과 완료 수는 유지한다.


11:32 KST 착수 전 확인 이력: 앱 상태 조회 지연으로 Codex의 해당 세션 로컬 기록에서 담당자의 최신 사용자 표시 메시지만 읽었다. A/B 응답 시각 `2026-10-23T02:32:05.490Z`, ROI 인계 응답 `2026-10-23T02:30:48.558Z` 및 최종 요약 `2026-10-23T02:32:04.805Z`. 현재 진행 중 시험을 통합 담당이 중지하지 않았으며, 당시에는 엔진 시작 허용 전 상태였다.


실제 반환 근거: A/B 담당의 `2026-10-23T02:38:38.191Z` commentary(확인 기준11:38:24 KST, 메시지 `msg_061a94f74c3c8f9a016adac8aace048191ace0e6593a4f09fb`)에서 시험 AI·CPU 부하 정리 및 반환을 명시했다. 통합 담당이11:53:24까지 엔진 사용을 배정했고, 엔진 담당의 직접 메시지로 같은 구간 수락·max4 빌드 착수를 확인했다. 이 기록은 후보 빌드/검증 합격이나 운영 반영을 뜻하지 않는다.

ROI 상세 인계 수신 — `2026-10-23T11:46:24+09:00`: 담당자의 호스트 `/proc`·Docker 읽기 전용 확인 기준 시각은 **2026-10-23 11:29:51 KST**다. 자체 GPU/카메라 시험 프로세스·컨테이너·ffmpeg/tegrastats 등 변환/수집기는 모두0이며 정리 대상이 없다. **ROI 작업의 반환 가능 확인으로 한정하며, 호스트 전체 독점 사용이나 A/B 종료를 대신 확인한 것이 아니다.** 운영·타 채팅 프로세스 변경0이다.

- 예정 범위: 전용 `two-camera-baseline` worktree에서1080p24·카메라별12/8/6Hz 독립2채널30조건. `two_camera_cadence_20261023.json`의 상태는 `PREPARED_BLOCKED_NOT_RUN`, 검증은 pytest18 PASS·Ruff PASS·Compose30개/격리 서비스60개다. 영상 변환·GPU 재생·성능·낙상 누락률은 미측정이며 공유batch2·실물2카메라·전체 부가 모델 검증으로 해석하지 않는다.
- 근거: `/home/sawwave/.codex/worktrees/two-camera-baseline/CCTV-project/docs/planning/TWO_CAMERA_BASELINE_HANDOFF_20261023.md`와 같은 worktree의 `docs/planning/evidence/two_camera_cadence_20261023.json`을 읽어 대조했다. 기준 HEAD `fefa4ab04ee2ca26bbc0ee433381e4b7648265e7`, 결과 커밋 없음·기존 미커밋 변경 보존은 담당 인계다.
- ROI는 엔진 작업 종료·프로세스 정리·반환 안내와 통합 담당의 다음 순서 확정까지 새 GPU/카메라/영상 변환 부하 시험·수집기를 보류하고, 이후에도 운영 사전점검을 통과해야 한다. **11:53:24 경과만으로 자동 재개하지 않는다.** 이번 기록에서 ROI 제품 변경을 원본에 통합하거나 실제 성능 합격으로 표시하지 않았다.

엔진 최종 반환: **2026-10-23T11:51:15.128141+09:00**. 최종 manifest는 `BUILT_BATCH_VALIDATED_NOT_DEPLOYED` / `PASS_BATCH_1_2_4_QUALITY_NOT_EVALUATED`로 갱신돼 대기 상태 불일치가 해소됐다. [공통 GPU 결과](../planning/evidence/engine_batch_gpu_20261023.json)·[상세 인계](../planning/handoffs/engine_batch_20261023.md). 최초 코드 패치는 재적용하지 않았으며 원본의 기존 변경/index를 유지했다.

### 기존 변경 보존과 경로 검증 — 2026-10-23

- 원본과 세 worktree의 변경 파일·미추적 개발 파일·Git index·staged/unstaged binary patch·기준 커밋을 `data/runtime/codex_parallel_separation_20261023_iqg6wycu/`에 보존했다. 디렉터리 권한은 `0700`이며 비밀값을 로그나 문서에 출력하지 않았다.
- 사본 내 파일 내용 재검증 및 복사 전후 상태 일치: 원본 765경로·자동 전환 121경로·엔진 배치 16경로·ROI 17경로, 약58MB. 이 숫자는 기록 시점의 변경·인계 자료 범위이며 이후 작업까지 고정한 것이 아니다.
- 원본 `data/`와 인수인계 자료의 미추적 대용량 1,269경로는 이동·삭제하지 않고 원래 위치에 유지했다. 사본의 `manifest.json`에 미복사 목록을 기록했다. 운영 DB·비밀 설정·모델/영상 전체를 백업한 것은 아니다.
- 원본 index SHA-256은 `805b110d7fe793abd2732c104080488b798cc73eda2404066ffd4f10a411fd6b`로 유지했다. 각 worktree는 별도 index와 서로 다른 `codex/` 브랜치를 사용한다. 기존 staged/unstaged 변경을 commit·stash·reset하지 않았다.
- 세 worktree의 최신 `AGENTS.md`·배정 안내 일치를 확인하고, 각 worktree의 `.gitignore`에도 루트 지침 포함 예외를 반영했다. 수정 전 문서는 사본의 `coordination-before/`, 해시 대조는 `coordination-updates.json`에 보존했다. 공통 지침 변경은 개발 작업의 제품 수정과 구분한다.
- 기존 채팅 세 개에 Codex CLI의 `queue`로 전용 경로 사용·원본 쓰기 금지·인계 요청을 전달했다. ROI와 엔진 배치의 확인 응답을 받았으며 오케스트레이션의 진행 중 시험은 유지했다. 메시지를 대기열에 넣은 것을 모든 시험의 종료로 해석하지 않는다.
- Git index 네 개의 경로 분리·기준 커밋/index 보존·원본 개발 파일 유지·최신 지침 일치·대표 수정 파일의 독립 inode·문서 검사 모두 통과했다. 상세 결과는 사본의 `verification.json`이다. 사본 작성 후 엔진 배치 채팅이 갱신한 세 인계 파일은 `late-worker-updates/`에 추가 보존하고 이전 사본도 유지했다.
- 복구 자료의 `RECOVERY.md`와 각 `manifest.json`을 기준으로 별도 복구 폴더에서 먼저 검증한다. 과거 HEAD와 현재 작업판의 차이가 커서 개발 worktree의 전체 diff를 원본에 일괄 적용하지 않는다. 엔진 배치 작업처럼 복사 기준 manifest와 작업 전용 패치를 사용한다.

### 채팅을 추가하거나 작업을 이어갈 때

- 기본은 통합 담당 1개와 개발 작업 2개부터 시작한다. 채팅 수 자체를 제한하기보다 동시에 수정하는 범위와 장비 사용을 조율한다. 별도 작업은 별도 worktree, 같은 작업의 후속 질문은 기존 채팅·worktree를 사용한다.
- 통합 담당은 새 작업을 위 배정표에 추가하고 `채팅 / 작업·WBS / 실제 경로·브랜치·기준 커밋 / 담당 파일·시험 / 장비·공유 자원 / 상태`를 기록한다. 작업자는 다른 checkout의 공통 표를 직접 고치지 않고 배정·완료 정보를 전달한다. 통합 담당은 필요할 때 안내 문서를 각 worktree에 전달하며 기존 작업자의 문서 변경은 덮어쓰지 않는다.
- 시작할 때 원본 폴더의 다른 작성자, 파일뿐 아니라 공통 인터페이스의 중복도 확인한다. 배정이 없는 독립 작업은 요청에서 범위를 정해 진행하되, 중복되거나 담당이 불명확한 부분만 읽기 전용으로 분석하며 통합 담당과 조정한다.
- 상태는 `분리 준비 → 작업 중 → 통합 대기 → 통합 완료`로 구분하고 막힌 이유는 해당 상태에 덧붙인다. 개발 채팅의 시험 통과만으로 공통 진행판을 완료 처리하지 않는다.
- 장비 시험은 한 번에 하나씩 진행한다. 통합 후 관련 시험·기록 갱신과 변경 보존을 확인한 worktree만 정리한다. Codex 관리 worktree는 앱의 보관 기능을 우선 사용하고, 채팅 중지·보관과 원격 시험 프로세스 종료는 별도로 확인한다.

새 채팅의 첫 요청 예시:

```text
이 프로젝트의 AGENTS.md와 docs/tooling/CODEX_SKILL_ADOPTION.md의 병렬 작업 규칙을 적용해 주세요.
작업: <이번에 진행할 한 가지 작업>
현재 작업 경로·기준 커밋·미커밋 상태와 다른 채팅의 담당 범위를 확인하고, 수정할 파일·시험·사용 장비를 먼저 알려주세요. 독립 개발은 별도 worktree에서 진행하고, 겹치는 부분은 읽기 전용 분석을 이어가며 조정해 주세요. 결과는 통합 담당이 반영할 수 있도록 정리해 주세요.
```

종료 인계에는 `작업 경로 / 기준·결과 커밋 또는 미커밋 상태 / 변경 파일 / 검증 성공·실패·미실행 / 운영 반영 여부 / 실행 중 프로세스 / 남은 일`을 포함한다. 통합 담당은 결과를 하나씩 반영하고 공통 WBS·PROGRESS·STATUS·진행판을 갱신한다. 새 작업이 없으면 채팅을 계속 실행하거나 새 기능을 임의로 추가하지 않는다.

### 실제 전환 순서

1. 아직 원본에서 수정 중인 개발 채팅에는 아래 문구를 전달하고 파일 수정이 멈춘 것을 확인한다. 이미 전용 worktree를 사용하는 채팅은 해당 경로와 변경 보존을 확인해 재사용한다. 채팅 중지가 이미 실행된 원격 프로세스나 컨테이너를 종료했다는 뜻은 아니므로 시험 상태와 담당도 인계한다.
2. 기존 staged/unstaged/미추적 상태를 보존하고 최신 소스의 기준을 확정한다. 비밀값·모델·영상·운영 데이터는 일괄 커밋하지 않는다. 현재 누적 변경 전체를 임의로 커밋하거나 다른 채팅의 index를 바꾸지 않는다.
3. 작업별 worktree를 생성하거나 앱의 `Hand off`를 사용한다. 현재 작업 브랜치/기준 커밋을 명시하고 새 경로에서 필요한 미커밋 소스와 시험이 빠지지 않았는지 대조한다. 도구의 `create_worktree`는 미커밋 변경을 자동 복사하지 않는다.
4. 작업마다 수정 가능 경로와 공유 자원을 정한다. 새 worktree의 루트 `AGENTS.md`가 최신인지 확인하고, 규칙 반영 전 커밋에서 만들었다면 별도로 전달해 기존 지침과 대조한 뒤 읽힌다. 모델은 읽기 전용 사용 여부를 확인하고 시험 출력·임시 DB·MQTT client ID와 topic·포트·컨테이너 이름은 분리한다. 실장비 성능 시험은 한 번에 하나씩 실행한다.
5. 검증된 변경만 통합 담당이 하나씩 반영하고 관련 시험 및 기존 진행 기록을 갱신한다. 다른 채팅의 작업 폴더를 직접 수정하지 않는다.

기존 채팅에 전달할 문구:

```text
병렬 작업 폴더를 분리할 예정입니다. 현재 명령이 안전한 지점에 도달하면 새 파일 수정·Git 변경·추가 실장비 시험을 멈추고, 수정 파일·미커밋 상태·진행 중 프로세스/컨테이너·사용 장비·남은 작업을 요약해 주세요. 운영 서비스를 임의로 중지하거나 기존 변경을 stash/reset하지 마세요. 새 작업 경로와 담당 범위를 전달받으면 AGENTS.md를 다시 읽고 재개하세요.
```

[공식 Worktree 안내](https://learn.chatgpt.com/docs/environments/git-worktrees)는 앱의 Worktree 선택과 `Hand off`, 로컬 변경 반영 방식을 설명한다. 앱 UI와 개별 생성 도구, 로컬과 원격 환경의 복사 동작이 같다고 가정하지 않고 생성 결과를 확인한다. 지침 추가만으로 실행 중 채팅의 경로가 바뀌거나 파일 잠금이 생기지는 않는다.

이번 작업은 개발 채팅의 worktree 분리 확인·변경 보존·공통 지침 전달이다. 제품 코드·배포·WBS 완료 상태는 변경하지 않았으며, 개발 결과의 실제 통합과 장비 시험은 각 인계의 남은 작업으로 유지한다.

2026-10-23 엔진 배치 사전점검 통합: [검토·검증·잔여](../planning/handoffs/engine_batch_20261023.md). 통합 직전 파일·패치·Git index와 검증 로그는 `data/runtime/engine_batch_integration_20261023_os2za4gp/`에 보존했다. GPU 빌드/실측은 진행 중인 오케스트레이션 시험과 순서를 조율하며, 개발 worktree는 max4 후보와 미커밋 작업이 남아 있으므로 유지한다.
