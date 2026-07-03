# 코드 디렉터리 이동 계획과 현재 상태

이 문서는 코드 디렉터리 재배치 계획과 현재까지 적용된 범위를 구분해 기록합니다.

## 현재 적용 상태

| 항목 | 상태 | 현재 기준 |
|---|---|---|
| `app/` 패키지 생성 | 완료 | `app/main.py`, `app/run_external_ingest.py` 존재 |
| 루트 CLI wrapper | 완료 | `main.py`, `run_external_ingest.py`가 `app.*` 호출 |
| `runners/` 이동 | 미진행 | Compose와 Dockerfile이 여전히 루트 `runners/` 사용 |
| `src/` 패키지 이동·rename | 미진행 | 코드와 테스트가 `src.*` import 사용 |
| `parser-python/` 이동 | 미진행 | 독립 서비스 구조 유지 |

따라서 Phase A는 일부만 적용되었습니다. 현재 문서의 목표 구조는 확정된 배포 구조가 아니라 후속 리팩터링 후보입니다.

## 목표

- 실행 진입점과 Docker 경로를 깨지 않고 코드 구조를 단계적으로 정리한다.
- `src` import 의존성을 한 번에 뒤엎지 않고, 무중단에 가까운 순서로 이동한다.
- `parser-python` 같은 별도 서비스는 주 애플리케이션과 분리된 단위로 유지한다.

## 현재 구조에서 확인된 제약

### 런타임 진입점

- 루트 CLI: `main.py`
- 서비스 엔트리포인트: `runners/run_public_api.py`, `runners/run_action_bridge.py`, `runners/run_alert_api.py`, `runners/run_edgex_adapter.py`, `runners/run_sensor_rule_bridge.py`, `runners/run_kuiper_rules.py`
- 별도 서비스: `parser-python/main.py`

### 강한 경로 의존성

- 애플리케이션 코드는 대부분 `from src...` import 를 사용한다.
- 테스트도 대다수가 `src.*` import 를 직접 사용한다.
- Docker/Compose 는 `./src -> /app/src`, `./runners -> /app/runners` bind mount 를 전제로 한다.
- `Dockerfile`의 `action` target 역시 `src`, `runners`, `kuiper` 를 `/app` 아래에 복사한다.
- 여러 runner 는 프로젝트 루트를 `sys.path` 에 넣고 `src.*` 를 import 한다.

따라서 첫 차수에서 `src` 이름 자체를 바꾸는 이동은 영향 범위가 너무 넓다.
안전한 순서는 디렉터리 묶음을 먼저 정리하고, import 루트 rename 은 마지막 차수로 미루는 것이다.

## 권장 목표 구조

```text
app/
├── cctv/                 # 현재 src 내용의 최종 목적지
├── runners/              # 현재 runners
├── main.py               # 현재 main.py 대체 진입점
└── services/
    └── parser_python/    # 현재 parser-python 의 후보 위치

config/                   # 유지
data/                     # 유지
deploy/                   # 유지
docs/                     # 유지
models/                   # 유지
scripts/                  # 유지
tests/                    # 유지
web/                      # 유지
```

단, 실제 이동은 아래처럼 단계적으로 진행한다.

## 차수별 이동 전략

### 1차: 컨테이너/런타임에 안전한 래핑 단계

목표:

- 디렉터리를 논리적으로 묶되 `src` import 와 `/app/src` 경로는 유지한다.
- 가장 작은 수정으로 이후 대규모 이동을 준비한다.

권장 작업:

1. `app/` 디렉터리를 새로 만들고 아래만 우선 이동한다.
2. `main.py` -> `app/main.py`
3. `runners/` -> `app/runners/`
4. 루트에는 호환용 `main.py`, `runners/` shim 또는 얇은 래퍼를 잠시 유지한다.
5. Compose/Dockerfile 는 먼저 `app/runners` 를 읽도록 바꾸되, `src` 는 그대로 둔다.

이 단계에서 유지할 것:

- `src/`
- `tests/` 의 `from src...`
- `/app/src` mount

이유:

- 러너와 엔트리포인트는 개수가 적고 수정 위치가 명확하다.
- 반면 `src` rename 은 import, 테스트, uvicorn app path, Docker mount 전체를 건드린다.

### 2차: 앱 코드 루트 이동 단계

목표:

- `src/` 를 `app/cctv/` 같은 명시적 패키지명으로 이동한다.

권장 작업:

1. `src/` -> `app/cctv/`
2. 애플리케이션 import 를 `from src...` 에서 `from app.cctv...` 또는 새 패키지명 기준으로 변경
3. 테스트 import 전면 교체
4. `uvicorn.run("src.api.app:app")` -> 새 모듈 경로로 변경
5. Compose volume `./src -> /app/src` 제거 또는 `./app/cctv -> /app/app/cctv` 로 변경
6. Dockerfile COPY 경로 변경

이 단계는 실제 코드 이동 작업의 핵심이며, 별도 차수로 분리해야 한다.

### 3차: 별도 서비스 정리 단계

목표:

- `parser-python/` 을 저장소 공용 구조 안으로 편입할지, 독립 서비스로 유지할지 결정한다.

권장 판단:

- 현재는 별도 서비스 성격이 강하므로 즉시 합치기보다 유지 권장
- 최종 후보 경로는 `services/parser-python/` 또는 `app/services/parser_python/`
- 단, 이 이동은 parser 전용 requirements, 테스트, 배포 스크립트가 함께 바뀌므로 마지막에 진행

## 다음 차수에서 먼저 검토할 후보

우선순위 상:

1. `runners/`
2. runner를 참조하는 문서/Compose/Dockerfile
3. 루트 `main.py` wrapper 제거 가능 시점

이번 차수에서 건드리지 말아야 할 것:

1. `src/`
2. `tests/` 의 대량 import
3. `parser-python/`
4. `scripts/` 전체 경로 체계

## 영향도 표

### `main.py` 이동 영향

- README 실행 예시
- 로컬 실행 명령
- systemd 또는 배포 스크립트가 루트 `main.py` 를 직접 호출하는지 추가 확인 필요

### `runners/` 이동 영향

- `docker-compose.yml`
- `docker-compose.jetson.yml`
- `Dockerfile`의 `action` target
- README 의 `python runners/...` 예시

### `src/` 이동 영향

- 거의 모든 테스트
- `main.py`
- 모든 runner
- FastAPI app import 문자열
- Docker COPY / volume / PYTHONPATH

### `parser-python/` 이동 영향

- parser 전용 테스트
- readiness/check 스크립트
- `collect_ignore_glob`
- parser 전용 requirements 와 운영 문서

## 실제 이동 순서 제안

### Phase A

1. `app/` 생성
2. `app/main.py`, `app/runners/` 배치
3. 루트 래퍼 유지
4. compose/Dockerfile/README 를 새 runner 경로 기준으로 교체
5. 테스트 및 서비스 기동 확인

### Phase B

1. `src/` -> 새 패키지명 이동
2. import 일괄 수정
3. container mount/COPY/PYTHONPATH 수정
4. 테스트 일괄 수정
5. API/Jetson compose 재검증

### Phase C

1. `parser-python/` 정리 여부 확정
2. 독립 서비스 유지 또는 `services/` 아래 편입
3. parser 테스트/문서/스크립트 정리

## 검증 명령 초안

### 최소 검증

```bash
pytest -q tests/test_public_api.py tests/test_action_bridge.py tests/test_zone_api.py
python main.py --help
python runners/run_public_api.py --help
sudo docker compose --env-file .env.jetson -f docker-compose.jetson.yml -p edgex-jetson config --services
```

### 이동 후 서비스 검증

```bash
curl -fsS http://localhost:9000/api/v1/health
curl -fsS http://localhost:9000/api/v1/readiness
curl -fsS http://localhost:7000/public-api/api/v1/health
```

## 롤백 원칙

- 차수마다 경로 래퍼를 잠시 유지해 한 번에 전부 되돌릴 수 있게 한다.
- `src` rename 전까지는 루트 import 체계와 container mount 를 보존한다.
- 실제 rename 은 테스트와 compose 검증이 모두 통과한 뒤에만 진행한다.

## 결론

이 저장소에서는 `src` 를 먼저 옮기면 안 된다.
가장 안전한 시작점은 `main.py` 와 `runners/` 를 `app/` 아래로 재배치하고 호환 래퍼를 유지하는 것이다.
그 다음 차수에서만 `src` 패키지 rename 과 Docker 경로 변경을 진행하는 것이 적절하다.
