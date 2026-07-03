# 빠른 시작

## 결론

- PC 개발·기능 확인은 Python/OpenCV 경로 또는 기본 `docker-compose.yml`을 사용합니다.
- Jetson 운영은 `docker-compose.jetson.yml`과 `.env.jetson`을 함께 사용합니다.
- 낙상 보조 모델의 `confirm` 모드는 현장 shadow 데이터 검수 전에는 활성화하지 않는 것을 권장합니다.

전체 운영 스택을 처음부터 모두 올리기보다, 아래 순서로 환경과 핵심 서비스부터 확인하는 방식이 장애 원인을 찾기 쉽습니다.

## 1. 공통 준비

```bash
git clone https://github.com/dih5156-lab/CCTV-project.git
cd CCTV-project
cp cameras.example.json cameras.json
```

`cameras.json`에는 실제 카메라 주소를 입력합니다. RTSP 계정과 비밀번호가 포함될 수 있으므로 저장소에 커밋하지 않습니다.

## 2. PC에서 Python으로 실행

Python 3.10 이상을 기준으로 합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/ai.txt
python main.py --cameras cameras.json --display
```

Windows PowerShell에서는 가상환경을 다음과 같이 활성화합니다.

```powershell
.venv\Scripts\Activate.ps1
python main.py --cameras cameras.json --display
```

모델 파일과 카메라 연결이 준비되지 않았다면 먼저 테스트를 실행해 코드 상태만 확인할 수 있습니다.

```bash
.venv/bin/python -m pytest -q
```

## 3. PC 또는 서버에서 Docker로 실행

```bash
cp .env.example .env
docker compose --env-file .env config --quiet
docker compose --env-file .env up -d --build
docker compose --env-file .env ps
```

처음에는 API와 Action Layer만 선택 실행할 수도 있습니다.

```bash
docker compose --env-file .env up -d \
  edgex-mqtt-broker cctv-alert-api cctv-action-layer cctv-public-api public-demo-ui
```

주요 확인 주소:

- 시연 UI: `http://127.0.0.1:7000`
- Public API 문서: `http://127.0.0.1:9000/docs`
- Stream API 상태: `http://127.0.0.1:8769/health`

기본 Compose의 일부 이미지는 ARM64에서 실행되지 않을 수 있습니다. Jetson에서는 다음 전용 구성을 사용합니다.

## 4. Jetson에서 실행

JetPack/L4T, NVIDIA Container Runtime, Docker Compose가 준비되어 있어야 합니다.

```bash
cp .env.jetson.example .env.jetson
```

`.env.jetson`에서 API 키, 내부 서비스 토큰, MQTT 인증값, 장비 주소를 운영 환경에 맞게 설정합니다. 비밀값은 예제 파일이나 문서에 기록하지 않습니다.

Jetson Compose는 외부 Docker volume을 사용합니다. 필요한 volume 이름과 초기 준비 방법은 [배포 환경변수 문서](DEPLOYMENT_ENVIRONMENT_VARIABLES.md)의 Jetson 배포 절차를 따릅니다.

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml config --quiet
docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --build
docker compose --env-file .env.jetson -f docker-compose.jetson.yml ps
```

기본 상태 확인:

```bash
curl -fsS http://127.0.0.1:9000/health
curl -fsS http://127.0.0.1:8765/health
curl -fsS http://127.0.0.1:8769/health
docker logs --tail 120 cctv-ai-engine
```

현장 투입 전 표준 점검:

```bash
RUNTIME_ENV_FILE=.env.jetson ./scripts/ops/run_operation_check.sh
./scripts/ops/run_deepstream_stability_watch.sh 60 60
```

첫 번째 인자는 관찰 시간(분), 두 번째 인자는 확인 간격(초)입니다. 운영 투입 전에는 1시간 확인 후 8시간, 24시간 순으로 늘리는 것을 권장합니다.

## 5. 종료와 다음 확인

PC/서버 기본 Compose:

```bash
docker compose --env-file .env down
```

Jetson Compose:

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml down
```

외부 volume과 그 안의 모델·DB·로그는 `down`만으로 삭제되지 않습니다.

- 환경변수 전체 기준: [배포 환경변수](DEPLOYMENT_ENVIRONMENT_VARIABLES.md)
- 장애 확인과 복구: [운영 Runbook](OPERATIONS_RUNBOOK.md)
- 현장 점검 순서: [운영 체크리스트](OPERATION_CHECKLIST.md)
- 서비스 구조: [프로젝트 개요](../modules/PROJECT_OVERVIEW.md)
- 현재 변경 및 검증 상태: [2026-07-03 변경 검증 요약](../reviews/CHANGESET_SUMMARY_2026-07-03.md)
